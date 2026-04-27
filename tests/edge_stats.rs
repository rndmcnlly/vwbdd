//! Edge distribution statistics for the vwbdd arena.
//!
//! Interleaved-encoding-specific: this test reconstructs per-node byte
//! lengths from the arena by scanning two LEB128 varints. Gated off when a
//! non-interleaved encoding backend is active.
//!
//! Question: in our current relative-only encoding, how are child references
//! distributed? The 2023 notebook that inspired this design observed a
//! *bimodal* distribution: children of a node at position ii were usually
//! either very low (small absolute index) OR very close to ii (small relative
//! delta). That insight motivated a 1-bit-mode hybrid encoding.
//!
//! We want to know: does that bimodality hold in vwbdd on the mult relation?
//! Or is "late-only" (pure back-reference) a good-enough heuristic, meaning
//! the hybrid wouldn't actually buy much?
//!
//! What we measure, per live arena:
//!   1. For every non-terminal child edge, record:
//!       - abs_idx: child's construction-order position [0, N-1]
//!       - rel_delta: current_offset - child_offset (bytes)
//!       - parent's construction-order position ii
//!   2. Build:
//!       - histogram of (abs_idx / ii)  -- is it bimodal at 0 and 1?
//!       - actual LEB128 byte cost with current scheme (sum of leb_len(rel_delta+2))
//!       - hypothetical best-case: sum of min(leb_len(abs_idx), leb_len(rel_delta))
//!         per edge, + 1 bit of mode per edge (amortize as ~0.125 bytes)
//!       - fraction of edges where abs_idx < rel_delta (in byte-cost terms)
//!
//! This is a pure measurement. It does not change the encoding.

use vwbdd::{Manager, Ref};

mod mult_shared;
use mult_shared::build_mult;

fn leb_len(mut x: u64) -> usize {
    let mut n = 1;
    while x >= 0x80 {
        x >>= 7;
        n += 1;
    }
    n
}

fn leb_len_u128(mut x: u128) -> usize {
    let mut n = 1;
    while x >= 0x80 {
        x >>= 7;
        n += 1;
    }
    n
}

/// Walk the live arena in construction order.
/// Returns a Vec<(offset, node)> where index i in the Vec is the ith node
/// constructed (i.e. its abs_idx).
fn walk_arena(m: &Manager) -> Vec<(u64, vwbdd::Node)> {
    let mut out = Vec::new();
    let buf_len = m.buf_len();
    let mut off: u64 = 0;
    while (off as usize) < buf_len {
        // Decode node at off. We need a raw buf view; use decode_node via Ref.
        let r = Ref::Node(off);
        let node = m.decode_node(r).unwrap();
        // Determine length consumed: re-encode and measure, OR decode_node_at
        // returns the length. We don't expose decode_node_at, but we can
        // reconstruct length: leb_len(var) + leb_len_u128(interleaved children).
        // That duplicates logic; instead, compute length from the two LEB128s
        // by consulting bytes directly.
        let len = {
            let slice = m.arena_slice(off as usize);
            let mut p = 0;
            // scan first LEB128
            while slice[p] & 0x80 != 0 { p += 1; }
            p += 1;
            // scan second LEB128
            while slice[p] & 0x80 != 0 { p += 1; }
            p += 1;
            p
        };
        out.push((off, node));
        off += len as u64;
    }
    out
}

struct EdgeStats {
    /// Per-edge records.
    edges: Vec<Edge>,
    num_nodes: usize,
}

#[derive(Debug, Clone, Copy)]
struct Edge {
    parent_idx: usize,
    parent_off: u64,
    child_idx: usize,       // abs_idx of child
    child_off: u64,
    rel_delta: u64,         // parent_off - child_off
    // Byte costs under different encodings:
    cost_rel: usize,        // LEB128(rel_delta + 2)
    cost_abs: usize,        // LEB128(child_idx + 2)
}

fn gather_edges(m: &Manager) -> EdgeStats {
    let nodes = walk_arena(m);
    // offset -> abs_idx
    let mut off_to_idx = std::collections::HashMap::new();
    for (i, (off, _)) in nodes.iter().enumerate() {
        off_to_idx.insert(*off, i);
    }
    let mut edges = Vec::new();
    for (i, (off, node)) in nodes.iter().enumerate() {
        for child in [node.lo, node.hi] {
            if let Ref::Node(child_off) = child {
                let child_idx = *off_to_idx.get(&child_off).expect("child in arena");
                let rel_delta = off - child_off;
                let cost_rel = leb_len(rel_delta + 2);
                let cost_abs = leb_len(child_idx as u64 + 2);
                edges.push(Edge {
                    parent_idx: i,
                    parent_off: *off,
                    child_idx,
                    child_off,
                    rel_delta,
                    cost_rel,
                    cost_abs,
                });
            }
        }
    }
    EdgeStats { edges, num_nodes: nodes.len() }
}

fn analyze(k: u32, stats: &EdgeStats) {
    let n = stats.num_nodes;
    let m = stats.edges.len();
    eprintln!("=== k={} : {} nodes, {} edges ===", k, n, m);

    // Bimodality: bucket child_idx / parent_idx into 10 bins.
    // Only for edges where parent_idx > 0.
    let mut bins = [0usize; 10];
    let mut bimodal_edges = 0;
    for e in &stats.edges {
        if e.parent_idx == 0 { continue; }
        let ratio = e.child_idx as f64 / e.parent_idx as f64;
        let b = (ratio * 10.0).min(9.0) as usize;
        bins[b] += 1;
        bimodal_edges += 1;
    }
    eprintln!("  child_idx/parent_idx distribution (fraction of edges):");
    for (i, &count) in bins.iter().enumerate() {
        let lo = i as f64 / 10.0;
        let hi = (i + 1) as f64 / 10.0;
        let frac = count as f64 / bimodal_edges.max(1) as f64;
        let bar = "#".repeat((frac * 60.0) as usize);
        eprintln!("    [{:.1},{:.1}) {:>7} {:>5.1}% {}", lo, hi, count, frac * 100.0, bar);
    }

    // Cost comparison.
    let total_rel: usize = stats.edges.iter().map(|e| e.cost_rel).sum();
    let total_abs: usize = stats.edges.iter().map(|e| e.cost_abs).sum();
    let total_hybrid: usize = stats.edges.iter().map(|e| e.cost_rel.min(e.cost_abs)).sum();
    // Mode bit: 1 bit per edge, amortized over edges.
    // To embed the mode, we'd likely pack it into the LSB of the code, which
    // means each code has to double in max value and LEB128 uses one more bit
    // -- not a full byte, but shifts the boundary. Approximate the cost as
    // ceil(total_edges / 8) extra bytes (one bit per edge).
    let mode_bits_bytes = (m + 7) / 8;

    eprintln!("  edge byte cost (LEB128-based):");
    eprintln!("    current (rel-only):      {:>8} B  ({:.3} B/edge)",
        total_rel, total_rel as f64 / m as f64);
    eprintln!("    hypothetical (abs-only): {:>8} B  ({:.3} B/edge)",
        total_abs, total_abs as f64 / m as f64);
    eprintln!("    hybrid min(rel,abs):     {:>8} B  ({:.3} B/edge)  +{} B mode bits",
        total_hybrid, total_hybrid as f64 / m as f64, mode_bits_bytes);
    eprintln!("    hybrid w/ mode:          {:>8} B  ({:.3} B/edge)",
        total_hybrid + mode_bits_bytes,
        (total_hybrid + mode_bits_bytes) as f64 / m as f64);

    let savings = total_rel as isize - (total_hybrid + mode_bits_bytes) as isize;
    let savings_pct = savings as f64 / total_rel as f64 * 100.0;
    eprintln!("    savings vs current:      {:>8} B  ({:.1}%)",
        savings, savings_pct);

    // Frac where abs cheaper.
    let abs_cheaper = stats.edges.iter().filter(|e| e.cost_abs < e.cost_rel).count();
    let rel_cheaper = stats.edges.iter().filter(|e| e.cost_rel < e.cost_abs).count();
    let tied = m - abs_cheaper - rel_cheaper;
    eprintln!("  per-edge winner (byte cost):");
    eprintln!("    abs strictly cheaper:    {:>7} ({:.1}%)", abs_cheaper, abs_cheaper as f64 / m as f64 * 100.0);
    eprintln!("    rel strictly cheaper:    {:>7} ({:.1}%)", rel_cheaper, rel_cheaper as f64 / m as f64 * 100.0);
    eprintln!("    tied:                    {:>7} ({:.1}%)", tied, tied as f64 / m as f64 * 100.0);

    // Secondary: is the "rel-cheap" band really at the high end of ii?
    // i.e. for edges where rel wins, where is child_idx relative to parent_idx?
    let rel_win_ratios: Vec<f64> = stats.edges.iter()
        .filter(|e| e.cost_rel < e.cost_abs && e.parent_idx > 0)
        .map(|e| e.child_idx as f64 / e.parent_idx as f64)
        .collect();
    let abs_win_ratios: Vec<f64> = stats.edges.iter()
        .filter(|e| e.cost_abs < e.cost_rel && e.parent_idx > 0)
        .map(|e| e.child_idx as f64 / e.parent_idx as f64)
        .collect();
    fn mean(xs: &[f64]) -> f64 {
        if xs.is_empty() { return f64::NAN; }
        xs.iter().sum::<f64>() / xs.len() as f64
    }
    eprintln!("  for edges where rel wins: mean child_idx/parent_idx = {:.3} (n={})",
        mean(&rel_win_ratios), rel_win_ratios.len());
    eprintln!("  for edges where abs wins: mean child_idx/parent_idx = {:.3} (n={})",
        mean(&abs_win_ratios), abs_win_ratios.len());

    eprintln!();
}

#[cfg(not(any(feature = "encoding-per-field", feature = "encoding-fixed")))]
#[test]
fn edge_distribution_sweep() {
    for k in 2..=8u32 {
        let handle = std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn(move || {
                let mut vw = Manager::new();
                let root = build_mult(&mut vw, k);
                let remapped = vw.gc(&[root]);
                let _ = remapped;
                let stats = gather_edges(&vw);
                analyze(k, &stats);
            })
            .unwrap();
        handle.join().unwrap();
    }
}
