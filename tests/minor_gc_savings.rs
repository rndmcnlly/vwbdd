//! Benchmark: what does the clean-bytes invariant cost at the wire?
//!
//! Under the invariant (see `src/slab.rs` module docs), every [`Diff`]
//! crossing a public boundary is function-canonical: `diff_since`
//! runs [`Manager::gc`] (tail-only variant) internally before
//! extracting bytes.
//!
//! This benchmark reports, across a sweep of trunc-mult base sizes
//! and a few extend shapes, the ratio of:
//!
//!   - "pre-cleaned tail bytes" (what the invariant ships)
//!   - "raw tail bytes" (what would have been shipped if we skipped
//!     the GC — a dirty-bytes counterfactual we can no longer produce
//!     via the public API, but can still measure by reaching into the
//!     internals via `apply_diff`-independent paths).
//!
//! Run explicitly:
//!
//! ```sh
//! cargo test --release --test minor_gc_savings -- --ignored --nocapture
//! ```

mod mult_shared;

use mult_shared::build_mult_trunc;
use vwbdd::{Manager, Ref};

/// A simple extend: conjoin the base root with `x[0]`.
fn ops_simple(m: &mut Manager, base_roots: &[Ref]) -> Vec<Ref> {
    let f = m.r#false();
    let t = m.r#true();
    let x0 = m.make_node(0, f, t);
    vec![m.and(base_roots[0], x0)]
}

/// A mixed extend: new root is a real non-terminal, ops also spawn scratch.
fn ops_mixed(m: &mut Manager, base_roots: &[Ref]) -> Vec<Ref> {
    let r = base_roots[0];
    let num_vars = m.num_vars();
    let k = num_vars / 3;
    let f = m.r#false();
    let t = m.r#true();
    let x0 = m.make_node(0, f, t);
    let y0 = m.make_node(k, f, t);
    let parity = m.xor(x0, y0);
    vec![m.and(r, parity)]
}

/// A fat extend: tautology/contradiction queries that fold to terminals
/// after traversing large subtrees. All intermediate nodes are scratch.
fn ops_fat(m: &mut Manager, base_roots: &[Ref]) -> Vec<Ref> {
    let r = base_roots[0];
    let nr = m.not(r);
    let either = m.or(r, nr);
    let both = m.and(r, nr);
    let x = m.xor(r, nr);
    vec![either, both, x]
}

fn measure_k(k: u32, ops_name: &str, ops: fn(&mut Manager, &[Ref]) -> Vec<Ref>) {
    let mut server = Manager::new();
    let base_root = build_mult_trunc(&mut server, k);
    let base_nodes = server.num_nodes();
    let base_slab = server.slab_for(&[base_root]);

    // Measure the clean (invariant-enforced) path: what extend_slab
    // actually ships.
    let clean_diff = Manager::extend_slab(&base_slab, |m, br| ops(m, br));
    let clean_bytes = clean_diff.tail.len();

    // Measure the would-be-dirty counterfactual: same ops, but we
    // extract the raw tail ourselves without pre-GC. We have to do
    // this manually since `diff_since` always cleans.
    let (dirty_bytes, dirty_nodes, clean_nodes): (usize, usize, usize) = {
        let mut m = Manager::new();
        let var_count = 3 * k;
        for _ in 0..var_count {
            m.new_var();
        }
        let base_roots = m.ingest_slab(&base_slab);
        let base_len = m.buf_len() as u64;
        let base_node_count = m.num_nodes();
        let _new_roots = ops(&mut m, &base_roots);
        let dirty_tail_bytes = m.buf_len() - base_len as usize;
        let dirty_tail_nodes = m.num_nodes() - base_node_count;
        // Now clean it.
        let cleaned = m.gc(base_len, &_new_roots);
        let _ = cleaned;
        let clean_tail_nodes = m.num_nodes() - base_node_count;
        (dirty_tail_bytes, dirty_tail_nodes, clean_tail_nodes)
    };

    let byte_ratio = if clean_bytes == 0 {
        if dirty_bytes == 0 { 1.0 } else { f64::INFINITY }
    } else {
        dirty_bytes as f64 / clean_bytes as f64
    };
    let node_ratio = if clean_nodes == 0 {
        if dirty_nodes == 0 { 1.0 } else { f64::INFINITY }
    } else {
        dirty_nodes as f64 / clean_nodes as f64
    };

    println!(
        "k={:2}  ops={:<8}  base={:>8} nodes   \
         tail_nodes {:>6} → {:>6} ({:.2}×)   \
         tail_bytes {:>8} → {:>8} ({:.2}×)",
        k, ops_name, base_nodes,
        dirty_nodes, clean_nodes, node_ratio,
        dirty_bytes, clean_bytes, byte_ratio,
    );
}

#[test]
#[ignore] // run with --ignored --nocapture
fn minor_gc_savings_sweep() {
    println!("\n=== what the clean-bytes invariant saves at the wire ===");
    println!("base = (x * y) mod 2^k == z\n");
    println!("shows: tail_bytes BEFORE vs AFTER tail-GC (what extend_slab ships)");
    println!("  simple: and(base, x[0])                    — 1 live node, ~no scratch");
    println!("  mixed:  and(base, xor(x[0], y[0]))         — small live result + ite scratch");
    println!("  fat:    or/and/xor(base, not(base))        — all scratch, results fold to terminals\n");
    for k in [5, 7, 9, 11] {
        measure_k(k, "simple", ops_simple);
        measure_k(k, "mixed", ops_mixed);
        measure_k(k, "fat", ops_fat);
        println!();
    }
}
