//! Truncated mult-relation timing, matching iota's demo.
//!
//! iota (oxidd-wasm) reports in-browser timings for `(x*y) mod 2^k = z`
//! where x, y, z are all k bits wide. This differs from the `tests/timing*`
//! harnesses, which build the full 2k-bit relation `x*y=z` with z stretched
//! to 2k bits. The truncated function is drastically smaller (discards the
//! high k bits of the product), so the same k values produce ~1-2 orders
//! of magnitude fewer nodes than the full-precision tests.
//!
//! This file exists so the vwbdd vs. oxidd-wasm vs. oxidd-native
//! comparison for the truncated workload is apples-to-apples.
//! `#[ignore]`d by default: run with `--ignored --nocapture`.

use std::time::Instant;

use oxidd::bdd::{new_manager as oxidd_new_manager, BDDFunction};
use oxidd::{BooleanFunction, Function, Manager as _, ManagerRef};

use vwbdd::Manager;

mod mult_shared;
use mult_shared::{build_mult_trunc, ox_eq, ox_mult_trunc, vw_reachable};

const BUDGET_MS: f64 = 600_000.0;

struct VwResult {
    dur_ms: f64,
    nodes: usize,
    arena_bytes_per_node: f64,
    total_bytes_per_node: f64,
}

fn run_vw(k: u32) -> VwResult {
    let t0 = Instant::now();
    let mut vw = Manager::new();
    let r = build_mult_trunc(&mut vw, k);
    let nodes = vw_reachable(&vw, r);
    let _ = vw.drop_roots(&[r]);
    let mem = vw.mem_stats();
    let dur_ms = t0.elapsed().as_secs_f64() * 1000.0;
    VwResult {
        dur_ms,
        nodes,
        arena_bytes_per_node: mem.arena_bytes_per_node(),
        total_bytes_per_node: mem.total_bytes_per_node(),
    }
}

fn run_ox(k: u32) -> (f64, usize) {
    let t0 = Instant::now();
    // Match oxidd-cli's default: 32 Mi apply-cache entries (~1.3 GB),
    // inner node capacity big enough for the truncated k=17 target
    // (~80M nodes). This lets each engine run at its own preferred
    // config rather than being handicapped by a too-small cache.
    let inner_node_cap = 1 << 27; // 128M slots
    let apply_cache_cap = 32 * 1024 * 1024; // oxidd-cli's default
    let mref = oxidd_new_manager(inner_node_cap, apply_cache_cap, 1);
    let (x, y, z, tt, ff) = mref.with_manager_exclusive(|mgr| {
        // Three k-bit uints, same declaration order as iota: x, y, z.
        let names: Vec<String> = (0..k)
            .map(|i| format!("x{}", i))
            .chain((0..k).map(|i| format!("y{}", i)))
            .chain((0..k).map(|i| format!("z{}", i)))
            .collect();
        mgr.add_named_vars(names.iter().map(|s| s.as_str())).unwrap();
        let x: Vec<_> = (0..k).map(|i| BDDFunction::var(mgr, i).unwrap()).collect();
        let y: Vec<_> = (k..2 * k).map(|i| BDDFunction::var(mgr, i).unwrap()).collect();
        let z: Vec<_> = (2 * k..3 * k).map(|i| BDDFunction::var(mgr, i).unwrap()).collect();
        (x, y, z, BDDFunction::t(mgr), BDDFunction::f(mgr))
    });
    let p = ox_mult_trunc(&x, &y, &ff);
    let r = ox_eq(&p, &z, &tt);
    let dur_ms = t0.elapsed().as_secs_f64() * 1000.0;
    (dur_ms, r.node_count())
}

#[test]
#[ignore] // expensive; run with `--ignored --nocapture`
fn mult_trunc_sweep() {
    eprintln!("Truncated mult relation: (x*y) mod 2^k = z (matches iota)");
    eprintln!(
        "{:>3} {:>11} {:>12} {:>12} {:>6}  {:>8} {:>10}",
        "k", "nodes", "vwbdd (ms)", "oxidd (ms)", "ratio", "arena B/n", "total B/n"
    );

    // iota's demo sweeps k = 7, 10, 12..17. Matching that lineup lets us
    // line rows up directly with the in-browser numbers. k=17 takes ~3 min
    // on an M3 Max; the full sweep is ~4 min total.
    let ks: &[u32] = &[7, 10, 12, 13, 14, 15, 16, 17];

    for &k in ks {
        let handle = std::thread::Builder::new()
            .stack_size(1024 << 20)
            .spawn(move || {
                let vw = run_vw(k);
                let (ox_ms, ox_nodes) = run_ox(k);
                (vw, ox_ms, ox_nodes)
            })
            .unwrap();
        let (vw, ox_ms, ox_nodes) = handle.join().unwrap();

        // Terminals convention: vwbdd excludes, oxidd includes; +2.
        assert_eq!(
            vw.nodes + 2, ox_nodes,
            "node count mismatch at k={}: vwbdd={} oxidd={}",
            k, vw.nodes, ox_nodes
        );

        let ratio = vw.dur_ms / ox_ms;
        eprintln!(
            "{:>3} {:>11} {:>12.1} {:>12.1} {:>5.2}x  {:>8.2} {:>10.2}",
            k, vw.nodes, vw.dur_ms, ox_ms, ratio,
            vw.arena_bytes_per_node, vw.total_bytes_per_node
        );

        if vw.dur_ms > BUDGET_MS {
            eprintln!("(stopping: vwbdd k={} exceeded {:.0}s budget)", k, BUDGET_MS / 1000.0);
            break;
        }
    }
}
