//! Larger-k timing sweep. Starts at k=8 and walks up until vwbdd crosses a
//! wall-clock budget. Also tracks arena B/node, unique table size, and the
//! vwbdd/oxidd ratio to see whether the ratio is stable, improving, or
//! degrading with scale.
//!
//! Runs both engines on each k but in separate threads with large stacks
//! (OxiDD's internal apply recurses deeply).
//!
//! Stopping rule: after a k whose vwbdd wall-clock exceeds BUDGET_MS we
//! don't attempt the next k.

use std::time::Instant;

use oxidd::bdd::{new_manager as oxidd_new_manager, BDDFunction};
use oxidd::{BooleanFunction, Function, Manager as _, ManagerRef};

use vwbdd::Manager;

mod mult_shared;
use mult_shared::{build_mult, ox_eq, ox_mult, vw_reachable};

// 10 min budget; k=11 is ~32 s on the default build, so k=12 is
// likely 2-3 min. Bump when pushing the sweep further.
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
    let r = build_mult(&mut vw, k);
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
    // Match oxidd-cli's default: 32 Mi apply-cache entries (~1.3 GB).
    // Lets oxidd run at its own preferred config.
    let inner_node_cap = 1 << 27; // 128M slots max (k=12 needs ~30M)
    let apply_cache_cap = 32 * 1024 * 1024; // oxidd-cli's default
    let mref = oxidd_new_manager(inner_node_cap, apply_cache_cap, 1);
    let (x, y, z, tt, ff) = mref.with_manager_exclusive(|mgr| {
        let names: Vec<String> = (0..k)
            .map(|i| format!("x{}", i))
            .chain((0..k).map(|i| format!("y{}", i)))
            .chain((0..2 * k).map(|i| format!("z{}", i)))
            .collect();
        mgr.add_named_vars(names.iter().map(|s| s.as_str())).unwrap();
        let x: Vec<_> = (0..k).map(|i| BDDFunction::var(mgr, i).unwrap()).collect();
        let y: Vec<_> = (k..2 * k).map(|i| BDDFunction::var(mgr, i).unwrap()).collect();
        let z: Vec<_> = (2 * k..4 * k).map(|i| BDDFunction::var(mgr, i).unwrap()).collect();
        (x, y, z, BDDFunction::t(mgr), BDDFunction::f(mgr))
    });
    let p = ox_mult(&x, &y, &ff);
    let r = ox_eq(&p, &z, &tt);
    let dur_ms = t0.elapsed().as_secs_f64() * 1000.0;
    (dur_ms, r.node_count())
}

#[test]
#[ignore] // expensive; run with `cargo test --release --test timing_large -- --ignored --nocapture`
fn timing_sweep_large() {
    eprintln!(
        "{:>3} {:>11} {:>12} {:>12} {:>6}  {:>8} {:>10}",
        "k", "nodes", "vwbdd (ms)", "oxidd (ms)", "ratio", "arena B/n", "total B/n"
    );

    let mut k: u32 = 8;
    loop {
        // Each run in its own thread with huge stack — OxiDD recurses deeply.
        let handle = std::thread::Builder::new()
            .stack_size(1024 << 20) // 1 GiB
            .spawn(move || {
                let vw = run_vw(k);
                let (ox_ms, ox_nodes) = run_ox(k);
                (vw, ox_ms, ox_nodes)
            })
            .unwrap();
        let (vw, ox_ms, ox_nodes) = handle.join().unwrap();

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
        k += 1;
        if k > 20 {
            break;
        }
    }
}
