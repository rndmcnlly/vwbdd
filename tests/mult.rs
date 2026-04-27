//! Mult relation x*y=z over k-bit unsigned ints. Bitblasts the multiplier in
//! both engines and compares node counts at each k. Prints bytes/node for
//! vwbdd at each k so we can see the curve.

use oxidd::bdd::{new_manager as oxidd_new_manager, BDDFunction};
use oxidd::{BooleanFunction, Function, Manager as _, ManagerRef};

use vwbdd::Manager;

mod mult_shared;
use mult_shared::{build_mult, ox_eq, ox_mult, vw_reachable};

/// Build mult at bit-width k, in both engines, assert matching node counts.
/// Returns (reachable_internal, oxidd_count, mem_after_gc).
fn run_mult(k: u32) -> (usize, usize, vwbdd::MemStats) {
    // --- OxiDD side ---
    let oxm = oxidd_new_manager(1 << 18, 1 << 14, 1);
    let (oxx, oxy, oxz, ox_true, ox_false) = oxm.with_manager_exclusive(|mgr| {
        let names: Vec<String> = (0..k)
            .map(|i| format!("x{}", i))
            .chain((0..k).map(|i| format!("y{}", i)))
            .chain((0..2 * k).map(|i| format!("z{}", i)))
            .collect();
        mgr.add_named_vars(names.iter().map(|s| s.as_str())).unwrap();
        let oxx: Vec<_> = (0..k).map(|i| BDDFunction::var(mgr, i).unwrap()).collect();
        let oxy: Vec<_> = (k..2 * k).map(|i| BDDFunction::var(mgr, i).unwrap()).collect();
        let oxz: Vec<_> = (2 * k..4 * k).map(|i| BDDFunction::var(mgr, i).unwrap()).collect();
        (oxx, oxy, oxz, BDDFunction::t(mgr), BDDFunction::f(mgr))
    });
    let ox_p = ox_mult(&oxx, &oxy, &ox_false);
    let ox_rel = ox_eq(&ox_p, &oxz, &ox_true);
    let oxidd_count = ox_rel.node_count();

    // --- vwbdd side ---
    let mut vw = Manager::new();
    let vrel = build_mult(&mut vw, k);

    let reachable = vw_reachable(&vw, vrel);
    assert_eq!(
        reachable + 2,
        oxidd_count,
        "node count mismatch at k={}: vwbdd={} oxidd={}",
        k,
        reachable + 2,
        oxidd_count
    );

    // GC to reachable set, report post-GC mem.
    let remapped = vw.gc(&[vrel]);
    let reachable_post = vw_reachable(&vw, remapped[0]);
    assert_eq!(reachable_post, reachable, "GC should preserve reachable count");
    assert_eq!(
        vw.num_nodes(),
        reachable,
        "post-GC manager should contain only reachable nodes"
    );

    (reachable, oxidd_count, vw.mem_stats())
}

#[test]
fn mult_sweep() {
    eprintln!(
        "{:>3} {:>10} {:>5} {:>7} {:>7} {:>11}",
        "k", "reachable", "arena", "uniq", "cache", "total_live"
    );
    eprintln!(
        "{:>3} {:>10} {:>5} {:>7} {:>7} {:>11}",
        "", "", "B/n", "B/n", "B/n", "B/n (w/ cache)"
    );
    for k in 2..=8 {
        // OxiDD's internal apply recurses; we need a big stack.
        let handle = std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn(move || run_mult(k))
            .unwrap();
        let (reachable, oxidd, mem) = handle.join().unwrap();
        let n = reachable.max(1);
        eprintln!(
            "{:>3} {:>10} {:>5.2} {:>7.2} {:>7.2} {:>5.2}+{:.2}={:.2}  (oxidd={})",
            k,
            reachable,
            mem.arena_bytes as f64 / n as f64,
            mem.unique_bytes as f64 / n as f64,
            mem.cache_bytes as f64 / n as f64,
            mem.total_live() as f64 / n as f64,
            mem.cache_bytes as f64 / n as f64,
            mem.total_with_cache() as f64 / n as f64,
            oxidd,
        );
    }
}
