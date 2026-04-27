//! Wall-clock timing: vwbdd vs OxiDD on the mult relation at matching k.
//! Uses identical bitblasting on both sides (ripple adders + bitwise equality)
//! so we're measuring engine perf, not bitblaster quality.

use std::time::Instant;

use oxidd::bdd::{new_manager as oxidd_new_manager, BDDFunction};
use oxidd::{BooleanFunction, Function, Manager as _, ManagerRef};

use vwbdd::Manager;

mod mult_shared;
use mult_shared::{build_mult, ox_eq, ox_mult, vw_reachable};

fn run_vw(k: u32) -> (std::time::Duration, usize) {
    let t0 = Instant::now();
    let mut vw = Manager::new();
    let r = build_mult(&mut vw, k);
    let d = t0.elapsed();
    (d, vw_reachable(&vw, r))
}

fn run_oxidd(k: u32) -> (std::time::Duration, usize) {
    let t0 = Instant::now();
    let mref = oxidd_new_manager(1 << 20, 1 << 16, 1);
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
    let d = t0.elapsed();
    (d, r.node_count())
}

#[test]
fn timing_sweep() {
    eprintln!(
        "{:>3} {:>10} {:>12} {:>12} {:>6}",
        "k", "nodes", "vwbdd (ms)", "oxidd (ms)", "ratio"
    );
    for k in 4..=8 {
        let handle = std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn(move || {
                let (vw_dur, vw_nodes) = run_vw(k);
                let (ox_dur, ox_nodes) = run_oxidd(k);
                (vw_dur, vw_nodes, ox_dur, ox_nodes)
            })
            .unwrap();
        let (vw_dur, vw_nodes, ox_dur, ox_nodes) = handle.join().unwrap();
        assert_eq!(vw_nodes + 2, ox_nodes, "node count mismatch at k={}", k);
        let ratio = vw_dur.as_secs_f64() / ox_dur.as_secs_f64();
        eprintln!(
            "{:>3} {:>10} {:>12.2} {:>12.2} {:>5.2}x",
            k,
            vw_nodes,
            vw_dur.as_secs_f64() * 1000.0,
            ox_dur.as_secs_f64() * 1000.0,
            ratio
        );
    }
}
