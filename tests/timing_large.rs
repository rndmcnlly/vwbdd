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

use vwbdd::{Manager, Ref};

// 30 s budget per the user's request.
const BUDGET_MS: f64 = 30_000.0;

fn vw_vars(vw: &mut Manager, k: u32) -> (Vec<Ref>, Vec<Ref>, Vec<Ref>) {
    let mk = |vw: &mut Manager| {
        let v = vw.new_var();
        let f = vw.r#false();
        let t = vw.r#true();
        vw.make_node(v, f, t)
    };
    let x: Vec<_> = (0..k).map(|_| mk(vw)).collect();
    let y: Vec<_> = (0..k).map(|_| mk(vw)).collect();
    let z: Vec<_> = (0..2 * k).map(|_| mk(vw)).collect();
    (x, y, z)
}

fn vw_full_adder(vw: &mut Manager, a: Ref, b: Ref, c: Ref) -> (Ref, Ref) {
    let ab = vw.xor(a, b);
    let sum = vw.xor(ab, c);
    let a_and_b = vw.and(a, b);
    let c_and_ab = vw.and(c, ab);
    let carry = vw.or(a_and_b, c_and_ab);
    (sum, carry)
}
fn ox_full_adder(a: &BDDFunction, b: &BDDFunction, c: &BDDFunction) -> (BDDFunction, BDDFunction) {
    let ab = a.xor(b).unwrap();
    let sum = ab.xor(c).unwrap();
    let a_and_b = a.and(b).unwrap();
    let c_and_ab = c.and(&ab).unwrap();
    (sum, a_and_b.or(&c_and_ab).unwrap())
}

fn vw_add(vw: &mut Manager, a: &[Ref], b: &[Ref]) -> Vec<Ref> {
    let mut out = Vec::with_capacity(a.len());
    let mut c = vw.r#false();
    for i in 0..a.len() {
        let (s, co) = vw_full_adder(vw, a[i], b[i], c);
        out.push(s); c = co;
    }
    out
}
fn ox_add(a: &[BDDFunction], b: &[BDDFunction], f: &BDDFunction) -> Vec<BDDFunction> {
    let mut out = Vec::with_capacity(a.len()); let mut c = f.clone();
    for i in 0..a.len() {
        let (s, co) = ox_full_adder(&a[i], &b[i], &c);
        out.push(s); c = co;
    }
    out
}

fn vw_mult(vw: &mut Manager, x: &[Ref], y: &[Ref]) -> Vec<Ref> {
    let k = x.len(); let n = 2 * k;
    let f = vw.r#false();
    let mut acc: Vec<Ref> = vec![f; n];
    for j in 0..k {
        let mut pp: Vec<Ref> = vec![f; n];
        for i in 0..k { if i + j < n { pp[i + j] = vw.and(x[i], y[j]); } }
        acc = vw_add(vw, &acc, &pp);
    }
    acc
}
fn ox_mult(x: &[BDDFunction], y: &[BDDFunction], f: &BDDFunction) -> Vec<BDDFunction> {
    let k = x.len(); let n = 2 * k;
    let mut acc: Vec<BDDFunction> = (0..n).map(|_| f.clone()).collect();
    for j in 0..k {
        let mut pp: Vec<BDDFunction> = (0..n).map(|_| f.clone()).collect();
        for i in 0..k { if i + j < n { pp[i + j] = x[i].and(&y[j]).unwrap(); } }
        acc = ox_add(&acc, &pp, f);
    }
    acc
}

fn vw_eq(vw: &mut Manager, p: &[Ref], z: &[Ref]) -> Ref {
    let mut acc = vw.r#true();
    for i in 0..p.len() {
        let d = vw.xor(p[i], z[i]);
        let s = vw.not(d);
        acc = vw.and(acc, s);
    }
    acc
}
fn ox_eq(p: &[BDDFunction], z: &[BDDFunction], t: &BDDFunction) -> BDDFunction {
    let mut acc = t.clone();
    for i in 0..p.len() {
        let d = p[i].xor(&z[i]).unwrap();
        let s = d.not().unwrap();
        acc = acc.and(&s).unwrap();
    }
    acc
}

fn reachable(m: &Manager, r: Ref) -> usize {
    let mut seen = std::collections::HashSet::new();
    let mut stk = vec![r];
    while let Some(r) = stk.pop() {
        if let Ref::Node(o) = r {
            if !seen.insert(o) { continue; }
            let n = m.decode_node(r).unwrap();
            stk.push(n.lo); stk.push(n.hi);
        }
    }
    seen.len()
}

struct VwResult {
    dur_ms: f64,
    nodes: usize,
    arena_bytes_per_node: f64,
    total_bytes_per_node: f64,
}

fn run_vw(k: u32) -> VwResult {
    vwbdd::profile::reset();
    let t0 = Instant::now();
    let mut vw = Manager::new();
    let (x, y, z) = vw_vars(&mut vw, k);
    let p = vw_mult(&mut vw, &x, &y);
    let r = vw_eq(&mut vw, &p, &z);
    let nodes = reachable(&vw, r);
    let remap = vw.gc(&[r]);
    let _ = remap;
    let mem = vw.mem_stats();
    let dur_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let snap = vwbdd::profile::snapshot();
    snap.report(&format!("k={} ({})", k, vwbdd::ENCODING_NAME), Some(dur_ms));
    VwResult {
        dur_ms,
        nodes,
        arena_bytes_per_node: mem.arena_bytes_per_node(),
        total_bytes_per_node: mem.total_bytes_per_node(),
    }
}

fn run_ox(k: u32) -> (f64, usize) {
    let t0 = Instant::now();
    // Generous table sizes; mult grows fast.
    let inner_node_cap = 1 << 24;   // 16M slots max
    let apply_cache_cap = 1 << 20;  // 1M entries
    let mref = oxidd_new_manager(inner_node_cap, apply_cache_cap, 1);
    let (x, y, z, tt, ff) = mref.with_manager_exclusive(|mgr| {
        let names: Vec<String> = (0..k).map(|i| format!("x{}", i))
            .chain((0..k).map(|i| format!("y{}", i)))
            .chain((0..2 * k).map(|i| format!("z{}", i))).collect();
        mgr.add_named_vars(names.iter().map(|s| s.as_str())).unwrap();
        let x: Vec<_> = (0..k).map(|i| BDDFunction::var(mgr, i).unwrap()).collect();
        let y: Vec<_> = (k..2 * k).map(|i| BDDFunction::var(mgr, i).unwrap()).collect();
        let z: Vec<_> = (2 * k..4 * k).map(|i| BDDFunction::var(mgr, i).unwrap()).collect();
        (x, y, z, BDDFunction::t(mgr), BDDFunction::f(mgr))
    });
    let p = ox_mult(&x, &y, &ff);
    let r = ox_eq(&p, &z, &tt);
    let nc = r.node_count();
    let dur_ms = t0.elapsed().as_secs_f64() * 1000.0;
    (dur_ms, nc)
}

#[test]
#[ignore] // expensive; run with `cargo test --release --test timing_large -- --ignored --nocapture`
fn timing_sweep_large() {
    eprintln!("# encoding: {}", vwbdd::ENCODING_NAME);
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
        // Hard safety cap.
        if k > 20 { break; }
    }
}
