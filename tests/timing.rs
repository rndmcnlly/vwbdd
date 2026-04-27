//! Wall-clock timing: vwbdd vs OxiDD on the mult relation at matching k.
//! Uses identical bitblasting on both sides (ripple adders + bitwise equality)
//! so we're measuring engine perf, not bitblaster quality.

use std::time::Instant;

use oxidd::bdd::{new_manager as oxidd_new_manager, BDDFunction};
use oxidd::{BooleanFunction, Function, Manager as _, ManagerRef};

use vwbdd::{Manager, Ref};

fn vw_vars(vw: &mut Manager, k: u32) -> (Vec<Ref>, Vec<Ref>, Vec<Ref>) {
    let nx: Vec<Ref> = (0..k).map(|_| {
        let v = vw.new_var(); let f = vw.r#false(); let t = vw.r#true();
        vw.make_node(v, f, t)
    }).collect();
    let ny: Vec<Ref> = (0..k).map(|_| {
        let v = vw.new_var(); let f = vw.r#false(); let t = vw.r#true();
        vw.make_node(v, f, t)
    }).collect();
    let nz: Vec<Ref> = (0..2*k).map(|_| {
        let v = vw.new_var(); let f = vw.r#false(); let t = vw.r#true();
        vw.make_node(v, f, t)
    }).collect();
    (nx, ny, nz)
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
    let k = x.len(); let n = 2*k;
    let f = vw.r#false();
    let mut acc: Vec<Ref> = vec![f; n];
    for j in 0..k {
        let mut pp: Vec<Ref> = vec![f; n];
        for i in 0..k { if i+j < n { pp[i+j] = vw.and(x[i], y[j]); } }
        acc = vw_add(vw, &acc, &pp);
    }
    acc
}
fn ox_mult(x: &[BDDFunction], y: &[BDDFunction], f: &BDDFunction) -> Vec<BDDFunction> {
    let k = x.len(); let n = 2*k;
    let mut acc: Vec<BDDFunction> = (0..n).map(|_| f.clone()).collect();
    for j in 0..k {
        let mut pp: Vec<BDDFunction> = (0..n).map(|_| f.clone()).collect();
        for i in 0..k { if i+j < n { pp[i+j] = x[i].and(&y[j]).unwrap(); } }
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

fn run_vw(k: u32) -> (std::time::Duration, usize) {
    let t0 = Instant::now();
    let mut vw = Manager::new();
    let (x, y, z) = vw_vars(&mut vw, k);
    let p = vw_mult(&mut vw, &x, &y);
    let r = vw_eq(&mut vw, &p, &z);
    let d = t0.elapsed();
    (d, vwbdd_reachable(&vw, r))
}

fn vwbdd_reachable(m: &Manager, r: Ref) -> usize {
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

fn run_oxidd(k: u32) -> (std::time::Duration, usize) {
    let t0 = Instant::now();
    let mref = oxidd_new_manager(1 << 20, 1 << 16, 1);
    let (x, y, z, tt, ff) = mref.with_manager_exclusive(|mgr| {
        let names: Vec<String> = (0..k).map(|i| format!("x{}", i))
            .chain((0..k).map(|i| format!("y{}", i)))
            .chain((0..2*k).map(|i| format!("z{}", i))).collect();
        mgr.add_named_vars(names.iter().map(|s| s.as_str())).unwrap();
        let x: Vec<_> = (0..k).map(|i| BDDFunction::var(mgr, i).unwrap()).collect();
        let y: Vec<_> = (k..2*k).map(|i| BDDFunction::var(mgr, i).unwrap()).collect();
        let z: Vec<_> = (2*k..4*k).map(|i| BDDFunction::var(mgr, i).unwrap()).collect();
        (x, y, z, BDDFunction::t(mgr), BDDFunction::f(mgr))
    });
    let p = ox_mult(&x, &y, &ff);
    let r = ox_eq(&p, &z, &tt);
    let d = t0.elapsed();
    let nc = r.node_count();
    (d, nc)
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
