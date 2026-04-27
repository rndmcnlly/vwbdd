//! Differential testing: build the same boolean functions in vwbdd and OxiDD,
//! assert that per-function reachable node counts match.
//!
//! OxiDD's BDDFunction::node_count() counts reachable nodes *including* the
//! two terminals (false, true). Our reachable_nodes() matches this convention.

use oxidd::bdd::{new_manager as oxidd_new_manager, BDDFunction};
use oxidd::{BooleanFunction, Function, Manager as _, ManagerRef};

use vwbdd::{Manager, Ref};

/// Helper: get vwbdd's count of distinct reachable nodes (including terminals)
/// from a given root. Matches OxiDD's `node_count` semantics.
fn vw_node_count(m: &Manager, r: Ref) -> usize {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![r];
    let mut terminals_seen = std::collections::HashSet::new();
    while let Some(r) = stack.pop() {
        match r {
            Ref::Terminal(v) => {
                terminals_seen.insert(v);
            }
            Ref::Node(off) => {
                if !seen.insert(off) {
                    continue;
                }
                let n = m.decode_node(r).unwrap();
                stack.push(n.lo);
                stack.push(n.hi);
            }
        }
    }
    seen.len() + terminals_seen.len()
}

fn var(m: &mut Manager, i: u32) -> Ref {
    let f = m.r#false();
    let t = m.r#true();
    m.make_node(i, f, t)
}

#[test]
fn terminals_have_count_1() {
    // OxiDD: ff.node_count() == 1, tt.node_count() == 1.
    let m = Manager::new();
    assert_eq!(vw_node_count(&m, m.r#false()), 1);
    assert_eq!(vw_node_count(&m, m.r#true()), 1);
}

#[test]
fn single_var_has_count_3() {
    // OxiDD: x.node_count() == 3 (x itself + false + true terminals).
    let mut m = Manager::new();
    let _ = m.new_var();
    let x = var(&mut m, 0);
    assert_eq!(vw_node_count(&m, x), 3);
}

#[test]
fn not_and_matches_oxidd() {
    // The OxiDD test: (x0 AND x1).not() has node_count == 4.
    //
    // ~(x0 AND x1) = (x0 -> ~x1) = if x0 then ~x1 else true
    //   root:        var=0, lo=true,  hi=~x1
    //   ~x1:         var=1, lo=true,  hi=false
    //   true, false: terminals
    // = 4 reachable nodes.
    let mut m = Manager::new();
    let _ = m.new_var();
    let _ = m.new_var();
    let x0 = var(&mut m, 0);
    let x1 = var(&mut m, 1);
    let and = m.and(x0, x1);
    let not_and = m.not(and);
    assert_eq!(vw_node_count(&m, not_and), 4);
}

/// Differential: build both engines side by side, assert counts match.
/// OxiDD and vwbdd both need to return the same node_count for
/// per-function reachability. This is the strong oracle.
#[test]
fn diff_basic_ops() {
    let oxm = oxidd_new_manager(1024, 128, 2);
    let (ox0, ox1) = oxm.with_manager_exclusive(|mgr| {
        mgr.add_named_vars(["x0", "x1"]).unwrap();
        (
            BDDFunction::var(mgr, 0).unwrap(),
            BDDFunction::var(mgr, 1).unwrap(),
        )
    });
    let ox_and = ox0.and(&ox1).unwrap();
    let ox_or = ox0.or(&ox1).unwrap();
    let ox_xor = ox0.xor(&ox1).unwrap();
    let ox_not_and = ox_and.not().unwrap();

    let mut vw = Manager::new();
    let _ = vw.new_var();
    let _ = vw.new_var();
    let vx0 = var(&mut vw, 0);
    let vx1 = var(&mut vw, 1);
    let va = vw.and(vx0, vx1);
    let vo = vw.or(vx0, vx1);
    let vxo = vw.xor(vx0, vx1);
    let vna = vw.not(va);

    assert_eq!(vw_node_count(&vw, vx0), ox0.node_count(), "x0");
    assert_eq!(vw_node_count(&vw, vx1), ox1.node_count(), "x1");
    assert_eq!(vw_node_count(&vw, va), ox_and.node_count(), "AND");
    assert_eq!(vw_node_count(&vw, vo), ox_or.node_count(), "OR");
    assert_eq!(vw_node_count(&vw, vxo), ox_xor.node_count(), "XOR");
    assert_eq!(
        vw_node_count(&vw, vna),
        ox_not_and.node_count(),
        "NOT(AND)"
    );
}

/// Build a more complex formula and cross-check reachable node count.
///   f = (x0 AND x1) OR (x2 AND x3) OR (x0 AND x2)
#[test]
fn diff_complex_formula() {
    let oxm = oxidd_new_manager(1024, 128, 2);
    let x: Vec<_> = oxm.with_manager_exclusive(|mgr| {
        mgr.add_named_vars(["x0", "x1", "x2", "x3"]).unwrap();
        (0..4).map(|i| BDDFunction::var(mgr, i).unwrap()).collect()
    });
    let a = x[0].and(&x[1]).unwrap();
    let b = x[2].and(&x[3]).unwrap();
    let c = x[0].and(&x[2]).unwrap();
    let ox_f = a.or(&b).unwrap().or(&c).unwrap();

    let mut vw = Manager::new();
    for _ in 0..4 {
        vw.new_var();
    }
    let vx: Vec<_> = (0..4).map(|i| var(&mut vw, i)).collect();
    let a = vw.and(vx[0], vx[1]);
    let b = vw.and(vx[2], vx[3]);
    let c = vw.and(vx[0], vx[2]);
    let ab = vw.or(a, b);
    let vf = vw.or(ab, c);

    assert_eq!(vw_node_count(&vw, vf), ox_f.node_count());
}

/// Parity function f = x0 XOR x1 XOR x2 XOR x3. Classic BDD workload; node
/// count should be exactly 2n+1 (with terminals) under natural variable order.
#[test]
fn diff_parity() {
    let oxm = oxidd_new_manager(1024, 128, 2);
    let x: Vec<_> = oxm.with_manager_exclusive(|mgr| {
        mgr.add_named_vars(["x0", "x1", "x2", "x3"]).unwrap();
        (0..4).map(|i| BDDFunction::var(mgr, i).unwrap()).collect()
    });
    let mut ox_f = x[0].clone();
    for xi in &x[1..] {
        ox_f = ox_f.xor(xi).unwrap();
    }

    let mut vw = Manager::new();
    for _ in 0..4 {
        vw.new_var();
    }
    let vx: Vec<_> = (0..4).map(|i| var(&mut vw, i)).collect();
    let mut vf = vx[0];
    for &xi in &vx[1..] {
        vf = vw.xor(vf, xi);
    }

    assert_eq!(vw_node_count(&vw, vf), ox_f.node_count());
}
