//! Byte-per-node measurements on known-correct BDDs (cross-checked against
//! OxiDD for node count). The hypothesis under test: variable-width encoding
//! achieves amortized bytes/node well below OxiDD's fixed 16 B/node at
//! realistic problem sizes.

use oxidd::bdd::{new_manager as oxidd_new_manager, BDDFunction};
use oxidd::{BooleanFunction, Function, Manager as _, ManagerRef};

use vwbdd::{Manager, Ref};

fn var(m: &mut Manager, i: u32) -> Ref {
    let f = m.r#false();
    let t = m.r#true();
    m.make_node(i, f, t)
}

/// Count reachable nodes (without terminals, matching our own num_nodes
/// semantics, i.e. internal BDD nodes only).
fn reachable_internal(m: &Manager, r: Ref) -> usize {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![r];
    while let Some(r) = stack.pop() {
        if let Ref::Node(off) = r {
            if !seen.insert(off) {
                continue;
            }
            let n = m.decode_node(r).unwrap();
            stack.push(n.lo);
            stack.push(n.hi);
        }
    }
    seen.len()
}

/// Emit a report line for a workload. Also cross-checks node count vs OxiDD.
fn report(name: &str, vw: &Manager, root: Ref, oxidd_count: usize) {
    let vw_internal = reachable_internal(vw, root);
    let vw_oxidd_style = vw_internal + 2; // oxidd counts terminals
    assert_eq!(
        vw_oxidd_style, oxidd_count,
        "[{}] node count mismatch: vwbdd={} oxidd={}",
        name, vw_oxidd_style, oxidd_count
    );

    let total_nodes_in_mgr = vw.num_nodes();
    let buf_len = vw.buf_len();
    let bytes_per_node = buf_len as f64 / total_nodes_in_mgr.max(1) as f64;
    eprintln!(
        "[{:28}] reachable(internal)={:>7} mgr nodes={:>7} bytes={:>9} B/node={:5.2} (oxidd fixed=16)",
        name, vw_internal, total_nodes_in_mgr, buf_len, bytes_per_node,
    );
}

#[test]
fn compression_wide_and_chain() {
    // n-way AND: x0 & x1 & ... & x_{n-1}. Reduced BDD has exactly n internal
    // nodes (a chain). A clean best-case for the encoding: nearby children.
    const N: u32 = 16;

    let oxm = oxidd_new_manager(8192, 1024, 2);
    let ox: Vec<_> = oxm.with_manager_exclusive(|mgr| {
        let names: Vec<String> = (0..N).map(|i| format!("x{}", i)).collect();
        mgr.add_named_vars(names.iter().map(|s| s.as_str())).unwrap();
        (0..N).map(|i| BDDFunction::var(mgr, i).unwrap()).collect()
    });
    let mut ox_and = ox[0].clone();
    for xi in &ox[1..] {
        ox_and = ox_and.and(xi).unwrap();
    }

    let mut vw = Manager::new();
    for _ in 0..N {
        vw.new_var();
    }
    let vx: Vec<_> = (0..N).map(|i| var(&mut vw, i)).collect();
    let mut vand = vx[0];
    for &xi in &vx[1..] {
        vand = vw.and(vand, xi);
    }

    report("AND chain n=16", &vw, vand, ox_and.node_count());
}

#[test]
fn compression_parity_chain() {
    // n-way XOR. Exactly 2n-1 internal nodes in the reduced BDD.
    const N: u32 = 16;

    let oxm = oxidd_new_manager(8192, 1024, 2);
    let ox: Vec<_> = oxm.with_manager_exclusive(|mgr| {
        let names: Vec<String> = (0..N).map(|i| format!("x{}", i)).collect();
        mgr.add_named_vars(names.iter().map(|s| s.as_str())).unwrap();
        (0..N).map(|i| BDDFunction::var(mgr, i).unwrap()).collect()
    });
    let mut ox_xor = ox[0].clone();
    for xi in &ox[1..] {
        ox_xor = ox_xor.xor(xi).unwrap();
    }

    let mut vw = Manager::new();
    for _ in 0..N {
        vw.new_var();
    }
    let vx: Vec<_> = (0..N).map(|i| var(&mut vw, i)).collect();
    let mut vxor = vx[0];
    for &xi in &vx[1..] {
        vxor = vw.xor(vxor, xi);
    }

    report("XOR parity n=16", &vw, vxor, ox_xor.node_count());
}

#[test]
fn compression_threshold() {
    // Threshold: at least K of N variables true. N=10, K=5.
    const N: u32 = 10;
    const K: u32 = 5;

    let oxm = oxidd_new_manager(65536, 4096, 2);
    let oxv: Vec<_> = oxm.with_manager_exclusive(|mgr| {
        let names: Vec<String> = (0..N).map(|i| format!("x{}", i)).collect();
        mgr.add_named_vars(names.iter().map(|s| s.as_str())).unwrap();
        (0..N).map(|i| BDDFunction::var(mgr, i).unwrap()).collect()
    });

    let false_ = oxm.with_manager_exclusive(|mgr| BDDFunction::f(mgr));
    let mut ox_acc = false_;
    for combo in k_combinations(N as usize, K as usize) {
        let mut cube = oxv[combo[0]].clone();
        for &i in &combo[1..] {
            cube = cube.and(&oxv[i]).unwrap();
        }
        ox_acc = ox_acc.or(&cube).unwrap();
    }

    let mut vw = Manager::new();
    for _ in 0..N {
        vw.new_var();
    }
    let vx: Vec<_> = (0..N).map(|i| var(&mut vw, i)).collect();
    let mut vacc = vw.r#false();
    for combo in k_combinations(N as usize, K as usize) {
        let mut cube = vx[combo[0]];
        for &i in &combo[1..] {
            cube = vw.and(cube, vx[i]);
        }
        vacc = vw.or(vacc, cube);
    }

    report("Threshold N=10 K=5", &vw, vacc, ox_acc.node_count());
}

fn k_combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut idx = (0..k).collect::<Vec<_>>();
    if k == 0 || k > n {
        return out;
    }
    loop {
        out.push(idx.clone());
        // Find rightmost index that can advance.
        let mut i = k;
        while i > 0 {
            i -= 1;
            if idx[i] < n - k + i {
                idx[i] += 1;
                for j in i + 1..k {
                    idx[j] = idx[j - 1] + 1;
                }
                break;
            }
            if i == 0 {
                return out;
            }
        }
    }
}
