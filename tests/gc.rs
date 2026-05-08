//! GC tests. After gc(roots), the manager should contain only nodes
//! reachable from the returned remapped roots. Node counts per root must be
//! unchanged. Subsequent operations on remapped roots must work and produce
//! the same answers as they would have on the original.

use vwbdd::{Manager, Ref};

fn var(m: &mut Manager, i: u32) -> Ref {
    let f = m.r#false();
    let t = m.r#true();
    m.make_node(i, f, t)
}

fn reachable(m: &Manager, r: Ref) -> usize {
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

#[test]
fn gc_with_no_roots_empties_manager() {
    let mut m = Manager::new();
    let _ = m.new_var();
    let _ = m.new_var();
    let x = var(&mut m, 0);
    let y = var(&mut m, 1);
    let _ = m.and(x, y);
    assert!(m.num_nodes() > 0);
    let _ = m.drop_roots(&[]);
    assert_eq!(m.num_nodes(), 0);
    assert_eq!(m.buf_len(), 0);
}

#[test]
fn gc_preserves_single_root_count() {
    let mut m = Manager::new();
    let _ = m.new_var();
    let _ = m.new_var();
    let x = var(&mut m, 0);
    let y = var(&mut m, 1);
    let and = m.and(x, y);
    let before = reachable(&m, and);
    let remapped = m.drop_roots(&[and]);
    assert_eq!(remapped.len(), 1);
    let after = reachable(&m, remapped[0]);
    assert_eq!(before, after);
    // No scaffolding left: manager nodes = reachable internal from root.
    assert_eq!(m.num_nodes(), after);
}

#[test]
fn gc_terminal_roots_yield_terminal_remap() {
    let mut m = Manager::new();
    let t = m.r#true();
    let f = m.r#false();
    let remapped = m.drop_roots(&[t, f]);
    assert_eq!(remapped[0], t);
    assert_eq!(remapped[1], f);
    assert_eq!(m.num_nodes(), 0);
}

#[test]
fn gc_post_ops_still_work() {
    // Build something, GC, keep building, check results consistent.
    let mut m = Manager::new();
    let _ = m.new_var();
    let _ = m.new_var();
    let _ = m.new_var();
    let x = var(&mut m, 0);
    let y = var(&mut m, 1);
    let z = var(&mut m, 2);
    let xy = m.and(x, y);
    // Compute both orderings of a 3-way AND before GC.
    let xyz_before = m.and(xy, z);
    let reachable_xyz_before = reachable(&m, xyz_before);

    // GC keeping only xy; drop xyz.
    let remapped = m.drop_roots(&[xy]);
    let xy_new = remapped[0];

    // Re-declare variables and rebuild (vars are still registered!).
    // Wait: after gc, vars are still registered but the variable-constructed
    // BDDs x, y, z are not alive. Rebuild them.
    let z_new = var(&mut m, 2);
    let xyz_after = m.and(xy_new, z_new);
    let reachable_xyz_after = reachable(&m, xyz_after);

    assert_eq!(reachable_xyz_before, reachable_xyz_after);
}

#[test]
fn gc_massively_shrinks_after_threshold_workload() {
    // Build the N=10, K=5 threshold function and GC; reachable set should be
    // much smaller than total nodes built.
    const N: u32 = 10;
    const K: usize = 5;
    let mut m = Manager::new();
    for _ in 0..N {
        m.new_var();
    }
    let vx: Vec<_> = (0..N).map(|i| var(&mut m, i)).collect();
    let mut acc = m.r#false();

    fn k_combos(n: usize, k: usize) -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        let mut idx: Vec<usize> = (0..k).collect();
        if k == 0 || k > n {
            return out;
        }
        loop {
            out.push(idx.clone());
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

    for combo in k_combos(N as usize, K) {
        let mut cube = vx[combo[0]];
        for &i in &combo[1..] {
            cube = m.and(cube, vx[i]);
        }
        acc = m.or(acc, cube);
    }

    let before_total = m.num_nodes();
    let reach = reachable(&m, acc);
    let remapped = m.drop_roots(&[acc]);
    let after_total = m.num_nodes();

    assert_eq!(after_total, reach, "GC should leave exactly reachable nodes");
    assert!(
        after_total < before_total / 10,
        "expected >10x shrinkage, got {} -> {}",
        before_total,
        after_total
    );
    // Root still works.
    assert_eq!(reachable(&m, remapped[0]), after_total);
}
