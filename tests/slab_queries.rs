//! Read-only queries on `Slab`: `support` and `sat_count`.
//!
//! These ops don't need a unique table or construction state: they
//! just decode nodes and walk the DAG. They live on `Slab` so a thin
//! client that only wants to answer "which vars does this depend on?"
//! or "how many satisfying assignments?" doesn't have to rebuild a
//! manager.

use std::collections::BTreeSet;

use vwbdd::{Manager, Ref, Slab};

/// Build vars x0..x(n-1) and return (manager, [x0..x(n-1)]).
fn mgr_with_vars(n: u32) -> (Manager, Vec<Ref>) {
    let mut m = Manager::new();
    let f = m.r#false();
    let t = m.r#true();
    let mut xs = Vec::new();
    for i in 0..n {
        let _ = m.new_var();
        xs.push(m.make_node(i, f, t));
    }
    (m, xs)
}

// --- support ---

#[test]
fn support_of_and_is_both_vars() {
    // (x0 ∧ x1) over 3 declared vars; x2 is unused.
    let (mut m, xs) = mgr_with_vars(3);
    let and = m.and(xs[0], xs[1]);
    let slab = m.slab_for(&[and]);

    let sup = slab.support(slab.roots[0]);
    assert_eq!(sup, BTreeSet::from([0u32, 1]));
}

#[test]
fn support_of_single_var_is_singleton() {
    let (mut m, xs) = mgr_with_vars(4);
    let slab = m.slab_for(&[xs[2]]);
    assert_eq!(slab.support(slab.roots[0]), BTreeSet::from([2u32]));
}

#[test]
fn support_of_terminals_is_empty() {
    let m = Manager::new();
    let t = m.r#true();
    let f = m.r#false();
    let slab_t = Slab::new(vec![], vec![t]);
    let slab_f = Slab::new(vec![], vec![f]);
    assert!(slab_t.support(t).is_empty());
    assert!(slab_f.support(f).is_empty());
}

#[test]
fn support_respects_cancellation() {
    // (x0 ∧ x1) ∨ (x0 ∧ ¬x1) reduces to x0; support is {0}.
    let (mut m, xs) = mgr_with_vars(2);
    let nx1 = m.not(xs[1]);
    let a = m.and(xs[0], xs[1]);
    let b = m.and(xs[0], nx1);
    let phi = m.or(a, b);
    let slab = m.slab_for(&[phi]);
    assert_eq!(slab.support(slab.roots[0]), BTreeSet::from([0u32]));
}

// --- sat_count ---

#[test]
fn sat_count_tautology_and_contradiction() {
    let m = Manager::new();
    let t = m.r#true();
    let f = m.r#false();
    let slab_t = Slab::new(vec![], vec![t]);
    let slab_f = Slab::new(vec![], vec![f]);
    for nvars in 0..6 {
        assert_eq!(slab_t.sat_count(t, nvars), (1u64 << nvars) as f64);
        assert_eq!(slab_f.sat_count(f, nvars), 0.0);
    }
}

#[test]
fn sat_count_invariant_under_unused_vars() {
    // f = x0. Universe grows from 1 to 5 vars; count doubles each time.
    let (mut m, xs) = mgr_with_vars(1);
    let slab = m.slab_for(&[xs[0]]);
    let r = slab.roots[0];
    for n in 1..=5 {
        let expected = (1u64 << (n - 1)) as f64;
        assert_eq!(
            slab.sat_count(r, n),
            expected,
            "sat_count(x0, {}) should be 2^{}",
            n,
            n - 1
        );
    }
}

#[test]
fn sat_count_and_plus_not_equals_full_cube() {
    // #sat(f) + #sat(¬f) = 2^n is the invariant that pins the
    // full-cube semantics. Check it on a non-trivial formula.
    let (mut m, xs) = mgr_with_vars(4);
    let nx3 = m.not(xs[3]);
    let a = m.and(xs[0], xs[1]);
    let b = m.and(xs[2], nx3);
    let phi = m.or(a, b);
    let nphi = m.not(phi);

    let slab = m.slab_for(&[phi, nphi]);
    let (rp, rn) = (slab.roots[0], slab.roots[1]);
    assert_eq!(slab.sat_count(rp, 4) + slab.sat_count(rn, 4), 16.0);
}

#[test]
fn sat_count_interleaving_invariant() {
    // Two relations on 4 bits with the same truth table but different
    // variable orders produce different BDD shapes yet the same count.
    //
    // Relation: exactly two of the four bits are 1. C(4,2) = 6.
    fn exactly_two_of_four(m: &mut Manager, order: [u32; 4]) -> Ref {
        let f = m.r#false();
        let t = m.r#true();
        let x: Vec<Ref> = order.iter().map(|&i| m.make_node(i, f, t)).collect();
        // Build "exactly two of x[0..4] are true" as an OR over the 6
        // pairs-of-indices cubes.
        let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let mut acc = f;
        for (i, j) in pairs {
            // cube: x[i] ∧ x[j] ∧ ¬x[k] ∧ ¬x[l]
            let others: Vec<usize> = (0..4).filter(|k| *k != i && *k != j).collect();
            let mut cube = m.and(x[i], x[j]);
            for k in others {
                let nk = m.not(x[k]);
                cube = m.and(cube, nk);
            }
            acc = m.or(acc, cube);
        }
        acc
    }

    let mut m1 = Manager::new();
    for _ in 0..4 {
        m1.new_var();
    }
    let phi1 = exactly_two_of_four(&mut m1, [0, 1, 2, 3]);
    let slab1 = m1.slab_for(&[phi1]);

    let mut m2 = Manager::new();
    for _ in 0..4 {
        m2.new_var();
    }
    // Different order: interleave high/low.
    let phi2 = exactly_two_of_four(&mut m2, [0, 2, 1, 3]);
    let slab2 = m2.slab_for(&[phi2]);

    assert_eq!(slab1.sat_count(slab1.roots[0], 4), 6.0);
    assert_eq!(slab2.sat_count(slab2.roots[0], 4), 6.0);
}

#[test]
fn sat_count_xor_chain_is_half_cube() {
    // x0 XOR x1 XOR x2 is satisfied by exactly half of {0,1}^3.
    let (mut m, xs) = mgr_with_vars(3);
    let a = m.xor(xs[0], xs[1]);
    let phi = m.xor(a, xs[2]);
    let slab = m.slab_for(&[phi]);
    assert_eq!(slab.sat_count(slab.roots[0], 3), 4.0);
    // And over a larger universe, still half.
    assert_eq!(slab.sat_count(slab.roots[0], 6), 32.0);
}
