//! Shared bitblasted x*y=z relation builders, used by every test that
//! compares vwbdd against OxiDD on the mult workload.
//!
//! Both sides use the same ripple-carry adder structure so the workload is
//! bitblaster-agnostic: any timing difference is pure engine perf.

#![allow(dead_code)] // not every test uses every helper

use oxidd::bdd::BDDFunction;
use oxidd::BooleanFunction;

use vwbdd::{Manager, Ref};

// ---------- vwbdd side ----------

pub fn vw_vars(vw: &mut Manager, k: u32) -> (Vec<Ref>, Vec<Ref>, Vec<Ref>) {
    let mk = |vw: &mut Manager| {
        let v = vw.new_var();
        let f = vw.r#false();
        let t = vw.r#true();
        vw.make_node(v, f, t)
    };
    let x: Vec<Ref> = (0..k).map(|_| mk(vw)).collect();
    let y: Vec<Ref> = (0..k).map(|_| mk(vw)).collect();
    let z: Vec<Ref> = (0..2 * k).map(|_| mk(vw)).collect();
    (x, y, z)
}

/// Variable layout for the **truncated** k-bit relation `(x*y) mod 2^k = z`:
/// three k-bit uints interleaved in declaration order (matches iota's
/// default variable order: x0,x1,..,y0,y1,..,z0,z1,..).
pub fn vw_vars_trunc(vw: &mut Manager, k: u32) -> (Vec<Ref>, Vec<Ref>, Vec<Ref>) {
    let mk = |vw: &mut Manager| {
        let v = vw.new_var();
        let f = vw.r#false();
        let t = vw.r#true();
        vw.make_node(v, f, t)
    };
    let x: Vec<Ref> = (0..k).map(|_| mk(vw)).collect();
    let y: Vec<Ref> = (0..k).map(|_| mk(vw)).collect();
    let z: Vec<Ref> = (0..k).map(|_| mk(vw)).collect();
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

pub fn vw_add(vw: &mut Manager, a: &[Ref], b: &[Ref]) -> Vec<Ref> {
    debug_assert_eq!(a.len(), b.len());
    let mut out = Vec::with_capacity(a.len());
    let mut c = vw.r#false();
    for i in 0..a.len() {
        let (s, co) = vw_full_adder(vw, a[i], b[i], c);
        out.push(s);
        c = co;
    }
    out
}

pub fn vw_mult(vw: &mut Manager, x: &[Ref], y: &[Ref]) -> Vec<Ref> {
    let k = x.len();
    let n = 2 * k;
    let f = vw.r#false();
    let mut acc: Vec<Ref> = vec![f; n];
    for j in 0..k {
        let mut pp: Vec<Ref> = vec![f; n];
        for i in 0..k {
            if i + j < n {
                pp[i + j] = vw.and(x[i], y[j]);
            }
        }
        acc = vw_add(vw, &acc, &pp);
    }
    acc
}

/// Truncated multiply: same shift-and-add structure as [`vw_mult`] but the
/// accumulator is k bits wide instead of 2k, matching iota's `mulBits` and
/// any other "modular" multiplier. Builds `(x * y) mod 2^k`, discarding the
/// high half. Substantially smaller BDDs than the full 2k-bit relation for
/// a given k.
pub fn vw_mult_trunc(vw: &mut Manager, x: &[Ref], y: &[Ref]) -> Vec<Ref> {
    let k = x.len();
    let f = vw.r#false();
    let mut acc: Vec<Ref> = vec![f; k];
    for i in 0..k {
        let mut partial: Vec<Ref> = vec![f; k];
        for j in 0..k {
            if j >= i {
                partial[j] = vw.and(x[j - i], y[i]);
            }
        }
        acc = vw_add(vw, &acc, &partial);
    }
    acc
}

pub fn vw_eq(vw: &mut Manager, p: &[Ref], z: &[Ref]) -> Ref {
    debug_assert_eq!(p.len(), z.len());
    let mut acc = vw.r#true();
    for i in 0..p.len() {
        let d = vw.xor(p[i], z[i]);
        let s = vw.not(d);
        acc = vw.and(acc, s);
    }
    acc
}

/// Build the x*y=z relation end-to-end in vwbdd, returning its root.
pub fn build_mult(vw: &mut Manager, k: u32) -> Ref {
    let (x, y, z) = vw_vars(vw, k);
    let p = vw_mult(vw, &x, &y);
    vw_eq(vw, &p, &z)
}

/// Build the **truncated** relation `(x * y) mod 2^k = z` where all three
/// operands are k bits wide. Matches iota's `ctx.eq(ctx.mul("x","y"), "z")`
/// with same-width x, y, z. Much smaller BDDs than [`build_mult`] at the
/// same k, since the high-order k bits of the product are discarded rather
/// than materialized into the relation.
pub fn build_mult_trunc(vw: &mut Manager, k: u32) -> Ref {
    let (x, y, z) = vw_vars_trunc(vw, k);
    let p = vw_mult_trunc(vw, &x, &y);
    vw_eq(vw, &p, &z)
}

/// Count reachable internal nodes from a root.
pub fn vw_reachable(m: &Manager, r: Ref) -> usize {
    let mut seen = std::collections::HashSet::new();
    let mut stk = vec![r];
    while let Some(r) = stk.pop() {
        if let Ref::Node(o) = r {
            if !seen.insert(o) {
                continue;
            }
            let n = m.decode_node(r).unwrap();
            stk.push(n.lo);
            stk.push(n.hi);
        }
    }
    seen.len()
}

// ---------- OxiDD side ----------

fn ox_full_adder(a: &BDDFunction, b: &BDDFunction, c: &BDDFunction) -> (BDDFunction, BDDFunction) {
    let ab = a.xor(b).unwrap();
    let sum = ab.xor(c).unwrap();
    let a_and_b = a.and(b).unwrap();
    let c_and_ab = c.and(&ab).unwrap();
    (sum, a_and_b.or(&c_and_ab).unwrap())
}

pub fn ox_add(a: &[BDDFunction], b: &[BDDFunction], f: &BDDFunction) -> Vec<BDDFunction> {
    debug_assert_eq!(a.len(), b.len());
    let mut out = Vec::with_capacity(a.len());
    let mut c = f.clone();
    for i in 0..a.len() {
        let (s, co) = ox_full_adder(&a[i], &b[i], &c);
        out.push(s);
        c = co;
    }
    out
}

pub fn ox_mult(x: &[BDDFunction], y: &[BDDFunction], f: &BDDFunction) -> Vec<BDDFunction> {
    let k = x.len();
    let n = 2 * k;
    let mut acc: Vec<BDDFunction> = (0..n).map(|_| f.clone()).collect();
    for j in 0..k {
        let mut pp: Vec<BDDFunction> = (0..n).map(|_| f.clone()).collect();
        for i in 0..k {
            if i + j < n {
                pp[i + j] = x[i].and(&y[j]).unwrap();
            }
        }
        acc = ox_add(&acc, &pp, f);
    }
    acc
}

/// Truncated OxiDD multiply: matches iota's `mulBits`. Accumulator stays
/// k bits wide; high-order bits of the product are discarded.
pub fn ox_mult_trunc(x: &[BDDFunction], y: &[BDDFunction], f: &BDDFunction) -> Vec<BDDFunction> {
    let k = x.len();
    let mut acc: Vec<BDDFunction> = (0..k).map(|_| f.clone()).collect();
    for i in 0..k {
        let mut partial: Vec<BDDFunction> = (0..k).map(|_| f.clone()).collect();
        for j in 0..k {
            if j >= i {
                partial[j] = x[j - i].and(&y[i]).unwrap();
            }
        }
        acc = ox_add(&acc, &partial, f);
    }
    acc
}

pub fn ox_eq(p: &[BDDFunction], z: &[BDDFunction], t: &BDDFunction) -> BDDFunction {
    debug_assert_eq!(p.len(), z.len());
    let mut acc = t.clone();
    for i in 0..p.len() {
        let d = p[i].xor(&z[i]).unwrap();
        let s = d.not().unwrap();
        acc = acc.and(&s).unwrap();
    }
    acc
}
