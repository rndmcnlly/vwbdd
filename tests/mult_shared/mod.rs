//! Shared mult-relation builder, reused by tests/mult.rs and tests/edge_stats.rs.

use vwbdd::{Manager, Ref};

fn vw_vars(vw: &mut Manager, k: u32) -> (Vec<Ref>, Vec<Ref>, Vec<Ref>) {
    let nx: Vec<Ref> = (0..k).map(|_| { let v = vw.new_var(); let f = vw.r#false(); let t = vw.r#true(); vw.make_node(v, f, t) }).collect();
    let ny: Vec<Ref> = (0..k).map(|_| { let v = vw.new_var(); let f = vw.r#false(); let t = vw.r#true(); vw.make_node(v, f, t) }).collect();
    let nz: Vec<Ref> = (0..2 * k).map(|_| { let v = vw.new_var(); let f = vw.r#false(); let t = vw.r#true(); vw.make_node(v, f, t) }).collect();
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

fn vw_add(vw: &mut Manager, a: &[Ref], b: &[Ref]) -> Vec<Ref> {
    assert_eq!(a.len(), b.len());
    let mut out = Vec::with_capacity(a.len());
    let mut c = vw.r#false();
    for i in 0..a.len() {
        let (s, c_out) = vw_full_adder(vw, a[i], b[i], c);
        out.push(s);
        c = c_out;
    }
    out
}

fn vw_mult(vw: &mut Manager, x: &[Ref], y: &[Ref]) -> Vec<Ref> {
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

fn vw_eq(vw: &mut Manager, p: &[Ref], z: &[Ref]) -> Ref {
    assert_eq!(p.len(), z.len());
    let mut acc = vw.r#true();
    for i in 0..p.len() {
        let diff = vw.xor(p[i], z[i]);
        let same = vw.not(diff);
        acc = vw.and(acc, same);
    }
    acc
}

/// Build the relation x*y=z for k-bit x,y into the given manager, returning
/// the root.
pub fn build_mult(vw: &mut Manager, k: u32) -> Ref {
    let (x, y, z) = vw_vars(vw, k);
    let p = vw_mult(vw, &x, &y);
    vw_eq(vw, &p, &z)
}
