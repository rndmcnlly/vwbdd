//! Profiling target: truncated mult at k=15.
//!
//! `cargo instruments -t time --release --example mult_trunc_k15`
//!
//! k=15 sits past the apply-cache-thrashing threshold and well into
//! DRAM-bound territory (~10M reachable nodes, ~14 s wall on an M3 Max)
//! but finishes fast enough for an interactive profiling loop. Use
//! this to answer 'where does vwbdd's time actually go at scale?'.
//!
//! The workload is the truncated mult relation `(x*y) mod 2^k = z`
//! over three k-bit uints, matching the iota demo and the numbers
//! recorded in VWBDD.md §3.4. No OxiDD here: we want a clean single-
//! engine profile, not a head-to-head.

use vwbdd::{Manager, Ref};

fn vw_vars_trunc(m: &mut Manager, k: u32) -> (Vec<Ref>, Vec<Ref>, Vec<Ref>) {
    let f = m.r#false();
    let t = m.r#true();
    let mut mk = |i: u32| {
        let _ = m.new_var();
        m.make_node(i, f, t)
    };
    let x: Vec<Ref> = (0..k).map(&mut mk).collect();
    let y: Vec<Ref> = (k..2 * k).map(&mut mk).collect();
    let z: Vec<Ref> = (2 * k..3 * k).map(&mut mk).collect();
    (x, y, z)
}

fn vw_add(m: &mut Manager, a: &[Ref], b: &[Ref]) -> Vec<Ref> {
    // Ripple-carry adder, k bits wide, same-width output (truncating).
    let f = m.r#false();
    let mut carry = f;
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        // sum = a ^ b ^ carry
        let ab = m.xor(a[i], b[i]);
        let s = m.xor(ab, carry);
        // new_carry = (a & b) | (carry & (a ^ b))
        let and_ab = m.and(a[i], b[i]);
        let and_c_ab = m.and(carry, ab);
        carry = m.or(and_ab, and_c_ab);
        out.push(s);
    }
    out
}

fn vw_mult_trunc(m: &mut Manager, x: &[Ref], y: &[Ref]) -> Vec<Ref> {
    // Shift-and-add, truncating at k bits.
    let k = x.len();
    let f = m.r#false();
    let mut acc: Vec<Ref> = vec![f; k];
    for j in 0..k {
        // partial_j[i] = x[i-j] & y[j] if i >= j else 0, truncated to k bits.
        let mut partial: Vec<Ref> = vec![f; k];
        for i in j..k {
            partial[i] = m.and(x[i - j], y[j]);
        }
        acc = vw_add(m, &acc, &partial);
    }
    acc
}

fn vw_eq(m: &mut Manager, p: &[Ref], z: &[Ref]) -> Ref {
    // Conjunction of bitwise equalities: AND_i (p[i] <=> z[i]).
    let t = m.r#true();
    let mut eq = t;
    for i in 0..p.len() {
        let xi = m.xor(p[i], z[i]);
        let eq_bit = m.not(xi);
        eq = m.and(eq, eq_bit);
    }
    eq
}

fn main() {
    let k: u32 = 15;
    let start = std::time::Instant::now();

    let mut m = Manager::new();
    let (x, y, z) = vw_vars_trunc(&mut m, k);
    let p = vw_mult_trunc(&mut m, &x, &y);
    let r = vw_eq(&mut m, &p, &z);

    let build_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mem = m.mem_stats();

    // Count reachable nodes by walking from the root.
    fn count_reachable(m: &Manager, r: Ref) -> usize {
        use std::collections::HashSet;
        let mut seen: HashSet<u64> = HashSet::new();
        let mut stack = Vec::new();
        if let Ref::Node(o) = r {
            stack.push(o);
        }
        while let Some(o) = stack.pop() {
            if !seen.insert(o) {
                continue;
            }
            if let Some(n) = m.decode_node(Ref::Node(o)) {
                if let Ref::Node(lo) = n.lo {
                    stack.push(lo);
                }
                if let Ref::Node(hi) = n.hi {
                    stack.push(hi);
                }
            }
        }
        seen.len()
    }
    let nodes = count_reachable(&m, r);

    eprintln!(
        "k={}: {} reachable nodes in {:.1} ms, arena {:.2} B/n, total {:.2} B/n",
        k, nodes, build_ms,
        mem.arena_bytes_per_node(),
        mem.total_bytes_per_node()
    );

    // Use `r` so the optimizer can't eliminate the build.
    std::hint::black_box(r);
}
