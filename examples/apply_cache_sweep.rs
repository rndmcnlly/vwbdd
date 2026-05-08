//! Apply-cache sizing sweep: how does hit rate and wall time change
//! with cache capacity? Supports §4.20.
//!
//! `cargo run --release --example apply_cache_sweep`
//!
//! Runs mult-trunc at k=12, 13, 14, 15 across cache sizes 2^18..2^24,
//! reporting hit/collision/empty rates plus wall time. Shows whether
//! the steady-state 50% hit rate observed at 2^21 is a working-set
//! ceiling or a hash-entropy ceiling. If hit rate climbs with size,
//! the working set is the bound. If it plateaus, the hash is.
//!
//! Takes a few minutes. The builders are inlined from the internal
//! tests/mult_shared/mod.rs helpers so this example stands alone.

use std::time::Instant;

use vwbdd::{
    apply_cache_stats, apply_cache_stats_enable, apply_cache_stats_reset, Manager,
    ManagerConfig, Ref,
};

fn vw_vars_trunc(m: &mut Manager, k: u32) -> (Vec<Ref>, Vec<Ref>, Vec<Ref>) {
    let f = m.r#false();
    let t = m.r#true();
    let mut mk = |i: u32, m: &mut Manager| -> Ref {
        let _ = m.new_var();
        m.make_node(i, f, t)
    };
    let x: Vec<Ref> = (0..k).map(|i| mk(i, m)).collect();
    let y: Vec<Ref> = (k..2 * k).map(|i| mk(i, m)).collect();
    let z: Vec<Ref> = (2 * k..3 * k).map(|i| mk(i, m)).collect();
    (x, y, z)
}

fn vw_add(m: &mut Manager, a: &[Ref], b: &[Ref]) -> Vec<Ref> {
    let f = m.r#false();
    let mut carry = f;
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        let ab = m.xor(a[i], b[i]);
        let s = m.xor(ab, carry);
        let and_ab = m.and(a[i], b[i]);
        let and_c_ab = m.and(carry, ab);
        carry = m.or(and_ab, and_c_ab);
        out.push(s);
    }
    out
}

fn vw_mult_trunc(m: &mut Manager, x: &[Ref], y: &[Ref]) -> Vec<Ref> {
    let k = x.len();
    let f = m.r#false();
    let mut acc: Vec<Ref> = vec![f; k];
    for j in 0..k {
        let mut partial: Vec<Ref> = vec![f; k];
        for i in j..k {
            partial[i] = m.and(x[i - j], y[j]);
        }
        acc = vw_add(m, &acc, &partial);
    }
    acc
}

fn vw_eq(m: &mut Manager, p: &[Ref], z: &[Ref]) -> Ref {
    let t = m.r#true();
    let mut eq = t;
    for i in 0..p.len() {
        let xi = m.xor(p[i], z[i]);
        let eq_bit = m.not(xi);
        eq = m.and(eq, eq_bit);
    }
    eq
}

fn run_once(k: u32, cache_slots: usize) -> (f64, u64, u64, u64) {
    // Reset counters, enable, run, read, disable.
    apply_cache_stats_reset();
    apply_cache_stats_enable(true);

    let start = Instant::now();
    let mut m = Manager::with_config(ManagerConfig::new().with_cache_slots(cache_slots));
    let (x, y, z) = vw_vars_trunc(&mut m, k);
    let p = vw_mult_trunc(&mut m, &x, &y);
    let _r = vw_eq(&mut m, &p, &z);
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    apply_cache_stats_enable(false);
    let (h, c, e) = apply_cache_stats();
    (elapsed, h, c, e)
}

fn main() {
    // Powers of two that span 256k up to 16M slots.
    // Each entry is 32 bytes, so 2^18 = 8 MiB, 2^24 = 512 MiB.
    let sizes: &[usize] = &[1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24];
    let ks: &[u32] = &[12, 13, 14, 15];

    println!();
    println!("apply-cache sizing sweep (mult-trunc, best-of-1)");
    println!();
    println!(
        "  {:>4} {:>7} {:>10} {:>8} {:>8} {:>8} {:>8}",
        "k", "slots", "wall (ms)", "total", "hit%", "coll%", "empty%"
    );
    println!("  {}", "-".repeat(60));

    for &k in ks {
        for &slots in sizes {
            let (ms, h, c, e) = run_once(k, slots);
            let total = h + c + e;
            let hp = if total > 0 { 100.0 * h as f64 / total as f64 } else { 0.0 };
            let cp = if total > 0 { 100.0 * c as f64 / total as f64 } else { 0.0 };
            let ep = if total > 0 { 100.0 * e as f64 / total as f64 } else { 0.0 };
            println!(
                "  {:>4} {:>7} {:>10.1} {:>8} {:>7.1}% {:>7.1}% {:>7.1}%",
                k,
                format_slots(slots),
                ms,
                format_count(total),
                hp,
                cp,
                ep
            );
        }
        println!();
    }
}

fn format_slots(s: usize) -> String {
    if s >= 1 << 20 {
        format!("{}M", s >> 20)
    } else if s >= 1 << 10 {
        format!("{}K", s >> 10)
    } else {
        format!("{}", s)
    }
}

fn format_count(c: u64) -> String {
    if c >= 1_000_000 {
        format!("{:.1}M", c as f64 / 1e6)
    } else if c >= 1_000 {
        format!("{:.0}K", c as f64 / 1e3)
    } else {
        format!("{}", c)
    }
}
