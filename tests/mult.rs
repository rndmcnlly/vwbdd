//! Mult relation x*y=z over k-bit unsigned ints. Bitblasts the multiplier in
//! both engines and compares node counts at each k. Prints bytes/node for
//! vwbdd at each k so we can see the curve.

use oxidd::bdd::{new_manager as oxidd_new_manager, BDDFunction};
use oxidd::{BooleanFunction, Function, Manager as _, ManagerRef};

use vwbdd::{Manager, Ref};

/// Variable order (both engines must agree): x_0..x_{k-1}, y_0..y_{k-1},
/// z_0..z_{2k-1}.  Low-order bit first.
fn vw_vars(vw: &mut Manager, k: u32) -> (Vec<Ref>, Vec<Ref>, Vec<Ref>) {
    let nx: Vec<Ref> = (0..k)
        .map(|_| {
            let v = vw.new_var();
            let f = vw.r#false();
            let t = vw.r#true();
            vw.make_node(v, f, t)
        })
        .collect();
    let ny: Vec<Ref> = (0..k)
        .map(|_| {
            let v = vw.new_var();
            let f = vw.r#false();
            let t = vw.r#true();
            vw.make_node(v, f, t)
        })
        .collect();
    let nz: Vec<Ref> = (0..2 * k)
        .map(|_| {
            let v = vw.new_var();
            let f = vw.r#false();
            let t = vw.r#true();
            vw.make_node(v, f, t)
        })
        .collect();
    (nx, ny, nz)
}

/// Full adder: (sum, carry_out) = a + b + c_in.
fn vw_full_adder(vw: &mut Manager, a: Ref, b: Ref, c: Ref) -> (Ref, Ref) {
    let ab = vw.xor(a, b);
    let sum = vw.xor(ab, c);
    // carry = (a AND b) OR (c AND (a XOR b))
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
    let carry = a_and_b.or(&c_and_ab).unwrap();
    (sum, carry)
}

/// Ripple-carry adder of two bit-vectors of length n. c_in starts at false.
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

fn ox_add(a: &[BDDFunction], b: &[BDDFunction], ox_false: &BDDFunction) -> Vec<BDDFunction> {
    assert_eq!(a.len(), b.len());
    let mut out = Vec::with_capacity(a.len());
    let mut c = ox_false.clone();
    for i in 0..a.len() {
        let (s, c_out) = ox_full_adder(&a[i], &b[i], &c);
        out.push(s);
        c = c_out;
    }
    out
}

/// Build p = x * y as a 2k-bit vector, using shift-and-add of partial products.
fn vw_mult(vw: &mut Manager, x: &[Ref], y: &[Ref]) -> Vec<Ref> {
    let k = x.len();
    let n = 2 * k;
    // Start with an accumulator of n zero bits.
    let f = vw.r#false();
    let mut acc: Vec<Ref> = vec![f; n];
    // For each bit j of y, build (x AND y_j) shifted left by j, add to acc.
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

fn ox_mult(x: &[BDDFunction], y: &[BDDFunction], ox_false: &BDDFunction) -> Vec<BDDFunction> {
    let k = x.len();
    let n = 2 * k;
    let mut acc: Vec<BDDFunction> = (0..n).map(|_| ox_false.clone()).collect();
    for j in 0..k {
        let mut pp: Vec<BDDFunction> = (0..n).map(|_| ox_false.clone()).collect();
        for i in 0..k {
            if i + j < n {
                pp[i + j] = x[i].and(&y[j]).unwrap();
            }
        }
        acc = ox_add(&acc, &pp, ox_false);
    }
    acc
}

/// Bitwise equivalence: AND over i of (p_i <=> z_i) = AND over i of NOT(p_i XOR z_i).
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

fn ox_eq(p: &[BDDFunction], z: &[BDDFunction], ox_true: &BDDFunction) -> BDDFunction {
    assert_eq!(p.len(), z.len());
    let mut acc = ox_true.clone();
    for i in 0..p.len() {
        let diff = p[i].xor(&z[i]).unwrap();
        let same = diff.not().unwrap();
        acc = acc.and(&same).unwrap();
    }
    acc
}

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

/// Build mult at bit-width k, in both engines, assert matching node counts.
/// Returns (reachable_internal, total_mgr_nodes, buf_len, oxidd_count).
fn run_mult(k: u32) -> (usize, usize, usize, usize, vwbdd::MemStats) {
    // --- OxiDD side ---
    let oxm = oxidd_new_manager(1 << 18, 1 << 14, 1);
    let (oxx, oxy, oxz, ox_true, ox_false) = oxm.with_manager_exclusive(|mgr| {
        let names: Vec<String> = (0..k)
            .map(|i| format!("x{}", i))
            .chain((0..k).map(|i| format!("y{}", i)))
            .chain((0..2 * k).map(|i| format!("z{}", i)))
            .collect();
        mgr.add_named_vars(names.iter().map(|s| s.as_str())).unwrap();
        let oxx: Vec<_> = (0..k)
            .map(|i| BDDFunction::var(mgr, i).unwrap())
            .collect();
        let oxy: Vec<_> = (k..2 * k)
            .map(|i| BDDFunction::var(mgr, i).unwrap())
            .collect();
        let oxz: Vec<_> = (2 * k..4 * k)
            .map(|i| BDDFunction::var(mgr, i).unwrap())
            .collect();
        (
            oxx,
            oxy,
            oxz,
            BDDFunction::t(mgr),
            BDDFunction::f(mgr),
        )
    });
    let ox_p = ox_mult(&oxx, &oxy, &ox_false);
    let ox_rel = ox_eq(&ox_p, &oxz, &ox_true);
    let oxidd_count = ox_rel.node_count();

    // --- vwbdd side ---
    let mut vw = Manager::new();
    let (vx, vy, vz) = vw_vars(&mut vw, k);
    let vp = vw_mult(&mut vw, &vx, &vy);
    let vrel = vw_eq(&mut vw, &vp, &vz);

    let reachable = reachable_internal(&vw, vrel);
    let oxidd_style = reachable + 2; // include terminals
    assert_eq!(
        oxidd_style, oxidd_count,
        "node count mismatch at k={}: vwbdd={} oxidd={}",
        k, oxidd_style, oxidd_count
    );

    // GC to reachable set, report post-GC mem.
    let remapped = vw.gc(&[vrel]);
    let post = vw.mem_stats();
    let reachable_post = reachable_internal(&vw, remapped[0]);
    assert_eq!(reachable_post, reachable, "GC should preserve reachable count");
    assert_eq!(
        vw.num_nodes(),
        reachable,
        "post-GC manager should contain only reachable nodes"
    );

    eprintln!(
        "  post-gc: arena={}B ({:.2} B/n), total_live={}B ({:.2} B/n)",
        post.arena_bytes,
        post.arena_bytes_per_node(),
        post.total_live(),
        post.total_bytes_per_node(),
    );

    (reachable, vw.num_nodes(), vw.buf_len(), oxidd_count, vw.mem_stats())
}

#[test]
fn mult_sweep() {
    // Recursive ite needs a big program stack. Spawn a dedicated thread per k
    // so stack frames don't accumulate across iterations.
    eprintln!(
        "{:>3} {:>10} {:>10} {:>5} {:>7} {:>7} {:>11}",
        "k", "reachable", "mgr_all", "arena", "uniq", "cache", "total_live"
    );
    eprintln!(
        "{:>3} {:>10} {:>10} {:>5} {:>7} {:>7} {:>11}",
        "", "", "", "B/n", "B/n", "B/n", "B/n (w/ cache)"
    );
    for k in 2..=8 {
        // OxiDD's internal apply recurses; we need a big stack for its worker.
        let handle = std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn(move || run_mult(k))
            .unwrap();
        let (reachable, all, _bytes, oxidd, mem) = handle.join().unwrap();
        let arena_bpn = mem.arena_bytes as f64 / all.max(1) as f64;
        let uniq_bpn = mem.unique_bytes as f64 / all.max(1) as f64;
        let cache_bpn = mem.cache_bytes as f64 / all.max(1) as f64;
        let total_bpn = mem.total_live() as f64 / all.max(1) as f64;
        let total_wcache_bpn =
            mem.total_with_cache() as f64 / all.max(1) as f64;
        eprintln!(
            "{:>3} {:>10} {:>10} {:>5.2} {:>7.2} {:>7.2} {:>5.2}+{:.2}={:.2}  (oxidd={})",
            k, reachable, all, arena_bpn, uniq_bpn, cache_bpn, total_bpn, cache_bpn, total_wcache_bpn, oxidd
        );
    }
}
