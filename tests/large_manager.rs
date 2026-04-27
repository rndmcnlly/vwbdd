//! Parameterization smoke tests: the u64 `LargeManager` alias compiles,
//! runs basic ite, and agrees with the default u32 `DefaultManager` on
//! a shared scenario.
//!
//! The meat of correctness (node counts, GC, OxiDD differential) is
//! exercised on `DefaultManager`; this file exists to make sure the
//! generic plumbing doesn't drift from the u32 path without a test catching it.

use vwbdd::{DefaultManager, LargeManager, Ref};

fn build_fixture_u32(m: &mut DefaultManager) -> Ref<u32> {
    let _ = m.new_var();
    let _ = m.new_var();
    let _ = m.new_var();
    // x0 variable
    let x0 = m.make_node(0, m.r#false(), m.r#true());
    let x1 = m.make_node(1, m.r#false(), m.r#true());
    let x2 = m.make_node(2, m.r#false(), m.r#true());
    // (x0 & x1) | x2
    let a = m.and(x0, x1);
    m.or(a, x2)
}

fn build_fixture_u64(m: &mut LargeManager) -> Ref<u64> {
    let _ = m.new_var();
    let _ = m.new_var();
    let _ = m.new_var();
    let x0 = m.make_node(0, m.r#false(), m.r#true());
    let x1 = m.make_node(1, m.r#false(), m.r#true());
    let x2 = m.make_node(2, m.r#false(), m.r#true());
    let a = m.and(x0, x1);
    m.or(a, x2)
}

#[test]
fn large_manager_builds_and_runs_ite() {
    let mut m = LargeManager::default();
    let _ = build_fixture_u64(&mut m);
    // The formula has three input vars and a single OR at the top;
    // the reduced OBDD shape is deterministic.
    assert!(m.num_nodes() >= 3);
}

#[test]
fn u32_and_u64_agree_on_node_count() {
    let mut m32 = DefaultManager::new();
    let mut m64 = LargeManager::default();

    let _ = build_fixture_u32(&mut m32);
    let _ = build_fixture_u64(&mut m64);

    assert_eq!(
        m32.num_nodes(),
        m64.num_nodes(),
        "u32 and u64 managers must produce identical node counts \
         (canonicity is a property of the variable order, not the offset width)"
    );
}

#[test]
fn large_manager_gc_roundtrip() {
    let mut m = LargeManager::default();
    let r = build_fixture_u64(&mut m);
    let pre = m.num_nodes();
    let kept = m.gc(&[r]);
    assert_eq!(kept.len(), 1);
    // GC drops intermediate `and` cofactors no longer reachable from the
    // root; keep the ≤ relation rather than equality.
    assert!(
        m.num_nodes() <= pre,
        "GC should not manufacture nodes (pre={}, post={})",
        pre,
        m.num_nodes()
    );
    // Semantics survive: we can still read the root's top var.
    let Ref::Node(_) = kept[0] else { panic!("root vanished"); };
    assert!(m.var_of(kept[0]).is_some());
}

#[test]
fn large_manager_reports_wider_unique_slots() {
    // Build a big-enough fixture that the unique table has inserted into
    // more than one slot, so its reported `bytes()` reflects the slot width.
    let mut m32 = DefaultManager::new();
    let mut m64 = LargeManager::default();
    let _ = build_fixture_u32(&mut m32);
    let _ = build_fixture_u64(&mut m64);

    let s32 = m32.mem_stats();
    let s64 = m64.mem_stats();
    // Both start with the same INITIAL_CAP, so u64 must be ~1.8× the size
    // of u32 in `unique_bytes` (9 B/slot vs 5 B/slot + the same tag vec).
    assert!(
        s64.unique_bytes > s32.unique_bytes,
        "u64 unique table should be larger than u32 at equal slot count"
    );
    // Cache-byte parity check: u64 IteCacheEntry is exactly twice u32's.
    assert_eq!(
        s64.cache_bytes,
        s32.cache_bytes * 2,
        "u64 cache entries are 32 B vs u32's 16 B"
    );
}

// ----------------------------------------------------------------------------
// mult(k=4) correctness cross-check: u32 and u64 must produce the same node
// count on a non-trivial workload. This catches bugs where an encoding bug
// affects only one width (e.g. a narrowing cast in the codec).
// ----------------------------------------------------------------------------

fn mult_node_count<M, O>(m: &mut M, k: u32) -> usize
where
    M: ManagerOps<O>,
    O: vwbdd::ArenaOffset,
{
    // Variables: x[0..k], y[0..k], z[0..2k].
    let xs: Vec<Ref<O>> = (0..k).map(|_| m.var_node()).collect();
    let ys: Vec<Ref<O>> = (0..k).map(|_| m.var_node()).collect();
    let zs: Vec<Ref<O>> = (0..2 * k).map(|_| m.var_node()).collect();

    // p = x * y via ripple-carry; eq = all bits of p match z.
    let n = 2 * (k as usize);
    let f = m.c_false();
    let mut acc: Vec<Ref<O>> = vec![f; n];
    for j in 0..(k as usize) {
        let mut pp: Vec<Ref<O>> = vec![f; n];
        for i in 0..(k as usize) {
            if i + j < n {
                pp[i + j] = m.and_(xs[i], ys[j]);
            }
        }
        let mut out = Vec::with_capacity(n);
        let mut c = f;
        for i in 0..n {
            let (s, co) = full_adder(m, acc[i], pp[i], c);
            out.push(s);
            c = co;
        }
        acc = out;
    }
    let t = m.c_true();
    let mut eq = t;
    for i in 0..n {
        let d = m.xor_(acc[i], zs[i]);
        let s = m.not_(d);
        eq = m.and_(eq, s);
    }
    // Count reachable.
    let mut seen = std::collections::HashSet::new();
    let mut stk = vec![eq];
    while let Some(r) = stk.pop() {
        if let Ref::Node(o) = r {
            if !seen.insert(m.off_to_u64(o)) {
                continue;
            }
            let (lo, hi) = m.children(r);
            stk.push(lo);
            stk.push(hi);
        }
    }
    seen.len()
}

fn full_adder<M, O>(m: &mut M, a: Ref<O>, b: Ref<O>, c: Ref<O>) -> (Ref<O>, Ref<O>)
where
    M: ManagerOps<O>,
    O: vwbdd::ArenaOffset,
{
    let ab = m.xor_(a, b);
    let sum = m.xor_(ab, c);
    let a_and_b = m.and_(a, b);
    let c_and_ab = m.and_(c, ab);
    let carry = m.or_(a_and_b, c_and_ab);
    (sum, carry)
}

/// Tiny trait so the mult-builder above can drive both `DefaultManager`
/// (u32) and `LargeManager` (u64) with the same code.
trait ManagerOps<O: vwbdd::ArenaOffset> {
    fn var_node(&mut self) -> Ref<O>;
    fn c_false(&self) -> Ref<O>;
    fn c_true(&self) -> Ref<O>;
    fn and_(&mut self, a: Ref<O>, b: Ref<O>) -> Ref<O>;
    fn or_(&mut self, a: Ref<O>, b: Ref<O>) -> Ref<O>;
    fn xor_(&mut self, a: Ref<O>, b: Ref<O>) -> Ref<O>;
    fn not_(&mut self, a: Ref<O>) -> Ref<O>;
    fn children(&self, r: Ref<O>) -> (Ref<O>, Ref<O>);
    fn off_to_u64(&self, o: O) -> u64;
}

impl ManagerOps<u32> for DefaultManager {
    fn var_node(&mut self) -> Ref<u32> {
        let v = self.new_var();
        let f = self.r#false();
        let t = self.r#true();
        self.make_node(v, f, t)
    }
    fn c_false(&self) -> Ref<u32> { self.r#false() }
    fn c_true(&self) -> Ref<u32> { self.r#true() }
    fn and_(&mut self, a: Ref<u32>, b: Ref<u32>) -> Ref<u32> { self.and(a, b) }
    fn or_(&mut self, a: Ref<u32>, b: Ref<u32>) -> Ref<u32> { self.or(a, b) }
    fn xor_(&mut self, a: Ref<u32>, b: Ref<u32>) -> Ref<u32> { self.xor(a, b) }
    fn not_(&mut self, a: Ref<u32>) -> Ref<u32> { self.not(a) }
    fn children(&self, r: Ref<u32>) -> (Ref<u32>, Ref<u32>) {
        let n = self.decode_node(r).unwrap();
        (n.lo, n.hi)
    }
    fn off_to_u64(&self, o: u32) -> u64 { o as u64 }
}

impl ManagerOps<u64> for LargeManager {
    fn var_node(&mut self) -> Ref<u64> {
        let v = self.new_var();
        let f = self.r#false();
        let t = self.r#true();
        self.make_node(v, f, t)
    }
    fn c_false(&self) -> Ref<u64> { self.r#false() }
    fn c_true(&self) -> Ref<u64> { self.r#true() }
    fn and_(&mut self, a: Ref<u64>, b: Ref<u64>) -> Ref<u64> { self.and(a, b) }
    fn or_(&mut self, a: Ref<u64>, b: Ref<u64>) -> Ref<u64> { self.or(a, b) }
    fn xor_(&mut self, a: Ref<u64>, b: Ref<u64>) -> Ref<u64> { self.xor(a, b) }
    fn not_(&mut self, a: Ref<u64>) -> Ref<u64> { self.not(a) }
    fn children(&self, r: Ref<u64>) -> (Ref<u64>, Ref<u64>) {
        let n = self.decode_node(r).unwrap();
        (n.lo, n.hi)
    }
    fn off_to_u64(&self, o: u64) -> u64 { o }
}

#[test]
fn mult_k4_node_count_matches_across_offset_widths() {
    let mut m32 = DefaultManager::new();
    let mut m64 = LargeManager::default();
    let c32 = mult_node_count(&mut m32, 4);
    let c64 = mult_node_count(&mut m64, 4);
    // Known value from tests/mult.rs: 498 reachable nodes at k=4.
    assert_eq!(c32, 498, "regression: u32 mult(k=4) node count drift");
    assert_eq!(
        c32, c64,
        "u32 and u64 mult(k=4) node counts must agree (canonicity is codec-independent)"
    );
}
