//! Tunable apply-cache smoke tests: building the same function with two
//! different cache sizes must produce identical node counts. Canonicity
//! is a property of the variable order, not the cache.

use vwbdd::{Manager, ManagerConfig, Ref};

fn build_and(m: &mut Manager) -> Ref {
    let _v0 = m.new_var();
    let _v1 = m.new_var();
    let _v2 = m.new_var();
    let f = m.r#false();
    let t = m.r#true();
    let x0 = m.make_node(0, f, t);
    let x1 = m.make_node(1, f, t);
    let x2 = m.make_node(2, f, t);
    let a = m.and(x0, x1);
    m.and(a, x2)
}

#[test]
fn with_cache_slots_small() {
    // 2^4 = 16 slots. Intentionally tiny: forces collision eviction even
    // on a trivial fixture. Correct results must survive.
    let mut m = Manager::with_cache_slots(1 << 4);
    let r = build_and(&mut m);
    // Root is a real node (not a terminal) for a 3-var AND.
    assert!(matches!(r, Ref::Node(_)));
}

#[test]
fn with_cache_slots_large() {
    let mut m = Manager::with_cache_slots(1 << 18);
    let r = build_and(&mut m);
    assert!(matches!(r, Ref::Node(_)));
}

#[test]
fn node_counts_agree_across_cache_sizes() {
    let mut m_small = Manager::with_cache_slots(1 << 4);
    let mut m_big = Manager::with_cache_slots(1 << 20);
    let _ = build_and(&mut m_small);
    let _ = build_and(&mut m_big);
    assert_eq!(
        m_small.num_nodes(),
        m_big.num_nodes(),
        "canonical node count is cache-size-independent"
    );
}

#[test]
fn mem_stats_reflect_cache_slots() {
    let m_small = Manager::with_cache_slots(1 << 10);
    let m_big = Manager::with_cache_slots(1 << 12);
    // Big has 4× the slots → 4× the cache_bytes (same entry size).
    assert_eq!(
        m_big.mem_stats().cache_bytes,
        m_small.mem_stats().cache_bytes * 4
    );
}

#[test]
#[should_panic(expected = "power of two")]
fn with_cache_slots_rejects_non_power_of_two() {
    let _ = ManagerConfig::new().with_cache_slots(1000);
}

#[test]
fn manager_config_default_matches_new() {
    // Guard against accidental divergence: Manager::new() should produce
    // a manager with the default config's cache size.
    let m_default = Manager::new();
    let m_config = Manager::with_cache_slots(vwbdd::DEFAULT_ITE_CACHE_SLOTS);
    assert_eq!(
        m_default.mem_stats().cache_bytes,
        m_config.mem_stats().cache_bytes
    );
}
