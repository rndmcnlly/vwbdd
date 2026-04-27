//! Microbenchmarks to decompose where time goes. No OxiDD in this file;
//! just vwbdd operations we can isolate.

use std::time::Instant;
use vwbdd::{Manager, Ref};

/// Measure: pure decode cost on a pre-built arena.
/// Build a BDD, then walk it N times via decode_node. Time the walks.
#[test]
fn decode_cost_per_node() {
    let mut m = Manager::new();
    let k = 6u32;
    for _ in 0..(4 * k) {
        m.new_var();
    }
    let vars: Vec<Ref> = (0..4 * k)
        .map(|i| {
            let f = m.r#false();
            let t = m.r#true();
            m.make_node(i, f, t)
        })
        .collect();

    // Build x*y=z mult relation inline (same as tests/mult.rs)
    let x = &vars[0..k as usize];
    let y = &vars[k as usize..2 * k as usize];
    let z = &vars[2 * k as usize..4 * k as usize];

    // Partial products
    let mut acc = vec![m.r#false(); 2 * k as usize];
    for j in 0..k as usize {
        let mut pp = vec![m.r#false(); 2 * k as usize];
        for i in 0..k as usize {
            if i + j < 2 * k as usize {
                pp[i + j] = m.and(x[i], y[j]);
            }
        }
        // ripple add
        let mut c = m.r#false();
        let mut out = vec![m.r#false(); 2 * k as usize];
        for i in 0..2 * k as usize {
            let ab = m.xor(acc[i], pp[i]);
            let sum = m.xor(ab, c);
            let a_and_b = m.and(acc[i], pp[i]);
            let c_and_ab = m.and(c, ab);
            let carry = m.or(a_and_b, c_and_ab);
            out[i] = sum;
            c = carry;
        }
        acc = out;
    }
    let mut rel = m.r#true();
    for i in 0..2 * k as usize {
        let d = m.xor(acc[i], z[i]);
        let s = m.not(d);
        rel = m.and(rel, s);
    }
    let remap = m.gc(&[rel]);
    let rel = remap[0];

    let n_nodes = m.num_nodes();
    eprintln!("prepared {} nodes", n_nodes);

    // Walk 100 times.
    let t0 = Instant::now();
    let iters = 100u32;
    let mut checksum: u64 = 0;
    for _ in 0..iters {
        // DFS walk decoding every node.
        let mut seen = vec![false; n_nodes];
        // Can't index by offset directly; use a HashSet.
        let mut seen_set = std::collections::HashSet::new();
        let mut stack = vec![rel];
        while let Some(r) = stack.pop() {
            if let Ref::Node(o) = r {
                if !seen_set.insert(o) {
                    continue;
                }
                let n = m.decode_node(r).unwrap();
                checksum = checksum.wrapping_add(n.var as u64);
                stack.push(n.lo);
                stack.push(n.hi);
            }
        }
        let _ = seen;
    }
    let dt = t0.elapsed();
    let total_decodes = iters as usize * n_nodes;
    let ns_per_decode = dt.as_nanos() as f64 / total_decodes as f64;
    eprintln!(
        "  {} walks x {} nodes = {} decodes in {:.2}ms = {:.1} ns/decode (checksum={})",
        iters, n_nodes, total_decodes,
        dt.as_secs_f64() * 1000.0, ns_per_decode, checksum,
    );

    // Now measure hashmap-only cost: iterate through offsets, just do
    // var_of_offset lookup. Uses the public var_of method.
    let all_refs: Vec<Ref> = {
        let mut seen_set = std::collections::HashSet::new();
        let mut stack = vec![rel];
        let mut out = Vec::new();
        while let Some(r) = stack.pop() {
            if let Ref::Node(o) = r {
                if !seen_set.insert(o) { continue; }
                let n = m.decode_node(r).unwrap();
                out.push(r);
                stack.push(n.lo); stack.push(n.hi);
            }
        }
        out
    };
    let t1 = Instant::now();
    let mut cksum = 0u64;
    for _ in 0..iters {
        for r in &all_refs {
            cksum = cksum.wrapping_add(m.var_of(*r).unwrap() as u64);
        }
    }
    let dt_varlookup = t1.elapsed();
    eprintln!(
        "  {} walks x {} var_of = {} lookups in {:.2}ms = {:.1} ns/lookup (checksum={})",
        iters,
        all_refs.len(),
        iters as usize * all_refs.len(),
        dt_varlookup.as_secs_f64() * 1000.0,
        dt_varlookup.as_nanos() as f64 / (iters as usize * all_refs.len()) as f64,
        cksum,
    );
}

/// Measure cost breakdown during construction (which also includes
/// make_node, unique-table probes, and ite_cache hits/misses).
#[test]
fn construction_cost_per_node() {
    use std::time::Instant;
    let k = 6u32;
    let t0 = Instant::now();
    let mut m = Manager::new();
    for _ in 0..(4 * k) {
        m.new_var();
    }
    let vars: Vec<Ref> = (0..4 * k)
        .map(|i| {
            let f = m.r#false();
            let t = m.r#true();
            m.make_node(i, f, t)
        })
        .collect();
    let x = &vars[0..k as usize];
    let y = &vars[k as usize..2 * k as usize];
    let z = &vars[2 * k as usize..4 * k as usize];
    let mut acc = vec![m.r#false(); 2 * k as usize];
    for j in 0..k as usize {
        let mut pp = vec![m.r#false(); 2 * k as usize];
        for i in 0..k as usize {
            if i + j < 2 * k as usize {
                pp[i + j] = m.and(x[i], y[j]);
            }
        }
        let mut c = m.r#false();
        let mut out = vec![m.r#false(); 2 * k as usize];
        for i in 0..2 * k as usize {
            let ab = m.xor(acc[i], pp[i]);
            let sum = m.xor(ab, c);
            let a_and_b = m.and(acc[i], pp[i]);
            let c_and_ab = m.and(c, ab);
            let carry = m.or(a_and_b, c_and_ab);
            out[i] = sum;
            c = carry;
        }
        acc = out;
    }
    let mut rel = m.r#true();
    for i in 0..2 * k as usize {
        let d = m.xor(acc[i], z[i]);
        let s = m.not(d);
        rel = m.and(rel, s);
    }
    let dt = t0.elapsed();
    eprintln!(
        "k={} build time: {:.2}ms for {} reachable nodes, {} total built",
        k,
        dt.as_secs_f64() * 1000.0,
        {
            let mut seen = std::collections::HashSet::new();
            let mut stack = vec![rel];
            while let Some(r) = stack.pop() {
                if let Ref::Node(o) = r {
                    if !seen.insert(o) { continue; }
                    let n = m.decode_node(r).unwrap();
                    stack.push(n.lo); stack.push(n.hi);
                }
            }
            seen.len()
        },
        m.num_nodes(),
    );
    let _ = dt;
}
