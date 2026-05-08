//! Slab/Diff round-trips: the "ship compact arenas, keep unique tables
//! ephemeral" story.
//!
//! Covers:
//!   - extend-then-apply: server builds a tail on top of a base, ships
//!     only the tail bytes, client reconstructs the full arena.
//!   - slab_for as the unification of gc + byte extraction.
//!   - byte-identical tails: what the server appends and what the
//!     client would have appended locally are the same bytes.
//!   - apply_diff mismatch: base_len check catches a corrupted base.

use vwbdd::{Diff, Manager, Ref, Slab};

/// Build a small base slab: (x0 ∧ x1) over 3 vars.
fn build_base() -> Slab {
    let mut m = Manager::new();
    let _ = m.new_var();
    let _ = m.new_var();
    let _ = m.new_var();
    let f = m.r#false();
    let t = m.r#true();
    let x0 = m.make_node(0, f, t);
    let x1 = m.make_node(1, f, t);
    let and = m.and(x0, x1);
    m.slab_for(&[and])
}

#[test]
fn slab_roundtrip_preserves_node_set() {
    // Build a formula, capture as slab, ingest into fresh manager,
    // assert the live node set is identical.
    let base = build_base();
    let base_bytes = base.bytes.len();
    let base_root_count = base.roots.len();

    let mut client = Manager::new();
    let _ = client.new_var();
    let _ = client.new_var();
    let _ = client.new_var();

    let roots = client.ingest_slab(&base);
    assert_eq!(roots.len(), base_root_count);
    assert_eq!(
        client.buf_len(),
        base_bytes,
        "ingested arena matches base slab byte-for-byte"
    );
    // Root survives as a real node (not a terminal reduction).
    assert!(matches!(roots[0], Ref::Node(_)));
}

#[test]
fn extend_produces_tail_only_diff() {
    // Server: given base = (x0 ∧ x1), produce NOT(base_root). The new
    // node references the base root, so the tail must be just the
    // handful of bytes that encode the negation — NOT a re-serialization
    // of the whole arena.
    let base = build_base();
    let base_len = base.bytes.len();

    let diff: Diff = Manager::extend_slab(&base, |m, base_roots| {
        let and = base_roots[0];
        let n = m.not(and);
        vec![n]
    });

    assert_eq!(
        diff.base_len, base_len as u64,
        "diff's claimed base matches the actual slab size"
    );
    // NOT(x0 ∧ x1) adds one new node (plus possibly reusing existing
    // terminals/nodes). It must NOT re-serialize the entire base. The
    // tight check: the tail encodes exactly the number of nodes the
    // ops created, not base.nodes + ops.nodes. At this scale the tail
    // is typically one node (5-7 bytes) regardless of base size.
    assert!(
        diff.tail.len() <= 16,
        "single-op extension tail should be ~1 node (≤16 bytes); got {}",
        diff.tail.len()
    );
    assert_eq!(diff.new_roots.len(), 1);
}

#[test]
fn client_applies_diff_and_gets_matching_nodes() {
    // Full server-client roundtrip:
    //   server builds base, computes NOT(root) via extend_slab,
    //   client ingests base, applies diff,
    //   we verify the client's resulting manager is functionally
    //   indistinguishable from a manager that did the whole build
    //   locally.
    let base = build_base();

    let diff = Manager::extend_slab(&base, |m, base_roots| {
        let and = base_roots[0];
        let n = m.not(and);
        vec![n]
    });

    // Client side.
    let mut client = Manager::new();
    let _ = client.new_var();
    let _ = client.new_var();
    let _ = client.new_var();
    let base_roots = client.ingest_slab(&base);
    let new_roots = client.apply_diff(&diff);

    assert_eq!(base_roots.len(), 1);
    assert_eq!(new_roots.len(), 1);

    // Reference: build base + not(base_root) locally in one manager.
    let mut reference = Manager::new();
    let _ = reference.new_var();
    let _ = reference.new_var();
    let _ = reference.new_var();
    let f = reference.r#false();
    let t = reference.r#true();
    let x0 = reference.make_node(0, f, t);
    let x1 = reference.make_node(1, f, t);
    let and = reference.and(x0, x1);
    let _nand = reference.not(and);
    // Reference may carry dead nodes from the `not` call's intermediates;
    // GC to match what slab_for does on the base + what apply_diff
    // leaves in place.
    let _ = reference.drop_roots(&[and, _nand]);

    // The client's live node set (reachable from base_roots + new_roots)
    // must match the reference's post-gc node set.
    let _ = client.drop_roots(&[base_roots[0], new_roots[0]]);
    assert_eq!(
        client.num_nodes(),
        reference.num_nodes(),
        "client post-diff live node count ({}) equals reference single-build count ({})",
        client.num_nodes(), reference.num_nodes(),
    );
}

#[test]
fn tail_bytes_are_position_independent() {
    // The structural claim: if two different processes both start from
    // the same base slab and build the same ops, they produce
    // bit-identical tail bytes. That's the property that makes
    // "ship the tail" a valid wire protocol.
    let base = build_base();

    let diff_a: Diff = Manager::extend_slab(&base, |m, base_roots| {
        let and = base_roots[0];
        let n = m.not(and);
        vec![n]
    });

    let diff_b: Diff = Manager::extend_slab(&base, |m, base_roots| {
        let and = base_roots[0];
        let n = m.not(and);
        vec![n]
    });

    assert_eq!(
        diff_a.tail, diff_b.tail,
        "two independent extends with the same base + same ops produce \
         bit-identical tail bytes (LEB128 delta codes are position-independent \
         once the base boundary is fixed)"
    );
    assert_eq!(diff_a.base_len, diff_b.base_len);
    assert_eq!(diff_a.new_roots, diff_b.new_roots);
}

#[test]
fn gc_via_slab_matches_direct_gc() {
    // slab_for is meant to be the unified form of gc. Demonstrate that
    // the arena byte-length after slab_for equals the byte-length the
    // manager would have after a raw gc call on the same roots.
    let mut m1 = Manager::new();
    let _ = m1.new_var();
    let _ = m1.new_var();
    let _ = m1.new_var();
    let f = m1.r#false();
    let t = m1.r#true();
    let x0 = m1.make_node(0, f, t);
    let x1 = m1.make_node(1, f, t);
    let x2 = m1.make_node(2, f, t);
    let a = m1.and(x0, x1);
    let _b = m1.and(x1, x2); // dead once we keep only `a` in gc
    let len_before_gc = m1.buf_len();

    let mut m2 = Manager::new();
    let _ = m2.new_var();
    let _ = m2.new_var();
    let _ = m2.new_var();
    let f2 = m2.r#false();
    let t2 = m2.r#true();
    let x0_2 = m2.make_node(0, f2, t2);
    let x1_2 = m2.make_node(1, f2, t2);
    let x2_2 = m2.make_node(2, f2, t2);
    let a2 = m2.and(x0_2, x1_2);
    let _b2 = m2.and(x1_2, x2_2);
    let slab = m2.slab_for(&[a2]);

    // m1: direct gc on m1's own `a`.
    let _ = m1.drop_roots(&[a]);
    let len_after_gc = m1.buf_len();

    assert!(
        len_after_gc < len_before_gc,
        "direct gc freed bytes ({}→{})", len_before_gc, len_after_gc
    );
    assert_eq!(
        slab.bytes.len(), len_after_gc,
        "slab_for's arena bytes ({}) match the direct gc result ({})",
        slab.bytes.len(), len_after_gc
    );
}

#[test]
#[should_panic(expected = "recipient arena is")]
fn apply_diff_detects_base_mismatch() {
    // If the client's base doesn't match the diff's base_len, applying
    // the diff would silently produce corrupt results (LEB128 deltas
    // would resolve against the wrong children). We want a loud panic.
    let base = build_base();
    let diff: Diff = Manager::extend_slab(&base, |m, base_roots| {
        let and = base_roots[0];
        vec![m.not(and)]
    });

    // Client never ingested the base: arena is empty but diff expects
    // base_len > 0.
    let mut client = Manager::new();
    let _ = client.new_var();
    let _ = client.new_var();
    let _ = client.new_var();
    let _ = client.apply_diff(&diff); // panics
}

#[test]
fn extend_with_no_ops_produces_empty_tail() {
    // A degenerate but important case: if the server doesn't build
    // anything new, the tail is zero bytes and no new roots.
    let base = build_base();

    // Type-annotate the closure arg so type inference sees `Manager`
    // even when the body gives it no disambiguating calls.
    let diff: Diff = Manager::extend_slab(&base, |_m: &mut Manager, _base_roots| Vec::new());

    assert_eq!(diff.tail.len(), 0, "no ops, no tail");
    assert_eq!(diff.new_roots.len(), 0);
    assert_eq!(diff.base_len, base.bytes.len() as u64);
}

#[test]
fn apply_empty_diff_is_noop() {
    let base = build_base();
    let diff: Diff = Manager::extend_slab(&base, |_m: &mut Manager, _base_roots| Vec::new());

    let mut client = Manager::new();
    let _ = client.new_var();
    let _ = client.new_var();
    let _ = client.new_var();
    let _ = client.ingest_slab(&base);
    let len_before = client.buf_len();

    let new_roots = client.apply_diff(&diff);
    assert_eq!(new_roots.len(), 0);
    assert_eq!(client.buf_len(), len_before, "empty diff didn't change arena");
}

// ---------------------------------------------------------------------------
// Minor (tail-only) GC: the generational story.
//
// The claim: `gc(base_len, roots)` with nonzero `base_len` collects
// only the tail, leaving the base bytes byte-identical. It's the
// natural pre-ship pass: after running ops that create dead
// intermediates, minor-GC'ing before
// `diff_since` produces a smaller shippable diff than dumping the raw
// tail would.
// ---------------------------------------------------------------------------

#[test]
fn minor_gc_preserves_base_bytes_exactly() {
    // Ingest a base, build some throwaway work in the tail, minor-GC
    // keeping nothing from the tail. Base bytes must be byte-identical.
    let base = build_base();
    let base_bytes = base.bytes.clone();

    let mut m = Manager::new();
    let _ = m.new_var();
    let _ = m.new_var();
    let _ = m.new_var();
    let base_roots = m.ingest_slab(&base);
    let base_len = m.buf_len() as u64;

    // Build some tail work that we'll throw away.
    let and_root = base_roots[0];
    let _nand = m.not(and_root);
    let _n2 = m.xor(and_root, and_root); // folds to false, but may generate scratch
    assert!(m.buf_len() as u64 > base_len, "some tail bytes were built");

    // Minor GC, keeping only the base root (tail-orphaned: nothing in
    // the tail is actually reachable from this root, since the root is
    // a base node).
    let kept = m.gc(base_len, &[and_root]);
    assert_eq!(kept[0], and_root, "base root passes through unchanged");

    // Base bytes untouched.
    assert_eq!(
        &m.arena_slice(0)[..base_len as usize],
        &base_bytes[..],
        "minor GC must not touch base bytes"
    );
    // Tail is now empty.
    assert_eq!(
        m.buf_len() as u64, base_len,
        "tail entirely reclaimed (nothing was actually needed)"
    );
}

#[test]
fn extend_slab_enforces_clean_bytes_invariant() {
    // The clean-bytes invariant: any Diff produced by extend_slab
    // contains no scratch, regardless of how ops-heavy the construction
    // was. Demonstrated by running ops that create lots of dead
    // intermediates and verifying the tail size matches what a
    // from-scratch construction of just the kept roots would produce.
    let base = build_base();

    // Build a diff that ships only the base root, after running a
    // bunch of abandoned work in the tail.
    let diff_with_scratch: Diff = Manager::extend_slab(&base, |m, base_roots| {
        let and = base_roots[0];
        let _dead = m.not(and);
        let _also_dead = m.xor(and, and);
        let _v = m.new_var();
        vec![and]
    });

    // Reference: extend without creating dead work.
    let diff_clean: Diff = Manager::extend_slab(&base, |_m, base_roots| {
        vec![base_roots[0]]
    });

    // Under the invariant, both must have zero-length tails (the kept
    // root is a base node, so no tail bytes are needed).
    assert_eq!(
        diff_with_scratch.tail.len(), 0,
        "extend_slab pre-GCs the tail; dead scratch must not appear in the diff"
    );
    assert_eq!(diff_clean.tail.len(), 0);
    assert_eq!(diff_with_scratch.new_roots, diff_clean.new_roots);
}

#[test]
fn minor_gc_keeps_live_tail_nodes() {
    // If we DO want to keep a tail node, minor GC must preserve it
    // (perhaps at a different offset), and its base-pointing children
    // must still resolve correctly after re-encoding.
    let base = build_base();

    let mut m = Manager::new();
    let _ = m.new_var();
    let _ = m.new_var();
    let _ = m.new_var();
    let base_roots = m.ingest_slab(&base);
    let base_len = m.buf_len() as u64;

    let and = base_roots[0];
    // Build a tail node we want to keep: NOT(and). Its structure
    // references base (the `and` subgraph lives in the base) — this
    // is a tail→base reference, the common case the backward-delta
    // codec has to handle across re-encoding.
    let nand = m.not(and);
    // Also build some scratch we don't keep.
    let _dead = m.xor(and, and);

    let live_before = m.num_nodes();
    let kept = m.gc(base_len, &[and, nand]);
    let live_after = m.num_nodes();

    assert_eq!(kept.len(), 2);
    assert_eq!(kept[0], and, "base root passes through");
    // `nand` may have moved to a different offset; we don't assert
    // kept[1] == nand, only that it resolves to the same logical node.
    let kept_nand = kept[1];
    assert!(matches!(kept_nand, Ref::Node(_)));

    assert!(
        live_after <= live_before,
        "minor GC can only shrink or preserve node count ({} → {})",
        live_before, live_after
    );
    // Verify functional equivalence: not(and) in a fresh manager
    // should produce a BDD with the same node count as the minor-GC'd
    // tail contents.
    let mut reference = Manager::new();
    let _ = reference.new_var();
    let _ = reference.new_var();
    let _ = reference.new_var();
    let ref_base_roots = reference.ingest_slab(&base);
    let _ref_nand = reference.not(ref_base_roots[0]);
    assert_eq!(
        m.num_nodes(), reference.num_nodes(),
        "post-minor-GC node count equals the 'build from fresh' node count for the same kept roots"
    );
}

#[test]
fn minor_gc_empty_tail_is_noop() {
    // If the tail is already empty, minor GC should be a cheap no-op
    // and return the roots unchanged.
    let base = build_base();

    let mut m = Manager::new();
    let _ = m.new_var();
    let _ = m.new_var();
    let _ = m.new_var();
    let base_roots = m.ingest_slab(&base);
    let base_len = m.buf_len() as u64;
    assert_eq!(base_len, m.buf_len() as u64);

    let kept = m.gc(base_len, &base_roots);
    assert_eq!(kept, base_roots);
    assert_eq!(m.buf_len() as u64, base_len);
}

#[test]
fn minor_gc_enables_tighter_extend_roundtrip() {
    // End-to-end generational roundtrip: server ingests base, runs a
    // fat extend with dead intermediates, minor-GCs the tail, ships
    // the skinnier diff; client ingests base + applies diff and gets
    // functionally equivalent results to the un-GC'd path.
    let base = build_base();

    // Server.
    let diff: Diff = {
        let mut m = Manager::new();
        let _ = m.new_var();
        let _ = m.new_var();
        let _ = m.new_var();
        let base_roots = m.ingest_slab(&base);
        let base_len = m.buf_len() as u64;
        let and = base_roots[0];
        // Build a real result we want to ship, plus dead scratch.
        let nand = m.not(and);
        let _throwaway = m.and(and, nand); // contradiction, folds
        let _more_throwaway = m.or(and, nand); // tautology, folds
        // Minor GC before shipping.
        let keep = m.gc(base_len, &[nand]);
        m.diff_since(base_len, &keep)
    };

    // Client.
    let mut client = Manager::new();
    let _ = client.new_var();
    let _ = client.new_var();
    let _ = client.new_var();
    let _ = client.ingest_slab(&base);
    let shipped_roots = client.apply_diff(&diff);
    assert_eq!(shipped_roots.len(), 1);

    // Compare against a from-scratch reference build.
    let mut reference = Manager::new();
    let _ = reference.new_var();
    let _ = reference.new_var();
    let _ = reference.new_var();
    let ref_base_roots = reference.ingest_slab(&base);
    let ref_nand = reference.not(ref_base_roots[0]);

    // GC both to the same root set to compare canonical sizes.
    let _ = client.drop_roots(&[shipped_roots[0]]);
    let _ = reference.drop_roots(&[ref_nand]);
    assert_eq!(
        client.num_nodes(), reference.num_nodes(),
        "shipped-via-minor-GC'd-diff produces the same BDD as the reference build"
    );
}
