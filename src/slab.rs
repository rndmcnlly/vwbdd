//! Slabs and diffs: the shipping units for compact-arena BDDs.
//!
//! ## The clean-bytes invariant
//!
//! Every [`Slab`] and [`Diff`] that crosses a public boundary is
//! **function-canonical**: running a GC on it would find nothing to
//! remove. The manager's internal arena may accumulate scratch from
//! `ite` and similar ops, but the moment bytes become visible to
//! another party (via [`Manager::slab_for`], [`Manager::diff_since`],
//! [`Manager::extend_slab`], or [`Manager::dump`]), they carry only
//! the closure of the declared roots.
//!
//! The invariant is inductive: `ingest_slab(clean_base)` leaves the
//! arena clean; `apply_diff(clean_base, clean_diff)` leaves it clean;
//! GC-before-ship preserves it on output. The only way to introduce
//! scratch is via construction ops (`make_node`, `ite`, `and`, …), and
//! that scratch is invisible until the next `diff_since` / `slab_for`
//! which cleans it out as part of emitting a public artifact.
//!
//! The one public operation that *actively* shrinks a live manager is
//! [`Manager::drop_roots`] (alias for `gc`, intent-level name). New
//! code should prefer `drop_roots(keep)` to `gc(roots)` because the
//! former names the mental model the caller is using: "these are the
//! roots I still care about; drop everything else."
//!
//! ## Three levels of canonicity
//!
//! It's worth naming the layers explicitly because they come apart:
//!
//! 1. **Function-canonical**: the DAG reachable from roots is the
//!    reduced BDD for the function. Enforced cheaply by the unique
//!    table. This is what the clean-bytes invariant delivers.
//!
//! 2. **Layout-canonical**: the byte encoding of that DAG depends only
//!    on the DAG, not on the construction history. *Not* guaranteed
//!    today — two managers that built the same BDD via different `ite`
//!    call sequences can produce different byte encodings of the same
//!    reduced DAG. A future `canonicalize_layout` pass (sort nodes by
//!    structural hash, re-emit in that order) would give this.
//!
//! 3. **Layout-minimal**: byte encoding is as short as possible under
//!    LEB128. Minimizing total `leb128_len(parent_offset - child_offset)`
//!    is a bandwidth-minimization problem (NP-hard in general). Any
//!    canonical layout rule is an arbitrary choice, not the minimum.
//!
//! The `gc_tail` benchmark in `tests/minor_gc_savings.rs` measured
//! this: compacting a tail by GC changed byte counts by ±1% on
//! near-dense tails because the new layout's delta distribution
//! crossed LEB128 boundaries differently. That's the gap between (2)
//! and (3) in action. For now we accept the gap; callers that need
//! content-addressed slabs (hash = identity) should layer (2) on top.
//!
//! ## The vocabulary
//!
//! The types here unify three things that previously wore different
//! names in the codebase:
//!
//!   - `gc()` produces a "just the live bytes" snapshot of an arena.
//!   - `absorb()` takes foreign arena bytes and re-canonicalizes them
//!     through the local unique table.
//!   - `dump()` / `load()` is the on-disk form of the above.
//!
//! Viewed from one level up, all three are the same operation on
//! different sides of the same wire: **a compact slab (arena bytes +
//! root references) is the durable artifact; the unique table is an
//! ephemeral, private index over it, rebuilt on demand.**
//!
//! This file defines:
//!
//!   - [`Slab`]: the durable artifact. Raw codec-encoded arena bytes
//!     plus a list of root [`Ref`]s that "export" something meaningful
//!     in those bytes. Equivalent to a loaded `.vwbdd` file, minus the
//!     on-disk framing (magic, version, CRC, names). Think of it as the
//!     tokenized sequence an LLM would see.
//!
//!   - [`Diff`]: an append-only delta against a known base [`Slab`].
//!     Carries only the tail bytes the recipient doesn't yet have plus
//!     any new root refs; the recipient reconstructs the full arena by
//!     concatenation. Relies on the LEB128 codec's position-independence
//!     property: because every child code is `2 + (cur - child)` (a
//!     backward delta), nodes appended on top of a base arena encode to
//!     the same bytes regardless of who is doing the appending, as long
//!     as everyone starts from the same base.
//!
//!   - Methods on [`Manager`] that tie the above together:
//!     [`Manager::ingest_slab`], [`Manager::slab_for`],
//!     [`Manager::diff_since`], [`Manager::apply_diff`], and the
//!     one-shot convenience [`Manager::extend_slab`].
//!
//! ## Why the tail-shipping works
//!
//! The on-disk `.vwbdd` format already ships arena bytes verbatim (see
//! `src/dump.rs`), and `absorb()` already re-canonicalizes them one
//! node at a time. The new piece is recognizing that after an `ingest`,
//! the manager's `buf.len()` marks a *base boundary*: any node built
//! after that point has an offset `>= base_len`, and its encoding
//! refers backwards either into the base (offset `< base_len`) or into
//! the tail. In both cases the child code is a backward delta, so the
//! byte sequence of the tail does not depend on who or where it was
//! built; it only depends on the offsets of its children, which are
//! shared between sender and recipient because they share the base.
//!
//! ## Example: the "extend" roundtrip
//!
//! ```
//! use vwbdd::{Manager, Slab};
//!
//! // Build a base slab with a single root.
//! let base: Slab = {
//!     let mut m = Manager::new();
//!     let _ = m.new_var();
//!     let _ = m.new_var();
//!     let f = m.r#false();
//!     let t = m.r#true();
//!     let x0 = m.make_node(0, f, t);
//!     let x1 = m.make_node(1, f, t);
//!     let and = m.and(x0, x1);
//!     m.slab_for(&[and])
//! };
//!
//! // "Server" side: ingest base, run ops, ship a diff back.
//! let diff = Manager::extend_slab(&base, |m, base_roots| {
//!     let and = base_roots[0];
//!     let nand = m.not(and);
//!     vec![nand]
//! });
//!
//! // "Client" side: ingest base, apply diff, now has both roots.
//! let mut client = Manager::new();
//! let _ = client.new_var();
//! let _ = client.new_var();
//! let base_roots = client.ingest_slab(&base);
//! let new_roots = client.apply_diff(&diff);
//! assert_eq!(base_roots.len(), 1);
//! assert_eq!(new_roots.len(), 1);
//! ```
//!
//! The client never saw the server's unique table. Neither side kept a
//! table around between requests. The wire payload is (base_len bytes
//! of arena ‖ roots) for the first trip and (tail bytes ‖ new_roots)
//! for every extension.

use crate::codec::{ArenaOffset, NodeCodec, Ref};
use crate::manager::Manager;

/// A compact shipping unit: the arena bytes and the exported root refs.
///
/// A slab is self-contained: the recipient can rebuild a full manager
/// (arena + unique table) from the bytes alone. The roots are just
/// offsets into those bytes, using the same delta-code convention that
/// appears inline in the node stream (terminals, or absolute offsets
/// resolved at ingest time).
#[derive(Debug, Clone)]
pub struct Slab<O: ArenaOffset = u32> {
    /// Raw codec-encoded node stream. Byte-identical to what the
    /// manager's internal arena buffer held at `slab_for` time, minus
    /// any dead (unreachable) nodes if the slab came from [`Manager::slab_for`].
    pub bytes: Vec<u8>,
    /// The roots the producer chose to export. Offsets refer into
    /// [`bytes`][Slab::bytes] (or are terminals).
    pub roots: Vec<Ref<O>>,
}

impl<O: ArenaOffset> Slab<O> {
    /// Construct a slab directly. Mostly for tests and in-memory
    /// transport; usual producers are [`Manager::slab_for`].
    pub fn new(bytes: Vec<u8>, roots: Vec<Ref<O>>) -> Self {
        Self { bytes, roots }
    }

    /// Byte length of the arena. The recipient will use this as the
    /// `base_len` boundary when applying any later [`Diff`].
    pub fn base_len(&self) -> u64 {
        self.bytes.len() as u64
    }
}

/// An append-only delta against a known base [`Slab`]. Contains only
/// the tail bytes that extend the base, plus any new root refs the
/// producer promised to deliver.
///
/// The recipient reconstructs the full arena by concatenating their
/// local base bytes with [`tail`][Diff::tail]. [`base_len`][Diff::base_len]
/// is a sanity check so we fail fast if the recipient's base doesn't
/// match the producer's assumption.
#[derive(Debug, Clone)]
pub struct Diff<O: ArenaOffset = u32> {
    /// The arena byte length the producer assumed the recipient already
    /// holds. If `recipient.arena_len() != diff.base_len`, applying
    /// will corrupt the arena; [`Manager::apply_diff`] panics instead.
    pub base_len: u64,
    /// Bytes to append verbatim to the recipient's arena.
    pub tail: Vec<u8>,
    /// New roots, expressed as refs into the **combined** arena
    /// (recipient base ‖ tail). Since the producer and recipient agree
    /// on the base and the tail is byte-identical on both sides, these
    /// offsets are valid for the recipient without any translation.
    pub new_roots: Vec<Ref<O>>,
}

impl<O: ArenaOffset> Diff<O> {
    /// Tail byte length. Diagnostic; the "ship size" of an extension.
    pub fn tail_len(&self) -> u64 {
        self.tail.len() as u64
    }
}

// ---- Manager methods: the unified primitive surface ----

impl<C: NodeCodec<O>, O: ArenaOffset> Manager<C, O> {
    /// Ingest a base [`Slab`] into this manager.
    ///
    /// Panics if the manager's arena is non-empty: a slab is a *base*,
    /// so this is only meaningful on a fresh manager. (If you want to
    /// layer a second slab on top of the first, think of that as a
    /// [`Diff`] — use [`apply_diff`][Self::apply_diff] instead.)
    ///
    /// **Precondition** (part of the clean-bytes invariant): `slab.bytes`
    /// must be function-canonical — every byte reachable from
    /// `slab.roots`, no scratch. Slabs produced by [`slab_for`][Self::slab_for]
    /// and by [`Diff::apply_diff`][Self::apply_diff] satisfy this by
    /// construction. A slab assembled manually from arbitrary bytes
    /// must be pre-cleaned by the caller.
    ///
    /// Rebuilds the unique table over the ingested bytes. Returns the
    /// roots, which are identical to `slab.roots` (nothing moved); the
    /// return for symmetry with [`apply_diff`][Self::apply_diff].
    pub fn ingest_slab(&mut self, slab: &Slab<O>) -> Vec<Ref<O>> {
        assert!(
            self.buf_len() == 0,
            "ingest_slab requires a fresh (empty-arena) manager; \
             got {} bytes already present. Use apply_diff to layer on top.",
            self.buf_len()
        );
        self.buf_mut().extend_from_slice(&slab.bytes);
        self.rebuild_unique_from_arena();
        slab.roots.clone()
    }

    /// Produce a [`Slab`] that captures the live closure of `roots`.
    ///
    /// Delivers a function-canonical slab (clean-bytes invariant): the
    /// returned `slab.bytes` is exactly the arena `drop_roots(roots)`
    /// would leave behind. Internally runs the copying GC (so the slab
    /// contains no scratch), then wraps `(arena_bytes, translated_roots)`.
    ///
    /// The manager itself is mutated (its arena is replaced with the
    /// live closure, its unique table rebuilt, its apply cache
    /// flushed) — exactly as `gc()` / `drop_roots()` does. If you want
    /// to preserve the current manager state, clone it first or build
    /// a fresh manager and run the ops there.
    pub fn slab_for(&mut self, roots: &[Ref<O>]) -> Slab<O> {
        let new_roots = self.gc(roots);
        Slab {
            bytes: self.arena_slice(0).to_vec(),
            roots: new_roots,
        }
    }

    /// Build a [`Diff`] describing the bytes appended to the arena
    /// since it had length `base_len`, plus the requested `new_roots`.
    ///
    /// **Clean-bytes invariant**: the returned diff's `tail` is
    /// function-canonical for `new_roots` — it contains no scratch
    /// (ite intermediates not reachable from `new_roots`). This is
    /// achieved by internally running a tail-only GC
    /// ([`Manager::gc_tail`]) against `new_roots` as a final step
    /// before extracting the tail bytes. The manager is mutated
    /// accordingly: its arena is also canonical after this call, and
    /// `new_roots` are translated to their post-GC offsets.
    ///
    /// Typical use: right after [`ingest_slab`][Self::ingest_slab],
    /// record `base_len = m.buf_len()`. Run ops. Call `diff_since`
    /// with that recorded length and the ops' results as `new_roots`.
    /// The returned diff's new_roots may differ from the roots you
    /// passed in (if they pointed into the tail), because the tail's
    /// offsets got compacted. Use the returned new_roots from then on.
    pub fn diff_since(&mut self, base_len: u64, new_roots: &[Ref<O>]) -> Diff<O> {
        let cur_len = self.buf_len() as u64;
        assert!(
            base_len <= cur_len,
            "diff_since: base_len {} > current arena {}",
            base_len,
            cur_len
        );
        // Enforce the clean-bytes invariant: compact the tail to just
        // the reachable closure of new_roots. This is a no-op when the
        // tail already held only the ops' live nodes; it reclaims
        // scratch when the ops recursed through ite and abandoned
        // intermediate results.
        let cleaned_roots = self.gc_tail(base_len, new_roots);
        let tail = self.arena_slice(base_len as usize).to_vec();
        Diff {
            base_len,
            tail,
            new_roots: cleaned_roots,
        }
    }

    /// Apply a [`Diff`] on top of the current arena.
    ///
    /// Panics if the recipient's arena length doesn't match
    /// `diff.base_len` — that would mean we're applying a diff against
    /// a base we don't actually hold, and the LEB128 backward-delta
    /// codes in the tail would resolve to the wrong child nodes.
    ///
    /// **Precondition** (clean-bytes invariant): `diff.tail` must be
    /// function-canonical for `diff.new_roots` — no scratch bytes.
    /// Diffs produced by [`diff_since`][Self::diff_since] and
    /// [`extend_slab_generic`][Self::extend_slab_generic] satisfy this
    /// by construction (they run `gc_tail` internally).
    ///
    /// **Postcondition**: if the recipient's arena was clean before
    /// this call (e.g. from `ingest_slab`), it's clean after. Clean
    /// base + clean diff = clean combined arena, inductively; append
    /// alone never introduces scratch.
    ///
    /// Returns `diff.new_roots` unchanged (they're already valid
    /// absolute offsets into the resulting arena; see the module-level
    /// discussion of position independence).
    pub fn apply_diff(&mut self, diff: &Diff<O>) -> Vec<Ref<O>> {
        let cur_len = self.buf_len() as u64;
        assert_eq!(
            cur_len, diff.base_len,
            "apply_diff: recipient arena is {} bytes but diff assumes base of {}",
            cur_len, diff.base_len
        );
        self.buf_mut().extend_from_slice(&diff.tail);
        // Rebuild the unique table over the full arena. This is
        // O(total nodes). A future refinement could insert only the
        // new nodes (walk from base_len to end), but rebuilding is
        // simpler and the tests depend on a uniformly-populated table.
        self.rebuild_unique_from_arena();
        diff.new_roots.clone()
    }

    /// The unified server-side primitive: ingest a base, run ops, emit
    /// a diff. Single call so the typical client never has to manage
    /// the `base_len` book-keeping manually.
    ///
    /// The `ops` closure receives the manager (already loaded with the
    /// base) and the base roots translated for local use. It returns
    /// the list of new roots the caller wants shipped back.
    ///
    /// Uses a fresh manager internally so the caller's existing state
    /// is untouched. If you want to run ops in your own manager,
    /// compose the pieces manually: [`ingest_slab`][Self::ingest_slab]
    /// → do work → [`diff_since`][Self::diff_since].
    ///
    /// This is the **generic** version. At a bare call site like
    /// `Manager::extend_slab(&base, |m, roots| { ... })` Rust's
    /// inference can't pick `C, O` because the closure body typically
    /// doesn't mention codec- or offset-specific types. The default
    /// build's inherent shortcut [`Manager::extend_slab`] in the
    /// `impl Manager<Leb128Codec, u32>` block pins those parameters so
    /// the bare call works. Non-default builds should call
    /// `Manager::<C, O>::extend_slab_generic(...)` explicitly.
    pub fn extend_slab_generic<F>(base: &Slab<O>, ops: F) -> Diff<O>
    where
        Self: Default,
        F: FnOnce(&mut Self, &[Ref<O>]) -> Vec<Ref<O>>,
    {
        let mut m = Self::default();
        // Re-declare variables as prefix of the base. We don't know
        // exactly how many the base used (the Slab intentionally
        // doesn't carry that metadata — it's purely bytes + roots),
        // so we scan the arena for the max var byte and declare that
        // many. Terminals-only slabs have zero vars.
        let var_count = scan_max_var::<C, O>(&base.bytes);
        for _ in 0..var_count {
            m.new_var();
        }
        let base_roots = m.ingest_slab(base);
        let base_len = m.buf_len() as u64;
        let new_roots = ops(&mut m, &base_roots);
        m.diff_since(base_len, &new_roots)
    }
}

/// Default-engine inherent shortcut for [`Manager::extend_slab_generic`].
/// Pins `(Leb128Codec, u32)` so the closure's `&mut Manager` argument
/// doesn't need a type annotation at the call site.
impl Manager<crate::codec::Leb128Codec, u32> {
    /// See [`Manager::extend_slab_generic`]. This shortcut lets callers
    /// write `Manager::extend_slab(&base, |m, roots| { ... })` with no
    /// turbofish and no closure-arg type annotation.
    pub fn extend_slab<F>(base: &Slab<u32>, ops: F) -> Diff<u32>
    where
        F: FnOnce(&mut Self, &[Ref<u32>]) -> Vec<Ref<u32>>,
    {
        Self::extend_slab_generic(base, ops)
    }
}

/// Scan an arena byte stream and return `max_var + 1`, i.e. the number
/// of variables that must be declared on the recipient for the nodes
/// in this slab to type-check.
///
/// The codec puts `var` as the first byte of each node (for
/// `Leb128Codec`), so we can do this without decoding children. For
/// other codecs this may need to go through `decode_var`.
fn scan_max_var<C: NodeCodec<O>, O: ArenaOffset>(bytes: &[u8]) -> u32 {
    let mut max_var: i64 = -1;
    let mut pos = 0usize;
    while pos < bytes.len() {
        let off = O::from_u64(pos as u64);
        let (node, len) = C::decode(&bytes[pos..], off);
        if node.var as i64 > max_var {
            max_var = node.var as i64;
        }
        pos += len;
    }
    (max_var + 1).max(0) as u32
}
