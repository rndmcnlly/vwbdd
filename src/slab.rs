//! Slabs and diffs: the shipping units for compact-arena BDDs.
//!
//! ## The clean-bytes invariant
//!
//! Every [`Slab`] and [`Diff`] that crosses a public boundary is
//! **function-canonical**: running a GC on it would find nothing to
//! remove. The manager's internal arena may accumulate scratch from
//! `ite` and similar ops, but the moment bytes become visible to
//! another party (via [`Manager::slab_for`], [`Manager::diff_since`],
//! or [`Manager::extend_slab`]), they carry only the closure of the
//! declared roots.
//!
//! The invariant is inductive: `ingest_slab(clean_base)` leaves the
//! arena clean; `apply_diff(clean_base, clean_diff)` leaves it clean;
//! a call to [`Manager::gc`] before ship preserves it on output. The
//! only way to introduce scratch is via construction ops; that scratch
//! is invisible until the next `diff_since` / `slab_for`, which cleans
//! it out as part of emitting a public artifact.
//!
//! See VWBDD.md §4.15 for the three-levels-of-canonicity discussion
//! (function-canonical vs layout-canonical vs layout-minimal) and the
//! measurements that drove the invariant.
//!
//! ## Types
//!
//! - [`Slab`]: raw codec-encoded arena bytes plus a list of root
//!   [`Ref`]s. The durable artifact; the unique table is an
//!   ephemeral, private index rebuilt on demand.
//!
//! - [`Diff`]: an append-only delta against a known base. Carries
//!   only the tail bytes the recipient doesn't yet have plus new
//!   roots; the recipient reconstructs the full arena by
//!   concatenation. Position-independent: because every child code
//!   is `2 + (cur − child)` (a backward delta), nodes appended on
//!   top of a shared base encode to the same bytes regardless of
//!   which process built them.
//!
//! ## Example: the extend roundtrip
//!
//! ```
//! use vwbdd::{Manager, Slab};
//!
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
//! // Server: ingest base, run ops, ship a diff back.
//! let diff = Manager::extend_slab(&base, |m, base_roots| {
//!     let nand = m.not(base_roots[0]);
//!     vec![nand]
//! });
//!
//! // Client: ingest base, apply diff.
//! let mut client = Manager::new();
//! let _ = client.new_var();
//! let _ = client.new_var();
//! let base_roots = client.ingest_slab(&base);
//! let new_roots = client.apply_diff(&diff);
//! assert_eq!(base_roots.len(), 1);
//! assert_eq!(new_roots.len(), 1);
//! ```

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::codec::{decode_node, Node, Ref};
use crate::manager::Manager;

/// A compact shipping unit: the arena bytes and the exported root refs.
#[derive(Debug, Clone)]
pub struct Slab {
    /// Raw codec-encoded node stream. Byte-identical to what the
    /// manager's internal arena held at `slab_for` time, minus any dead
    /// (unreachable) nodes if the slab came from [`Manager::slab_for`].
    pub bytes: Vec<u8>,
    /// The roots the producer chose to export. Offsets refer into
    /// [`bytes`][Slab::bytes] (or are terminals).
    pub roots: Vec<Ref>,
}

impl Slab {
    /// Construct a slab directly. Mostly for tests and in-memory
    /// transport; usual producers are [`Manager::slab_for`].
    pub fn new(bytes: Vec<u8>, roots: Vec<Ref>) -> Self {
        Self { bytes, roots }
    }

    /// Byte length of the arena. The recipient will use this as the
    /// `base_len` boundary when applying any later [`Diff`].
    pub fn base_len(&self) -> u64 {
        self.bytes.len() as u64
    }

    /// Decode the node whose header starts at byte offset `off`.
    /// A clean slab's refs always land on valid node headers.
    #[inline]
    fn node_at(&self, off: u64) -> Node {
        let (node, _consumed) = decode_node(&self.bytes[off as usize..], off);
        node
    }

    /// The set of variables that `root` actually branches on.
    ///
    /// Read-only DFS over the reachable subgraph: no unique table or
    /// manager required. Runs in O(|reachable nodes|) via a visited
    /// memo so shared subDAGs are walked once.
    pub fn support(&self, root: Ref) -> BTreeSet<u32> {
        let mut out = BTreeSet::new();
        let mut visited: HashSet<u64> = HashSet::new();
        let mut stack: Vec<u64> = Vec::new();
        if let Ref::Node(off) = root {
            stack.push(off);
        }
        while let Some(off) = stack.pop() {
            if !visited.insert(off) {
                continue;
            }
            let n = self.node_at(off);
            out.insert(n.var);
            if let Ref::Node(lo) = n.lo {
                stack.push(lo);
            }
            if let Ref::Node(hi) = n.hi {
                stack.push(hi);
            }
        }
        out
    }

    /// Number of satisfying assignments of `root` over `{0,1}^nvars`.
    ///
    /// Variables not branched on each contribute a factor of 2, so the
    /// count is invariant under variable order and interleaving: it's
    /// a property of the function, not the BDD shape. Result is `f64`
    /// (exact up to 2^53).
    ///
    /// Implemented via fractional counts: each node memoizes "fraction
    /// of the sub-cube below me that satisfies", in `[0, 1]`. Skipped
    /// levels are free because a child that doesn't mention a variable
    /// has the same fraction regardless of its value.
    pub fn sat_count(&self, root: Ref, nvars: u32) -> f64 {
        let mut memo: HashMap<u64, f64> = HashMap::new();
        let frac = self.sat_frac(root, &mut memo);
        frac * (1u64 << nvars) as f64
    }

    fn sat_frac(&self, r: Ref, memo: &mut HashMap<u64, f64>) -> f64 {
        match r {
            Ref::Terminal(false) => 0.0,
            Ref::Terminal(true) => 1.0,
            Ref::Node(off) => {
                if let Some(&c) = memo.get(&off) {
                    return c;
                }
                let n = self.node_at(off);
                let lo = self.sat_frac(n.lo, memo);
                let hi = self.sat_frac(n.hi, memo);
                let c = 0.5 * lo + 0.5 * hi;
                memo.insert(off, c);
                c
            }
        }
    }
}

/// An append-only delta against a known base [`Slab`].
#[derive(Debug, Clone)]
pub struct Diff {
    /// The arena byte length the producer assumed the recipient holds.
    /// If it doesn't match, [`Manager::apply_diff`] panics.
    pub base_len: u64,
    /// Bytes to append verbatim to the recipient's arena.
    pub tail: Vec<u8>,
    /// New roots, expressed as refs into the **combined** arena
    /// (recipient base ‖ tail).
    pub new_roots: Vec<Ref>,
}

impl Diff {
    /// Tail byte length. Diagnostic; the "ship size" of an extension.
    pub fn tail_len(&self) -> u64 {
        self.tail.len() as u64
    }
}

// ---- Manager methods: the unified primitive surface ----

impl Manager {
    /// Ingest a base [`Slab`] into this manager.
    ///
    /// Panics if the manager's arena is non-empty: a slab is a *base*,
    /// so this is only meaningful on a fresh manager.
    ///
    /// **Precondition**: `slab.bytes` must be function-canonical.
    /// Slabs produced by [`Self::slab_for`] and by
    /// [`Self::apply_diff`] satisfy this by construction.
    pub fn ingest_slab(&mut self, slab: &Slab) -> Vec<Ref> {
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
    /// Delivers a function-canonical slab. Internally runs the copying
    /// GC, then wraps `(arena_bytes, translated_roots)`. The manager
    /// itself is mutated (its arena is replaced, its unique table
    /// rebuilt, its apply cache flushed).
    pub fn slab_for(&mut self, roots: &[Ref]) -> Slab {
        let new_roots = self.drop_roots(roots);
        Slab {
            bytes: self.arena_slice(0).to_vec(),
            roots: new_roots,
        }
    }

    /// Build a [`Diff`] describing the bytes appended since the arena
    /// had length `base_len`, plus the requested `new_roots`.
    ///
    /// **Clean-bytes invariant**: the returned diff's `tail` is
    /// function-canonical for `new_roots`. Internally runs a tail-only
    /// GC ([`Manager::gc`] with nonzero `base_len`) before extracting
    /// the tail bytes.
    ///
    /// Typical use: right after [`Self::ingest_slab`], record
    /// `base_len = m.buf_len()`. Run ops. Call `diff_since` with that
    /// length and the ops' results as `new_roots`. The returned
    /// `new_roots` may differ from what you passed in (if they pointed
    /// into the tail), because the tail's offsets got compacted. Use
    /// the returned ones from then on.
    pub fn diff_since(&mut self, base_len: u64, new_roots: &[Ref]) -> Diff {
        let cur_len = self.buf_len() as u64;
        assert!(
            base_len <= cur_len,
            "diff_since: base_len {} > current arena {}",
            base_len, cur_len
        );
        // Enforce the clean-bytes invariant.
        let cleaned_roots = self.gc(base_len, new_roots);
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
    /// `diff.base_len` — that would mean applying a diff against a base
    /// we don't hold, and the LEB128 backward-delta codes in the tail
    /// would resolve to the wrong child nodes.
    ///
    /// **Postcondition**: clean base + clean diff = clean combined
    /// arena, inductively; append alone never introduces scratch.
    pub fn apply_diff(&mut self, diff: &Diff) -> Vec<Ref> {
        let cur_len = self.buf_len() as u64;
        assert_eq!(
            cur_len, diff.base_len,
            "apply_diff: recipient arena is {} bytes but diff assumes base of {}",
            cur_len, diff.base_len
        );
        self.buf_mut().extend_from_slice(&diff.tail);
        // Rebuild the unique table over the full arena.
        self.rebuild_unique_from_arena();
        diff.new_roots.clone()
    }

    /// The unified server-side primitive: ingest a base, run ops, emit
    /// a diff. Single call so the typical client never has to manage
    /// the `base_len` book-keeping manually.
    ///
    /// Uses a fresh manager internally so the caller's existing state
    /// is untouched. If you want to run ops in your own manager,
    /// compose the pieces manually: [`Self::ingest_slab`] → do work →
    /// [`Self::diff_since`].
    pub fn extend_slab<F>(base: &Slab, ops: F) -> Diff
    where
        F: FnOnce(&mut Self, &[Ref]) -> Vec<Ref>,
    {
        let mut m = Self::default();
        // Re-declare variables as prefix of the base. The slab doesn't
        // carry var-count metadata (it's purely bytes + roots), so we
        // scan the arena for the max var byte.
        let var_count = scan_max_var(&base.bytes);
        for _ in 0..var_count {
            m.new_var();
        }
        let base_roots = m.ingest_slab(base);
        let base_len = m.buf_len() as u64;
        let new_roots = ops(&mut m, &base_roots);
        m.diff_since(base_len, &new_roots)
    }
}

/// Scan an arena byte stream and return `max_var + 1`, i.e. the number
/// of variables that must be declared on the recipient for the nodes
/// to type-check.
fn scan_max_var(bytes: &[u8]) -> u32 {
    let mut max_var: i64 = -1;
    let mut pos = 0usize;
    while pos < bytes.len() {
        let off = pos as u64;
        let (node, len) = decode_node(&bytes[pos..], off);
        if node.var as i64 > max_var {
            max_var = node.var as i64;
        }
        pos += len;
    }
    (max_var + 1).max(0) as u32
}
