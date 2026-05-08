//! The live BDD manager. Append-only buffer + unique table for canonicity +
//! an apply cache for ite.
//!
//! Generic over [`NodeCodec`] (how nodes are laid out in bytes) and
//! [`ArenaOffset`] (the numeric type used to address arena bytes). See
//! `crate::codec` for the trait definitions and the shipped defaults
//! ([`Leb128Codec`] and `u32` / `u64`).

use std::collections::HashMap;
use std::marker::PhantomData;

use crate::codec::{ArenaOffset, Leb128Codec, Node, NodeCodec, Ref, ref_to_u64};
use crate::unique::{unique_key_hash, CompactUnique};

/// Fast mixing hash for an (f, g, h) ite-cache triple. Splitmix-style chain.
#[inline]
fn ite_key_hash<O: ArenaOffset>(f: Ref<O>, g: Ref<O>, h: Ref<O>) -> u64 {
    let mut x = ref_to_u64::<O>(f);
    x = x.wrapping_mul(0x9e3779b97f4a7c15);
    x ^= ref_to_u64::<O>(g);
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x ^= ref_to_u64::<O>(h);
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

/// A packed `Ref`, used only in bulk storage (the apply cache). One native
/// offset word wide: 4 bytes at `O = u32`, 8 bytes at `O = u64`.
///
/// Encoding (shared across widths):
///   0          -> Terminal(false)
///   1          -> Terminal(true)
///   2          -> empty sentinel (replaces a `filled: bool` field)
///   3..=MAX    -> Node(value - 3), i.e. arena offset
///
/// At `O = u32` this gives us 16 B/IteCacheEntry = exactly one cache line
/// (§4.8's original win). At `O = u64` the entry is 32 B = half a line (two
/// per line); the §4.11 trade for removing the 4 GiB arena ceiling.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
struct PackedRef<O: ArenaOffset>(O);

impl<O: ArenaOffset> PackedRef<O> {
    fn r#false() -> Self {
        Self(O::ZERO)
    }
    fn r#true() -> Self {
        Self(O::ONE)
    }
    fn empty() -> Self {
        Self(O::from_u64(2))
    }

    #[inline]
    fn pack(r: Ref<O>) -> Self {
        match r {
            Ref::Terminal(false) => Self::r#false(),
            Ref::Terminal(true) => Self::r#true(),
            Ref::Node(off) => {
                // `off + 3` must still fit in `O`. At u32 this excludes the
                // top three byte-offsets in a 4 GiB arena, which is already
                // forbidden by CompactUnique's slot-width promise. At u64 the
                // bound is irrelevant in practice.
                let packed = off
                    .checked_add(O::from_u64(3))
                    .expect("arena offset exceeds PackedRef limit");
                Self(packed)
            }
        }
    }

    #[inline]
    fn unpack(self) -> Ref<O> {
        let v = self.0.to_u64();
        match v {
            0 => Ref::Terminal(false),
            1 => Ref::Terminal(true),
            2 => unreachable!("EMPTY PackedRef should never be unpacked"),
            _ => Ref::Node(O::from_u64(v - 3)),
        }
    }
}

/// Direct-mapped apply cache entry. Empty slot is marked by `f == EMPTY`.
/// 4 × `O` bytes — 16 B at u32, 32 B at u64.
#[repr(C)]
#[derive(Copy, Clone)]
struct IteCacheEntry<O: ArenaOffset> {
    f: PackedRef<O>,
    g: PackedRef<O>,
    h: PackedRef<O>,
    r: PackedRef<O>,
}

impl<O: ArenaOffset> IteCacheEntry<O> {
    fn empty() -> Self {
        Self {
            f: PackedRef::empty(),
            g: PackedRef::empty(),
            h: PackedRef::empty(),
            r: PackedRef::empty(),
        }
    }
}

/// Default number of slots in the direct-mapped ite apply cache.
/// 2^21 = 2,097,152 entries:
/// - u32: 32 MiB cache
/// - u64: 64 MiB cache
///
/// Chosen to clear the k=15 truncated-mult working set (~10M edges) while
/// staying trivial relative to any real workload's RAM budget. The earlier
/// 2^17 default (§4.5, §4.8) turned out to be a silent perf trap once
/// workloads grew past k=11: throughput fell 2-3× when the working set
/// overflowed the cache. Override via [`ManagerConfig::with_cache_slots`]
/// for small-arena workloads where the default's 32 MiB is wasteful.
pub const DEFAULT_ITE_CACHE_SLOTS: usize = 1 << 21;

/// Construction-time options for [`Manager`]. Builder-shaped so that future
/// knobs (per-level unique-table partitioning, GC-trigger heuristics, etc.)
/// can be added without breaking existing call sites.
///
/// Mirrors oxidd's approach: the library doesn't guess — the caller names
/// the capacities explicitly, but a `default()` gives a sensible baseline.
#[derive(Debug, Clone, Copy)]
pub struct ManagerConfig {
    ite_cache_slots: usize,
}

impl ManagerConfig {
    /// Default config: [`DEFAULT_ITE_CACHE_SLOTS`] entries in the apply cache.
    pub const fn new() -> Self {
        Self {
            ite_cache_slots: DEFAULT_ITE_CACHE_SLOTS,
        }
    }

    /// Set the direct-mapped ite apply cache size in slots. Must be a
    /// power of two so the probe can use `hash & (slots - 1)` instead of
    /// a modulo. Panics otherwise.
    ///
    /// Typical values: `1 << 17` (128k, 2 MiB) for tight-memory builds;
    /// `1 << 21` (2M, 32 MiB, the default); `1 << 25` (32M, 512 MiB) for
    /// oxidd-cli-scale workloads.
    #[must_use]
    pub const fn with_cache_slots(mut self, slots: usize) -> Self {
        assert!(
            slots.is_power_of_two(),
            "ite_cache_slots must be a power of two"
        );
        assert!(slots >= 2, "ite_cache_slots must be at least 2");
        self.ite_cache_slots = slots;
        self
    }

    /// Current cache-slot count.
    pub const fn ite_cache_slots(self) -> usize {
        self.ite_cache_slots
    }
}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// The live BDD engine.
///
/// Default type parameters match the wasm-friendly compact build:
/// `Leb128Codec` + `u32` offsets (4 GiB arena cap, ~6.7 B/node unique-table
/// density). For the large-arena server build use
/// [`crate::LargeManager`] (`Manager<Leb128Codec, u64>`).
pub struct Manager<C: NodeCodec<O> = Leb128Codec, O: ArenaOffset = u32> {
    buf: Vec<u8>,
    /// Unique table: compact linear-probe, `O`-width offsets + u8 tags,
    /// 0.75 load factor on power-of-two sizes (§4.7, §4.11).
    unique: CompactUnique<C, O>,
    num_vars: u32,
    /// Direct-mapped ite apply cache. Fixed-size (set at construction);
    /// collisions evict. See [`ManagerConfig::with_cache_slots`].
    ite_cache: Box<[IteCacheEntry<O>]>,
    /// `ite_cache.len() - 1`, precomputed for the `hash & mask` fast path.
    /// Since `ite_cache.len()` is always a power of two (enforced by
    /// [`ManagerConfig::with_cache_slots`]), this mask gives the slot index.
    ite_cache_mask: u64,
    /// Live entry count (for diagnostics; approximate under eviction).
    ite_cache_len: usize,
    _codec: PhantomData<C>,
}

impl<C: NodeCodec<O>, O: ArenaOffset> Manager<C, O> {
    /// Construct a manager with explicit config. The generic-accepting path;
    /// callers using the default type parameters can use the inherent
    /// `Manager::new()` or `Manager::with_cache_slots(n)`.
    pub fn with_config(config: ManagerConfig) -> Self {
        let slots = config.ite_cache_slots;
        debug_assert!(slots.is_power_of_two());
        debug_assert!(slots >= 2);
        // vec! + into_boxed_slice keeps the heap allocation contiguous
        // without ever building the huge array on the stack.
        let ite_cache: Box<[IteCacheEntry<O>]> =
            vec![IteCacheEntry::empty(); slots].into_boxed_slice();
        Self {
            buf: Vec::new(),
            unique: CompactUnique::new(),
            num_vars: 0,
            ite_cache,
            ite_cache_mask: (slots as u64) - 1,
            ite_cache_len: 0,
            _codec: PhantomData,
        }
    }

    pub fn new_var(&mut self) -> u32 {
        let v = self.num_vars;
        self.num_vars += 1;
        v
    }

    pub fn num_vars(&self) -> u32 {
        self.num_vars
    }

    pub fn constant(&self, value: bool) -> Ref<O> {
        Ref::Terminal(value)
    }

    pub fn r#false(&self) -> Ref<O> {
        Ref::Terminal(false)
    }

    pub fn r#true(&self) -> Ref<O> {
        Ref::Terminal(true)
    }

    pub fn num_nodes(&self) -> usize {
        self.unique.len()
    }

    pub fn buf_len(&self) -> usize {
        self.buf.len()
    }

    /// Raw arena view from an offset. For diagnostic tooling; callers should
    /// know they're reading codec-encoded nodes.
    pub fn arena_slice(&self, off: usize) -> &[u8] {
        &self.buf[off..]
    }

    /// Mutable access to the arena buffer. Used only by the crate-internal
    /// dump/load path (`src/dump.rs`) to append raw bytes after a file
    /// read. Callers must follow up with
    /// [`Manager::rebuild_unique_from_arena`] so the unique table reflects
    /// the new nodes.
    pub(crate) fn buf_mut(&mut self) -> &mut Vec<u8> {
        &mut self.buf
    }

    /// Walk the current arena in construction order and rebuild the unique
    /// table from scratch. Used internally by [`Self::gc`] and by the
    /// dump/load path in `src/dump.rs`.
    pub(crate) fn rebuild_unique_from_arena(&mut self) {
        self.unique_rebuild_from_arena();
    }

    /// Memory breakdown (arena + unique + cache). `unique_bytes` includes
    /// empty slots; at 0.75 load the linear-probe table runs ~(1.33 × slot_width)
    /// bytes per node. `cache_bytes` reflects the cache size chosen at
    /// construction via [`ManagerConfig`].
    pub fn mem_stats(&self) -> MemStats {
        MemStats {
            arena_bytes: self.buf.len(),
            unique_bytes: self.unique.bytes(),
            cache_bytes: self.ite_cache.len() * std::mem::size_of::<IteCacheEntry<O>>(),
            unique_entries: self.num_nodes(),
            cache_entries: self.ite_cache_len,
        }
    }

    pub fn var_of(&self, r: Ref<O>) -> Option<u32> {
        match r {
            Ref::Terminal(_) => None,
            Ref::Node(off) => {
                let (var, _) = C::decode_var(&self.buf[off.to_u64() as usize..], off);
                Some(var)
            }
        }
    }

    pub fn decode_node(&self, r: Ref<O>) -> Option<Node<O>> {
        match r {
            Ref::Terminal(_) => None,
            Ref::Node(off) => {
                let (n, _) = C::decode(&self.buf[off.to_u64() as usize..], off);
                Some(n)
            }
        }
    }

    pub fn make_node(&mut self, var: u32, lo: Ref<O>, hi: Ref<O>) -> Ref<O> {
        assert!(var < self.num_vars, "var {} not declared", var);
        if lo == hi {
            return lo;
        }
        if let Ref::Node(_) = lo {
            let lv = self.var_of(lo).unwrap();
            assert!(
                lv > var,
                "variable ordering violated: parent var={} lo var={}",
                var, lv
            );
        }
        if let Ref::Node(_) = hi {
            let hv = self.var_of(hi).unwrap();
            assert!(
                hv > var,
                "variable ordering violated: parent var={} hi var={}",
                var, hv
            );
        }
        let hash = unique_key_hash::<O>(var, lo, hi);
        if let Some(off) = self.unique.lookup(hash, var, lo, hi, &self.buf) {
            return Ref::Node(off);
        }
        let new_off = O::from_u64(self.buf.len() as u64);
        C::encode(var, lo, hi, new_off, &mut self.buf);
        self.unique.insert(hash, new_off, &self.buf);
        Ref::Node(new_off)
    }

    /// Top variable of a ref. Terminals return None (treated as infinity).
    fn top_var(&self, r: Ref<O>) -> Option<u32> {
        self.var_of(r)
    }

    /// Cofactor r at variable v (that is, substitute v = val and return the
    /// simplified function). If r's top var is deeper than v, r is unchanged.
    fn cofactor(&self, r: Ref<O>, v: u32, val: bool) -> Ref<O> {
        match r {
            Ref::Terminal(_) => r,
            Ref::Node(_off) => {
                let node_var = self.var_of(r).unwrap();
                if node_var > v {
                    // r doesn't depend on v at this level.
                    r
                } else if node_var == v {
                    // Pluck the appropriate child.
                    let node = self.decode_node(r).unwrap();
                    if val {
                        node.hi
                    } else {
                        node.lo
                    }
                } else {
                    // node_var < v: shouldn't happen if we're cofactoring at
                    // the top var of the current frame.
                    panic!(
                        "cofactor called below current level: node_var={}, v={}",
                        node_var, v
                    );
                }
            }
        }
    }

    /// Short-circuit terminal cases for ite. Returns Some(result) if
    /// applicable without recursion; None if real work is needed.
    fn ite_trivial(&self, f: Ref<O>, g: Ref<O>, h: Ref<O>) -> Option<Ref<O>> {
        if let Ref::Terminal(fv) = f {
            return Some(if fv { g } else { h });
        }
        if g == h {
            return Some(g);
        }
        if matches!(g, Ref::Terminal(true)) && matches!(h, Ref::Terminal(false)) {
            return Some(f);
        }
        None
    }

    /// Probe the direct-mapped apply cache. Returns Some(r) on hit, None on
    /// empty slot or collision miss.
    #[inline]
    fn ite_cache_get(&self, f: Ref<O>, g: Ref<O>, h: Ref<O>) -> Option<Ref<O>> {
        let slot = (ite_key_hash::<O>(f, g, h) & self.ite_cache_mask) as usize;
        let e = &self.ite_cache[slot];
        let pf = PackedRef::<O>::pack(f);
        if e.f == pf && e.g == PackedRef::<O>::pack(g) && e.h == PackedRef::<O>::pack(h) {
            Some(e.r.unpack())
        } else {
            None
        }
    }

    /// Insert into the direct-mapped apply cache. Overwrites on collision.
    #[inline]
    fn ite_cache_put(&mut self, f: Ref<O>, g: Ref<O>, h: Ref<O>, r: Ref<O>) {
        let slot = (ite_key_hash::<O>(f, g, h) & self.ite_cache_mask) as usize;
        let e = &mut self.ite_cache[slot];
        if e.f == PackedRef::<O>::empty() {
            self.ite_cache_len += 1;
        }
        *e = IteCacheEntry {
            f: PackedRef::pack(f),
            g: PackedRef::pack(g),
            h: PackedRef::pack(h),
            r: PackedRef::pack(r),
        };
    }

    /// Iterative Shannon-expansion ite. Explicit work-stack instead of
    /// program-stack recursion, so we don't blow the thread stack on deep
    /// BDDs.
    ///
    /// Because the apply cache is direct-mapped (evicts on collision), we
    /// *cannot* rely on it to carry intermediate results between child
    /// completion and parent combine. Instead, we maintain a parallel result
    /// stack: each Enter that does real work pushes a Combine and two child
    /// Enters; when a child completes it pushes its result onto `results`;
    /// Combine pops two results, builds the node, and pushes its own.
    pub fn ite(&mut self, f: Ref<O>, g: Ref<O>, h: Ref<O>) -> Ref<O> {
        if let Some(r) = self.ite_trivial(f, g, h) {
            return r;
        }
        if let Some(r) = self.ite_cache_get(f, g, h) {
            return r;
        }

        enum Frame<O: ArenaOffset> {
            Enter { f: Ref<O>, g: Ref<O>, h: Ref<O> },
            Combine { f: Ref<O>, g: Ref<O>, h: Ref<O>, tv: u32 },
        }
        let mut stack: Vec<Frame<O>> = Vec::with_capacity(64);
        let mut results: Vec<Ref<O>> = Vec::with_capacity(64);
        stack.push(Frame::Enter { f, g, h });

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter { f, g, h } => {
                    if let Some(r) = self.ite_trivial(f, g, h) {
                        results.push(r);
                        continue;
                    }
                    if let Some(r) = self.ite_cache_get(f, g, h) {
                        results.push(r);
                        continue;
                    }
                    let tv = [self.top_var(f), self.top_var(g), self.top_var(h)]
                        .iter()
                        .filter_map(|x| *x)
                        .min()
                        .unwrap();
                    let f0 = self.cofactor(f, tv, false);
                    let f1 = self.cofactor(f, tv, true);
                    let g0 = self.cofactor(g, tv, false);
                    let g1 = self.cofactor(g, tv, true);
                    let h0 = self.cofactor(h, tv, false);
                    let h1 = self.cofactor(h, tv, true);

                    // Combine pops 2 results (lo then hi) and pushes 1.
                    stack.push(Frame::Combine { f, g, h, tv });
                    // Push hi first so lo completes first (LIFO); Combine
                    // expects hi on top, then lo below it.
                    stack.push(Frame::Enter { f: f1, g: g1, h: h1 });
                    stack.push(Frame::Enter { f: f0, g: g0, h: h0 });
                }
                Frame::Combine { f, g, h, tv } => {
                    let hi = results.pop().expect("hi result missing");
                    let lo = results.pop().expect("lo result missing");
                    let r = self.make_node(tv, lo, hi);
                    self.ite_cache_put(f, g, h, r);
                    results.push(r);
                }
            }
        }

        debug_assert_eq!(results.len(), 1);
        results.pop().expect("ite produced no result")
    }

    // Derived boolean operations.
    pub fn not(&mut self, f: Ref<O>) -> Ref<O> {
        let t = self.r#true();
        let fa = self.r#false();
        self.ite(f, fa, t)
    }

    pub fn and(&mut self, f: Ref<O>, g: Ref<O>) -> Ref<O> {
        let fa = self.r#false();
        self.ite(f, g, fa)
    }

    pub fn or(&mut self, f: Ref<O>, g: Ref<O>) -> Ref<O> {
        let t = self.r#true();
        self.ite(f, t, g)
    }

    pub fn xor(&mut self, f: Ref<O>, g: Ref<O>) -> Ref<O> {
        let ng = self.not(g);
        self.ite(f, ng, g)
    }

    /// **Drop everything except these roots.** Major garbage collection:
    /// replaces the arena with a fresh one containing only the closure
    /// of `keep`, returning the translated roots.
    ///
    /// This is the only public operation that shrinks a manager. The
    /// invariant it supports: after `drop_roots`, running a GC again
    /// would find nothing to remove — the arena is *function-canonical*
    /// for the given roots, carrying no scratch from prior operations.
    /// (Not *layout-canonical*: node construction order during ops
    /// determined the current byte layout, which `drop_roots` preserves
    /// in relative order but compacts. Two managers that built the same
    /// BDD via different op sequences can produce different byte
    /// encodings of the same canonical DAG. See VWBDD.md's discussion
    /// of the three canonicity levels.)
    ///
    /// Replaces the older name [`Self::gc`] at the caller's level of
    /// abstraction: `drop_roots` names the *intent* (these are the
    /// roots I still care about), while `gc` names the *mechanism*.
    /// Both are public; callers should prefer `drop_roots` for new
    /// code. The apply cache is flushed as a side effect.
    pub fn drop_roots(&mut self, keep: &[Ref<O>]) -> Vec<Ref<O>> {
        self.gc(keep)
    }

    /// Copying garbage collector. Takes a slice of roots, builds a fresh arena
    /// containing only the nodes reachable from those roots, and returns the
    /// remapped roots. The apply cache is flushed; callers must replace their
    /// own held refs with the returned ones.
    ///
    /// Mechanism-level name; see [`Self::drop_roots`] for the intent-level
    /// name recommended for new code.
    pub fn gc(&mut self, roots: &[Ref<O>]) -> Vec<Ref<O>> {
        let mut remap: HashMap<u64, Ref<O>> = HashMap::new();
        let mut new_buf: Vec<u8> = Vec::new();

        enum Frame<O: ArenaOffset> {
            Enter(O),
            Exit(O),
        }
        let old_buf = &self.buf;

        for &root in roots {
            if let Ref::Node(off) = root {
                if remap.contains_key(&off.to_u64()) { continue; }
                let mut stack: Vec<Frame<O>> = vec![Frame::Enter(off)];
                while let Some(frame) = stack.pop() {
                    match frame {
                        Frame::Enter(o) => {
                            if remap.contains_key(&o.to_u64()) { continue; }
                            let (node, _) = C::decode(&old_buf[o.to_u64() as usize..], o);
                            stack.push(Frame::Exit(o));
                            if let Ref::Node(lo_off) = node.lo {
                                if !remap.contains_key(&lo_off.to_u64()) {
                                    stack.push(Frame::Enter(lo_off));
                                }
                            }
                            if let Ref::Node(hi_off) = node.hi {
                                if !remap.contains_key(&hi_off.to_u64()) {
                                    stack.push(Frame::Enter(hi_off));
                                }
                            }
                        }
                        Frame::Exit(o) => {
                            if remap.contains_key(&o.to_u64()) { continue; }
                            let (node, _) = C::decode(&old_buf[o.to_u64() as usize..], o);
                            let new_lo = translate::<O>(&remap, node.lo);
                            let new_hi = translate::<O>(&remap, node.hi);
                            // Reduction (shouldn't fire on canonical input).
                            let new_ref = if new_lo == new_hi {
                                new_lo
                            } else {
                                let new_off = O::from_u64(new_buf.len() as u64);
                                C::encode(node.var, new_lo, new_hi, new_off, &mut new_buf);
                                Ref::Node(new_off)
                            };
                            remap.insert(o.to_u64(), new_ref);
                        }
                    }
                }
            }
        }

        // Swap in the new arena.
        self.buf = new_buf;

        // Rebuild unique table from the new arena.
        self.unique_rebuild_from_arena();

        // Flush apply cache: old offsets are dead.
        for e in self.ite_cache.iter_mut() {
            *e = IteCacheEntry::empty();
        }
        self.ite_cache_len = 0;

        // Translate roots.
        roots.iter().map(|&r| translate::<O>(&remap, r)).collect()
    }

    /// **Minor (tail-only) garbage collection.** Frees unreachable bytes
    /// in the *tail* of the arena (the region past `base_len`), leaving
    /// the base bytes `[0..base_len)` untouched byte-for-byte.
    ///
    /// **Most callers don't need this directly.**
    /// [`Manager::diff_since`](crate::slab) runs `gc_tail` internally
    /// to enforce the clean-bytes invariant on diffs it produces. This
    /// method is exposed for workloads that want to minor-GC without
    /// immediately producing a diff (e.g., long-running queries that
    /// accumulate scratch across many ops and periodically want to
    /// reclaim it without a full major GC).
    ///
    /// This is the generational counterpart to [`Self::gc`]. Where
    /// `gc` is a major collection that rebuilds the entire arena,
    /// `gc_tail` touches only the young generation: nodes built since
    /// we last "promoted" a base. The base/tail boundary is a parameter;
    /// the typical value comes from recording `m.buf_len()` right after
    /// [`Self::ingest_slab`] (see `src/slab.rs`).
    ///
    /// **Why this works without a write barrier.** A generational GC
    /// usually needs a remembered set to track old→young pointers. We
    /// don't, because the LEB128 codec encodes children as backward
    /// byte deltas: a node at offset `o` can only reference children
    /// at offsets `< o`. So a base node (offset `< base_len`) can never
    /// reference a tail node (offset `>= base_len`) — the direction is
    /// syntactically forbidden. Tail→base references, on the other
    /// hand, are fine and must be preserved verbatim: the base doesn't
    /// move, so a tail-to-base child keeps the same absolute offset
    /// (though its backward-delta encoding will change because the
    /// *parent* offset may have shifted after compaction).
    ///
    /// **Roots handling.** `roots` may contain terminals, base nodes,
    /// or tail nodes. Terminals and base-node roots pass through
    /// unchanged. Tail-node roots get remapped to their new tail
    /// offsets.
    ///
    /// **What you get back.** The returned vector is `roots` translated
    /// through the remap. After this call, `self.buf_len()` is typically
    /// `<` the pre-call value (tail shrank); `base_len` is unchanged.
    /// A subsequent `diff_since(base_len, translated_roots)` produces
    /// the minimum shippable diff for those roots — this is the natural
    /// pre-ship optimization pass.
    ///
    /// **Cost.** O(live tail nodes) for the walk + rebuild. The unique
    /// table is rebuilt from the *full* arena (base + new tail) because
    /// the base nodes' slots were fine but the tail nodes got new
    /// offsets; simplest correct option, and base nodes re-insert with
    /// the same hash-to-slot mapping they had before (deterministic
    /// splitmix). A future refinement could preserve the base's unique
    /// table entries and only reinsert tail entries; left for a
    /// profile-driven session.
    ///
    /// **When this wins and when it doesn't** (measured in
    /// `tests/minor_gc_savings.rs` on `trunc-mult` at k=5..11):
    ///
    /// | tail shape                              | byte shrink      |
    /// |-----------------------------------------|------------------|
    /// | all scratch (query folds to terminals)  | ∞ (tail → 0 B)   |
    /// | mostly-live (~1% dead)                  | ~0.99-1.04× (wash)|
    /// | tiny result (~1-2 nodes)                | ~1.6-1.75×       |
    ///
    /// The surprise: at very low dead-node fractions, re-encoding
    /// surviving nodes at their new (compacted) parent offsets can
    /// push a handful of LEB128 child-delta codes across a 7-bit
    /// boundary, costing more bytes than the dead nodes would have
    /// saved. Net negative by ~1% at k=11 when only 4 of 3075 tail
    /// nodes were dead. The call is therefore a *choice the caller
    /// makes* based on knowledge of the ops: reach for it when you
    /// know the query is scratch-dominated (model-counting-style
    /// questions, satisfiability probes), skip it when you're
    /// accumulating live structure.
    pub fn gc_tail(&mut self, base_len: u64, roots: &[Ref<O>]) -> Vec<Ref<O>> {
        let buf_len = self.buf.len() as u64;
        assert!(
            base_len <= buf_len,
            "gc_tail: base_len {} > arena len {}",
            base_len, buf_len
        );
        if base_len == buf_len {
            // Nothing to collect; tail is empty. Still translate roots
            // (they must all be terminals or base nodes).
            return roots.to_vec();
        }

        // remap: tail-offset (u64) → new Ref<O>. Terminals and base
        // nodes are NOT stored here; they pass through translate_young.
        let mut remap: HashMap<u64, Ref<O>> = HashMap::new();

        // Strategy: snapshot the old tail bytes, truncate `self.buf`
        // to `base_len`, then re-encode surviving tail nodes directly
        // into `self.buf`. This satisfies the codec's invariant
        // (out.len() == current_offset at encode time) without having
        // to pre-fill a scratch buffer.
        let old_tail: Vec<u8> = self.buf[base_len as usize..].to_vec();

        enum Frame<O: ArenaOffset> {
            Enter(O),
            Exit(O),
        }

        // Helper: is this offset in the young generation?
        let is_young = |o: u64| -> bool { o >= base_len };

        // Walk each root over the OLD arena bytes. We keep a separate
        // closure over the full old buffer (base + old_tail) by
        // concatenating base bytes (still in self.buf up to base_len)
        // with old_tail — but actually we can just read directly from
        // self.buf.as_slice() BEFORE truncation. Do the walk first,
        // collect the post-order list of surviving offsets, then
        // truncate and re-encode.
        let mut post_order: Vec<O> = Vec::new();
        {
            let old_buf = &self.buf;
            for &root in roots {
                if let Ref::Node(off) = root {
                    let ov = off.to_u64();
                    if !is_young(ov) {
                        continue; // base root; nothing to do
                    }
                    if remap.contains_key(&ov) {
                        continue;
                    }
                    let mut stack: Vec<Frame<O>> = vec![Frame::Enter(off)];
                    while let Some(frame) = stack.pop() {
                        match frame {
                            Frame::Enter(o) => {
                                let ov = o.to_u64();
                                if remap.contains_key(&ov) { continue; }
                                // Sentinel: mark we're walking this
                                // node. We'll fill in remap at Exit.
                                // Use a placeholder to avoid
                                // re-enqueueing; simplest is to check
                                // a separate "seen" set. Reuse remap
                                // with a temporary sentinel doesn't
                                // work because we need the real value
                                // at Exit. Use a small visited set.
                                debug_assert!(is_young(ov));
                                let (node, _) = C::decode(&old_buf[ov as usize..], o);
                                stack.push(Frame::Exit(o));
                                if let Ref::Node(lo_off) = node.lo {
                                    let lv = lo_off.to_u64();
                                    if is_young(lv) && !remap.contains_key(&lv) {
                                        stack.push(Frame::Enter(lo_off));
                                    }
                                }
                                if let Ref::Node(hi_off) = node.hi {
                                    let hv = hi_off.to_u64();
                                    if is_young(hv) && !remap.contains_key(&hv) {
                                        stack.push(Frame::Enter(hi_off));
                                    }
                                }
                            }
                            Frame::Exit(o) => {
                                let ov = o.to_u64();
                                if remap.contains_key(&ov) { continue; }
                                // Mark seen by inserting a placeholder;
                                // we'll overwrite with the real new ref
                                // after re-encoding below.
                                remap.insert(ov, Ref::Terminal(false));
                                post_order.push(o);
                            }
                        }
                    }
                }
            }
        }

        // Now truncate the buffer to just the base, and re-encode
        // surviving tail nodes directly into it.
        //
        // We need to decode children from the OLD tail bytes (which
        // we just snapshotted into `old_tail`) — the self.buf tail
        // region will be progressively rewritten as we go, so we can't
        // decode from it. Base children are at offsets < base_len and
        // remain in self.buf untouched.
        self.buf.truncate(base_len as usize);

        // Clear the placeholder entries so translate_young's lookup
        // gets the real values we're about to insert.
        remap.clear();

        // Re-walk post_order (already in topological Exit order) and
        // encode. Decode each node's original children from old_tail
        // (for young children) or from self.buf (for base children,
        // though we only need them via translate_young which is
        // identity on base).
        for old_off in post_order {
            let ov = old_off.to_u64();
            // Read the node from old_tail. Offset within old_tail is
            // (ov - base_len).
            let tail_rel = (ov - base_len) as usize;
            let (node, _) = C::decode(&old_tail[tail_rel..], old_off);
            let new_lo = translate_young::<O>(&remap, node.lo, base_len);
            let new_hi = translate_young::<O>(&remap, node.hi, base_len);
            let new_ref = if new_lo == new_hi {
                new_lo
            } else {
                let new_off = O::from_u64(self.buf.len() as u64);
                C::encode(node.var, new_lo, new_hi, new_off, &mut self.buf);
                Ref::Node(new_off)
            };
            remap.insert(ov, new_ref);
        }

        // Rebuild the unique table over the full arena. Both base and
        // tail entries get reinserted; the base-hashes are deterministic
        // under splitmix, so nothing changes for them structurally — but
        // we still have to refill the table because we don't have a
        // per-generation unique-table today.
        self.unique_rebuild_from_arena();

        // Flush apply cache: any entry that referenced a stale tail
        // offset would now resolve to a different node. Conservative
        // choice; base-only entries would technically survive but we
        // don't distinguish them.
        for e in self.ite_cache.iter_mut() {
            *e = IteCacheEntry::empty();
        }
        self.ite_cache_len = 0;

        // Translate roots for the caller.
        roots.iter()
            .map(|&r| translate_young::<O>(&remap, r, base_len))
            .collect()
    }

    /// Rebuild the unique table by scanning the current arena in construction
    /// order. Called after GC, which overwrites `self.buf` with a fresh arena.
    fn unique_rebuild_from_arena(&mut self) {
        let mut live = 0usize;
        {
            let mut pos: usize = 0;
            while pos < self.buf.len() {
                let off = O::from_u64(pos as u64);
                let (_, len) = C::decode(&self.buf[pos..], off);
                live += 1;
                pos += len;
            }
        }
        self.unique.resize_for(live);
        let mut pos: usize = 0;
        while pos < self.buf.len() {
            let off = O::from_u64(pos as u64);
            let (node, len) = C::decode(&self.buf[pos..], off);
            let hash = unique_key_hash::<O>(node.var, node.lo, node.hi);
            self.unique.insert(hash, off, &self.buf);
            pos += len;
        }
    }
}

fn translate<O: ArenaOffset>(remap: &HashMap<u64, Ref<O>>, r: Ref<O>) -> Ref<O> {
    match r {
        Ref::Terminal(_) => r,
        Ref::Node(off) => *remap.get(&off.to_u64()).expect("root unreachable in remap"),
    }
}

/// Minor-GC variant of [`translate`]: base-generation node refs pass
/// through unchanged (they weren't walked, so they aren't in the remap),
/// young-generation refs go through the remap.
fn translate_young<O: ArenaOffset>(
    remap: &HashMap<u64, Ref<O>>,
    r: Ref<O>,
    base_len: u64,
) -> Ref<O> {
    match r {
        Ref::Terminal(_) => r,
        Ref::Node(off) => {
            let ov = off.to_u64();
            if ov < base_len {
                // Base node: identity.
                r
            } else {
                // Young node: must be in the remap because we walked
                // every reachable tail node.
                *remap.get(&ov).expect("young root unreachable in minor-GC remap")
            }
        }
    }
}

impl<C: NodeCodec<O>, O: ArenaOffset> Default for Manager<C, O> {
    fn default() -> Self {
        Self::with_config(ManagerConfig::default())
    }
}

/// Convenience inherent impls on the default (u32, Leb128) type so callers
/// can write `Manager::new()` and `Manager::with_cache_slots(n)` without
/// turbofish. Type inference picks these over the generic `with_config`
/// path. For other configurations use a type alias plus
/// `::with_config(...)` or `::default()`:
///
/// ```ignore
/// let m: LargeManager = LargeManager::default();
/// let m: LargeManager = LargeManager::with_config(
///     ManagerConfig::default().with_cache_slots(1 << 20)
/// );
/// ```
impl Manager<Leb128Codec, u32> {
    /// Default-config manager: compact u32 arena, Leb128 codec, 2^21-slot
    /// apply cache.
    pub fn new() -> Self {
        Self::with_config(ManagerConfig::default())
    }

    /// Convenience: build a default-width engine with an overridden apply
    /// cache slot count. Equivalent to
    /// `Self::with_config(ManagerConfig::new().with_cache_slots(slots))`.
    pub fn with_cache_slots(slots: usize) -> Self {
        Self::with_config(ManagerConfig::new().with_cache_slots(slots))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MemStats {
    pub arena_bytes: usize,
    pub unique_bytes: usize,
    pub cache_bytes: usize,
    pub unique_entries: usize,
    pub cache_entries: usize,
}

impl MemStats {
    pub fn total_live(&self) -> usize {
        self.arena_bytes + self.unique_bytes
    }
    pub fn total_with_cache(&self) -> usize {
        self.total_live() + self.cache_bytes
    }
    pub fn arena_bytes_per_node(&self) -> f64 {
        self.arena_bytes as f64 / self.unique_entries.max(1) as f64
    }
    pub fn total_bytes_per_node(&self) -> f64 {
        self.total_live() as f64 / self.unique_entries.max(1) as f64
    }
}
