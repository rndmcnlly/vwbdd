//! The live BDD manager. Append-only byte buffer + unique table for
//! canonicity + a direct-mapped apply cache for ite.
//!
//! Single-engine design: arena offsets are `u64` so the same code path
//! handles a 4 GiB wasm arena and a multi-terabyte server arena. The
//! unique table and apply cache both store u64 offsets.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::codec::{decode_node, decode_var, encode_node, Node, Ref, ref_to_u64};
use crate::unique::{unique_key_hash, UniqueTables};

// --- Apply-cache instrumentation (see §4.20) ---
//
// Counters for `ite_cache_get` outcomes. Gated on a single Relaxed
// bool load (branch-predictor friendly; disabled by default) so the
// baseline hot path stays unchanged. The sweep harness
// (`examples/apply_cache_sweep.rs`) flips the gate on, runs, reads,
// resets.

static APPLY_ENABLED: AtomicBool = AtomicBool::new(false);
static APPLY_HITS: AtomicU64 = AtomicU64::new(0);
static APPLY_COLL: AtomicU64 = AtomicU64::new(0);
static APPLY_EMPTY: AtomicU64 = AtomicU64::new(0);

// Pattern-distribution counters: what shape are the (f,g,h) triples
// hitting the cache? Each call is classified into exactly one bucket.
// Used to design cache-key canonicalization (§4.21).
static PAT_AND: AtomicU64 = AtomicU64::new(0); // h == F (so ite(f,g,F) = f ∧ g)
static PAT_OR: AtomicU64 = AtomicU64::new(0); // g == T (so ite(f,T,h) = f ∨ h)
static PAT_NOT: AtomicU64 = AtomicU64::new(0); // g == F, h == T (so ite(f,F,T) = ¬f)
static PAT_OTHER: AtomicU64 = AtomicU64::new(0); // three-way ite with no shortcut pattern

enum ApplyEvent { Hit, Coll, Empty }

#[inline]
fn apply_stats_bump(ev: ApplyEvent) {
    if !APPLY_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    match ev {
        ApplyEvent::Hit => APPLY_HITS.fetch_add(1, Ordering::Relaxed),
        ApplyEvent::Coll => APPLY_COLL.fetch_add(1, Ordering::Relaxed),
        ApplyEvent::Empty => APPLY_EMPTY.fetch_add(1, Ordering::Relaxed),
    };
}

#[inline]
fn apply_stats_bump_pattern(f: Ref, g: Ref, h: Ref) {
    if !APPLY_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let _ = f;
    let t = Ref::Terminal(true);
    let ff = Ref::Terminal(false);
    let counter: &AtomicU64 = if g == ff && h == t {
        &PAT_NOT
    } else if g == t {
        &PAT_OR
    } else if h == ff {
        &PAT_AND
    } else {
        &PAT_OTHER
    };
    counter.fetch_add(1, Ordering::Relaxed);
}

/// Enable or disable the apply-cache counters. Default off. Turn on
/// around the region of interest; read via `apply_cache_stats`.
pub fn apply_cache_stats_enable(on: bool) {
    APPLY_ENABLED.store(on, Ordering::Relaxed);
}

/// Read-only snapshot of the apply-cache counters since the last
/// reset. `(hits, collision-misses, empty-misses)`.
pub fn apply_cache_stats() -> (u64, u64, u64) {
    (
        APPLY_HITS.load(Ordering::Relaxed),
        APPLY_COLL.load(Ordering::Relaxed),
        APPLY_EMPTY.load(Ordering::Relaxed),
    )
}

/// Read-only snapshot of the ite-pattern-distribution counters.
/// `(and, or, not, other)` — how many calls matched each shape?
/// Useful for designing cache-key canonicalization.
pub fn apply_cache_patterns() -> (u64, u64, u64, u64) {
    (
        PAT_AND.load(Ordering::Relaxed),
        PAT_OR.load(Ordering::Relaxed),
        PAT_NOT.load(Ordering::Relaxed),
        PAT_OTHER.load(Ordering::Relaxed),
    )
}

/// Reset the apply-cache counters to zero. Used by the sweep harness
/// between runs so each cache-size sample is clean.
pub fn apply_cache_stats_reset() {
    APPLY_HITS.store(0, Ordering::Relaxed);
    APPLY_COLL.store(0, Ordering::Relaxed);
    APPLY_EMPTY.store(0, Ordering::Relaxed);
    PAT_AND.store(0, Ordering::Relaxed);
    PAT_OR.store(0, Ordering::Relaxed);
    PAT_NOT.store(0, Ordering::Relaxed);
    PAT_OTHER.store(0, Ordering::Relaxed);
}

/// Fast mixing hash for an (f, g, h) ite-cache triple. Splitmix-style chain.
#[inline]
fn ite_key_hash(f: Ref, g: Ref, h: Ref) -> u64 {
    let mut x = ref_to_u64(f);
    x = x.wrapping_mul(0x9e3779b97f4a7c15);
    x ^= ref_to_u64(g);
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x ^= ref_to_u64(h);
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

/// A `Ref` packed into a single u64 for bulk storage in the apply cache.
///
///   0          -> Terminal(false)
///   1          -> Terminal(true)
///   2          -> empty sentinel (no `filled` bool needed)
///   3..        -> Node(value - 3), i.e. arena offset
///
/// `IteCacheEntry` is 4 × 8 B = 32 B, or half a cache line: two entries
/// per line, still L1-friendly. The u32-era trick of packing an entry
/// into a single cache line is no longer available; simplifying by
/// committing to a single arena-offset width (see VWBDD.md §8)
/// outweighed it.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
struct PackedRef(u64);

impl PackedRef {
    const EMPTY: Self = Self(2);

    #[inline]
    fn pack(r: Ref) -> Self {
        match r {
            Ref::Terminal(false) => Self(0),
            Ref::Terminal(true) => Self(1),
            Ref::Node(off) => Self(off.checked_add(3).expect("arena offset overflow in cache")),
        }
    }

    #[inline]
    fn unpack(self) -> Ref {
        match self.0 {
            0 => Ref::Terminal(false),
            1 => Ref::Terminal(true),
            2 => unreachable!("EMPTY PackedRef should never be unpacked"),
            v => Ref::Node(v - 3),
        }
    }
}

/// Direct-mapped apply cache entry. Empty slot is marked by `f == EMPTY`.
/// 32 B (half a cache line).
#[repr(C)]
#[derive(Copy, Clone)]
struct IteCacheEntry {
    f: PackedRef,
    g: PackedRef,
    h: PackedRef,
    r: PackedRef,
}

impl IteCacheEntry {
    const EMPTY: Self = Self {
        f: PackedRef::EMPTY,
        g: PackedRef::EMPTY,
        h: PackedRef::EMPTY,
        r: PackedRef::EMPTY,
    };
}

/// Default number of slots in the direct-mapped ite apply cache.
/// 2^21 = 2,097,152 entries × 32 B = 64 MiB.
///
/// Chosen to clear the k=15 truncated-mult working set (~10M edges) while
/// staying trivial relative to any real workload's RAM budget. The earlier
/// 2^17 default turned out to be a silent perf trap once workloads grew
/// past k=11: throughput fell 2-3× when the working set overflowed the
/// cache. Override via [`ManagerConfig::with_cache_slots`] for small-arena
/// workloads where the default's 64 MiB is wasteful.
pub const DEFAULT_ITE_CACHE_SLOTS: usize = 1 << 21;

/// Construction-time options for [`Manager`]. Builder-shaped so future
/// knobs (per-level unique-table partitioning, GC-trigger heuristics, etc.)
/// can be added without breaking existing call sites.
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
    /// Typical values: `1 << 17` (128k, 4 MiB) for tight-memory builds;
    /// `1 << 21` (2M, 64 MiB, the default); `1 << 25` (32M, 1 GiB) for
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
pub struct Manager {
    buf: Vec<u8>,
    /// Unique table: compact linear-probe, u64 offsets + u8 tags,
    /// 0.75 load factor on power-of-two sizes.
    unique: UniqueTables,
    num_vars: u32,
    /// Direct-mapped ite apply cache. Fixed-size (set at construction);
    /// collisions evict. See [`ManagerConfig::with_cache_slots`].
    ite_cache: Box<[IteCacheEntry]>,
    /// `ite_cache.len() - 1`, precomputed for the `hash & mask` fast path.
    ite_cache_mask: u64,
    /// Live entry count (for diagnostics; approximate under eviction).
    ite_cache_len: usize,
}

impl Manager {
    /// Default-config manager: 2^21-slot apply cache.
    pub fn new() -> Self {
        Self::with_config(ManagerConfig::default())
    }

    /// Convenience: override the apply cache slot count. Equivalent to
    /// `Self::with_config(ManagerConfig::new().with_cache_slots(slots))`.
    pub fn with_cache_slots(slots: usize) -> Self {
        Self::with_config(ManagerConfig::new().with_cache_slots(slots))
    }

    /// Construct a manager with explicit config.
    pub fn with_config(config: ManagerConfig) -> Self {
        let slots = config.ite_cache_slots;
        debug_assert!(slots.is_power_of_two());
        debug_assert!(slots >= 2);
        // vec! + into_boxed_slice keeps the heap allocation contiguous
        // without ever building the huge array on the stack.
        let ite_cache: Box<[IteCacheEntry]> =
            vec![IteCacheEntry::EMPTY; slots].into_boxed_slice();
        Self {
            buf: Vec::new(),
            unique: UniqueTables::new(),
            num_vars: 0,
            ite_cache,
            ite_cache_mask: (slots as u64) - 1,
            ite_cache_len: 0,
        }
    }

    pub fn new_var(&mut self) -> u32 {
        let v = self.num_vars;
        self.num_vars += 1;
        self.unique.ensure_var(v);
        v
    }

    pub fn num_vars(&self) -> u32 {
        self.num_vars
    }

    pub fn constant(&self, value: bool) -> Ref {
        Ref::Terminal(value)
    }

    pub fn r#false(&self) -> Ref {
        Ref::Terminal(false)
    }

    pub fn r#true(&self) -> Ref {
        Ref::Terminal(true)
    }

    pub fn num_nodes(&self) -> usize {
        self.unique.len()
    }

    pub fn buf_len(&self) -> usize {
        self.buf.len()
    }

    /// Raw arena view from an offset. For diagnostic tooling; callers
    /// should know they're reading codec-encoded nodes.
    pub fn arena_slice(&self, off: usize) -> &[u8] {
        &self.buf[off..]
    }

    /// Mutable access to the arena buffer. Used by the slab ingest path
    /// (`src/slab.rs`) to append raw bytes after a transport read.
    /// Callers must follow up with [`Self::rebuild_unique_from_arena`]
    /// so the unique table reflects the new nodes.
    pub(crate) fn buf_mut(&mut self) -> &mut Vec<u8> {
        &mut self.buf
    }

    /// Walk the current arena in construction order and rebuild the
    /// unique table from scratch. Used internally by [`Self::gc`] and
    /// by the slab ingest path.
    pub(crate) fn rebuild_unique_from_arena(&mut self) {
        self.unique_rebuild_from_arena();
    }

    /// Memory breakdown (arena + unique + cache).
    pub fn mem_stats(&self) -> MemStats {
        MemStats {
            arena_bytes: self.buf.len(),
            unique_bytes: self.unique.bytes(),
            cache_bytes: self.ite_cache.len() * std::mem::size_of::<IteCacheEntry>(),
            unique_entries: self.num_nodes(),
            cache_entries: self.ite_cache_len,
        }
    }

    pub fn var_of(&self, r: Ref) -> Option<u32> {
        match r {
            Ref::Terminal(_) => None,
            Ref::Node(off) => {
                let (var, _) = decode_var(&self.buf[off as usize..], off);
                Some(var)
            }
        }
    }

    pub fn decode_node(&self, r: Ref) -> Option<Node> {
        match r {
            Ref::Terminal(_) => None,
            Ref::Node(off) => {
                let (n, _) = decode_node(&self.buf[off as usize..], off);
                Some(n)
            }
        }
    }

    pub fn make_node(&mut self, var: u32, lo: Ref, hi: Ref) -> Ref {
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
        let hash = unique_key_hash(var, lo, hi);
        if let Some(off) = self.unique.lookup(hash, var, lo, hi, &self.buf) {
            return Ref::Node(off);
        }
        let new_off = self.buf.len() as u64;
        encode_node(var, lo, hi, new_off, &mut self.buf);
        self.unique.insert(hash, var, new_off, &self.buf);
        Ref::Node(new_off)
    }

    /// Top variable of a ref. Terminals return None (treated as infinity).
    fn top_var(&self, r: Ref) -> Option<u32> {
        self.var_of(r)
    }

    /// Cofactor r at variable v (substitute v = val and return the
    /// simplified function). If r's top var is deeper than v, r is unchanged.
    fn cofactor(&self, r: Ref, v: u32, val: bool) -> Ref {
        match r {
            Ref::Terminal(_) => r,
            Ref::Node(_) => {
                let node_var = self.var_of(r).unwrap();
                if node_var > v {
                    // r doesn't depend on v at this level.
                    r
                } else if node_var == v {
                    // Pluck the appropriate child.
                    let node = self.decode_node(r).unwrap();
                    if val { node.hi } else { node.lo }
                } else {
                    panic!(
                        "cofactor called below current level: node_var={}, v={}",
                        node_var, v
                    );
                }
            }
        }
    }

    /// Short-circuit terminal cases for ite.
    fn ite_trivial(&self, f: Ref, g: Ref, h: Ref) -> Option<Ref> {
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

    /// Probe the direct-mapped apply cache. Returns Some(r) on hit,
    /// None on empty slot or collision miss.
    #[inline]
    fn ite_cache_get(&self, f: Ref, g: Ref, h: Ref) -> Option<Ref> {
        apply_stats_bump_pattern(f, g, h);
        let slot = (ite_key_hash(f, g, h) & self.ite_cache_mask) as usize;
        let e = &self.ite_cache[slot];
        let pf = PackedRef::pack(f);
        if e.f == pf && e.g == PackedRef::pack(g) && e.h == PackedRef::pack(h) {
            apply_stats_bump(ApplyEvent::Hit);
            Some(e.r.unpack())
        } else if e.f == PackedRef::EMPTY {
            apply_stats_bump(ApplyEvent::Empty);
            None
        } else {
            apply_stats_bump(ApplyEvent::Coll);
            None
        }
    }

    /// Insert into the direct-mapped apply cache. Overwrites on collision.
    #[inline]
    fn ite_cache_put(&mut self, f: Ref, g: Ref, h: Ref, r: Ref) {
        let slot = (ite_key_hash(f, g, h) & self.ite_cache_mask) as usize;
        let e = &mut self.ite_cache[slot];
        if e.f == PackedRef::EMPTY {
            self.ite_cache_len += 1;
        }
        *e = IteCacheEntry {
            f: PackedRef::pack(f),
            g: PackedRef::pack(g),
            h: PackedRef::pack(h),
            r: PackedRef::pack(r),
        };
    }

    /// Iterative Shannon-expansion ite. Explicit work-stack so we don't
    /// blow the thread stack on deep BDDs.
    ///
    /// Because the apply cache is direct-mapped (evicts on collision),
    /// we cannot rely on it to carry intermediate results between child
    /// completion and parent combine. Instead, we maintain a parallel
    /// result stack: each Enter that does real work pushes a Combine
    /// and two child Enters; when a child completes it pushes its
    /// result onto `results`; Combine pops two results, builds the
    /// node, and pushes its own.
    pub fn ite(&mut self, f: Ref, g: Ref, h: Ref) -> Ref {
        if let Some(r) = self.ite_trivial(f, g, h) {
            return r;
        }
        if let Some(r) = self.ite_cache_get(f, g, h) {
            return r;
        }

        enum Frame {
            Enter { f: Ref, g: Ref, h: Ref },
            Combine { f: Ref, g: Ref, h: Ref, tv: u32 },
        }
        let mut stack: Vec<Frame> = Vec::with_capacity(64);
        let mut results: Vec<Ref> = Vec::with_capacity(64);
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
    pub fn not(&mut self, f: Ref) -> Ref {
        let t = self.r#true();
        let fa = self.r#false();
        self.ite(f, fa, t)
    }

    pub fn and(&mut self, f: Ref, g: Ref) -> Ref {
        let fa = self.r#false();
        self.ite(f, g, fa)
    }

    pub fn or(&mut self, f: Ref, g: Ref) -> Ref {
        let t = self.r#true();
        self.ite(f, t, g)
    }

    pub fn xor(&mut self, f: Ref, g: Ref) -> Ref {
        let ng = self.not(g);
        self.ite(f, ng, g)
    }

    /// **Drop everything except these roots.** Replaces the arena with
    /// a fresh one containing only the closure of `keep`, returning the
    /// translated roots. Equivalent to `self.gc(0, keep)`; this is the
    /// intent-level name recommended for full collections.
    ///
    /// After this call the arena is function-canonical for `keep`
    /// (running GC again would find nothing to remove). Not
    /// *layout-canonical*: two managers that built the same BDD via
    /// different op sequences can produce different byte encodings of
    /// the same reduced DAG. The apply cache is flushed.
    pub fn drop_roots(&mut self, keep: &[Ref]) -> Vec<Ref> {
        self.gc(0, keep)
    }

    /// **Generational garbage collector.** Frees unreachable bytes in
    /// the tail of the arena (offsets `>= base_len`), leaving the base
    /// bytes byte-for-byte untouched. Pass `base_len = 0` for a full
    /// collection; [`Self::drop_roots`] is the ergonomic alias for
    /// that. Used by [`Self::diff_since`](crate::slab) to enforce the
    /// clean-bytes invariant on emitted diffs.
    ///
    /// No write barrier is needed: the LEB128 codec encodes children
    /// as backward deltas, so a base node (offset `< base_len`) cannot
    /// reference a tail node. Tail→base edges preserve absolute
    /// offsets (delta encodings may shift).
    ///
    /// Full-collection cost: one `buf.clone()` into `old_tail` before
    /// re-encoding, a transient 2× memory blip. Tail-collection byte
    /// savings measured on truncated-mult k=5..11:
    ///
    /// | tail shape                 | byte shrink        |
    /// |----------------------------|--------------------|
    /// | all scratch                | ∞ (tail → 0 B)     |
    /// | mostly-live (~1% dead)     | ~0.99-1.04× (wash) |
    /// | tiny result (1-2 nodes)    | ~1.6-1.75×         |
    ///
    /// The near-dense wash is real: re-encoding surviving nodes at
    /// compacted offsets can push LEB128 deltas across 7-bit
    /// boundaries, costing more than the dead nodes saved.
    pub fn gc(&mut self, base_len: u64, roots: &[Ref]) -> Vec<Ref> {
        let buf_len = self.buf.len() as u64;
        assert!(
            base_len <= buf_len,
            "gc: base_len {} > arena len {}",
            base_len, buf_len
        );
        if base_len == buf_len {
            // Nothing to collect; tail is empty.
            return roots.to_vec();
        }

        // remap: tail-offset → new Ref. Terminals and base nodes are
        // NOT stored here; they pass through `translate`.
        let mut remap: HashMap<u64, Ref> = HashMap::new();

        // Snapshot the old tail bytes so we can read children from it
        // while re-encoding into the truncated self.buf.
        let old_tail: Vec<u8> = self.buf[base_len as usize..].to_vec();

        enum Frame {
            Enter(u64),
            Exit(u64),
        }

        let is_young = |o: u64| -> bool { o >= base_len };

        // Walk each root over the OLD arena bytes, collecting a
        // topologically ordered post-order list of surviving tail nodes.
        let mut post_order: Vec<u64> = Vec::new();
        {
            let old_buf = &self.buf;
            for &root in roots {
                if let Ref::Node(off) = root {
                    if !is_young(off) {
                        continue; // base root; nothing to do
                    }
                    if remap.contains_key(&off) {
                        continue;
                    }
                    let mut stack: Vec<Frame> = vec![Frame::Enter(off)];
                    while let Some(frame) = stack.pop() {
                        match frame {
                            Frame::Enter(o) => {
                                if remap.contains_key(&o) {
                                    continue;
                                }
                                debug_assert!(is_young(o));
                                let (node, _) = decode_node(&old_buf[o as usize..], o);
                                stack.push(Frame::Exit(o));
                                if let Ref::Node(lo_off) = node.lo {
                                    if is_young(lo_off) && !remap.contains_key(&lo_off) {
                                        stack.push(Frame::Enter(lo_off));
                                    }
                                }
                                if let Ref::Node(hi_off) = node.hi {
                                    if is_young(hi_off) && !remap.contains_key(&hi_off) {
                                        stack.push(Frame::Enter(hi_off));
                                    }
                                }
                            }
                            Frame::Exit(o) => {
                                if remap.contains_key(&o) {
                                    continue;
                                }
                                // Placeholder marks "visited"; we
                                // overwrite it below.
                                remap.insert(o, Ref::Terminal(false));
                                post_order.push(o);
                            }
                        }
                    }
                }
            }
        }

        // Truncate the buffer to just the base, then re-encode
        // surviving tail nodes directly into it. We decode children
        // from `old_tail` (self.buf's tail region is being rewritten).
        self.buf.truncate(base_len as usize);
        remap.clear();

        for old_off in post_order {
            let tail_rel = (old_off - base_len) as usize;
            let (node, _) = decode_node(&old_tail[tail_rel..], old_off);
            let new_lo = translate(&remap, node.lo, base_len);
            let new_hi = translate(&remap, node.hi, base_len);
            let new_ref = if new_lo == new_hi {
                new_lo
            } else {
                let new_off = self.buf.len() as u64;
                encode_node(node.var, new_lo, new_hi, new_off, &mut self.buf);
                Ref::Node(new_off)
            };
            remap.insert(old_off, new_ref);
        }

        // Rebuild the unique table over the full arena.
        self.unique_rebuild_from_arena();

        // Flush apply cache: any entry that referenced a stale tail
        // offset would now resolve to a different node.
        for e in self.ite_cache.iter_mut() {
            *e = IteCacheEntry::EMPTY;
        }
        self.ite_cache_len = 0;

        // Translate roots for the caller.
        roots
            .iter()
            .map(|&r| translate(&remap, r, base_len))
            .collect()
    }

    /// Rebuild the unique table by scanning the current arena in
    /// construction order. Called after GC or after a slab ingest.
    ///
    /// Two passes: first count per-variable live nodes so each per-var
    /// table can be pre-sized at 0.75 load (avoiding a cascade of
    /// resizes during reinsertion); second pass actually inserts.
    fn unique_rebuild_from_arena(&mut self) {
        // Pass 1: per-var histogram.
        let mut per_var: Vec<usize> = vec![0; self.num_vars as usize];
        {
            let mut pos: usize = 0;
            while pos < self.buf.len() {
                let off = pos as u64;
                let (node, len) = decode_node(&self.buf[pos..], off);
                if (node.var as usize) >= per_var.len() {
                    per_var.resize(node.var as usize + 1, 0);
                }
                per_var[node.var as usize] += 1;
                pos += len;
                let _ = off;
            }
        }
        self.unique.reset_for(&per_var);
        // Pass 2: reinsert.
        let mut pos: usize = 0;
        while pos < self.buf.len() {
            let off = pos as u64;
            let (node, len) = decode_node(&self.buf[pos..], off);
            let hash = unique_key_hash(node.var, node.lo, node.hi);
            self.unique.insert(hash, node.var, off, &self.buf);
            pos += len;
        }
    }
}

/// Translate a ref through a GC remap. Base-generation refs (offset
/// `< base_len`) pass through unchanged — they weren't walked, so they
/// aren't in the remap. Young-generation refs are looked up; their
/// absence is a bug (every reachable tail node should have been
/// visited). Terminals pass through.
///
/// At `base_len = 0` this is the full-gc translator: every Node ref
/// goes through the remap.
fn translate(remap: &HashMap<u64, Ref>, r: Ref, base_len: u64) -> Ref {
    match r {
        Ref::Terminal(_) => r,
        Ref::Node(off) => {
            if off < base_len {
                r
            } else {
                *remap
                    .get(&off)
                    .expect("young root unreachable in GC remap")
            }
        }
    }
}

impl Default for Manager {
    fn default() -> Self {
        Self::with_config(ManagerConfig::default())
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
