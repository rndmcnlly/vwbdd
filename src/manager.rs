//! The live BDD manager. Append-only buffer + unique table for canonicity +
//! an apply cache for ite.

use std::collections::HashMap;

use crate::node::{decode_node_at, decode_var_at, encode_node_at, Node, Ref};

/// Compact open-addressed unique table. Struct-of-arrays: one `Vec<u32>` of
/// arena offsets (plus one, so 0 means empty) alongside a parallel `Vec<u8>`
/// of hash-derived tags. The full key (var, lo, hi) lives in the arena; we
/// recover it on probe verify via LEB128 decode.
///
/// **Why u32 offsets are safe.** Slot values are arena byte offsets, not
/// node indices. Our arena runs ~4 B/node, so u32 offsets address up to
/// 4 GB arena = ~900M nodes. Well beyond anything we run today (7.6M nodes
/// in 34 MB arena at k=11). We assert this bound at insert time. If broken,
/// switch to u64 slots and accept 2x memory on the table.
///
/// **Why u8 hash tags.** A 1-byte tag gives us ~1/256 false-positive rate
/// per probe, which filters almost all mismatched probes before the
/// expensive LEB128 decode+compare. This mirrors hashbrown's design without
/// the SIMD group probe (we probe one slot at a time). Tag 0 is reserved
/// for empty slots, so live tags are in 1..=255 (lose 1 bit of entropy;
/// overall false-positive rate is ~1/255).
///
/// Combined footprint: 5 B/slot * 1.33 (0.75 load) = **~6.7 B/node**.
///
/// Collisions (both hash collisions and probe collisions) resolve by linear
/// probe: tag mismatch skips decode; tag match triggers verify-on-decode.
/// No separate overflow bucket needed.
struct CompactUnique {
    /// Power-of-two-sized slot array. `slots[i] == 0` means empty; otherwise
    /// `offset = slots[i] - 1`.
    slots: Vec<u32>,
    /// Parallel hash-tag array; same length as `slots`. `tags[i] == 0` when
    /// `slots[i] == 0`; otherwise `tags[i] = tag_of_hash(hash) != 0`.
    tags: Vec<u8>,
    /// Number of live entries (non-zero slots).
    len: usize,
    /// When `len` reaches this threshold, resize 2x. Recomputed on resize
    /// as `new_cap * 3 / 4` (0.75 load factor).
    resize_at: usize,
}

/// Derive a nonzero u8 tag from a 64-bit hash.
#[inline]
fn tag_of_hash(h: u64) -> u8 {
    let t = (h >> 56) as u8;
    t | 1
}

/// Initial slot count. 1024 slots * 5 B = 5 KB, fits in L1. Grows as needed.
const COMPACT_UNIQUE_INITIAL_CAP: usize = 1024;

impl CompactUnique {
    fn new() -> Self {
        let slots = vec![0u32; COMPACT_UNIQUE_INITIAL_CAP];
        let tags = vec![0u8; COMPACT_UNIQUE_INITIAL_CAP];
        Self {
            slots,
            tags,
            len: 0,
            resize_at: COMPACT_UNIQUE_INITIAL_CAP * 3 / 4,
        }
    }

    #[inline]
    fn cap(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    fn mask(&self) -> usize {
        self.cap() - 1
    }

    /// Look up `(var, lo, hi)` by hash. Returns the existing offset on
    /// match, or `None` if absent (linear probe terminates at empty slot).
    /// Tag check comes first; decode only on tag match.
    fn lookup(
        &self,
        hash: u64,
        var: u32,
        lo: Ref,
        hi: Ref,
        buf: &[u8],
    ) -> Option<u64> {
        let mask = self.mask();
        let tag = tag_of_hash(hash);
        let mut i = (hash as usize) & mask;
        loop {
            let t = self.tags[i];
            if t == 0 {
                return None;
            }
            if t == tag {
                let slot = self.slots[i];
                let off = (slot - 1) as u64;
                let (n, _) = decode_node_at(&buf[off as usize..], off);
                if n.var == var && n.lo == lo && n.hi == hi {
                    return Some(off);
                }
            }
            i = (i + 1) & mask;
        }
    }

    /// Insert `offset` at the slot determined by `hash`. Caller must have
    /// verified (via `lookup`) that the key is not already present. Handles
    /// resize when load factor exceeds 0.75.
    fn insert(&mut self, hash: u64, offset: u64, buf: &[u8]) {
        debug_assert!(
            offset < u32::MAX as u64,
            "arena exceeded 4 GB; CompactUnique u32 slot overflow"
        );
        if self.len + 1 > self.resize_at {
            self.resize(buf);
        }
        let mask = self.mask();
        let tag = tag_of_hash(hash);
        let mut i = (hash as usize) & mask;
        while self.tags[i] != 0 {
            i = (i + 1) & mask;
        }
        self.slots[i] = (offset as u32) + 1;
        self.tags[i] = tag;
        self.len += 1;
    }

    /// Grow to 2x capacity and reinsert every live entry at its new slot.
    /// We re-hash from the decoded (var, lo, hi) to avoid storing hashes.
    fn resize(&mut self, buf: &[u8]) {
        let new_cap = self.cap() * 2;
        let mut new_slots = vec![0u32; new_cap];
        let mut new_tags = vec![0u8; new_cap];
        let new_mask = new_cap - 1;
        for (i, &slot) in self.slots.iter().enumerate() {
            if self.tags[i] == 0 {
                continue;
            }
            let off = (slot - 1) as u64;
            let (n, _) = decode_node_at(&buf[off as usize..], off);
            let h = unique_key_hash(n.var, n.lo, n.hi);
            let tag = tag_of_hash(h);
            let mut j = (h as usize) & new_mask;
            while new_tags[j] != 0 {
                j = (j + 1) & new_mask;
            }
            new_slots[j] = slot;
            new_tags[j] = tag;
        }
        self.slots = new_slots;
        self.tags = new_tags;
        self.resize_at = new_cap * 3 / 4;
    }

    /// Resize to fit approximately `expected` entries at 0.75 load, rounding
    /// up to a power of two. Called post-GC to shrink back after construction
    /// has inflated the table past the live-node footprint.
    fn resize_for(&mut self, expected: usize) {
        let needed = ((expected * 4 + 2) / 3).max(COMPACT_UNIQUE_INITIAL_CAP);
        let mut new_cap = COMPACT_UNIQUE_INITIAL_CAP;
        while new_cap < needed {
            new_cap *= 2;
        }
        self.slots = vec![0u32; new_cap];
        self.tags = vec![0u8; new_cap];
        self.len = 0;
        self.resize_at = new_cap * 3 / 4;
    }

    fn bytes(&self) -> usize {
        self.slots.len() * std::mem::size_of::<u32>()
            + self.tags.len() * std::mem::size_of::<u8>()
    }
}

#[inline]
fn ref_to_u64(r: Ref) -> u64 {
    match r {
        Ref::Terminal(false) => 0x1,
        Ref::Terminal(true) => 0x2,
        Ref::Node(o) => 0x1000_0000_0000_0000u64 ^ o,
    }
}

/// A fast mixing hash for (var, lo, hi) triples. Uses splitmix64 on the
/// concatenation of the three pieces. Good enough for unique-table keys;
/// collisions are resolved by verify-on-decode.
#[inline]
fn unique_key_hash(var: u32, lo: Ref, hi: Ref) -> u64 {
    let mut h = var as u64;
    h = h.wrapping_mul(0x9e3779b97f4a7c15);
    h ^= ref_to_u64(lo);
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h ^= ref_to_u64(hi);
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    h
}

/// Fast mixing hash for an (f, g, h) ite-cache triple. Same splitmix-style
/// chain as unique_key_hash but over three Refs.
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

/// A 4-byte packed `Ref`, used only in bulk storage (the apply cache).
///
/// The public `Ref` enum is 16 bytes (8-byte tag + 8-byte u64 payload, no
/// niche). That bloated `IteCacheEntry` to 72 bytes, so the 2^17-slot cache
/// was taking 9 MB and spanning 1.5 cache lines per entry. Packing each Ref
/// into u32 cuts the entry to 16 bytes (exactly one cache line) and the cache
/// to 2 MB.
///
/// Encoding:
///   0          -> Terminal(false)
///   1          -> Terminal(true)
///   2          -> empty sentinel (replaces the `filled: bool` field)
///   3..=MAX    -> Node(value - 3), i.e. arena offset
///
/// This requires arena offsets < u32::MAX - 3 (~4 GiB), which we already
/// promised for the unique table's u32 offset+1 encoding (§4.7).
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
struct PackedRef(u32);

impl PackedRef {
    const FALSE: Self = Self(0);
    const TRUE: Self = Self(1);
    const EMPTY: Self = Self(2);

    #[inline]
    fn pack(r: Ref) -> Self {
        match r {
            Ref::Terminal(false) => Self::FALSE,
            Ref::Terminal(true) => Self::TRUE,
            Ref::Node(off) => {
                // Arena bound (<4 GiB) is also enforced in the unique table;
                // this debug_assert catches regressions early.
                debug_assert!(
                    off <= (u32::MAX as u64) - 3,
                    "arena offset {} exceeds PackedRef limit",
                    off
                );
                Self((off as u32).wrapping_add(3))
            }
        }
    }

    #[inline]
    fn unpack(self) -> Ref {
        match self.0 {
            0 => Ref::Terminal(false),
            1 => Ref::Terminal(true),
            2 => unreachable!("EMPTY PackedRef should never be unpacked"),
            v => Ref::Node((v - 3) as u64),
        }
    }
}

/// Direct-mapped apply cache entry. Empty slot is marked by `f == PackedRef::EMPTY`.
/// 4 × u32 = 16 bytes = exactly one cache line on aarch64 / x86_64.
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

/// Power-of-two number of slots. 2^17 = 131,072 entries × 16 B = 2 MiB.
/// Large enough to hold most of k=8's working set with low collision eviction.
const ITE_CACHE_SLOTS: usize = 1 << 17;
const ITE_CACHE_MASK: u64 = (ITE_CACHE_SLOTS as u64) - 1;

pub struct Manager {
    buf: Vec<u8>,
    /// Unique table: compact linear-probe, u32 offsets + u8 tags, 0.75 load
    /// factor on power-of-two sizes (§4.7). ~10.7 B/node average.
    ///
    /// History:
    /// - §4.6: four hand-rolled HashMap replacements all regressed.
    /// - §4.7 (current): `CompactUnique`. Key insight: don't store the
    ///   key in the slot — the arena already has it; verify on decode.
    /// - §4.9 (rejected): tried cuckoo at 0.85 load for -33% memory;
    ///   actual cost was ~2× speed, much higher than predicted.
    unique: CompactUnique,
    num_vars: u32,
    /// Direct-mapped ite apply cache. Fixed-size; collisions evict.
    ite_cache: Box<[IteCacheEntry; ITE_CACHE_SLOTS]>,
    /// Live entry count (for diagnostics; approximate under eviction).
    ite_cache_len: usize,
}

impl Manager {
    pub fn new() -> Self {
        // Box::new on a huge array would overflow the stack; build on heap.
        let entries: Vec<IteCacheEntry> = vec![IteCacheEntry::EMPTY; ITE_CACHE_SLOTS];
        let boxed_slice: Box<[IteCacheEntry]> = entries.into_boxed_slice();
        let ite_cache: Box<[IteCacheEntry; ITE_CACHE_SLOTS]> =
            boxed_slice.try_into().ok().expect("ite_cache size matches");
        Self {
            buf: Vec::new(),
            unique: CompactUnique::new(),
            num_vars: 0,
            ite_cache,
            ite_cache_len: 0,
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
        self.unique.len
    }

    pub fn buf_len(&self) -> usize {
        self.buf.len()
    }

    /// Raw arena view from an offset. For diagnostic tooling; callers should
    /// know they're reading LEB128-encoded nodes.
    pub fn arena_slice(&self, off: usize) -> &[u8] {
        &self.buf[off..]
    }

    /// Memory breakdown (arena + unique + cache). The unique-table number
    /// includes empty slots; at target 0.85 load, slot overhead is ~5.9 B/node.
    pub fn mem_stats(&self) -> MemStats {
        // Cuckoo unique table: 20 B/bucket × bucket_count, each bucket
        // holding at most 4 (offset, tag) pairs.
        let uniq_entries = self.num_nodes();
        let uniq_bytes = self.unique.bytes();

        // Direct-mapped ite_cache: fixed-size array of IteCacheEntry
        // (16 B each, one cache line). Allocated regardless of fill level.
        let cache_entries = self.ite_cache_len;
        let cache_bytes = ITE_CACHE_SLOTS * std::mem::size_of::<IteCacheEntry>();

        let var_bytes = 0usize;

        MemStats {
            arena_bytes: self.buf.len(),
            unique_bytes: uniq_bytes,
            cache_bytes,
            var_table_bytes: var_bytes,
            unique_entries: uniq_entries,
            cache_entries,
        }
    }

    pub fn var_of(&self, r: Ref) -> Option<u32> {
        match r {
            Ref::Terminal(_) => None,
            Ref::Node(off) => {
                let (var, _) = decode_var_at(&self.buf[off as usize..], off);
                Some(var)
            }
        }
    }

    pub fn decode_node(&self, r: Ref) -> Option<Node> {
        match r {
            Ref::Terminal(_) => None,
            Ref::Node(off) => {
                let (n, _) = decode_node_at(&self.buf[off as usize..], off);
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
        encode_node_at(var, lo, hi, new_off, &mut self.buf);
        self.unique.insert(hash, new_off, &self.buf);
        Ref::Node(new_off)
    }

    /// Top variable of a ref. Terminals return None (treated as infinity).
    fn top_var(&self, r: Ref) -> Option<u32> {
        self.var_of(r)
    }

    /// Cofactor r at variable v (that is, substitute v = val and return the
    /// simplified function). If r's top var is deeper than v, r is unchanged.
    fn cofactor(&self, r: Ref, v: u32, val: bool) -> Ref {
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

    /// Probe the direct-mapped apply cache. Returns Some(r) on hit, None on
    /// empty slot or collision miss.
    #[inline]
    fn ite_cache_get(&self, f: Ref, g: Ref, h: Ref) -> Option<Ref> {
        let slot = (ite_key_hash(f, g, h) & ITE_CACHE_MASK) as usize;
        let e = &self.ite_cache[slot];
        let pf = PackedRef::pack(f);
        // Empty slot: f == EMPTY. A real `f` can never pack to EMPTY (2),
        // so a single equality check both rejects empties and filters
        // collision misses.
        if e.f == pf && e.g == PackedRef::pack(g) && e.h == PackedRef::pack(h) {
            Some(e.r.unpack())
        } else {
            None
        }
    }

    /// Insert into the direct-mapped apply cache. Overwrites on collision.
    #[inline]
    fn ite_cache_put(&mut self, f: Ref, g: Ref, h: Ref, r: Ref) {
        let slot = (ite_key_hash(f, g, h) & ITE_CACHE_MASK) as usize;
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
                    // lo was pushed first and should come out second via pop().
                    // Order: we pushed f0_enter last, so f0 result lands on
                    // results stack first, then f1's. Hence:
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

    /// Copying garbage collector. Takes a slice of roots, builds a fresh arena
    /// containing only the nodes reachable from those roots, and returns the
    /// remapped roots. The apply cache is flushed; callers must replace their
    /// own held refs with the returned ones.
    ///
    /// Implementation: topological DFS from roots, emitting each reached node
    /// into a fresh buffer in a post-order that respects the variable-ordering
    /// invariant (children before parents, which they already are by
    /// construction — parent var < child var, and children are written first).
    pub fn gc(&mut self, roots: &[Ref]) -> Vec<Ref> {
        let mut remap: HashMap<u64, Ref> = HashMap::new();
        let mut new_buf: Vec<u8> = Vec::new();

        enum Frame {
            Enter(u64),
            Exit(u64),
        }
        let old_buf = &self.buf;

        for &root in roots {
            if let Ref::Node(off) = root {
                if remap.contains_key(&off) { continue; }
                let mut stack: Vec<Frame> = vec![Frame::Enter(off)];
                while let Some(frame) = stack.pop() {
                    match frame {
                        Frame::Enter(o) => {
                            if remap.contains_key(&o) { continue; }
                            let (node, _) = decode_node_at(&old_buf[o as usize..], o);
                            stack.push(Frame::Exit(o));
                            if let Ref::Node(lo_off) = node.lo {
                                if !remap.contains_key(&lo_off) {
                                    stack.push(Frame::Enter(lo_off));
                                }
                            }
                            if let Ref::Node(hi_off) = node.hi {
                                if !remap.contains_key(&hi_off) {
                                    stack.push(Frame::Enter(hi_off));
                                }
                            }
                        }
                        Frame::Exit(o) => {
                            if remap.contains_key(&o) { continue; }
                            let (node, _) = decode_node_at(&old_buf[o as usize..], o);
                            let new_lo = translate(&remap, node.lo);
                            let new_hi = translate(&remap, node.hi);
                            // Reduction (shouldn't fire on canonical input).
                            let new_ref = if new_lo == new_hi {
                                new_lo
                            } else {
                                // Emit into new_buf. Canonicity is preserved
                                // because the old arena was already canonical
                                // and the DFS visits each unique offset once.
                                let new_off = new_buf.len() as u64;
                                encode_node_at(node.var, new_lo, new_hi, new_off, &mut new_buf);
                                Ref::Node(new_off)
                            };
                            remap.insert(o, new_ref);
                        }
                    }
                }
            }
        }

        // Swap in the new arena.
        self.buf = new_buf;

        // Rebuild unique table from the new arena. Walk it in construction
        // order (same order we emitted), LEB128-decoding each node, inserting
        // into a fresh slot array sized for the known entry count.
        self.unique_rebuild_from_arena();

        // Flush apply cache: old offsets are dead.
        for e in self.ite_cache.iter_mut() {
            *e = IteCacheEntry::EMPTY;
        }
        self.ite_cache_len = 0;

        // Translate roots.
        roots.iter().map(|&r| translate(&remap, r)).collect()
    }

    /// Rebuild the unique table by scanning the current arena in construction
    /// order. Called after GC, which overwrites `self.buf` with a fresh arena.
    ///
    /// Note: insertion requires `&self.buf` for the load-factor-triggered
    /// resize (which re-decodes every live node). We work around the
    /// self-borrow by inserting via the `CompactUnique` methods directly.
    fn unique_rebuild_from_arena(&mut self) {
        // Count live nodes to pre-size the table. Two decode passes total,
        // but resize-on-the-fly during insert would copy and re-decode
        // repeatedly — this is cheaper.
        let mut live = 0usize;
        {
            let mut pos: usize = 0;
            while pos < self.buf.len() {
                let off = pos as u64;
                let (_, len) = decode_node_at(&self.buf[pos..], off);
                live += 1;
                pos += len;
            }
        }
        self.unique.resize_for(live);
        let mut pos: usize = 0;
        while pos < self.buf.len() {
            let off = pos as u64;
            let (node, len) = decode_node_at(&self.buf[pos..], off);
            let hash = unique_key_hash(node.var, node.lo, node.hi);
            self.unique.insert(hash, off, &self.buf);
            pos += len;
        }
    }
}

fn translate(remap: &HashMap<u64, Ref>, r: Ref) -> Ref {
    match r {
        Ref::Terminal(_) => r,
        Ref::Node(off) => *remap.get(&off).expect("root unreachable in remap"),
    }
}

impl Default for Manager {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MemStats {
    pub arena_bytes: usize,
    pub unique_bytes: usize,
    pub cache_bytes: usize,
    pub var_table_bytes: usize,
    pub unique_entries: usize,
    pub cache_entries: usize,
}

impl MemStats {
    pub fn total_live(&self) -> usize {
        self.arena_bytes + self.unique_bytes + self.var_table_bytes
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
    pub fn total_with_cache_per_node(&self) -> f64 {
        self.total_with_cache() as f64 / self.unique_entries.max(1) as f64
    }
}
