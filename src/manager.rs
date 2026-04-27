//! The live BDD manager. Append-only buffer + unique table for canonicity +
//! an apply cache for ite.

use std::collections::HashMap;

use crate::node::{decode_node_at, decode_var_at, encode_node_at, ref_to_u64, Node, Ref};
use crate::unique::{unique_key_hash, CompactUnique};

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
        self.unique.len()
    }

    pub fn buf_len(&self) -> usize {
        self.buf.len()
    }

    /// Raw arena view from an offset. For diagnostic tooling; callers should
    /// know they're reading LEB128-encoded nodes.
    pub fn arena_slice(&self, off: usize) -> &[u8] {
        &self.buf[off..]
    }

    /// Memory breakdown (arena + unique + cache). `unique_bytes` includes
    /// empty slots; at 0.75 load the linear-probe table runs ~6.7 B/node
    /// (5 B/slot × 1.33).
    pub fn mem_stats(&self) -> MemStats {
        MemStats {
            arena_bytes: self.buf.len(),
            unique_bytes: self.unique.bytes(),
            cache_bytes: ITE_CACHE_SLOTS * std::mem::size_of::<IteCacheEntry>(),
            unique_entries: self.num_nodes(),
            cache_entries: self.ite_cache_len,
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
