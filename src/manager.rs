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

/// Power-of-two number of slots. 2^17 = 131,072 entries.
/// - u32: 2 MiB cache
/// - u64: 4 MiB cache
/// Large enough to hold most of k=8's mult working set with low collision eviction.
const ITE_CACHE_SLOTS: usize = 1 << 17;
const ITE_CACHE_MASK: u64 = (ITE_CACHE_SLOTS as u64) - 1;

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
    /// Direct-mapped ite apply cache. Fixed-size; collisions evict.
    ite_cache: Box<[IteCacheEntry<O>; ITE_CACHE_SLOTS]>,
    /// Live entry count (for diagnostics; approximate under eviction).
    ite_cache_len: usize,
    _codec: PhantomData<C>,
}

impl<C: NodeCodec<O>, O: ArenaOffset> Manager<C, O> {
    /// Construct an empty manager. Prefer the `Default` impl or a specific
    /// type alias (`DefaultManager`, `LargeManager`) for ergonomic
    /// type inference.
    pub fn new_parameterized() -> Self {
        // Box::new on a huge array would overflow the stack; build on heap.
        let entries: Vec<IteCacheEntry<O>> = vec![IteCacheEntry::empty(); ITE_CACHE_SLOTS];
        let boxed_slice: Box<[IteCacheEntry<O>]> = entries.into_boxed_slice();
        let ite_cache: Box<[IteCacheEntry<O>; ITE_CACHE_SLOTS]> =
            boxed_slice.try_into().ok().expect("ite_cache size matches");
        Self {
            buf: Vec::new(),
            unique: CompactUnique::new(),
            num_vars: 0,
            ite_cache,
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

    /// Memory breakdown (arena + unique + cache). `unique_bytes` includes
    /// empty slots; at 0.75 load the linear-probe table runs ~(1.33 × slot_width)
    /// bytes per node.
    pub fn mem_stats(&self) -> MemStats {
        MemStats {
            arena_bytes: self.buf.len(),
            unique_bytes: self.unique.bytes(),
            cache_bytes: ITE_CACHE_SLOTS * std::mem::size_of::<IteCacheEntry<O>>(),
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
        let slot = (ite_key_hash::<O>(f, g, h) & ITE_CACHE_MASK) as usize;
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
        let slot = (ite_key_hash::<O>(f, g, h) & ITE_CACHE_MASK) as usize;
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

    /// Copying garbage collector. Takes a slice of roots, builds a fresh arena
    /// containing only the nodes reachable from those roots, and returns the
    /// remapped roots. The apply cache is flushed; callers must replace their
    /// own held refs with the returned ones.
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

impl<C: NodeCodec<O>, O: ArenaOffset> Default for Manager<C, O> {
    fn default() -> Self {
        Self::new_parameterized()
    }
}

/// Convenience: `Manager::new()` returns the default compact (u32, Leb128)
/// engine. This is deliberately a concrete-type method rather than a generic
/// one, so tests can write `Manager::new()` and have inference pick the
/// defaults. For other configurations use a type alias plus `::default()`:
///
/// ```ignore
/// let m: LargeManager = LargeManager::default();
/// ```
impl Manager<Leb128Codec, u32> {
    pub fn new() -> Self {
        Self::new_parameterized()
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
