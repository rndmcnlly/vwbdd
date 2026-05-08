//! Compact open-addressed unique table for BDD canonicity.
//!
//! Struct-of-arrays: one `Vec<u64>` of arena offsets (plus one, so `0`
//! means empty) alongside a parallel `Vec<u8>` of hash-derived tags. The
//! full key `(var, lo, hi)` lives in the arena; we recover it on probe
//! verify via [`decode_node`].
//!
//! Sizing: 9 B/slot, ~12 B/node at 0.75 load. For arenas under 4 GiB a
//! future shrink to `u32` slots would recover ~25% of the unique-table
//! footprint; see VWBDD.md's discussion of the width trade.
//!
//! **Why u8 hash tags.** A 1-byte tag gives ~1/256 false-positive rate
//! per probe, filtering almost all mismatched probes before the expensive
//! node decode. Tag 0 is reserved for empty slots, so live tags are in
//! 1..=255 (we lose 1 bit of entropy; overall FP rate ~1/255).
//!
//! Collisions resolve by linear probe: tag mismatch skips decode; tag
//! match triggers verify-on-decode. No separate overflow bucket.

use crate::codec::{decode_node, Ref};

/// Initial slot count. Fits in L1.
const INITIAL_CAP: usize = 1024;

pub struct CompactUnique {
    slots: Vec<u64>,
    tags: Vec<u8>,
    len: usize,
    /// When `len` reaches this threshold, resize 2×. Recomputed on resize
    /// as `new_cap * 3 / 4` (0.75 load).
    resize_at: usize,
}

/// Derive a nonzero u8 tag from a 64-bit hash.
#[inline]
fn tag_of_hash(h: u64) -> u8 {
    ((h >> 56) as u8) | 1
}

/// A fast mixing hash for `(var, lo, hi)` triples. Splitmix64-style chain.
/// Good enough for unique-table keys; collisions resolve by verify-on-decode.
#[inline]
pub fn unique_key_hash(var: u32, lo: Ref, hi: Ref) -> u64 {
    use crate::codec::ref_to_u64;
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

impl CompactUnique {
    pub fn new() -> Self {
        Self {
            slots: vec![0u64; INITIAL_CAP],
            tags: vec![0u8; INITIAL_CAP],
            len: 0,
            resize_at: INITIAL_CAP * 3 / 4,
        }
    }

    pub fn len(&self) -> usize {
        self.len
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
    /// match or `None` on empty-slot terminator. Tag check comes first;
    /// decode only on tag match.
    pub fn lookup(&self, hash: u64, var: u32, lo: Ref, hi: Ref, buf: &[u8]) -> Option<u64> {
        let mask = self.mask();
        let tag = tag_of_hash(hash);
        let mut i = (hash as usize) & mask;
        loop {
            let t = self.tags[i];
            if t == 0 {
                return None;
            }
            if t == tag {
                // slot stores offset+1 (0 is the empty sentinel).
                let off = self.slots[i] - 1;
                let (n, _) = decode_node(&buf[off as usize..], off);
                if n.var == var && n.lo == lo && n.hi == hi {
                    return Some(off);
                }
            }
            i = (i + 1) & mask;
        }
    }

    /// Insert `offset` at the slot determined by `hash`. Caller must have
    /// verified (via `lookup`) that the key is not already present.
    /// Triggers resize when load exceeds 0.75.
    pub fn insert(&mut self, hash: u64, offset: u64, buf: &[u8]) {
        debug_assert!(
            offset < u64::MAX,
            "arena offset overflow in CompactUnique"
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
        self.slots[i] = offset + 1;
        self.tags[i] = tag;
        self.len += 1;
    }

    /// Grow to 2× capacity and reinsert every live entry at its new slot.
    /// We re-hash from the decoded `(var, lo, hi)` to avoid storing hashes.
    fn resize(&mut self, buf: &[u8]) {
        let new_cap = self.cap() * 2;
        let mut new_slots = vec![0u64; new_cap];
        let mut new_tags = vec![0u8; new_cap];
        let new_mask = new_cap - 1;
        for (i, &slot) in self.slots.iter().enumerate() {
            if self.tags[i] == 0 {
                continue;
            }
            let off = slot - 1;
            let (n, _) = decode_node(&buf[off as usize..], off);
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

    /// Resize to fit approximately `expected` entries at 0.75 load,
    /// rounding up to a power of two. Called post-GC to shrink the table
    /// back down after construction has inflated it past the live-node
    /// footprint.
    pub fn resize_for(&mut self, expected: usize) {
        let needed = ((expected * 4 + 2) / 3).max(INITIAL_CAP);
        let mut new_cap = INITIAL_CAP;
        while new_cap < needed {
            new_cap *= 2;
        }
        self.slots = vec![0u64; new_cap];
        self.tags = vec![0u8; new_cap];
        self.len = 0;
        self.resize_at = new_cap * 3 / 4;
    }

    pub fn bytes(&self) -> usize {
        self.slots.len() * std::mem::size_of::<u64>() + self.tags.len()
    }
}

impl Default for CompactUnique {
    fn default() -> Self {
        Self::new()
    }
}

/// A family of per-variable unique tables.
///
/// Motivation: a single global `CompactUnique` at k=15 mult-trunc (10M
/// nodes, ~150 MB table) is the #1 hot spot in a time-profile, with
/// most cost attributable to DRAM-random-access probes. Partitioning
/// by variable gives smaller per-table working sets, many of which fit
/// in L2 during the construction phases where that variable is active.
/// This is what CUDD and OxiDD do.
///
/// Each variable's nodes are hash-consed only against other nodes with
/// the same variable, which is sound because the canonicity predicate
/// `(var, lo, hi)` already partitions by `var`: no cross-var collisions
/// are possible.
///
/// Empty-var tables (variable declared but never used as a node's `var`)
/// stay at `INITIAL_CAP` slots — 9 KiB each — so the per-var overhead is
/// ~9 KiB/var before the first insert. At the scales we target (fewer
/// than 1000 vars), that's under 9 MiB of fixed overhead, an acceptable
/// trade for the locality win on the hot probe path.
pub struct UniqueTables {
    tables: Vec<CompactUnique>,
}

impl UniqueTables {
    pub fn new() -> Self {
        Self { tables: Vec::new() }
    }

    /// Ensure there's a table for variable `var`. Called from `new_var`.
    /// Idempotent (no-op if `var` is already covered).
    pub fn ensure_var(&mut self, var: u32) {
        let needed = var as usize + 1;
        while self.tables.len() < needed {
            self.tables.push(CompactUnique::new());
        }
    }

    /// Total live entries across all per-var tables.
    pub fn len(&self) -> usize {
        self.tables.iter().map(|t| t.len()).sum()
    }

    /// Total bytes across all per-var tables, including empty ones.
    pub fn bytes(&self) -> usize {
        self.tables.iter().map(|t| t.bytes()).sum()
    }

    /// Look up `(var, lo, hi)` in the table for `var`. Returns the
    /// existing offset on match or `None` on miss.
    ///
    /// Panics if `var` has not been registered via `ensure_var`.
    #[inline]
    pub fn lookup(&self, hash: u64, var: u32, lo: Ref, hi: Ref, buf: &[u8]) -> Option<u64> {
        self.tables[var as usize].lookup(hash, var, lo, hi, buf)
    }

    /// Insert `offset` into the table for `var`. Caller must have
    /// verified via `lookup` that the key is not present.
    #[inline]
    pub fn insert(&mut self, hash: u64, var: u32, offset: u64, buf: &[u8]) {
        self.tables[var as usize].insert(hash, offset, buf);
    }

    /// Clear all per-var tables and pre-size each one for its expected
    /// load after a full arena scan. Used by `rebuild_from_arena` /
    /// `resize_for` equivalents.
    pub fn reset_for(&mut self, per_var_expected: &[usize]) {
        self.ensure_var(per_var_expected.len().saturating_sub(1) as u32);
        for (v, table) in self.tables.iter_mut().enumerate() {
            let expected = per_var_expected.get(v).copied().unwrap_or(0);
            table.resize_for(expected);
        }
    }

    /// Number of declared variable tables.
    pub fn num_vars(&self) -> usize {
        self.tables.len()
    }
}

impl Default for UniqueTables {
    fn default() -> Self {
        Self::new()
    }
}
