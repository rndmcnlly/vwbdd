//! Compact open-addressed unique table for BDD canonicity.
//!
//! Struct-of-arrays: one `Vec<O>` of arena offsets (plus one, so `O::ZERO`
//! means empty) alongside a parallel `Vec<u8>` of hash-derived tags. The full
//! key `(var, lo, hi)` lives in the arena; we recover it on probe verify via
//! the codec's `decode`.
//!
//! **Offset width is a compile-time parameter** via [`ArenaOffset`]. The two
//! instantiations we ship:
//!
//! - `O = u32` (default): 5 B/slot, ~6.7 B/node at 0.75 load, arena capped
//!   at 4 GiB. The right choice for wasm and any deployment where the
//!   arena is known to fit.
//! - `O = u64` (large-arena): 9 B/slot, ~12 B/node at 0.75 load, arena
//!   limited only by host RAM (multi-TB feasible). The right choice for
//!   server-side builders that will later serialize the arena for a
//!   smaller client.
//!
//! **Why u8 hash tags (shared across widths).** A 1-byte tag gives ~1/256
//! false-positive rate per probe, filtering almost all mismatched probes
//! before the expensive codec decode. Tag 0 is reserved for empty slots, so
//! live tags are in 1..=255 (we lose 1 bit of entropy; overall FP rate ~1/255).
//!
//! Collisions resolve by linear probe: tag mismatch skips decode; tag match
//! triggers verify-on-decode through the codec. No separate overflow bucket.

use std::marker::PhantomData;

use crate::codec::{ArenaOffset, NodeCodec, Ref};

/// Initial slot count. Fits in L1 at either offset width.
const INITIAL_CAP: usize = 1024;

pub struct CompactUnique<C: NodeCodec<O>, O: ArenaOffset> {
    slots: Vec<O>,
    tags: Vec<u8>,
    len: usize,
    /// When `len` reaches this threshold, resize 2×. Recomputed on resize as
    /// `new_cap * 3 / 4` (0.75 load).
    resize_at: usize,
    _codec: PhantomData<C>,
}

/// Derive a nonzero u8 tag from a 64-bit hash.
#[inline]
fn tag_of_hash(h: u64) -> u8 {
    ((h >> 56) as u8) | 1
}

/// A fast mixing hash for `(var, lo, hi)` triples. Splitmix64-style chain.
/// Good enough for unique-table keys; collisions resolve by verify-on-decode.
#[inline]
pub fn unique_key_hash<O: ArenaOffset>(var: u32, lo: Ref<O>, hi: Ref<O>) -> u64 {
    use crate::codec::ref_to_u64;
    let mut h = var as u64;
    h = h.wrapping_mul(0x9e3779b97f4a7c15);
    h ^= ref_to_u64::<O>(lo);
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h ^= ref_to_u64::<O>(hi);
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    h
}

impl<C: NodeCodec<O>, O: ArenaOffset> CompactUnique<C, O> {
    pub fn new() -> Self {
        Self {
            slots: vec![O::ZERO; INITIAL_CAP],
            tags: vec![0u8; INITIAL_CAP],
            len: 0,
            resize_at: INITIAL_CAP * 3 / 4,
            _codec: PhantomData,
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

    /// Look up `(var, lo, hi)` by hash. Returns the existing offset on match
    /// or `None` on empty-slot terminator. Tag check comes first; decode
    /// only on tag match.
    pub fn lookup(&self, hash: u64, var: u32, lo: Ref<O>, hi: Ref<O>, buf: &[u8]) -> Option<O> {
        let mask = self.mask();
        let tag = tag_of_hash(hash);
        let mut i = (hash as usize) & mask;
        loop {
            let t = self.tags[i];
            if t == 0 {
                return None;
            }
            if t == tag {
                // slot stores offset+1 (O::ZERO is the empty sentinel).
                let off = self.slots[i].checked_sub(O::ONE).expect("slot underflow");
                let (n, _) = C::decode(&buf[off.to_u64() as usize..], off);
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
    pub fn insert(&mut self, hash: u64, offset: O, buf: &[u8]) {
        debug_assert!(
            offset.checked_add(O::ONE).is_some(),
            "arena offset overflow in CompactUnique (width exhausted)"
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
        self.slots[i] = offset.wrapping_add(O::ONE);
        self.tags[i] = tag;
        self.len += 1;
    }

    /// Grow to 2× capacity and reinsert every live entry at its new slot.
    /// We re-hash from the decoded `(var, lo, hi)` to avoid storing hashes.
    fn resize(&mut self, buf: &[u8]) {
        let new_cap = self.cap() * 2;
        let mut new_slots = vec![O::ZERO; new_cap];
        let mut new_tags = vec![0u8; new_cap];
        let new_mask = new_cap - 1;
        for (i, &slot) in self.slots.iter().enumerate() {
            if self.tags[i] == 0 {
                continue;
            }
            let off = slot.checked_sub(O::ONE).expect("slot underflow on resize");
            let (n, _) = C::decode(&buf[off.to_u64() as usize..], off);
            let h = unique_key_hash::<O>(n.var, n.lo, n.hi);
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
    /// up to a power of two. Called post-GC to shrink the table back down
    /// after construction has inflated it past the live-node footprint.
    pub fn resize_for(&mut self, expected: usize) {
        let needed = ((expected * 4 + 2) / 3).max(INITIAL_CAP);
        let mut new_cap = INITIAL_CAP;
        while new_cap < needed {
            new_cap *= 2;
        }
        self.slots = vec![O::ZERO; new_cap];
        self.tags = vec![0u8; new_cap];
        self.len = 0;
        self.resize_at = new_cap * 3 / 4;
    }

    pub fn bytes(&self) -> usize {
        self.slots.len() * std::mem::size_of::<O>() + self.tags.len()
    }
}

impl<C: NodeCodec<O>, O: ArenaOffset> Default for CompactUnique<C, O> {
    fn default() -> Self {
        Self::new()
    }
}
