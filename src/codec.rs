//! Parameterization axes for the variable-width BDD engine.
//!
//! Two orthogonal dimensions:
//!
//! - [`ArenaOffset`]: the numeric type used to address bytes in the arena.
//!   `u32` is the compact default (4 GiB max arena, ~25% smaller unique
//!   table than u64) and is what wasm targets want. `u64` lifts the ceiling
//!   at a modest density cost — what you want on a 1 TB server building
//!   a large BDD before handing the compressed arena off to a smaller
//!   client.
//!
//! - [`NodeCodec`]: how a `(var, lo, hi)` triple is laid out in bytes.
//!   Currently only [`Leb128Codec`] (variable-width, the original design)
//!   is materialized. The trait exists so future experiments (interleaved
//!   LEB128, fixed12 ceiling, v_skip, hybrid abs+rel) can slot in without
//!   touching `Manager` plumbing.
//!
//! The [`Manager`] is generic over both: `Manager<C: NodeCodec, O: ArenaOffset>`
//! with `Leb128Codec` and `u32` as defaults.

use std::fmt::Debug;
use std::hash::Hash;

use crate::leb::{decode_u128, encode_u128};

/// Numeric trait abstracting the arena offset width. Implemented for `u32`
/// and `u64`. Callers shouldn't need to implement this themselves.
///
/// We deliberately keep the surface area small: enough to increment/decrement
/// offsets, convert to/from `u64` for hashing and LEB128 encoding, and
/// compare. The `to_u64` / `from_u64` conversions are cheap (both widths
/// are ≤ 64 bits) and are what lets us reuse the splitmix-style hashers
/// and the generic LEB128 codec across offset widths.
pub trait ArenaOffset:
    Copy + Clone + Debug + Eq + Ord + Hash + Default + 'static
{
    const ZERO: Self;
    const ONE: Self;
    /// Largest representable offset (used for overflow checks).
    const MAX: Self;

    fn from_u64(v: u64) -> Self;
    fn to_u64(self) -> u64;

    fn checked_add(self, rhs: Self) -> Option<Self>;
    fn checked_sub(self, rhs: Self) -> Option<Self>;
    fn saturating_add(self, rhs: Self) -> Self;
    fn wrapping_add(self, rhs: Self) -> Self;
}

impl ArenaOffset for u32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MAX: Self = u32::MAX;

    #[inline]
    fn from_u64(v: u64) -> Self {
        debug_assert!(v <= u32::MAX as u64, "u32 arena offset overflow: {}", v);
        v as u32
    }
    #[inline]
    fn to_u64(self) -> u64 {
        self as u64
    }
    #[inline]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        u32::checked_add(self, rhs)
    }
    #[inline]
    fn checked_sub(self, rhs: Self) -> Option<Self> {
        u32::checked_sub(self, rhs)
    }
    #[inline]
    fn saturating_add(self, rhs: Self) -> Self {
        u32::saturating_add(self, rhs)
    }
    #[inline]
    fn wrapping_add(self, rhs: Self) -> Self {
        u32::wrapping_add(self, rhs)
    }
}

impl ArenaOffset for u64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MAX: Self = u64::MAX;

    #[inline]
    fn from_u64(v: u64) -> Self {
        v
    }
    #[inline]
    fn to_u64(self) -> u64 {
        self
    }
    #[inline]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        u64::checked_add(self, rhs)
    }
    #[inline]
    fn checked_sub(self, rhs: Self) -> Option<Self> {
        u64::checked_sub(self, rhs)
    }
    #[inline]
    fn saturating_add(self, rhs: Self) -> Self {
        u64::saturating_add(self, rhs)
    }
    #[inline]
    fn wrapping_add(self, rhs: Self) -> Self {
        u64::wrapping_add(self, rhs)
    }
}

/// Node-layout strategy. Implementors are typically zero-sized marker
/// structs; the trait methods are free-function-style and capture the
/// three codec responsibilities:
///
/// 1. [`encode`][NodeCodec::encode]: serialize `(var, lo, hi)` into the arena.
/// 2. [`decode`][NodeCodec::decode]: recover the full node (for unique-table verify).
/// 3. [`decode_var`][NodeCodec::decode_var]: fast path for `var_of` —
///    hot enough that it gets its own method.
///
/// All three take `current_offset: O` so that child back-references can be
/// resolved to absolute `Ref<O>::Node` values without touching a side table.
///
/// Invariants implementors must uphold:
/// - `encode` appends exactly one node's worth of bytes to `out` and returns nothing.
/// - `decode` returns `(node, len)` where `len` is the number of bytes consumed
///   from the start of `buf`, such that `encode(...); decode(out, off)` is a roundtrip.
/// - `decode_var(buf, off)` returns the same `var` as `decode(buf, off).0.var`,
///   but should be cheaper when possible.
pub trait NodeCodec<O: ArenaOffset>: Copy + Clone + Debug + 'static {
    /// Diagnostic name, used in test sweeps and `MemStats`-adjacent logging.
    const NAME: &'static str;

    fn encode(var: u32, lo: Ref<O>, hi: Ref<O>, current_offset: O, out: &mut Vec<u8>);
    fn decode(buf: &[u8], current_offset: O) -> (Node<O>, usize);
    /// Fast-path `var` decode. Implementors that store `var` inline at the
    /// head of the node can make this a single byte read.
    fn decode_var(buf: &[u8], current_offset: O) -> (u32, usize);
}

// --- Core types, parameterized over offset width ---

/// A BDD reference: a terminal or a node at some arena offset.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ref<O: ArenaOffset = u32> {
    Terminal(bool),
    Node(O),
}

/// A decoded node. The codec is what knows how to produce this from bytes
/// and vice versa; this struct is the codec-independent in-memory form.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Node<O: ArenaOffset = u32> {
    pub var: u32,
    pub lo: Ref<O>,
    pub hi: Ref<O>,
}

/// Encode a child reference as a child code relative to the parent's offset.
/// 0/1 are the terminal sentinels; 2+delta means "node at parent_offset − delta".
///
/// This helper is codec-independent: any codec that uses delta-coded children
/// can share it. A codec that wanted absolute-indexed children would roll its own.
#[inline]
pub fn ref_to_code<O: ArenaOffset>(r: Ref<O>, current_offset: O) -> u64 {
    match r {
        Ref::Terminal(false) => 0,
        Ref::Terminal(true) => 1,
        Ref::Node(child_off) => {
            let co = child_off.to_u64();
            let cur = current_offset.to_u64();
            assert!(
                co < cur,
                "child offset {} must precede current offset {}",
                co, cur
            );
            2 + (cur - co)
        }
    }
}

#[inline]
pub fn code_to_ref<O: ArenaOffset>(code: u64, current_offset: O) -> Ref<O> {
    match code {
        0 => Ref::Terminal(false),
        1 => Ref::Terminal(true),
        c => Ref::Node(O::from_u64(current_offset.to_u64() - (c - 2))),
    }
}

/// Pack a `Ref` into a u64 for hashing. Terminals get small distinct tags;
/// node offsets get a high bit set so they can't collide with them.
///
/// The 0x1000_0000_0000_0000 Node-bit is above any plausible offset at any
/// width we support (u32 tops out at 2^32; u64 is limited by host RAM to
/// well below 2^60 at any realistic scale).
#[inline]
pub fn ref_to_u64<O: ArenaOffset>(r: Ref<O>) -> u64 {
    match r {
        Ref::Terminal(false) => 0x1,
        Ref::Terminal(true) => 0x2,
        Ref::Node(o) => 0x1000_0000_0000_0000u64 ^ o.to_u64(),
    }
}

// --- The one concrete codec we ship today ---

/// The original variable-width codec: `[u8 var][LEB128(lo_code)][LEB128(hi_code)]`.
///
/// Design notes (see VWBDD.md §4.2, §4.10):
/// - `var` is a raw byte (not LEB128) so `decode_var` is a single `buf[0]`.
///   Debug-asserted to stay below 256.
/// - Children are encoded as back-pointer deltas via [`ref_to_code`].
/// - LEB128 spends one byte per 7 bits of magnitude, so small deltas (the
///   common case: children adjacent to parent) encode in 1 byte; deep-DAG
///   edges grow gracefully.
#[derive(Copy, Clone, Debug, Default)]
pub struct Leb128Codec;

impl<O: ArenaOffset> NodeCodec<O> for Leb128Codec {
    const NAME: &'static str = "leb128";

    #[inline]
    fn encode(var: u32, lo: Ref<O>, hi: Ref<O>, current_offset: O, out: &mut Vec<u8>) {
        debug_assert_eq!(out.len() as u64, current_offset.to_u64());
        debug_assert!(var < 256, "var {} exceeds u8 (raise cap if needed)", var);
        out.push(var as u8);
        let lo_code = ref_to_code::<O>(lo, current_offset);
        let hi_code = ref_to_code::<O>(hi, current_offset);
        encode_u128(lo_code as u128, out);
        encode_u128(hi_code as u128, out);
    }

    #[inline]
    fn decode(buf: &[u8], current_offset: O) -> (Node<O>, usize) {
        let var = buf[0] as u32;
        let (lo_code, n1) = decode_u128(&buf[1..]);
        let (hi_code, n2) = decode_u128(&buf[1 + n1..]);
        let lo = code_to_ref::<O>(lo_code as u64, current_offset);
        let hi = code_to_ref::<O>(hi_code as u64, current_offset);
        (Node { var, lo, hi }, 1 + n1 + n2)
    }

    #[inline]
    fn decode_var(buf: &[u8], _current_offset: O) -> (u32, usize) {
        (buf[0] as u32, 1)
    }
}
