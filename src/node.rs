//! Variable-width node encoding: one raw `u8 var` prefix, then two
//! LEB128-encoded child codes.
//!
//! Layout:
//!   [u8 var] LEB128(lo_code) LEB128(hi_code)
//!
//! Child reference encoding:
//!   0             -> Terminal(false)
//!   1             -> Terminal(true)
//!   2 + delta     -> Node at arena offset (current_offset - delta)
//!
//! Design notes:
//! - `var` is a raw byte (not LEB128) so `var_of` is a single buf[off] read.
//!   At any realistic BDD scale there are < 256 variables; debug-asserted.
//! - Child codes are back-pointer deltas so a node's children always come
//!   before it in the append-only arena. Deltas shrink to 1-2 LEB bytes for
//!   nearby children, grow as deep DAGs accumulate skips.
//! - An earlier design stored `v_skip` (parent's var minus own var) instead
//!   of `var` directly; that saved ~0.5-1 B/node but forced decode to look
//!   up children's vars in a side HashMap (~10 ns/lookup × 2). Inline var
//!   pays the byte for a 2-3× decode speedup. (§4.2 in VWBDD.md.)
//! - An earlier design interleaved lo/hi into a single LEB128 varint; on
//!   large workloads that was a near-wash in bytes and cost ~2-6% in decode
//!   time, so we dropped it (§4.10).

use crate::leb::{decode_u128, encode_u128};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ref {
    Terminal(bool),
    Node(u64),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Node {
    pub var: u32,
    pub lo: Ref,
    pub hi: Ref,
}

pub fn ref_to_code(r: Ref, current_offset: u64) -> u64 {
    match r {
        Ref::Terminal(false) => 0,
        Ref::Terminal(true) => 1,
        Ref::Node(child_off) => {
            assert!(
                child_off < current_offset,
                "child offset {} must precede current offset {}",
                child_off,
                current_offset
            );
            2 + (current_offset - child_off)
        }
    }
}

pub fn code_to_ref(code: u64, current_offset: u64) -> Ref {
    match code {
        0 => Ref::Terminal(false),
        1 => Ref::Terminal(true),
        c => Ref::Node(current_offset - (c - 2)),
    }
}

/// Pack a `Ref` into a u64 for hashing. Terminals get small distinct tags;
/// node offsets get a high bit set so they can't collide with them.
#[inline]
pub fn ref_to_u64(r: Ref) -> u64 {
    match r {
        Ref::Terminal(false) => 0x1,
        Ref::Terminal(true) => 0x2,
        Ref::Node(o) => 0x1000_0000_0000_0000u64 ^ o,
    }
}

pub fn encode_node_at(
    var: u32,
    lo: Ref,
    hi: Ref,
    current_offset: u64,
    out: &mut Vec<u8>,
) {
    debug_assert_eq!(out.len() as u64, current_offset);
    debug_assert!(var < 256, "var {} exceeds u8 (raise cap if needed)", var);
    out.push(var as u8);
    let lo_code = ref_to_code(lo, current_offset);
    let hi_code = ref_to_code(hi, current_offset);
    encode_u128(lo_code as u128, out);
    encode_u128(hi_code as u128, out);
}

pub fn decode_node_at(buf: &[u8], current_offset: u64) -> (Node, usize) {
    let var = buf[0] as u32;
    let (lo_code, n1) = decode_u128(&buf[1..]);
    let (hi_code, n2) = decode_u128(&buf[1 + n1..]);
    let lo = code_to_ref(lo_code as u64, current_offset);
    let hi = code_to_ref(hi_code as u64, current_offset);
    (Node { var, lo, hi }, 1 + n1 + n2)
}

/// Fast path for `var_of`: one byte, no varint.
#[inline]
pub fn decode_var_at(buf: &[u8], _current_offset: u64) -> (u32, usize) {
    (buf[0] as u32, 1)
}
