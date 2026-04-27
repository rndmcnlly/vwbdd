//! Variable-width node encoding.
//!
//! Each node is two LEB128 varints:
//!
//!   LEB128(var)
//!   LEB128(interleave(lo_code, hi_code))    as u128
//!
//! Child reference encoding (u64):
//!   0                 -> false terminal
//!   1                 -> true terminal
//!   2 + delta         -> node at (current_node_offset - delta)
//!
//! Rationale: we tried a compressed form that stored v_skip instead of var
//! (var could be reconstructed from children's vars). That saved ~0.5-1 byte
//! per node but forced decode to look up each child's var in a side HashMap,
//! costing ~10 ns/lookup × 2 = ~20 ns per decode on top of the ~20 ns of
//! LEB128/pairing work. By storing var inline, decode needs no side table
//! at all: ~25 ns total, 2-3x faster. Worth the bytes.

use crate::leb::{decode_u128, encode_u128};
use crate::pair::{deinterleave, interleave};

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

pub fn encode_node_at(
    var: u32,
    lo: Ref,
    hi: Ref,
    current_offset: u64,
    out: &mut Vec<u8>,
) {
    debug_assert_eq!(out.len() as u64, current_offset);
    encode_u128(var as u128, out);
    let lo_code = ref_to_code(lo, current_offset);
    let hi_code = ref_to_code(hi, current_offset);
    let children = interleave(lo_code, hi_code);
    encode_u128(children, out);
}

pub fn decode_node_at(buf: &[u8], current_offset: u64) -> (Node, usize) {
    let (var, n0) = decode_u128(&buf[..]);
    let (packed, n1) = decode_u128(&buf[n0..]);
    let (lo_code, hi_code) = deinterleave(packed);
    let lo = code_to_ref(lo_code, current_offset);
    let hi = code_to_ref(hi_code, current_offset);
    (Node { var: var as u32, lo, hi }, n0 + n1)
}
