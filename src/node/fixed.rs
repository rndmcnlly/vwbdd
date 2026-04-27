//! Fixed-width backend (§4.10 ceiling comparison).
//!
//! Layout: each node is exactly 12 bytes, written as three little-endian u32s:
//!
//!   [0..4]  var
//!   [4..8]  lo_packed
//!   [8..12] hi_packed
//!
//! Packed-ref encoding (absolute, not delta):
//!   0                -> Terminal(false)
//!   1                -> Terminal(true)
//!   2..=u32::MAX     -> Node at arena offset (value - 2)
//!
//! Requires arena byte-length < u32::MAX - 1. At 12 B/node that's ~358M nodes,
//! way above our k=11 scale (~7.6M). Uses no LEB128, no pairing math, no
//! delta arithmetic: a decode is three u32 reads plus one compare per child.

use super::{Node, Ref};

pub const ENCODING_NAME: &str = "fixed12";

#[inline]
fn pack_ref(r: Ref) -> u32 {
    match r {
        Ref::Terminal(false) => 0,
        Ref::Terminal(true) => 1,
        Ref::Node(off) => {
            debug_assert!(off + 2 < u32::MAX as u64, "arena exceeded fixed12 capacity");
            (off as u32) + 2
        }
    }
}

#[inline]
fn unpack_ref(x: u32) -> Ref {
    match x {
        0 => Ref::Terminal(false),
        1 => Ref::Terminal(true),
        n => Ref::Node((n - 2) as u64),
    }
}

#[inline]
fn read_u32_le(buf: &[u8]) -> u32 {
    u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])
}

pub fn encode_node_at(
    var: u32,
    lo: Ref,
    hi: Ref,
    current_offset: u64,
    out: &mut Vec<u8>,
) {
    debug_assert_eq!(out.len() as u64, current_offset);
    out.extend_from_slice(&var.to_le_bytes());
    out.extend_from_slice(&pack_ref(lo).to_le_bytes());
    out.extend_from_slice(&pack_ref(hi).to_le_bytes());
}

pub fn decode_node_at(buf: &[u8], _current_offset: u64) -> (Node, usize) {
    let var = read_u32_le(&buf[0..4]);
    let lo = unpack_ref(read_u32_le(&buf[4..8]));
    let hi = unpack_ref(read_u32_le(&buf[8..12]));
    (Node { var, lo, hi }, 12)
}

#[inline]
pub fn decode_var_at(buf: &[u8], _current_offset: u64) -> (u32, usize) {
    (read_u32_le(&buf[0..4]), 4)
}
