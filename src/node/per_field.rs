//! Per-field LEB128 backend (§4.10 candidate).
//!
//!   LEB128(var)
//!   LEB128(lo_code)
//!   LEB128(hi_code)
//!
//! Drops the interleave/deinterleave pair math. Expected byte cost:
//! each code fits in fewer LEB bytes than their interleaving (which doubles
//! bit-width before varint-encoding), so the arena should actually shrink on
//! workloads where most child deltas are small. Decode is three independent
//! varints with no pairing math.

use super::{code_to_ref, ref_to_code, Node, Ref};
use crate::leb::{decode_u128, encode_u128};

pub const ENCODING_NAME: &str = "per-field";

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
    encode_u128(lo_code as u128, out);
    encode_u128(hi_code as u128, out);
}

pub fn decode_node_at(buf: &[u8], current_offset: u64) -> (Node, usize) {
    let (var, n0) = decode_u128(&buf[..]);
    let (lo_code, n1) = decode_u128(&buf[n0..]);
    let (hi_code, n2) = decode_u128(&buf[n0 + n1..]);
    let lo = code_to_ref(lo_code as u64, current_offset);
    let hi = code_to_ref(hi_code as u64, current_offset);
    (Node { var: var as u32, lo, hi }, n0 + n1 + n2)
}

#[inline]
pub fn decode_var_at(buf: &[u8], _current_offset: u64) -> (u32, usize) {
    let (var, n) = decode_u128(&buf[..]);
    (var as u32, n)
}
