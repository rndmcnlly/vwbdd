//! Interleaved LEB128 backend (§4.2 historical default).
//!
//!   LEB128(var)
//!   LEB128(interleave(lo_code, hi_code)) as u128
//!
//! Rationale, preserved from earlier notes: we tried a more-compressed form
//! that stored v_skip instead of var (var could be reconstructed from the
//! children's vars). That saved ~0.5-1 byte per node but forced decode to look
//! up each child's var in a side HashMap, costing ~10 ns/lookup × 2 = ~20 ns
//! per decode on top of the ~20 ns of LEB128/pairing work. By storing var
//! inline, decode needs no side table at all: ~25 ns total, 2-3× faster.

use super::{code_to_ref, ref_to_code, Node, Ref};
use crate::leb::{decode_u128, encode_u128};
use crate::pair::{deinterleave, interleave};

pub const ENCODING_NAME: &str = "interleaved";

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

/// Decode just the `var` field. Used by `var_of`, which is hotter than full
/// decode (ordering asserts, top_var in ite). Returns (var, bytes consumed by
/// the var field only).
#[inline]
pub fn decode_var_at(buf: &[u8], _current_offset: u64) -> (u32, usize) {
    let (var, n) = decode_u128(&buf[..]);
    (var as u32, n)
}
