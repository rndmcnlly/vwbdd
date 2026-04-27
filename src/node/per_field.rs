//! Per-field backend with raw-u8 `var` prefix (§4.10 update).
//!
//! Layout:
//!   [u8 var]
//!   LEB128(lo_code)
//!   LEB128(hi_code)
//!
//! Three fields, no pairing math. var is raw u8 (not LEB128) for fast
//! var_of. See interleaved.rs for rationale.

use super::{code_to_ref, ref_to_code, Node, Ref};
use crate::leb::{decode_u128, encode_u128};

pub const ENCODING_NAME: &str = "per-field+u8var";

pub fn encode_node_at(
    var: u32,
    lo: Ref,
    hi: Ref,
    current_offset: u64,
    out: &mut Vec<u8>,
) {
    debug_assert_eq!(out.len() as u64, current_offset);
    debug_assert!(var < 256, "var {} exceeds u8", var);
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

#[inline]
pub fn decode_var_at(buf: &[u8], _current_offset: u64) -> (u32, usize) {
    (buf[0] as u32, 1)
}
