//! Interleaved backend with raw-u8 `var` prefix (§4.10 update).
//!
//! Layout:
//!   [u8 var]
//!   LEB128(interleave(lo_code, hi_code)) as u128
//!
//! The var is a raw byte, not LEB128. `var_of` becomes a single `buf[off]`
//! read, no state machine. At our working scale (k=11, 44 vars) every var
//! already fit in one LEB byte, so this change is **free on arena size** and
//! makes the hottest decode path ~constant-time.
//!
//! Assumes `var < 256`. At k=64 we'd need 256 vars which is beyond any
//! realistic BDD workload; asserted in debug builds on encode.

use super::{code_to_ref, ref_to_code, Node, Ref};
use crate::leb::{decode_u128, encode_u128};
use crate::pair::{deinterleave, interleave};

pub const ENCODING_NAME: &str = "interleaved+u8var";

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
    let children = interleave(lo_code, hi_code);
    encode_u128(children, out);
}

pub fn decode_node_at(buf: &[u8], current_offset: u64) -> (Node, usize) {
    let var = buf[0] as u32;
    let (packed, n1) = decode_u128(&buf[1..]);
    let (lo_code, hi_code) = deinterleave(packed);
    let lo = code_to_ref(lo_code, current_offset);
    let hi = code_to_ref(hi_code, current_offset);
    (Node { var, lo, hi }, 1 + n1)
}

/// Fast path for `var_of`: one byte read, no varint.
#[inline]
pub fn decode_var_at(buf: &[u8], _current_offset: u64) -> (u32, usize) {
    (buf[0] as u32, 1)
}
