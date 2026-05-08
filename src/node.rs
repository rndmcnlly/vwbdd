//! Free-function wrappers over the node codec for tests that roundtrip
//! bytes directly. New code should use the functions in `codec` directly.

pub use crate::codec::{code_to_ref, decode_node, decode_var, encode_node, ref_to_code, ref_to_u64, Node, Ref};

// Thin aliases for the pre-refactor names still used by `tests/node.rs`.

#[inline]
pub fn encode_node_at(var: u32, lo: Ref, hi: Ref, current_offset: u64, out: &mut Vec<u8>) {
    encode_node(var, lo, hi, current_offset, out)
}

#[inline]
pub fn decode_node_at(buf: &[u8], current_offset: u64) -> (Node, usize) {
    decode_node(buf, current_offset)
}

#[inline]
pub fn decode_var_at(buf: &[u8], current_offset: u64) -> (u32, usize) {
    decode_var(buf, current_offset)
}
