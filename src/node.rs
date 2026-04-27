//! Compatibility shim re-exporting the default (u32-offset Leb128) codec
//! surface as free functions. This file existed before the codec abstraction
//! landed; keeping the free functions lets existing tests that poke the
//! codec directly (e.g. `tests/node.rs`) stay readable and unchanged.
//!
//! For the parameterized types and the `NodeCodec` / `ArenaOffset` traits,
//! see `src/codec.rs`.

pub use crate::codec::{code_to_ref, ref_to_code, ref_to_u64, Leb128Codec, Node, NodeCodec, Ref};

// Free-function wrappers over the default codec at the default (u32) width.
// Matches the pre-parameterization API; exists for backward compat of the
// `tests/node.rs` roundtrip tests. New code should go through
// `<Leb128Codec as NodeCodec<O>>::encode` directly.

#[inline]
pub fn encode_node_at(var: u32, lo: Ref, hi: Ref, current_offset: u64, out: &mut Vec<u8>) {
    <Leb128Codec as NodeCodec<u32>>::encode(var, lo, hi, current_offset as u32, out)
}

#[inline]
pub fn decode_node_at(buf: &[u8], current_offset: u64) -> (Node, usize) {
    <Leb128Codec as NodeCodec<u32>>::decode(buf, current_offset as u32)
}

#[inline]
pub fn decode_var_at(buf: &[u8], current_offset: u64) -> (u32, usize) {
    <Leb128Codec as NodeCodec<u32>>::decode_var(buf, current_offset as u32)
}
