//! Node layout and core types.
//!
//! The node format is `[u8 var][LEB128(lo_code)][LEB128(hi_code)]`:
//! - `var` is a raw byte (not LEB128) so `decode_var` is a single `buf[0]`.
//!   Debug-asserted to stay below 256.
//! - Children are encoded as back-pointer deltas via [`ref_to_code`].
//! - LEB128 spends one byte per 7 bits of magnitude, so small deltas (the
//!   common case: children adjacent to parent) encode in 1 byte; deep-DAG
//!   edges grow gracefully.
//!
//! Arena offsets are `u64`: the native word on server-class builds, and
//! the simplest choice. A wasm client holding a ≤ 4 GiB arena will still
//! only populate the low 32 bits of each offset.

use crate::leb::{decode_u128, encode_u128};

/// A BDD reference: a terminal or a node at some arena offset.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ref {
    Terminal(bool),
    Node(u64),
}

/// A decoded node, in-memory form.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Node {
    pub var: u32,
    pub lo: Ref,
    pub hi: Ref,
}

/// Encode a child reference as a child code relative to the parent's offset.
/// 0/1 are the terminal sentinels; 2+delta means "node at parent_offset − delta".
#[inline]
pub fn ref_to_code(r: Ref, current_offset: u64) -> u64 {
    match r {
        Ref::Terminal(false) => 0,
        Ref::Terminal(true) => 1,
        Ref::Node(child_off) => {
            assert!(
                child_off < current_offset,
                "child offset {} must precede current offset {}",
                child_off, current_offset
            );
            2 + (current_offset - child_off)
        }
    }
}

#[inline]
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

// --- Node encode / decode ---

/// Encode `(var, lo, hi)` at `current_offset` into `out`.
/// Appends exactly one node's worth of bytes.
#[inline]
pub fn encode_node(var: u32, lo: Ref, hi: Ref, current_offset: u64, out: &mut Vec<u8>) {
    debug_assert_eq!(out.len() as u64, current_offset);
    debug_assert!(var < 256, "var {} exceeds u8 (raise cap if needed)", var);
    out.push(var as u8);
    let lo_code = ref_to_code(lo, current_offset);
    let hi_code = ref_to_code(hi, current_offset);
    encode_u128(lo_code as u128, out);
    encode_u128(hi_code as u128, out);
}

/// Decode the node at the head of `buf`. `current_offset` is the parent's
/// offset (used to resolve child back-deltas into absolute refs).
/// Returns `(node, bytes_consumed)`.
#[inline]
pub fn decode_node(buf: &[u8], current_offset: u64) -> (Node, usize) {
    let var = buf[0] as u32;
    let (lo_code, n1) = decode_u128(&buf[1..]);
    let (hi_code, n2) = decode_u128(&buf[1 + n1..]);
    let lo = code_to_ref(lo_code as u64, current_offset);
    let hi = code_to_ref(hi_code as u64, current_offset);
    (Node { var, lo, hi }, 1 + n1 + n2)
}

/// Fast path for `var_of`: read the leading byte only.
#[inline]
pub fn decode_var(buf: &[u8], _current_offset: u64) -> (u32, usize) {
    (buf[0] as u32, 1)
}
