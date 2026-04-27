//! Node encoding backends.
//!
//! Three mutually exclusive encodings of `(var, lo, hi)` into the arena byte
//! buffer are selectable via Cargo features. All assume `var < 256` (we never
//! realistically run with more variables; at k=11 we use 44):
//!
//! - `encoding-interleaved` (default):
//!     [u8 var] LEB128(interleave(lo_code, hi_code))
//!   The §4.2 layout with the §4.10 `var`-is-raw-u8 fast path.
//!
//! - `encoding-per-field`:
//!     [u8 var] LEB128(lo_code) LEB128(hi_code)
//!   Three fields; no pair math. Same u8-var fast path.
//!
//! - `encoding-fixed`:
//!     repr-C [u32; 3] = (var, lo_packed, hi_packed), 12 B/node total.
//!   Ceiling comparison with no varint decode at all.
//!
//! Child reference encoding (`ref_to_code`/`code_to_ref`):
//!   0                 -> false terminal
//!   1                 -> true terminal
//!   2 + delta         -> node at (current_node_offset - delta)
//!
//! All backends expose the same API:
//!   encode_node_at(var, lo, hi, current_offset, out)
//!   decode_node_at(buf, current_offset) -> (Node, bytes_consumed)
//!   decode_var_at(buf, current_offset)  -> (var, bytes_consumed_if_only_var)
//!
//! The last is a fast path for `var_of` which is called twice per `make_node`
//! (ordering assertion) and once per `cofactor` / `top_var`.

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

// ---- Feature dispatch -------------------------------------------------------

#[cfg(all(feature = "encoding-per-field", feature = "encoding-interleaved"))]
compile_error!("encoding-per-field and encoding-interleaved are mutually exclusive");
#[cfg(all(feature = "encoding-fixed", feature = "encoding-interleaved"))]
compile_error!("encoding-fixed and encoding-interleaved are mutually exclusive");
#[cfg(all(feature = "encoding-fixed", feature = "encoding-per-field"))]
compile_error!("encoding-fixed and encoding-per-field are mutually exclusive");

// Default: interleaved (the §4.2 historical baseline).
#[cfg(not(any(feature = "encoding-per-field", feature = "encoding-fixed")))]
mod interleaved;
#[cfg(not(any(feature = "encoding-per-field", feature = "encoding-fixed")))]
pub use interleaved::{decode_node_at, decode_var_at, encode_node_at, ENCODING_NAME};

#[cfg(feature = "encoding-per-field")]
mod per_field;
#[cfg(feature = "encoding-per-field")]
pub use per_field::{decode_node_at, decode_var_at, encode_node_at, ENCODING_NAME};

#[cfg(feature = "encoding-fixed")]
mod fixed;
#[cfg(feature = "encoding-fixed")]
pub use fixed::{decode_node_at, decode_var_at, encode_node_at, ENCODING_NAME};
