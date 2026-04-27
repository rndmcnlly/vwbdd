//! vwbdd: a variable-width BDD engine.
//!
//! Nodes live in an append-only byte buffer encoded as:
//!   [u8 var] LEB128(lo_code) LEB128(hi_code)
//! where child codes are back-pointer deltas (0=F, 1=T, 2+k=node at -k).
//!
//! Single-threaded `&mut self` design, plain BDDs (no complement edges) to
//! match oxidd::bdd::BDDFunction canonicity so we can differentially test.
//! Optimized for arena density (fit more nodes in fixed memory) at a modest
//! speed cost vs oxidd.

pub mod leb;
pub mod manager;
pub mod node;
pub mod unique;

pub use manager::{Manager, MemStats};
pub use node::{Node, Ref};
