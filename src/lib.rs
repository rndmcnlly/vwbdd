//! vwbdd: a variable-width BDD engine.
//!
//! Nodes live in an append-only byte buffer encoded as:
//!   [u8 var] LEB128(lo_code) LEB128(hi_code)
//! where child codes are back-pointer deltas (0=F, 1=T, 2+k=node at -k).
//!
//! Single-threaded `&mut self` design, plain BDDs (no complement edges)
//! to match `oxidd::bdd::BDDFunction` canonicity so we can differentially
//! test. Optimized for arena density (fit more nodes in fixed memory) at
//! a modest speed cost vs oxidd.
//!
//! Arena offsets are `u64`: simplest choice at a small density cost vs
//! `u32` (~12 B/node unique table vs ~6.7 B/node). A future `u32` variant
//! can be gated by a cargo feature if a wasm-sized deployment needs it;
//! for now the code stays single-width.
//!
//! The shipping primitives live in [`slab`]: [`Slab`] / [`Diff`] are the
//! in-memory transport units. Persistence is left to callers (wrap the
//! raw slab bytes in any container you like).

pub mod codec;
pub mod leb;
pub mod manager;
pub mod node;
pub mod slab;
pub mod unique;

pub use codec::{Node, Ref};
pub use manager::{Manager, ManagerConfig, MemStats, DEFAULT_ITE_CACHE_SLOTS};
pub use slab::{Diff, Slab};
