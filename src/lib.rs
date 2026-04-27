//! vwbdd: variable-width BDD engine (experiment).
//!
//! A live BDD engine where nodes are stored in an append-only byte buffer using
//! a LEB128-based variable-width encoding. Children are referenced by backward
//! byte offsets from the current node.
//!
//! This is v0: just enough to drive TDD toward a working `ite`-based apply.
//! Plain BDDs, no complement edges, matching OxiDD's BDD (not BCDD) canonicity.

pub mod leb;
pub mod manager;
pub mod node;
pub mod pair;
pub mod profile;

pub use manager::{Manager, MemStats};
pub use node::{Node, Ref};
pub use node::ENCODING_NAME;
