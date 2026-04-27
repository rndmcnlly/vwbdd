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
//!
//! ## Parameterization
//!
//! Two orthogonal axes expressed as type parameters on [`Manager`]:
//!
//! - **[`ArenaOffset`]**: `u32` (default, 4 GiB cap, compact — the right
//!   choice for wasm and any deployment where the arena is known to fit)
//!   or `u64` (lifts the ceiling to host RAM for server-side builds).
//! - **[`NodeCodec`]**: the wire format for a `(var, lo, hi)` triple.
//!   [`Leb128Codec`] is the only one shipped today.
//!
//! Two convenience aliases:
//! - [`DefaultManager`] = `Manager<Leb128Codec, u32>` — the compact build.
//! - [`LargeManager`] = `Manager<Leb128Codec, u64>` — the unlimited-arena build.
//!
//! Server/client split: build a large BDD with `LargeManager`, serialize
//! the arena, ship it to a wasm client running `DefaultManager` for
//! inference-only workloads under 4 GiB.

pub mod codec;
pub mod dump;
pub mod leb;
pub mod manager;
pub mod node;
pub mod unique;

pub use codec::{ArenaOffset, Leb128Codec, Node, NodeCodec, Ref};
pub use dump::{DumpError, LoadedRoots};
pub use manager::{Manager, ManagerConfig, MemStats, DEFAULT_ITE_CACHE_SLOTS};

/// Compact-arena manager: `u32` offsets, 4 GiB max, ~6.7 B/node unique table.
/// Defaults-compatible alias so existing code written as `Manager` keeps working.
pub type DefaultManager = Manager<Leb128Codec, u32>;

/// Large-arena manager: `u64` offsets, host-RAM limit, ~12 B/node unique table.
/// For server-side BDD builders that later hand off a serialized arena.
pub type LargeManager = Manager<Leb128Codec, u64>;
