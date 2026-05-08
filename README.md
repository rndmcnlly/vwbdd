# vwbdd

A variable-width BDD engine in Rust: append-only LEB128 arena + verify-on-decode unique table + an in-memory `Slab`/`Diff` exchange format that lets a server ship a delta to a client that already holds the base arena.

> **Status: research prototype.** The design note, **[VWBDD.md](./VWBDD.md)**, is the primary artifact; the code is how the claims were validated. Read that first if you want to know what this is really about.

## The claim in one paragraph

Every widely-used BDD engine (CUDD, BuDDy, Sylvan, OxiDD, dd.autoref) stores nodes as fixed-width records in a flat array: 16 bytes per node, chosen so the node fits a hardware cache line. That's a correct local optimum. This repo explores a different one: store BDD nodes as LEB128-encoded variable-width records in an append-only byte arena, with children referenced by backward byte deltas. The result, on a truncated-multiplication benchmark at k=17 (82M reachable nodes):

- **~3× smaller arena bytes per node** than fixed-width engines (3.9–5.4 B/node across k=7..17 vs 16 B/node).
- **~20–26 B/node total engine footprint** (including unique table), vs OxiDD's ~32 B/node.
- Wall clock within a small-constant factor (~3×) of OxiDD on the same workloads.
- **Byte-level position-independence**: tail bytes appended by any process on top of a shared base are bit-identical, so a compact server can ship a ~KB delta to a client holding megabytes of shared base.

For a deployment where working-set size is the binding constraint (browser clients, embedded contexts, server-to-client BDD shipping), the compact-at-a-cost trade is the right one. For a workload where wall time is the binding constraint and the BDD fits comfortably in RAM, use [OxiDD](https://github.com/oxidd/oxidd) or CUDD.

## What's here

```
src/
├── lib.rs         module glue and public re-exports
├── leb.rs         LEB128 u128 encode/decode
├── codec.rs       Ref, Node, encode_node/decode_node/decode_var
├── node.rs        compatibility shim for free-function test access
├── unique.rs      CompactUnique: linear-probe verify-on-decode (§4.7)
└── manager.rs     Manager, ManagerConfig, make_node, ite, gc, gc_tail, apply cache

tests/
├── (basic)          leb, node, manager, ite, gc
├── differential     node-count equality vs OxiDD on the same formulas
├── mult             full x*y=z relation, k=2..8
├── mult_trunc_*     truncated (x*y) mod 2^k = z, matches iota demo
├── timing           wall-clock vs OxiDD on mult, k=4..8
├── slab             Slab/Diff roundtrips, extend_slab, clean-bytes invariant
├── slab_queries     Slab::support and Slab::sat_count read-only queries
└── cache_config     ManagerConfig builder tests
```

Total: 70 tests passing. No runtime dependencies. See §8 of VWBDD.md for the agility-driven cut that retired the `NodeCodec`/`ArenaOffset` traits and the `.vwbdd` file format.

## Minimal example

```rust
use vwbdd::{Manager, Slab};

let mut m = Manager::new();
let x = m.new_var();
let y = m.new_var();
let f = m.r#false();
let t = m.r#true();

let vx = m.make_node(x, f, t);
let vy = m.make_node(y, f, t);
let and = m.and(vx, vy);

println!("x ∧ y has {} reachable nodes", m.num_nodes());

// Ship the arena as a compact in-memory slab (bytes + roots).
// Persistence is caller-supplied: wrap `slab.bytes` in any container.
let slab: Slab = m.slab_for(&[and]);
```

## Running the benchmarks

The comparison harnesses build both engines in the same test binary and run them on identical formulas:

```sh
cargo test --release                # 70 tests, including oxidd differential
cargo test --release --test timing  # k=4..8 mult timing vs oxidd
```

The first run pulls OxiDD from [rndmcnlly/oxidd @ b7fdc97](https://github.com/rndmcnlly/oxidd/tree/wasm32-support) and compiles it (~1–2 min).

## Design note

The full design writeup, **[VWBDD.md](./VWBDD.md)**, covers:

- §1–3: motivation, design, measurements
- §4.1–4.15: the optimization-and-rejection history
- §5: what's still missing
- §6: the verdict
- §7: file index
- §8: the agility cut (retired `NodeCodec`, `ArenaOffset` trait, `.vwbdd` file format; source surface reduced by ~37% LOC)

Several of the §4 sections record optimizations we tried and rejected (cuckoo hashing, hybrid abs+rel addressing, four hand-rolled unique tables, three codec alternatives). They're the most useful sections for anyone thinking about a similar project.

## Related work

- [**OxiDD**](https://github.com/oxidd/oxidd): the reference modern BDD engine in Rust that vwbdd's measurements compare against. Has multi-threaded apply, lock-free caches, atomic refcounts. If you want a production BDD engine in Rust, use OxiDD.
- [**iota**](https://github.com/rndmcnlly/oxidd-wasm): typed symbolic-programming layer over OxiDD compiled to wasm32. vwbdd's measurements at larger k use iota's truncated-mult workload as a reference point.

## License

MIT. See [LICENSE](./LICENSE).
