# vwbdd

A variable-width BDD engine in Rust: append-only LEB128 arena, verify-on-decode unique table, and a byte-level dump format that doubles as an IPC payload.

> **Status: research prototype.** The design note, **[VWBDD.md](./VWBDD.md)**, is the primary artifact; the code is how the claims were validated. Read that first if you want to know what this is really about.

## The claim in one paragraph

Every widely-used BDD engine (CUDD, BuDDy, Sylvan, OxiDD, dd.autoref) stores nodes as fixed-width records in a flat array — 16 bytes per node, chosen so that the node fits a hardware cache line. That's a correct local optimum. This repo explores a different local optimum: store BDD nodes as LEB128-encoded variable-width records in an append-only byte arena, with children referenced by backward byte deltas. The result, at k=15 on a truncated-multiplication benchmark, is:

- **3.5-4× smaller arena bytes per node** than fixed-width engines (2.6-4.6 B/node vs 16 B/node).
- **~16 B/node total engine footprint** (including unique table + apply cache), vs OxiDD's ~32 B/node.
- **2.5-3.1× slower wall clock** than OxiDD native in compute-heavy workloads, flat across a 176× growth in node count (k=12 → k=17).
- **Runs problems that the reference wasm BDD engine can't**: at k=17 on the iota mult benchmark, OxiDD-wasm OOMs inside the 4 GiB wasm32 linear-memory ceiling; vwbdd finishes the same workload in 1.12 GB.

For a deployment where working-set size is the binding constraint — browser clients, embedded contexts, or server-to-client BDD shipping — the compact-at-a-cost trade is the right one.

For a workload where wall time is the binding constraint and the BDD fits comfortably in RAM, use [OxiDD](https://github.com/oxidd/oxidd) or CUDD. This repo doesn't try to be faster than them.

## What's here

```
src/
├── lib.rs         module glue, public re-exports, type aliases
├── leb.rs         LEB128 u128 encode/decode
├── codec.rs       ArenaOffset + NodeCodec traits; Leb128Codec (§4.12)
├── node.rs        compatibility shim for free-function test access
├── unique.rs      CompactUnique<C, O>: linear-probe verify-on-decode (§4.7)
├── dump.rs        .vwbdd native format: dump/load/absorb with CRC32 (§4.14)
└── manager.rs     Manager<C, O>, ManagerConfig, make_node, ite, gc, apply cache

tests/
├── (basic)        leb, node, manager, ite, gc
├── differential   node-count equality vs OxiDD on the same formulas
├── compression    bytes/node scaling on AND/XOR chains
├── mult           full x*y=z relation, k=2..8
├── mult_trunc_*   truncated (x*y) mod 2^k = z, matches iota demo
├── timing*        wall-clock comparisons vs OxiDD (ignored; run with --ignored)
├── large_manager  u32/u64 cross-width correctness
├── cache_config   ManagerConfig builder tests
└── dump           .vwbdd roundtrip, multi-root, absorb dedup, error paths
```

Total: 65 tests passing, 3 `#[ignore]`d timing sweeps. The code is deliberately small and pedagogically readable: ~1200 LOC in `src/` with no runtime dependencies. Heavy commenting throughout; every architectural choice documented.

## Minimal example

```rust
use vwbdd::{Manager, Ref};

let mut m = Manager::new();
let x = m.new_var();
let y = m.new_var();
let f = m.r#false();
let t = m.r#true();

let vx = m.make_node(x, f, t);
let vy = m.make_node(y, f, t);
let and = m.and(vx, vy);

println!("x ∧ y has {} reachable nodes", m.num_nodes());

// Dump to a file; load later or in another process
m.dump("and.vwbdd", &[and])?;

let (mut m2, loaded) = Manager::load("and.vwbdd")?;
assert_eq!(loaded.roots.len(), 1);
```

## Running the benchmarks

The comparison harnesses build both engines in the same test binary and run them on identical formulas:

```sh
cargo test --release                    # 65 tests, including oxidd differential
cargo test --release --test timing      # k=4..8 mult timing vs oxidd
cargo test --release --test timing_large --  --ignored --nocapture   # k=8..11
cargo test --release --test mult_trunc_timing -- --ignored --nocapture  # matches iota
```

The first run pulls OxiDD from [rndmcnlly/oxidd @ b7fdc97](https://github.com/rndmcnlly/oxidd/tree/wasm32-support) and compiles it (~1-2 min).

## Design note

The full design writeup, **[VWBDD.md](./VWBDD.md)** (~800 lines), covers:

- §1-3: motivation, design, measurements
- §4.1-4.13: the optimization-and-rejection history — what we tried, what worked, what didn't, and why
- §4.14: the native dump format and its role as a multi-process parallelism primitive
- §5: what's still missing
- §6: the verdict — for whom is this worth using, and for whom isn't it
- §7: file index

Several of the §4 sections record optimizations we tried and rejected (cuckoo hashing, hybrid abs+rel addressing, four hand-rolled unique tables, three codec alternatives). They're the most useful sections for anyone thinking about a similar project.

## Related work

- [**OxiDD**](https://github.com/oxidd/oxidd) — the reference modern BDD engine in Rust that vwbdd's measurements compare against. Has multi-threaded apply, lock-free caches, atomic refcounts. If you want a production BDD engine in Rust, use OxiDD.
- [**iota**](https://github.com/rndmcnlly/oxidd-wasm) — typed symbolic-programming layer over OxiDD compiled to wasm32. vwbdd's k=15-17 measurements use iota's truncated-mult workload as a reference point.
- [**DDDMP**](https://github.com/ssoelvsten/cudd/tree/main/dddmp) — CUDD's native BDD dump format. vwbdd's `.vwbdd` format adopts multi-root support from DDDMP's convention (`.nroots` + `.rootids`) while using a more compact binary layout.

## License

MIT. See [LICENSE](./LICENSE).
