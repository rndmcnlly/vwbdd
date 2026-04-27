# VWBDD: a variable-width BDD engine experiment

A design note. How and why we built a live BDD engine where nodes are stored in an append-only byte buffer using LEB128-based variable-width encoding, with children referenced by backward byte offsets.

Audience: the author, future collaborators, and anyone curious whether variable-width node layouts can win on modern hardware. This document records the shape of the decisions we made and the measurements that justify or refute them.

---

## TL;DR

We built a single-file Rust BDD engine where each node is encoded as a raw u8 `var` byte followed by two LEB128-encoded back-reference child codes, stored in an append-only byte buffer. Canonicity is enforced by a unique table keyed on a cheap hash of `(var, lo, hi)`; on probe, collisions are resolved by decoding the arena node and verifying.

Across the mult-relation `x*y=z` from k=2 to k=11 (reachable nodes 29 to 7.6M):

- **Correctness**: node counts match OxiDD exactly at every scale, verified differentially via a shared bitblaster running both engines in the same test.
- **Arena compression**: post-GC, we store BDD nodes at **2.6 to 4.6 bytes each** (k=2 to k=11) vs OxiDD's fixed 16 B/node. Dominant factor is the byte-offset distance to child nodes, which LEB128 compresses well.
- **Parameterized engine** (§4.12): `Manager<C: NodeCodec, O: ArenaOffset>`. Default `DefaultManager = Manager<Leb128Codec, u32>` is the compact build (4 GiB arena cap, ~6.7 B/node unique-table density — right for wasm clients and other memory-tight deployments). `LargeManager = Manager<Leb128Codec, u64>` lifts the arena ceiling to host RAM for server-side builders, at ~12 B/node density. Single library, two monomorphizations, picked by type alias at the call site.
- **Unique-table compression** (§4.7): `CompactUnique` — struct-of-arrays offset slots + parallel hash tags. u32 default at **~6.7 B/node**; u64 large-arena at **~12 B/node**. Both far below the 32 B/node a generic `HashMap<u64,u64>` would cost.
- **Apply-cache compression** (§4.8, §4.11): a packed-ref encoding per offset width. Default (u32) gives 16 B entries (one 64 B cache line); u64 gives 32 B entries (two per line). Default cache total: **2 MB**. Large-arena cache total: 4 MB.
- **Decode fast path** (§4.10): raw u8 `var` + three per-field LEB128s, after an A/B showed that interleaving pair-math and LEB-encoding the var were both costly without winning bytes at our scale.
- **Total live memory (default build)**: **~15-16 B/node** (arena + unique table), plus a 2 MB fixed apply cache. Grand total at k=11: **~121 MB for 7.6M nodes**. About half of OxiDD's ~30-32 B/live-node.
- **Wall-clock time**: **1.75× slower than OxiDD** at k=11 on the default build (§4.12 post-parameterization; the monomorphized u32 engine is 14-21% faster than the single-width u64 engine of §4.11).

**Design trade-off, accepted explicitly**: within ~3× of OxiDD on runtime, at ~half OxiDD's total engine memory *and* unlimited-arena scaling when you need it. The §4.12 parameterization removed the prior either/or: no longer does the server-side build and the wasm-client build have to choose between compactness and capacity. The wasm client runs `DefaultManager` at 15-16 B/node with a 4 GiB arena ceiling; the server builds with `LargeManager` at ~25 B/node and can absorb up to ~40B live nodes in 1 TB of RAM. The serialized arena is codec-compatible across widths because `Leb128Codec` encodes the same bytes either way; the receiving engine just uses its own offset type to index them.

Four rejected optimizations earn their own sections: §4.4 (abs+rel hybrid addressing; the 2023 notebook's trick didn't transfer), §4.6 (four hand-rolled unique-table variants, all regressed), and §4.9 (cuckoo-4-slot at 0.85 load — accurate memory prediction, speed prediction wrong by 2-3×). §4.10 documents the A/B that validated the current simple per-field design. §4.11 records the u32 → u64 widening that lifted the 4 GiB arena ceiling (but ran into footprint regression on small cases). §4.12 resolved that tension with a compile-time parameterization.


## 1. Motivation

Modern BDD engines (OxiDD, CUDD, Sylvan, BuDDy) all use fixed-width 16-byte nodes in flat arrays. At k=10 mult-relation, that's ~200 MiB just for the nodes. The laptop cache hierarchy makes working-set size a first-order perf variable: an L2-resident BDD runs 2-3× faster per op than a DRAM-resident one (see the roofline analysis in the parent repo's session notes).

A 2023 notebook experiment (`CompressWithBDDs.ipynb` in the parent dir) showed that BDDs can be **serialized** at ~3 bytes/node using LEB128 + pairing-function packing, a ~5× size reduction. That codec was offline: the author built the BDD in `dd.autoref`, then walked it to emit a compressed byte stream.

The question that wouldn't leave me alone: what if you **computed** on the compressed form directly? Decode a node on-demand into a 16-byte local struct, do the apply work, write back. Append-only storage means child references can be byte offsets that are monotonically backward — which the LEB128 bias toward small values compresses well by construction.

If this works:
- Smaller working set → better cache residency → closer to memory-latency roofline.
- Append-only → mark-and-sweep GC at batch boundaries, no concurrent-access contention.
- Single codec for compute, storage, and transfer: dump the arena verbatim to disk; load is memcpy + a DFS to rebuild the unique table.

If it doesn't: at minimum we have a good serialization codec (which the SHIM.md already flagged as a todo), and we've learned where the variable-width hypothesis breaks down on compute.


## 2. The design

### 2.1 Wire format

Each node is one raw `u8 var` byte followed by two LEB128 varints:

```
u8  var                          // level number (0 = top), raw byte
LEB128(lo_code)                  // child reference, u128 encoding
LEB128(hi_code)                  // child reference
```

Child reference encoding:

```
0                 -> false terminal
1                 -> true terminal
2 + delta         -> node at (current_offset - delta), delta >= 0
```

**Why raw u8 var, not LEB128?** Any realistic BDD workload has far fewer than 256 variables (mult at k=11 uses 44), so a 1-byte var is always cheap. Raw-u8 makes `var_of(r)` a single `buf[off]` read — no state-machine LEB128 parse — and `var_of` is the hottest decode path (2× per `make_node` for ordering asserts, 3× per `ite` frame for the top-var pick, etc). See §4.10.

**Why three separate LEB128s, not one combined varint?** An earlier version bit-interleaved `lo_code` and `hi_code` into a single u128, then LEB128-encoded that. The self-framing interleave avoided storing a length prefix for lo. On small workloads it saved ~17-29% of arena bytes; on the mult-k≥6 workload the pair-math cost matched the compression win, and wall time was ~2-6% slower than the simpler per-field design. We kept simplicity. (§4.10.)

**Why LEB128 at all?** Child deltas have wide dynamic range: ~1 byte for adjacent-node back-references (the common case, because ripple-adder structure produces lots of locality), several bytes for cross-level edges in deep DAGs. LEB128 is the right encoding shape.

**Why not v_skip instead of var?** An even earlier design stored `v_skip = min_child_var - var - 1` (usually 0) so the var field LEB128'd to fewer bytes. Decode had to reconstruct `var` by looking up both children's vars in a side `HashMap<offset, var>` — ~20 ns extra per decode for ~0.5-1 byte saved. Inline `var` is a 2-3× decode speedup for 1 byte/node. Easy call for a live engine; a pure storage codec where decode cost is paid once would keep v_skip.


### 2.2 Append-only arena + unique table + apply cache

The manager (`src/manager.rs`) owns three pieces:

1. `buf: Vec<u8>` — the arena. Nodes are appended as they're built, never mutated.
2. `unique: CompactUnique` (§4.7, `src/unique.rs`) — open-addressed table with `Vec<u32>` offsets and a parallel `Vec<u8>` of hash tags. Key is a hash of `(var, lo, hi)`; the canonical offset lives in the slot. On tag match we decode the arena node and verify the full key.
3. `ite_cache: Box<[IteCacheEntry; 2^17]>` (§4.5, §4.8) — direct-mapped apply cache for `ite`. Each entry is a 16 B (`PackedRef × 4`) one-cache-line record; collisions evict.

`make_node(var, lo, hi)` enforces both reduction (if `lo == hi`, return `lo`) and canonicalization (same triple → same offset). Variable ordering is asserted: parent's var must be strictly less than each non-terminal child's var.

`ite` is a Shannon expansion with memoization. v0 was recursive; the current version is iterative with an explicit `Vec<Frame>` worklist to avoid stack overflow on deep BDDs. Each frame is either `Enter` (compute cofactors, push children) or `Combine` (children's results are in the cache, make this node).

All operations are `&mut self`. The Rust borrow checker gives us "one writer at a time" for free, at the type level, with no runtime locking cost. This sidesteps the footgun that OxiDD's multi-reader/single-writer design imposes on callers (see footgun section in the parent `SHIM.md`). For workloads that are inherently serial (fixpoint loops), this is a strict upgrade.


### 2.3 Copying GC at batch boundaries

`gc(roots: &[Ref]) -> Vec<Ref>` is a copying collector. Given a slice of root `Ref`s the caller wants to keep, it:

1. DFS-walks reachable nodes from each root (iterative, explicit stack).
2. In post-order (children before parents), emits each reached node into a fresh buffer.
3. Builds a new unique table over the new offsets.
4. Replaces `buf`, `unique`, `unique_collisions`. Flushes the apply cache (old offsets are invalid).
5. Returns the roots remapped to new offsets.

Why copying instead of mark-and-sweep? The arena is variable-width, so there's no "free slot" to mark. You have to either leave byte-level holes (compaction becomes mandatory eventually anyway) or copy live nodes to a fresh arena (what we do). Copying is O(live nodes), which is the right complexity for the work we actually want to do.

Post-GC, the manager contains exactly the reachable set. Dead scaffolding from intermediate computations is gone. In our mult-relation benchmarks, this routinely reduces the live node count by 5-10× vs the "all ever built" total.

Subtle caveat: after GC, all `Ref` values held by the caller outside the `roots` array are dangling. Callers must thread their roots through gc calls. The type system could enforce this with lifetime-parameterized handles, but v0 trusts the caller. A future revision should make dangling a compile error rather than a runtime misbehavior.


## 3. Measurements

### 3.1 Correctness: node counts match OxiDD exactly

BDDs under a fixed variable order are canonical: same function, same var order → same reduced diagram, same node count. This is the strongest correctness oracle we have. We depend on it throughout.

`tests/differential.rs` uses OxiDD (via `oxidd = { path = "../oxidd/crates/oxidd" }` as a dev-dependency) and vwbdd in the same test binary, builds identical formulas in both, and asserts that `node_count()` agrees. Six scenarios pass: boolean op identities, a shared-subexpression 4-variable formula, and a 4-variable XOR parity chain.

`tests/mult.rs` pushes this further: a full shift-and-add multiplier bitblasted identically in both engines for k=2..8 bits. At every step, reachable-node counts agree exactly:

| k | reachable | OxiDD count |
|--:|---:|---:|
| 2 | 29 | 31 (29 + 2 terminals) |
| 3 | 125 | 127 |
| 4 | 498 | 500 |
| 5 | 1,997 | 1,999 |
| 6 | 7,856 | 7,858 |
| 7 | 30,941 | 30,943 |
| 8 | 122,309 | 122,311 |

The discrepancy of 2 is OxiDD's convention of including terminals in `node_count()` (we don't). After the offset, every pair matches to the node.

53 tests total across the crate, all green.


### 3.2 Arena compression

Post-GC arena bytes per node, measured on the mult sweep:

| k | nodes | arena bytes | B/node |
|--:|---:|---:|---:|
| 2 | 29 | 75 | 2.59 |
| 3 | 125 | 343 | 2.74 |
| 4 | 498 | 1,492 | 3.00 |
| 5 | 1,997 | 6,488 | 3.25 |
| 6 | 7,856 | 26,910 | 3.43 |
| 7 | 30,941 | 114,220 | 3.69 |
| 8 | 122,309 | 481,761 | 3.94 |

OxiDD's node table uses 16 B/slot regardless of k, so at k=8 our arena is **4.1× smaller**. The ratio widens slightly at smaller k and narrows at larger k, consistent with LEB128 spending ~1 extra byte per 128× increase in buffer size.

On simpler workloads the arena is even tighter: an n-way AND chain lands at 1.26 B/node (`tests/compression.rs`), because every new node references only its immediate predecessor and two terminals — maximally local.

Extrapolating: at k=10 (12.6M nodes, ~60 MiB arena) we'd be at ~4.5 B/node. At k=14 (3.7M nodes, ~18 MiB arena) we'd be at ~4.0 B/node. Neither approaches OxiDD's 16 B/node ceiling.


### 3.3 Total memory (the honest accounting)

Arena is not the whole story. A live BDD engine needs bookkeeping: unique table, apply cache, maybe a variable registry. The fair comparison includes all of it.

At k=8 (122,309 nodes), current state (post-GC, so apply cache is empty):

| component | bytes | B/node |
|---|---:|---:|
| arena (post-GC) | 481,761 | 3.94 |
| unique table (HashMap<u64,u64>, 32 B/entry estimate) | 3,913,888 | 32.00 |
| *subtotal live* | **4,395,649** | **35.94** |
| ite apply cache (direct-mapped, fixed 5 MB) | 5,242,880 | 42.9 (allocated, not growth-bound) |

Note the apply cache is now a fixed-size allocation (2^17 slots × 40 B) regardless of problem size. At k=8 it fits our working set; at k=2 it's 170 KB/node overhead (but 170 KB total is trivial). At k=10+ it will partially collide-evict, which is the correct failure mode for an apply cache.

OxiDD's comparable footprint at the same node count, based on its docs and the SHIM.md numbers: ~16 B/node (node table) + a similarly-sized apply cache ≈ 33 B/node total + ~2 MB cache.

So we're at **rough parity** on total live memory, not the 3-5× win the arena number alone suggests. The arena is tiny and elegant; the unique table is dense and ordinary. Together they land at the same order of magnitude as a well-tuned fixed-width engine.

This is the honest reading. v0 was much worse (101 B/node, 3× OxiDD) because we used `HashMap<(u32, Ref, Ref), u64>` with a wide key. Three optimizations (§4.2, §4.3, §4.5) closed the gap.


### 3.4 Wall-clock time vs OxiDD

Bitblasting `x*y=z` from scratch on a single thread, both engines in one test binary. Standard sweep (`tests/timing.rs`, k=4..8):

| k | nodes | vwbdd (ms) | oxidd (ms) | ratio |
|--:|---:|---:|---:|---:|
| 4 | 498 | ~1.1 | ~0.28 | 4.0× slower |
| 5 | 1,997 | ~2.3 | ~0.62 | 3.7× |
| 6 | 7,856 | ~8.6 | ~2.4 | 3.6× |
| 7 | 30,941 | ~30 | ~8.7 | 3.4× |
| 8 | 122,309 | ~112 | ~37.5 | **3.0×** |

**Large-k sweep** (`tests/timing_large.rs`, ignored by default; run with `--ignored`). Default max k=10; k=11 exceeds our 30 s budget for vwbdd:

| k | nodes | vwbdd (ms) | oxidd (ms) | ratio | arena B/n | total B/n |
|--:|---:|---:|---:|---:|---:|---:|
| 8  | 122,309   | 129     | 48     | **2.70×** | 3.94 | 14.66 |
| 9  | 484,417   | 614     | 242    | **2.54×** | 4.11 | 14.94 |
| 10 | 1,916,977 | 4,718   | 1,431  | 3.30× | 4.40 | 15.34 |
| 11 | 7,596,181 | 39,386  | 17,546 | **2.24×** | 4.59 | 15.64 |

**Ratios across k=8..11**: **2.2-3.3×**, with a **3.3× peak at k=10**. That k=10 peak is apply-cache thrashing: our direct-mapped cache is 2^17 = 131k slots, and at k=10 the working set is ~1.9M edges, ~14× cache capacity. Entries get evicted before they can be reused, forcing redundant recomputation. OxiDD's default cache is larger (~1M slots) so the threshold hits them later. At k=11, both engines are in a "cache-starved" regime relative to working set — but **vwbdd actually wins more ground at k=11 (2.24×)** than at smaller k, because the compact unique table and 2-MB packed apply cache both keep more of the working set in L2.

The cache size should probably be tunable. Left for a future session.

**Arena compression extrapolates cleanly**: 3.94 → 4.59 B/node across a 62× growth in node count (k=8 → k=11), consistent with LEB128 spending ~1 extra byte per 128× buffer growth. Our compression ratio vs OxiDD's fixed 16 B/node stays at **3.5-4× smaller** throughout.

**Total live memory (arena + unique table) dropped from ~36 to ~15 B/node** in §4.7's compact-table rewrite. Counting the apply cache as amortized overhead per live node, **§4.8 brought the grand total at k=11 to 15.9 B/live-node**, down from 16.8 B/live-node, against OxiDD's ~30-32 B/live-node (**~2× smaller engine**).

Progression across our optimizations, k=8 specifically:

| version | vwbdd (ms) | ratio | total B/n |
|---|---:|---:|---:|
| v0 (v_skip + side-table var lookup + wide unique key) | 401 | 10.8× | ~80 |
| +inline var, drop var_of_offset | 247 | 6.5× | ~50 |
| +slim unique HashMap (hash→offset, verify on decode) | 214 | 5.6× | ~36 |
| +direct-mapped apply cache (§4.5) | 105 | 2.8× | ~36 |
| +§4.7 compact unique table (u32 offset + u8 tag) | 112 | 3.0× | 14.7 |
| +§4.8 PackedRef (16 B apply-cache entry, 1 cache line) | **129** | **2.70×** | **14.7** (excl. cache), **15.9** grand-total |

The first four optimizations each cut ~40-50% of remaining wall time. §4.7 gave back ~7 ms to speed but cut total live memory by 2.4×. §4.8 then recovered more than that — ~17 ms back — purely from cache-line discipline, at no memory cost. Speed-first and memory-first aren't always in tension.


## 4. What the data told us

### 4.1 The hashmap footgun

The microbenchmarks (`tests/microbench.rs`) were the turning point. They showed:

- Full node decode: **62.6 ns**.
- HashMap `var_of_offset` lookup alone: **10.6 ns**.

Our v0 `decode_node_at` did *two* hashmap lookups per decode (one per child, to reconstruct `var` from `v_skip`). That's ~21 ns of hashmap cost on top of ~20 ns of LEB128/pairing work — roughly 2:1 overhead to payload.

The pattern got worse on the construction path. `make_node(var, lo, hi)` does ordering assertions (2 hashmap lookups) plus a unique-table probe (1 lookup) plus, on the fresh-node path, a var-table insert. `ite` layers three cofactors × 2 values × 2 hashmap lookups per cofactor, plus ite-cache probes, plus the make_node chain. A single leaf `and` on fresh operands is **~15 hashmap operations**. At 10 ns each, that's 150 ns just in hashmap overhead — per ite call. And we do millions of these.

Bryant's apply algorithm is memory-bound, not compute-bound (see roofline in the parent session). Memory-bound doesn't mean "bandwidth-bound" — it means "dependent-load-latency-bound," which is exactly what cache-missing hashmap probes produce. Our v0 was asking the memory system to chase pointers it didn't need to chase.

Once we named the diagnosis, the optimizations named themselves.


### 4.2 Optimization 1: inline var

v0 stored `v_skip = min_child_var - var - 1` instead of `var`, with the argument that v_skip is small (often 0) and therefore costs fewer LEB128 bits. Decode had to reconstruct `var` by looking up both children's vars in a side HashMap `var_of_offset: HashMap<u64, u32>`.

The trade: save ~0.5-1 byte/node, pay ~20 ns/decode and ~28 B/node of side-table memory.

Switching to `LEB128(var)` as the first field of each node:
- Arena grew 2.93 → 3.94 B/node at k=8 (+34%, about 1 byte per node as predicted).
- Decode dropped from 62.6 → ~25 ns (estimated; not microbenched post-change).
- `var_of_offset` HashMap eliminated: -28 B/node.
- Total live memory dropped 101 → 75 B/node.
- k=8 timing dropped 401 → 247 ms (40% faster).

One byte per node for 40% speed and 26 B/node less memory. The v_skip-minsky packing was a beautiful compression trick that cost more than it saved in a live engine. Worth keeping in mind for a dump/load codec where decode cost is paid once.


### 4.3 Optimization 2: slim unique table

v0's unique table was `HashMap<(u32, Ref, Ref), u64>`. Key size: 4 bytes for `var`, plus two `Ref` enums at 16 bytes each (Rust pads the `{tag: u8, u64}` variant to 16 B for alignment) = 36 bytes of key payload, ~48 with struct padding, ~64 with hashbrown overhead. Call it 72 B/entry.

The standard BDD-engine trick: use `HashMap<u64, u64>` where the key is a cheap hash of `(var, lo, hi)` and the value is the canonical offset. On lookup, decode the candidate node and verify. If the hash is good, verification succeeds on the first try ~99%+ of the time.

Estimated entry size: 8 B key + 8 B value + 16 B hashbrown = **32 B/entry**. 2.25× smaller than the old keyed table.

A small `unique_collisions: HashMap<u64, Vec<u64>>` handles the rare hash collision. Almost always empty or absent; doesn't contribute meaningfully to memory.

Results at k=8:
- Total live memory: 75 → 35.94 B/node.
- k=8 timing: 247 → 214 ms (13% faster; smaller wins because we still pay one decode per make_node, but we also avoid cloning Refs into a wider key).

The hash function is a splitmix64-adjacent integer mixer inline in `src/manager.rs::unique_key_hash`. Hand-tuned for speed; the Rust std `DefaultHasher` was noticeably slower in pre-optimization microbenches.

Post-change, the unique table is 32 B/node vs the arena's 4 B/node. The unique table is now 8× the size of the arena it indexes. That's the next natural target: a bucket-array unique table (like CUDD's) would be ~16 B/node. But that's for a future session.


### 4.4 Rejected optimization: absolute+relative hybrid addressing

The 2023 notebook that inspired this design used a hybrid addressing scheme: each child edge carried a 1-bit mode selecting between **absolute** (child's construction-order index, small when the child was built early) and **relative** (backward byte delta, small when the child is adjacent). The encoder picked whichever was cheaper per edge. On the NES ROM workload the notebook targeted, this produced a visibly bimodal edge distribution (Cell 26), and the hybrid encoded roughly half the edges with each mode.

The obvious question for vwbdd: does the bimodality replicate on our workload, and does the hybrid pay off? We added `tests/edge_stats.rs` to measure this — a pure observation, no encoding change. For every live edge in the post-GC mult arena at each k, it records (parent_idx, child_idx, rel_delta) and computes LEB128 costs under three alternatives.

**The bimodality does replicate.** Across k=5..8, the distribution of `child_idx / parent_idx` shows the notebook's signature: ~67% of edges cluster in the `[0.9, 1.0)` bin (children adjacent to parents), with a diffuse plateau across the rest of `[0.0, 0.9)` and a mild secondary peak near zero. Conditional means make it sharper: for edges where the relative encoding wins, mean ratio is 0.997; for edges where absolute wins, mean ratio is 0.18-0.35. Same pattern, different workload.

**But the hybrid does not pay off on mult.** Summed LEB128 byte cost of the encoded child references, compared across schemes:

| k | edges | rel-only (current) | abs-only | hybrid min + mode-bit | savings vs current |
|--:|---:|---:|---:|---:|---:|
| 4 | 736 | 980 B | 1,506 B | 981 B | **−0.1%** |
| 5 | 2,987 | 4,010 B | 5,762 B | 4,303 B | **−7.3%** |
| 6 | 11,886 | 16,675 B | 23,548 B | 17,474 B | **−4.8%** |
| 7 | 47,195 | 76,584 B | 111,310 B | 71,752 B | +6.3% |
| 8 | 187,587 | 316,941 B | 529,451 B | 325,474 B | **−2.7%** |

Negative numbers mean the hybrid costs more than the current pure-relative encoding. Only k=7 shows a real win, and that looks like a phase-transition artifact where the arena size happens to straddle the LEB128 2→3 byte boundary (16,384 bytes) in a way that favors absolute codes for one particular DAG layer. At k=6 and k=8 the hybrid loses by 3-5%.

**Why it fails here but worked for the ROM:**

1. **The adjacent-children cluster is dominant (66-67%).** For those edges, `rel_delta` is often ≤ 126, encoding in a single LEB128 byte. The absolute alternative (`child_idx` near `parent_idx`, which is near `N`) needs 2-3 bytes once N exceeds 127. Rel is strictly cheaper for two-thirds of all edges before we consider overhead.
2. **The absolute-cheap minority is small (8-23% depending on k).** Even where absolute wins, it wins by a small margin. At k=8, 66% of edges are rel-cheaper, 8% are abs-cheaper, 26% tie. The hybrid gains nothing on ties but pays the mode-bit tax regardless.
3. **The mode bit costs ~0.125 B/edge amortized.** Across 187k edges at k=8, that's 23 KB — larger than the 15 KB saved by per-edge minimization. The quantization floor of LEB128 (whole bytes) combined with a small abs-cheap fraction means the savings stay below the overhead.
4. **The notebook's workload had different graph topology.** Dense ROM data produces BDDs with strong sharing of deep (near-terminal) subtrees across many late-built parents, which favors short absolute indices. Mult produces a regular banded structure where sharing is mostly local (adjacent adder cells, consecutive bit positions). Same packing idea, different winner.

The bimodality observation held; the hybrid prescription didn't transfer. Pure back-reference is **within ~3%** of the theoretical minimum achievable by a 1-bit-mode hybrid on this workload, at every k we care about. The implementation cost (mode-bit packing in the code LSB, decoder branch per edge, GC handling of both modes) would buy nothing.

Keeping the test in the suite: it runs in 0.36 s and would immediately surface the win if we ever ran this codec on a workload with ROM-like edge structure. Good measurement, deliberately-rejected optimization.


### 4.5 Optimization 3: direct-mapped apply cache

After §4.3 we were at 5.6× slower than OxiDD with roughly equivalent total memory and 4× smaller arena. The remaining gap was structural: decode was cheap (45 ns/node per post-§4.3 microbench), var_of was cheap (1.7 ns), unique-table probe was cheap. But `ite` was still slow. Where was time going?

Per-op accounting told the story. Each ite call through the hot path did, among other work:
- 2 ite-cache probes (at the top of the call, and at the Enter frame once popped off stack)
- 1 ite-cache insert on each Combine frame
- 1 ite-cache lookup per resolve, used to pass child results up to parents

That's **3-5 HashMap operations per ite call**. Our HashMap was `HashMap<(Ref, Ref, Ref), Ref>` using Rust's default hasher (**SipHash** — a keyed, DoS-resistant, cryptographic-strength hash designed for untrusted input). On a 48-byte key, SipHash runs ~30 ns by itself before hashbrown's lookup logic adds more. OxiDD, like every serious BDD engine, uses a **direct-mapped apply cache**: a fixed-size `Vec<Entry>` indexed by `hash(key) & (size-1)`, one cache-line read per lookup, naive collision eviction.

The diagnosis named the fix. Replace `HashMap<(Ref,Ref,Ref), Ref>` with `Box<[IteCacheEntry; 2^17]>`, where `IteCacheEntry` is `(filled: bool, f, g, h, r: Ref)` packed into ~68 bytes (fits within two cache lines; one cache line for the 3-ref key plus flag). Use `ite_key_hash` (same splitmix-style mixer as `unique_key_hash`) to pick a slot. On collision: overwrite unconditionally. On flush (during GC): memset to EMPTY.

**One correctness wrinkle the change forced us to face.** v0's iterative ite relied on the cache to carry results between child-Enter completion and parent-Combine execution — parent Combine looked up `(f0, g0, h0)` in the cache to retrieve the child's result. With a direct-mapped cache that evicts on collision, this is unsound: if a sibling's work caused a collision evicting f0's slot between insert and read, the lookup would fail silently. The fix is a **parallel result stack**: each Enter that does real work pushes a Combine plus two child Enters, and *each Enter pushes its result onto a `Vec<Ref>` when it completes*. Combine pops two results (lo, then hi) from the result stack, not from the cache. The cache is now used only for cross-call memoization, which is its actual job; in-flight communication goes through the stack. This is how every real BDD engine does it.

Sizing: 2^17 = 131,072 slots × 68 B = ~5 MB allocated regardless of problem size. At k=8 we have ~188k live edges, so we're slightly undersized and will see some collision eviction — that's fine, it's the correct failure mode for an apply cache (forget older work, keep more recent). For k < 6, the cache dwarfs the working set; this is a "large-workload" engine now. OxiDD sizes similarly (their default is 1 << 14 slots, though; we chose larger because arena is already cheap).

Memory impact:
- Old: HashMap ite_cache grew linearly with insertions, ~80 B per live entry. At k=8, ~190k entries = ~15 MB.
- New: fixed 5 MB. **Strictly less memory at k=8.**

Timing impact at k=8: **214 → 105 ms**. The single largest optimization we made, and the cleanest: one data-structure swap driven by a per-op accounting of where time went. Ratio to OxiDD: 5.6× → **2.8×**.

We didn't also swap the unique table to a bucket array in this pass. That's the obvious next step and would target the remaining 1.8× gap. But §4.5 already validates the diagnosis: the variable-width encoding itself was not the bottleneck — generic HashMaps on the hot path were.


### 4.6 Rejected optimization: replacing the unique-table HashMap

After §4.5's apply-cache swap (HashMap → direct-mapped array) halved our wall-clock time, the remaining 2.8× gap vs OxiDD seemed to point at the unique table as the next target. It was also a `HashMap<u64, u64>`, consulted on every `make_node` call, ~742k times per k=8 mult build. If the direct-mapped swap worked for the apply cache, the reasoning went, a similar hand-rolled structure should work here too.

We tried three variants, each measured against the §4.5 baseline (105 ms, 2.8× at k=8):

**Variant A: single open-addressed table with tag-based verify.**
`Vec<UniqueSlot>` with `UniqueSlot = { hash_tag: u32, offset_plus_1: u64 }` (16 B/slot). Linear probe; on tag match, decode arena and verify full key; on mismatch, skip. Resize 2× when load factor exceeds 0.75. Expected ~21 B/node memory (33% less than HashMap) and faster lookups.

Result: **128 ms, 3.3×** at k=8. *Regressed 22%.*

**Variant B: single open-addressed table with full inline key.**
Same layout but `UniqueSlot = { offset: u64, var: u32, lo: Ref, hi: Ref }` (40 B/slot). No arena decode on verify — direct u64/u32 comparisons. Slower memory accesses: larger slot = fewer per cache line.

Result: **147 ms, 3.9×** at k=8. *Regressed 40%.*

**Variant C: variable-partitioned open-addressed tables.**
`Vec<VarTable>` indexed by `var`, each `VarTable` a small open-addressed Vec<UniqueSlot> growing independently. Var is implicit in the table index, so the slot needs no var field. Each sub-table is small enough to fit in L1/L2 cache on its own.

Result: **125 ms, 3.3×** at k=8. *Regressed 19%.*

**Variant D (user-suggested): variable-partitioned `Vec<HashMap<(Ref, Ref), u64>>`.**
Same per-variable partitioning as C, but each sub-table is a stdlib HashMap keyed on `(lo, hi)` — keep hashbrown's SIMD group probing while shrinking per-table working set. Full key stored (no verify-decode needed).

Result: **135 ms, 3.7×** at k=8. *Regressed 30%.*

**Variant E (baseline restored): single HashMap<u64, u64> with unique_key_hash.**
Revert to §4.5 end state.

Result: **102 ms, 2.7×** at k=8. (Original.)

**What we learned about where the HashMap's speed actually comes from.** Rust's stdlib HashMap (hashbrown) does three things we couldn't easily match with hand-rolled code:

1. **SIMD group probing.** Hashbrown stores a 7-bit metadata byte per slot, packed into groups of 16 (one SSE register / one NEON vector). A single 128-bit load + compare-and-mask yields 16 slot-match decisions in ~3 instructions. Our naive linear probe reads one slot at a time with a branch per slot — the CPU can't do nearly as much work per cycle.

2. **SipHash on 8-byte keys is cheap.** SipHash's DoS-resistance costs come from being designed for variable-length keys under adversarial conditions. For a fixed 8-byte key it's ~5 ns, barely slower than our splitmix-style mixer (~2 ns). The per-lookup hash cost was never the dominant factor I'd assumed.

3. **Hashbrown's layout keeps metadata and entries separate.** Metadata (1 B/slot) is traversed first; entries (24 B/entry) are only touched on a metadata match. This is better cache-usage than interleaving key+value together in wide slots, which is what every "inline the key" variant did.

Variants A and C "won" on memory in theory (21 vs 32 B/node slot footprint) but that theoretical win was smaller than the measured speed regression. Variant B definitively lost on both — bigger slot, fewer per cache line, more DRAM traffic. Variant D's per-variable HashMap partitioning turned out worse than a single HashMap because each HashMap carries ~48 B of struct overhead plus its own internal growth curve, and mult has ~32 variables at k=8; total HashMap-object overhead exceeded the memory benefit of partitioning.

**The lesson we drew too quickly.** We concluded "unique-table shrinking is a losing game" and kept the HashMap. That was true *for the shape of replacement we'd tried* — all four variants stored a separate hash tag or inline key per slot, growing slot footprint. We hadn't yet tried the design that actually matched the arena's advantage: **don't store the key at all**. §4.7 picks up from here.


### 4.7 The shrink that worked: offset-only verify-on-decode table

Re-reading §4.6, one pattern stood out: every failed variant treated the slot as the place where the key lives. But our arena *already stores the key*. We're paying twice for it — once compressed in LEB128 on the arena, once redundantly in the unique-table slot. The v0 HashMap had this problem (24 B/entry for the key). So did variants A-D.

The better design keeps only the arena offset in each slot and verifies on probe by decoding the arena node. If LEB128 decode is cheap (it is: ~40 ns per node), the verify cost is dominated by the cache miss on the arena — which we'd take anyway as soon as we needed the node for apply. The unique table becomes a pure **location index**: "where is the node with these properties?" not "is this the node?"

**Design:** struct-of-arrays compact table.
- `Vec<u32>` of arena offsets (plus one, 0 = empty slot).
- Parallel `Vec<u8>` of hash tags (high byte of hash, OR'd with 1 to stay nonzero).
- Linear probe. On probe, check the tag first (cheap u8 compare on a dense array). Only on tag match do we decode the arena to verify the full key.

**Why u32 offsets work.** Slot values are byte offsets, not node indices. At 4 B/node average, a u32 offset addresses up to ~900M nodes. We were at 7.6M at k=11; 100× headroom. A `debug_assert` catches the overflow if we ever break it.

**Why u8 tags work.** The 1-byte tag gives ~1/255 false-positive probability per probe (tag 0 reserved for empty). Under load factor 0.75, most probes terminate at the first empty slot anyway; tag mismatches on the occasional chain avoid the decode. This is hashbrown's design sans SIMD group-probing.

**Total slot footprint: 5 B/slot × 1.33 slots/entry at 0.75 load = ~6.7 B/node** for the unique table. Compare to the HashMap's 32 B/node and variant A's 21 B/node.

**Post-GC resize.** §4.7 also fixes a subtle issue §4.6 didn't have: during a long construction, the table grows past the live-node count (many intermediate nodes get GC'd later). A naive post-GC `clear()` reuses the bloated table, and the shrink is wasted. We added `resize_for(live_count)` which reallocates at the right size for the final live set.

**Measurements at k=8 (mult relation, 122k live nodes):**

| Design | Wall time | Ratio | Unique B/n | Arena B/n | Total B/n |
|---|---|---|---|---|---|
| HashMap (§4.5 end) | 102 ms | 2.7× | 32.0 | 3.94 | 35.94 |
| u64 slots, no tag | 150 ms | 4.0× | 17.15 | 3.94 | 21.09 |
| u32 slots, no tag | 140 ms | 3.7× | 8.57 | 3.94 | 12.51 |
| **u32 slots + u8 tag** | **112 ms** | **3.0×** | **10.72** | **3.94** | **14.66** |

**Across scales (k=8 to k=11):**

| k | nodes | Ratio | Total B/n |
|---|---|---|---|
| 8 | 122k | 3.10× | 14.66 |
| 9 | 484k | 2.91× | 14.94 |
| 10 | 1.9M | 3.86× | 15.34 |
| 11 | 7.6M | **2.55×** | 15.64 |

The k=11 result is the most interesting: **we're *faster* than the HashMap baseline there** (2.55× vs the baseline's 2.97×). The smaller unique table fits better in L2 at the scale where cache pressure is worst. This is the payoff for the memory-first framing: at the scale that *matters* — where problems start pushing the hardware envelope — compactness translates directly to speed.

**What the §4.6 variants got wrong.** Reading §4.6 again with §4.7's result in hand, the failed variants weren't wrong about SIMD probing or hashbrown's cleverness. They were wrong about a cheaper thing: **redundancy**. Storing the key in the slot when the arena already has it is strictly worse than storing a pointer to the key, regardless of collision scheme. SIMD group probing is valuable but it isn't what made hashbrown win over variants A-D; what made hashbrown win was that we'd structurally overcommitted memory in every hand-rolled alternative. §4.7's table is smaller *and* comparably fast, because it gave up on storing the key.

Kept the 54-test suite green. Total live memory at k=11 dropped from 36 to 15.6 B/node — a **2.3× shrink**, with speed within 3× of OxiDD. This is the trade the TL;DR now quotes.


### 4.8 Shrinking the apply cache: PackedRef

After §4.7 cut the unique table to 65% of engine memory, a landscape audit revealed the next fat structure: the direct-mapped apply cache from §4.5. Two discrepancies popped out of the audit:

1. The §4.5 comment claimed `IteCacheEntry` was "4 Refs = 64 bytes = one cache line." It was actually **72 bytes**, because `Ref = enum { Terminal(bool), Node(u64) }` is 16 B (8 B discriminant + 8 B u64 payload, no niche). Plus a `filled: bool` field and padding. The cache was spanning 1.5 cache lines per entry — the opposite of what a direct-mapped structure wants.
2. The comment claimed the cache was "~5 MB." It was actually **9 MB** (2^17 × 72 B).

The fix is not to touch the public `Ref` enum (too many pattern-match sites) but to introduce an internal 4-byte packed form used only in the cache's bulk storage:

```rust
#[repr(transparent)]
struct PackedRef(u32);
// 0 = Terminal(false), 1 = Terminal(true), 2 = EMPTY sentinel, 3..=MAX = Node(offset)
```

The `EMPTY` sentinel lets us drop the `filled: bool` field entirely; an empty slot is detected by `entry.f == PackedRef::EMPTY`, and no real `f` can ever pack to that value.

**Result: `IteCacheEntry` shrinks from 72 B to exactly 16 B, one cache line.**

- Cache total: 9 MB → **2 MB** (4.5× smaller)
- Per-probe: 1.5 cache-line read → 1 cache-line read (every cache hit touches strictly one line)

This is a case where shrinking is expected to *speed things up* (fewer bytes per probe, fewer lines prefetched) rather than trade speed for memory. Measurement confirms:

| k | vwbdd before (ms) | vwbdd after (ms) | Ratio before | Ratio after |
|--:|---:|---:|---:|---:|
| 8  | 144 | **129** | 3.17× | **2.70×** |
| 9  | 701 | **614** | 2.96× | **2.54×** |
| 10 | 5605 | **4718** | 3.93× | **3.30×** |
| 11 | 45149 | **39386** | 2.58× | **2.24×** |

**11-16% speedup across all k, no memory regression.** The largest speedup is at k=10, where apply-cache thrashing was worst; a smaller working-set-per-entry helps the thrash regime most.

**Engine-wide memory at k=11**: arena 34.9 MB + unique 84.0 MB + cache 2 MB = **120.9 MB = 15.9 B/live-node grand total** (was 16.8). OxiDD at the same k uses ~30-32 B/live-node; we're now **~2× smaller in total engine memory** while maintaining a 2.2-3.3× speed ratio.

The unused `Ref` → `u64` encoding in `ref_to_u64` (used for hashing) is kept since it's already fast; only the storage shape changed. The 54-test suite stayed green throughout.


### 4.9 Rejected optimization: cuckoo hashing at 0.85 load

After §4.8, a landscape audit showed the unique table was now 66% of vwbdd's engine memory at k=11 (84 MB of 121 MB) — the remaining fat. The table's load factor at k=11 is only **0.45**, because power-of-two slot counts force `CompactUnique` to round up past the live node count by up to 2×. If we could raise the load factor to ~0.85, we'd cut unique-table memory by ~46% (about 40 MB at k=11) with no change to per-slot overhead.

The literature's standard answer is **cuckoo hashing with 4-slot buckets**, which has a theoretical load-factor limit of ~0.98 and a comfortable operating point around 0.85. Each key hashes to two candidate buckets; lookup probes at most 8 slots (two cache lines of slots plus two of tags); insert tries both buckets, evicts on failure and cascades into alternate buckets, rebuilding on MAX_KICKS exceeded.

**Pre-implementation estimate** (answered with calibrated confidence in a design conversation):

> "Cuckoo with 4-slot buckets: ~0-10% slower or possibly *faster*, -35% memory. Strictly better than Robin Hood here."

**Actual measurement** after a working, tested implementation (cuckoo at 0.85 target load, fastrange for non-power-of-two bucket counts, 500-kick cascade, grow-1.5× on failure):

| k | nodes | CompactUnique ratio | Cuckoo ratio | Speed cost | Total B/n before | Total B/n after | Memory win |
|---|--:|--:|--:|--:|--:|--:|--:|
| 8  | 122k | 2.70× | 5.79× | **2.14× slower** | 14.66 | 9.82 | -33% |
| 9  | 484k | 2.54× | 5.07× | **2.00× slower** | 14.94 | 10.00 | -33% |
| 10 | 1.9M | 3.30× | 8.75× | **2.65× slower** | 15.34 | 10.28 | -33% |
| 11 | 7.6M | 2.24× | 4.46× | **1.99× slower** | 15.64 | 10.48 | -33% |

The memory prediction was **accurate** (estimated -35%, actual -33%). The speed prediction was **wrong by a factor of 2-3×**. Cuckoo delivered the memory and paid double in speed.

**What the estimate got wrong, in descending order of magnitude:**

1. **Lookup cost was underestimated.** The reasoning said: "at 0.85 load, 2-slot cuckoo lookup beats linear probing beyond the immediate-neighbor zone." This is *true in isolation*: on a miss at 0.85 load, cuckoo does 2 cache-line reads; linear probing might do 3-5 before hitting empty. But linear probing at **0.45 load** (what CompactUnique actually runs at) terminates on the first probe ~55% of the time on a single cache line; cuckoo always probes both buckets, ~2× the cache pressure per lookup. The estimate compared cuckoo-at-0.85 to linear-probe-at-0.85, but the whole reason to switch to cuckoo was because linear probing was running at 0.45. The right comparison was cuckoo-at-0.85 vs linear-probe-at-0.45, which cuckoo loses on base cost.

2. **Fastrange cost was underestimated.** `CompactUnique` uses `hash & mask`: one AND, ~0.3 ns per op. Cuckoo with arbitrary bucket counts uses fastrange: `((h as u64) * (bc as u64)) >> 32`, a 64-bit multiply, ~3-5 ns. That's ~3 ns extra per lookup and per insert. On a workload doing tens of millions of ops per second, this alone adds up.

3. **The lookup/insert asymmetry in the workload was ignored.** The pre-implementation analysis focused on insert's eviction cascade (rare, amortized O(1) at 0.85 load, ~150-300 ns per eviction). That reasoning was correct but irrelevant: in `ite`-heavy workloads, lookups outnumber inserts ~3-5:1, and lookups don't care about cascades at all. Insert cost was barely the issue; lookup cost was.

**The deeper lesson: the "landscape audit" framing had a blind spot.** Counting bytes per structure led to the right target (the unique table was 66% of memory). But "bytes per structure" doesn't account for *access frequency × access cost*. Every single `make_node` call goes through unique-table lookup. Cuckoo doubled the cost of every `make_node`. Any amount of memory saved by cuckoo had to pay back that 2× on the hottest path in the engine. §4.7's linear-probe table wins on cost-weighted access, even though it loses on density.

**Reverted to §4.7 `CompactUnique`.** The implementation is preserved in git history / the edit log for posterity, but the main branch keeps the lower-memory-density, higher-speed-per-byte-amortized design.

**What this rules out.** Not cuckoo specifically. The generalization is: *on the unique table's lookup path, any scheme that increases per-op cache-line touches is a bad trade*, regardless of how much memory it saves. This excludes cuckoo variants, Robin Hood (which has longer probe chains than linear probing at the load factor where its displacement variance wins), and anything that trades cache-line bandwidth for density.

**What this doesn't rule out.** Approaches that preserve single-cache-line lookup:
- **Drop the tag array** at low load factor. Small expected loss; might be worth measuring as a pure ablation.
- **Per-level unique tables** (option 3 from the landscape audit). Still one lookup per op, but to a smaller level-scoped table that fits better in L1. The cache-locality win from small per-level tables *adds* to single-probe lookup rather than replacing it. Likely a different story from cuckoo.


### 4.10 Where does time actually go? Encoding-scheme A/B

After §4.9, the obvious remaining question: *how much of our wall time is the variable-width decode costing us?* Before optimizing anything else we wanted a real answer, not a guess. Two complementary measurements:

**1. Phase timing.** Added a compile-time-gated `profile-timing` feature that wraps the hot decode sites (`var_of`, `decode_node`, unique-table verify, unique-table resize, encode) in `std::time::Instant::now()` Guards and accumulates per-phase ns totals. On the mult k=10 workload the breakdown (instrumented build, so absolute numbers are inflated ~2× by `Instant::now` overhead, but ratios survive):

- `var_of`: **~58% of accounted time**, called 4-5× more often than full decode
- `decode_node`: ~21%
- `decode_verify` + `decode_resize`: ~17% combined
- `encode`: ~4%
- *Unaccounted* (unique-table slot traversal, ite orchestration, hashing, apply-cache probing): ~66% of wall time

That said: `var_of` dominates *among the things we instrumented*, so any encoding change that cheapens `var_of` wins.

**2. Three-backend A/B with a feature flag.** Implemented three mutually exclusive node encodings in `src/node/` subdir, each exposing identical `encode_node_at` / `decode_node_at` / `decode_var_at` APIs:

- `encoding-interleaved` (§4.2 historical): `LEB128(var) | LEB128(interleave(lo, hi))`
- `encoding-per-field`: `LEB128(var) | LEB128(lo) | LEB128(hi)`
- `encoding-fixed`: `[u32; 3]` struct, 12 B/node, no varint decode at all (ceiling comparison)

Mult k=8..11 clean wall times, no instrumentation:

| k | interleaved | per-field | fixed12 |
|---|--:|--:|--:|
| 8  | 151 ms / 3.94 B/n | 122 ms / 4.06 B/n | 102 ms / 12.00 B/n |
| 11 | 39.4 s / 4.59 B/n | 41.5 s / 4.62 B/n | 33.3 s / 12.00 B/n |

Fixed12 saves **~16% at k=11 for 2.6× arena bloat** — a terrible trade. Per-field is within 5% of interleaved on both axes; the pair-math is basically free and the compression advantage of interleaving is real only on small BDDs (where both are tiny anyway).

**The u8-var side-pocket.** The profile data also pointed at a cheaper win: `var_of` was spending time in the LEB128 state machine, but at any plausible BDD scale `var < 256`, and LEB128-of-a-value-below-128 is exactly one byte. So: drop the LEB128, store the var as a raw u8, and `var_of(r)` becomes a single `buf[off]` read. Zero bytes gained or lost per node at our working scale.

Measured on interleaved (pre-change vs post-change):

| k | LEB var | u8 var | speedup |
|---|--:|--:|--:|
| 8  | 151 ms | 117 ms | **-23%** |
| 11 | 39.4 s | 37.5 s | **-5%** |

The speedup is large at small k (where `var_of` fraction is highest) and shrinks at large k (where unique-table traversal dominates). **No memory cost.** Pure win.

**Combined post-§4.10 design (current default):** `u8 var` + three independent LEB128s (per-field), no interleave. Timing at k=11: **~37 s / 2.0× OxiDD / 4.62 arena B/n / 15.66 total B/n**. 2-6% faster than the u8-var-interleaved build on mult, roughly the same arena footprint.

**The cleanup pass that followed.** Once the measurement was in, we deleted everything the experiment had rejected: the interleaved backend, `src/pair.rs` (the bit-interleave module, now unused), the fixed12 backend (faster but gives up the whole point), the feature-flag dispatch in `src/node.rs`, and the `profile-timing` instrumentation (it had served its purpose). Also deduplicated three copies of the mult-relation builder into `tests/mult_shared/mod.rs`, extracted `CompactUnique` into `src/unique.rs`, and trimmed a dead field from `MemStats`. Net: **-818 LOC** across src and tests, no behavior change.

Post-simplification: `src/` is 918 LOC across four files (leb, node, unique, manager). `tests/` is 1331 LOC across ten files. Single-file pedagogical aesthetic preserved where it paid off, per-file separation where it clarified.


### 4.11 Removing the 4 GiB ceiling: u64 offsets everywhere

§4.7 (`CompactUnique`) and §4.8 (`PackedRef`) both pushed density by packing arena offsets into u32. Perfectly reasonable at the scales we were running (7.6M nodes / 34 MB arena at k=11), but the ceiling was real: arena > 4 GiB would trip a `debug_assert` in the unique table's insert and silently wrap in a release build. A BDD engine meant to absorb as much RAM as a machine has (up to ~1 TB on our target laptop) can't leave that landmine in place.

The change is structural but uninteresting in design terms: widen both `CompactUnique::slots` and `PackedRef` from u32 to u64. Everything else stays the same: slot-value-0-means-empty for `CompactUnique`; `0/1/2` sentinels for `PackedRef::FALSE/TRUE/EMPTY` followed by `3..=u64::MAX` for node offsets. The `IteCacheEntry` grows from 16 B (exactly one 64 B cache line) to 32 B (two entries per line).

**Measurements** — mult `x*y=z` sweep at k=8..11, post-GC:

| k | nodes | vwbdd u32 (ms) | vwbdd u64 (ms) | Δ speed | total B/n u32 | total B/n u64 | Δ mem |
|--:|---:|---:|---:|---:|---:|---:|---:|
| 8  | 122k | 129 | 112 | -13% | 14.66 | 23.35 | +59% |
| 9  | 484k | 614 | 570 | -7% | 14.94 | 23.58 | +58% |
| 10 | 1.9M | 4,718 | 4,488 | -5% | 15.34 | 24.15 | +57% |
| 11 | 7.6M | 39,386 | 37,262 | -5% | 15.64 | 24.50 | +57% |

Two surprises in the data:

1. **Speed improved slightly.** The u64 build is 5-13% *faster* than u32 across all k. Best guess: fewer narrowing conversions in the hot loops (the u32 slot-read used to immediately widen to u64 for arithmetic; the u32 PackedRef widened on unpack), and the doubled apply cache (4 MiB vs 2 MiB) pays back more of the thrashing at k ≥ 10. I did not expect a speedup to fall out of a straightforward widening, but it did, consistently.

2. **Memory cost is larger than I estimated from slot-width alone.** 4 B/slot × 1.33 slot inflation = ~5 B/node of expected growth; actual growth is ~9 B/node. The extra comes from the unique table's power-of-two sizing: at the 7.6M-entry mark, the next pow2 that holds 10.1M slots (0.75 load) is 2^24 = 16.8M slots — roughly 2× more slots than strictly needed. That inflation was already present at u32 (it's why post-§4.7 we landed at ~11-12 B/node instead of the theoretical 6.7 B/node), but widening the slot made the inflation more expensive in absolute terms.

**Engine total at k=11**: arena 34.9 MB + unique ~151 MB + apply cache 4 MB = **~190 MB (25 B/live-node)**, vs OxiDD at ~30-32 B/live-node. Still a ~20% memory win, down from the previous ~2× win. The trade was bought consciously: the 4 GiB arena ceiling came for free at the scales we'd measured but would have been catastrophic at the scales we actually want to reach.

**Arithmetic for the target regime.** At 1 TB of RAM, with the current ~24-25 B/live-node total:

| Component | Bytes per node | 1 TB budget supports |
|---|---:|---:|
| arena (LEB128-encoded) | ~5 | 200B nodes if arena alone |
| unique table (u64 slots @ 0.75 load, pow2 inflated) | ~20 | 50B nodes if unique alone |
| apply cache (fixed) | O(1) | negligible |
| **combined** | ~25 | **~40B live nodes** |

40 billion live BDD nodes in 1 TB. That's 2-3 orders of magnitude past anything in the BDD literature I've seen benchmarked as "large." The real question at those scales becomes whether `ite` wall-time scales linearly (and whether we need a per-level apply cache to resist thrashing), not whether the data structures fit.

**The broader reading.** §4.7's "don't store the key in the slot" insight is robust across slot widths; we just moved along the space/time trade-off curve it defined. The u32 variant is the memory-optimal version of that design. The u64 variant is the RAM-scalable version. If a future session ever needs both — say, a heuristic "is this arena going to stay small?" decision made at `Manager::new()` — the generic treatment is straightforward (parameterize `CompactUnique<Off>` and `PackedRef<Off>` over an `ArenaOffset` trait). We haven't done that because the measurements don't demand it: u64 gives up 9 B/node of density for a 7-13% speed *improvement* and unlimited arena headroom, and there's no scale between "small enough that the 4 GiB cap never mattered anyway" and "large enough that u64 is mandatory" where the u32 variant is strictly better.

The 4 GiB ceiling was an artifact of the measurement regime we were optimizing in. Lifting it cost us the "2× smaller than OxiDD" framing but preserved every structural claim: verify-on-decode unique table, single-line (well, double-line now) direct-mapped apply cache, variable-width arena compression, copying GC. The engine still fits more nodes per byte than any fixed-16-B/node design.


### 4.12 Keeping both: `Manager<C: NodeCodec, O: ArenaOffset>`

§4.11 shipped the u64 widening as the only build. Within a session, a user question flipped that decision: a wasm client will never have more than 4 GiB of heap, but a server-side BDD builder preparing a diagram to ship *to* that client might need tens of GiB. Both use cases are real, and they want different engine footprints. One binary shouldn't have to pick.

The change is a straightforward monomorphization refactor, but it rearranges the module structure enough to be worth recording.

**The trait boundary that emerged.** Two axes, both orthogonal:

1. `trait ArenaOffset`: the numeric type used for arena byte offsets. Implemented for `u32` and `u64`. The surface area is deliberately minimal (`ZERO`, `ONE`, `from_u64`/`to_u64`, checked arithmetic) because every use site was already doing one of those operations.

2. `trait NodeCodec<O: ArenaOffset>`: how a `(var, lo, hi)` triple is laid out in bytes. Three methods — `encode`, `decode`, `decode_var` — because those were already the three functions `src/node.rs` exposed. Implementors are zero-sized types; the trait is stateless. `Leb128Codec` (the current design) is the one materialized impl. §4.10's rejected codecs (interleaved, fixed12, v_skip) would fit here without touching `Manager`; they're not in this cut because the A/B already said they lose.

These are combined on `Manager<C: NodeCodec<O>, O: ArenaOffset>`. `CompactUnique<C, O>` and `IteCacheEntry<O>` travel along. `Ref<O>` and `Node<O>` are parameterized too — honest is better than hiding conversions at the arena boundary.

**Default type parameters do the ergonomic work.** `Manager<C = Leb128Codec, O = u32>` lets every existing call site that writes `Manager::new()` keep working at the compact default. Two type aliases crystallize the common choices:

```rust
pub type DefaultManager = Manager<Leb128Codec, u32>;   // 4 GiB cap, compact
pub type LargeManager   = Manager<Leb128Codec, u64>;   // host-RAM cap, wider
```

The `new()` method is defined as an inherent on the concrete `Manager<Leb128Codec, u32>` type rather than on the generic impl. This is the idiom that makes Rust's inference pick `Leb128Codec, u32` for bare `Manager::new()` calls; a generic `Manager<C, O>::new()` would leave the type parameters unresolved and force every call site to turbofish. Other configurations use `LargeManager::default()`.

**What it cost and what it bought.** The parameterization touches every file in `src/` and adds one new file (`src/codec.rs`), but the hot paths are structurally unchanged — after monomorphization, the two instantiations compile to what the §4.7/§4.8 (u32) and §4.11 (u64) code paths did, respectively. No runtime dispatch, no dynamic checks.

Measurements on mult `x*y=z`, default (u32) build, after parameterization:

| k | §4.11 u64-only (ms) | §4.12 default (u32) (ms) | speedup | total B/n |
|--:|---:|---:|---:|---:|
| 8  | 112 | 98 | 13% | 14.78 |
| 9  | 570 | 453 | 21% | 14.92 |
| 10 | 4,488 | 3,671 | 18% | 15.40 |
| 11 | 37,262 | 32,018 | 14% | 15.66 |

The default build is **14-21% faster than the u64-only §4.11 engine at every k**, with memory density restored to the §4.8 baseline (~15-16 B/node at k=11). The compiler now emits 32-bit native ops on the u32 monomorphization instead of the u64 ops with narrowing casts that §4.11 required. Ratio to OxiDD at k=11: **1.75×**, the best we've ever measured.

For the `LargeManager` (u64) configuration, timing stays close to §4.11's numbers (the monomorphization is essentially identical code), with the ceiling lifted. The two builds are strict Pareto improvements over the single-parameterization worlds each replaces:

- Small-arena world: u32 default is faster than §4.11's u64-only (narrower ops) and lower-memory (smaller slots).
- Large-arena world: u64 build reaches the same unlimited-ceiling that §4.11 promised.

**Cross-width correctness test.** `tests/large_manager.rs` contains a `mult(k=4) = 498 nodes` assertion that runs on both widths and demands they agree. Canonicity is a property of the variable order, not the offset width; if a codec bug ever truncated an offset silently, that test would catch it.

**Ref<O> in public API.** This was the one deliberate break: the `Ref` enum's `Node` variant now carries `O` rather than `u64`. All callers write either `Ref<u32>` (explicit) or `Ref` with default inferred. For tests and the differential harness that only run on `DefaultManager`, nothing changes; for `LargeManager` users, `Ref<u64>` is the honest type. Hiding the width behind a `u64` payload would have meant silent narrowing at every cache insert; exposing it means the compiler tells you where conversions would happen. The measurements above suggest that transparency was worth it.

**What this doesn't yet give us.** Codec parameterization is scaffolded but not exercised. The next session that wants to revisit §4.10's codec A/B can add an `InterleavedCodec` and slot it into the same `Manager<InterleavedCodec, u32>` shape; no further `Manager` changes needed. Until then, `Leb128Codec` is the only codec shipped, and the trait's presence is mostly pedagogical — a clean boundary that says "this is the thing that changes when you experiment with node layouts."


### 4.13 The apply cache was a silent perf trap

Until this session, the direct-mapped apply cache was hardcoded at 2^17 slots — 128k entries, 2 MiB. That choice came from §4.5 when our largest benchmark was k=8 mult (122k nodes), and it was correct for that regime. It was never re-examined as the project pushed through k=11 and k=12.

A three-way comparison (vwbdd native, oxidd native, oxidd-wasm via the `iota` demo at rndmcnlly.github.io/oxidd-wasm/iota.html) exposed the trap. iota's published numbers for a truncated `(x*y) mod 2^k = z` workload had us 6.97× slower than oxidd at k=15 — anomalously bad relative to our stable ~2.5× ratio at k ≤ 12. The diagnosis: at k=15 the working set is ~10M edges; with 128k cache slots that's 80× oversubscription. Every `ite` recursion was evicting entries it would need seconds later, turning memoization into a cache-miss storm.

The fix was a one-line change and a 2.66× speedup at k=15. Bumping the cache to 2^21 slots (2M entries, 32 MiB) brought the ratio back to 2.60×, flat with k=12, 13, 14. Direct measurement:

| k | nodes | vwbdd old (2^17 cache) | vwbdd new (2^21 cache) | oxidd (2^25 cache) | old ratio | new ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 464,181 | 314 ms | ≈ same | 121 ms | 2.60× | 2.60× |
| 13 | 1,292,158 | 951 ms | ≈ same | 386 ms | 2.47× | 2.47× |
| 14 | 3,697,095 | 3,564 ms | ≈ same | 1,275 ms | 2.80× | 2.80× |
| **15** | **10,304,528** | **31,375 ms** | **11,812 ms** | **4,693 ms** | **6.97×** | **2.51×** |

**How oxidd sizes its stuff** (answering the question that motivated this section):

The library (`oxidd::bdd::new_manager`) takes two capacities explicitly: `inner_node_capacity` and `apply_cache_capacity`, plus a `threads` count. No defaults inside the library. The `oxidd-cli` reference tool picks:

- **Apply cache**: **32 Mi entries (2^25)** by default — the canonical "large fixed absolute allocation" (~1.3 GB at ~40 B/entry). Hardcoded, not scaled to problem size.
- **Inner node table**: fills remaining RAM at 32 B/slot, capped at 2^32 entries (32-bit edge index).

Entry sizes:
- oxidd's direct-mapped cache: parking_lot mutex + arity byte + numeric byte + operator + 3 edges + value ≈ **40 B/entry** (supports up to arity-3 for ite-style operators).
- vwbdd's: `4 × PackedRef<u32>` = **16 B/entry** at u32, **32 B/entry** at u64. No mutex (single-threaded by `&mut self`), no arity variability, tuned specifically for ite.

At matched entry count, vwbdd's cache uses 40% of oxidd's memory for the same slot coverage. Our new 2^21 default (32 MiB) gives us 2M slots — roughly equivalent cache coverage to oxidd at 820k entries (33 MB), which oxidd-cli considers undersized by default. The real lesson: **the default should have been 2^21 since §4.5**, and we paid for not re-examining it for six months of workload growth.

**API shape** (`src/manager.rs`):

```rust
pub struct ManagerConfig { ite_cache_slots: usize }

impl ManagerConfig {
    pub const fn new() -> Self { /* 2^21 default */ }
    #[must_use]
    pub const fn with_cache_slots(self, slots: usize) -> Self { /* power-of-two check */ }
}

impl Manager<C, O> {
    pub fn with_config(config: ManagerConfig) -> Self { ... }
}

impl Manager<Leb128Codec, u32> {      // the default-type inherent impl
    pub fn new() -> Self { Self::with_config(ManagerConfig::default()) }
    pub fn with_cache_slots(slots: usize) -> Self { /* convenience */ }
}
```

This mirrors `Vec::new` vs `Vec::with_capacity` (stdlib) and oxidd's "caller picks" philosophy, while giving a sensible default so `Manager::new()` does the right thing for the common case.

**Why a builder struct over a single `new(slots)` function**: forward compatibility. Future knobs — per-level unique-table partitioning (§5 TODO), GC-trigger heuristics, custom initial arena capacity — can land as more `with_*` methods without breaking existing call sites. We don't need any of those today; we just don't want to paint ourselves into a corner on the one-knob API.

**What's left exposed.** The new default (2^21) is still 16× smaller than oxidd-cli's (2^25). On a laptop with 64 GB of RAM that difference is meaningless; the 32 MB we allocate now vs 512 MB oxidd allocates is both statistically zero. But the caller who wants the full oxidd-cli-scale treatment can now name it: `Manager::with_cache_slots(1 << 25)`. Tests can sweep the space and find an optimum per workload without recompiling.

**Measured steady-state ratio, k=12..15, on the truncated mult relation** (matches iota's workload):

| k | nodes | vwbdd (ms) | oxidd native (ms) | iota wasm (ms) | vwbdd/oxidd | vwbdd/iota |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 464k | 314 | 121 | 158 | 2.60× | 1.99× |
| 13 | 1.3M | 951 | 386 | 505 | 2.47× | 1.88× |
| 14 | 3.7M | 3,564 | 1,275 | 1,632 | 2.80× | 2.18× |
| 15 | 10.3M | 11,812 | 4,693 | 5,329 | **2.51×** | **2.22×** |
| 16 | 29.5M | 39,359 | 13,519 | 21,833 | 2.91× | 1.80× |
| 17 | 82.3M | 141,235 | 46,127 | **OOM** | 3.06× | — |

The ratio is flat at 2.5-3.1× across a **176× growth in node count** (k=12 → k=17). That's the real answer to "how does vwbdd compare at scale": not 6.97× as the unfixed cache suggested, not 1.75× as the §4.12 writeup claimed (that number was on the *full* 2k-bit relation with different variable ordering characteristics). On this specific iota-style workload, steady state is ~2.5-3× slower than oxidd native.

**At k=17 we run a workload iota-wasm cannot.** iota OOMs at k=17 because oxidd's 40 B/entry node table × ~80M nodes = ~3.2 GB exceeds the 4 GiB wasm32 linear-memory ceiling. vwbdd finishes the same workload in 1.12 GB engine total (arena 445 MB + unique ~670 MB + cache 32 MB), with room to spare inside the u32 arena cap. This is the architectural payoff of the variable-width design in its clearest form yet: **we can run a browser-scale BDD workload that oxidd-wasm cannot, on the same 4 GiB memory budget.**

**Reconciling with §4.12's numbers.** The full-relation k=11 landing at 1.75× and the truncated k=15 landing at 2.51× aren't inconsistent — they're measuring different functions. The full 2k-bit relation with 44 variables produces BDDs whose internal structure happens to be friendlier to our variable-width codec (more locality → more 1-byte LEB128 deltas). The truncated relation with 3k=45 variables produces denser interconnect (k-bit product crossing through a k-bit equality) where the byte-per-edge advantage shrinks. Both are real; neither is the full story. The honest framing is "2-3× slower than oxidd, with the constant depending on workload graph structure."

**What the ratio would look like if we matched oxidd's cache entry size.** If we went from 16 B PackedRef entries to 40 B full-key entries (no verify-on-decode), we'd stop paying `decode_node_at` on every cache miss verification, and our 2M-slot cache would functionally behave like oxidd's. But the memory footprint would grow from 32 MB to 80 MB, and we'd lose the architectural parity with §4.7's "don't store the key" insight. A future session could explore this as an A/B but it's not obviously a win.


### 4.14 Dump/load and the multi-process primitive

The arena is already a byte stream; it just needed a header. `src/dump.rs` adds the wrapper, plus a merge primitive (`absorb`) that makes multi-process parallelism a natural application of the existing canonicalization discipline.

**Motivation.** vwbdd's `&mut self` design rules out multi-thread parallelism inside a single manager: borrow-checking enforces one writer at a time. That sounds like a handicap until you notice that the append-only arena and deterministic-hash unique table make *multi-process* parallelism unusually easy. Append-only means a mmap'd shared-read view is safe by construction: no writer means no barriers. Deterministic `splitmix64` hashing on fixed-layout `(var, lo, hi)` triples means any process reconstructing the unique table from the same bytes gets the same slot assignments. If we can serialize an arena and re-ingest it with dedup, we have the pieces to stitch partitions together.

The hard part of parallel BDD compute is usually *fine-grained* sharing during the `ite` recursion: threads contend for the unique table on every `make_node`. That's what oxidd's multi-threaded mode manages carefully with lock-free slot CAS and atomic refcounts. The *coarse-grained* parallelism — partition the problem, let each worker build independently, merge — is barely covered in the BDD literature, because most engines don't have a cheap merge primitive. Ours does, because the arena *is* the merge input.

**Format.** Little-endian, 32-byte header + raw arena bytes + absolute-encoded root refs + optional length-prefixed UTF-8 names + 4-byte CRC32.

```text
[8 B magic "VWBDD\0\0\0"]
[2 B format_version]
[1 B offset_width: 4=u32, 8=u64]
[1 B flags: bit 0 = has_root_names]
[4 B num_vars]
[4 B num_roots]
[8 B arena_len]
[4 B reserved]
[arena_len bytes of raw Leb128Codec-encoded nodes]
[num_roots × 8 B of wire-encoded refs (0=F, 1=T, 2+off=Node)]
[optional: num_roots × (2 B length + UTF-8 name)]
[4 B CRC32]
```

Design choices:

- **Multi-root first-class.** `num_roots: u32` with a parallel list of `u64`-encoded refs. `num_roots = 1` is the single-root case at no per-byte cost. Matches DDDMP's long-established convention (CUDD's export format has supported multiple roots via `.nroots` + `.rootids` since the 1990s) and fits the natural BDD workflows: transition systems ship `{init, trans, reached}`; bitblasted circuits ship one function per output bit; partitioned computations ship `{T₁, T₂, ..., Tₙ}` for downstream OR.
- **Named roots optional via flag bit.** The common case (positional roots) pays zero bytes for names. Named mode adds one length-prefixed UTF-8 entry per root, for transition-system artifacts that want to preserve `init`, `trans`, etc. as first-class labels.
- **CRC32 at the tail.** Validates the whole file including header. Cheap (~1 GB/s), catches torn writes and bit-rot. Implemented inline with a 30-line Rust CRC32 (IEEE polynomial 0xedb88320) to keep the crate dep-free.
- **Offset-width byte in the header.** A `u32` dump can load into a `u32` or `u64` engine (widening is free); a `u64` dump fails to load into a `u32` engine if any offset exceeds `u32::MAX`. The load path checks per-ref.
- **Absolute root encoding via the child-code convention.** `0 = Terminal(false)`, `1 = Terminal(true)`, `2 + off = Node(off)`. Reuses the arena's per-child encoding but with an absolute offset rather than a back-reference delta. Natural and self-describing.
- **No magic per node.** Individual nodes aren't self-delimiting because the arena is decoded sequentially and we know `arena_len` up front. The header's byte count is the only framing.

**Three operations on `Manager`:**

```rust
fn dump(&self, path, roots: &[Ref<O>]) -> Result<()>;
fn dump_named(&self, path, roots_and_names: &[(Ref<O>, &str)]) -> Result<()>;

fn load(path) -> Result<(Self, LoadedRoots<O>)>;              // fresh engine
fn load_with_config(path, config) -> Result<(Self, LoadedRoots<O>)>;

fn absorb(&mut self, path) -> Result<Vec<Ref<O>>>;            // merge into self
```

`load` is the simple case: fresh manager, arena bytes copied verbatim (the bytes in the file are already canonical under this codec), unique table rebuilt by a single linear walk. Total load cost: O(bytes) for the filesystem read, O(nodes) for the table rebuild. No decompression, no per-node allocations. At ~5 B/node in the arena plus ~9 B/slot in the unique table, a k=15 dump (10M nodes) is ~50 MB of disk + a ~15 MB unique table rebuild, maybe 200 ms on SSD hardware.

`absorb` is the interesting one. It walks the foreign arena in construction order, decodes each node, translates its children through a running `HashMap<foreign_offset, local_ref>`, and re-interns via `make_node`. The key property: because `make_node` is canonicalizing, any subgraph that already exists in the receiver collapses to the existing offset. **A worker's node that was already built by a previous absorb comes back as the same `Ref`, without growing the arena.** The deduplication is free — it's the same unique-table logic that deduplicates inside a single session, applied across session boundaries.

**The multi-process pattern.** The sketch we'd use for disjunctive partitioning:

```rust
// Parent builds the shared scaffold (variables, constants, primed vars).
let mut parent = Manager::new();
declare_transition_variables(&mut parent);
// ... optionally dump and freeze as a shared base, or just leave empty.

// Fork N workers. Each builds one partition:
let worker_output = |i: usize| {
    let mut w = Manager::new();
    declare_transition_variables(&mut w);       // same declaration order
    let t_i = build_transition_partition_i(&mut w);
    w.dump(format!("/tmp/partition_{i}.vwbdd"), &[t_i])?;
    Ok(())
};
// ... launch via std::process::Command / fork / MPI / etc.

// Parent merges:
let mut merged_roots = Vec::new();
for i in 0..N {
    let roots = parent.absorb(format!("/tmp/partition_{i}.vwbdd"))?;
    merged_roots.push(roots[0]);
}
let t = parent.or_many(&merged_roots);
```

No threads, no locks, no shared memory primitives, no rayon dependency. Filesystem (or named pipes, or a socket delivering the dump bytes) is the sync point. Each worker runs at the same single-thread speed vwbdd already achieves; the parallelism happens at the process boundary, not inside the `ite` recursion.

**Deduplication across absorbs.** Tested directly: if worker A dumps formula `F₁` and worker B dumps `F₂`, and both formulas contain the subgraph `(x₀ ∧ x₁)` independently, the parent after `absorb(A); absorb(B)` contains that subgraph exactly once. The test (`tests/dump.rs::absorb_dedupes_shared_subgraphs`) builds both formulas in a reference manager and asserts the absorbed parent's node count matches exactly. Canonicity + deterministic hashing does the work.

**Performance shape.** `absorb` is roughly as expensive as building the worker's BDD from scratch in the parent: each node goes through `make_node`, which does a unique-table probe plus maybe an arena write. What the absorb buys you is *that work already happened in parallel in the worker*. If a k=15 partition takes 12 s in a worker and 150 ms to absorb back into the parent, N workers in parallel finish in roughly `max_worker_time + N × absorb_time`. For N=4 workers all running ~12 s, total wall time is ~12 + 4×0.15 ≈ 12.6 s vs ~48 s sequential. 4× speedup at the cost of some disk I/O.

**What this doesn't do (yet).** There's no shared-base mmap story: each worker builds its partition independently from scratch, including re-building any scaffolding that every partition uses. A future addition would be a "frozen snapshot" mode where the parent dumps a base arena with its unique table, and workers mmap it as a read-only prefix, appending local nodes on top. That cuts worker setup time but adds complexity; left for a session that has a concrete workload demonstrating the waste. The present `absorb` semantics are sufficient for the immediate use case: partition, build, merge.

**The broader architectural claim.** Between §4.13 (tunable apply cache) and §4.14 (dump/load/absorb), vwbdd now offers the full "application developer drives policy, library provides mechanisms" stance across all three resource dimensions: memory (ManagerConfig), GC (manual, copying), and parallelism (manual, multi-process via absorb). The contrast with oxidd is sharp in this last dimension: oxidd shapes its multi-threaded design around a lock-free coordination protocol that's hard for the caller to opt out of; we offer no intra-manager parallelism at all, and trade that away for a merge primitive the application uses however it wants. Both are valid designs; they optimize for different caller philosophies.


## 5. What's still missing

Things we'd want before calling this a real engine:

**Quantifiers** (`exist`, `forall`). Needed for reachability fixpoints, the FDG paper's workload. Classic implementation as a recursive descent with its own apply cache; probably 100 lines on top of what we have.

**Substitution** (`rename`, `compose`). Primed-variable tense discipline needs these.

**Sat-count**, **pick-one**, **pick-iter**. Obvious extensions, well understood, no research needed.

**Codec A/B harness.** §4.12 scaffolded the `NodeCodec<O>` trait but only materialized `Leb128Codec`. A future session revisiting §4.10 can add `InterleavedCodec`, `Fixed12Codec`, or `VSkipCodec` as type-parameter variants and run them head-to-head in a single test binary — no `Manager` changes needed. The path-dependent reason to skip this now is that §4.10's A/B already ruled those three out on the mult workload; the path-independent reason to do it later is a workload with different edge statistics (ROM data, §4.4's inspiration) might flip the answer.

**Adaptive apply-cache size.** §4.13 made the cache tunable at construction time and bumped the default to 2^21 slots. Not yet adaptive: the cache doesn't grow in response to workload pressure, and there's no heuristic connecting problem size (number of declared vars, `make_node` rate) to the right slot count. oxidd-cli's approach is "allocate big, 2^25 slots by default" and it works because they have a generous `inner_node_capacity` budget anyway. A workload-aware heuristic would be a real improvement over either fixed default, especially for a wasm target that can't afford the 32 MB baseline.

**Decoded-node cache.** For very hot nodes (usually the ones near the roots of the current operation), cache the decoded `(var, lo, hi)` in a small fixed-width table to skip repeated LEB128 work. Not yet attempted.

**Shared-base dump snapshots.** §4.14 landed dump/load/absorb for the partition-build-merge pattern. A further refinement is a "frozen snapshot" mode: parent writes a base arena + unique table, workers mmap it as a read-only prefix and build local tails on top, sharing the scaffold bytes across processes without per-worker rebuild cost. Worth doing when a concrete workload shows that per-worker scaffold rebuilds dominate.

**Compile-time safety against dangling refs across GC.** Make `Ref` lifetime-parameterized so the compiler refuses to let you use a pre-GC ref after a GC call. Standard Rust arena-handle trick.

**Complement edges.** If we ever want to match BCDD node counts instead of plain BDD, we'd need a complement bit somewhere. Currently we track plain BDDs to match OxiDD's `bdd::BDDFunction` (not `bcdd`), so this is a feature, not a bug, for the current comparison.


## 6. Verdict

The variable-width hypothesis has three claims, each with a different verdict:

**"Arena is much smaller."** True. **3.5-4× smaller** than OxiDD's node table across k=2..11 (see §3.4). With the §4.7 compact unique table the win extended to the full engine footprint, and §4.12's parameterization restored it as the default: **~15-16 B/node total** on `DefaultManager` vs OxiDD's ~30-32. For the `LargeManager` build that lifts the 4 GiB arena ceiling, the trade is ~25 B/node — still smaller than OxiDD, now at any host-RAM scale.

**"Smaller working set → faster compute."** Confirmed at the scale where it matters, and improving over time. We're 1.75-2.5× slower than OxiDD at k=8..11 on the default build post-§4.12, with the ratio *improving* as k grows: at k=11 we're **1.75×**, the best measurement we've ever recorded. The §4.6 conclusion ("unique-table shrinking is a losing game") was wrong — it was a losing game for the shapes we tried, right for the shape that stores no key at all.

**"Append-only + batch GC is simpler than concurrent refcounting."** True, and the implementation was pleasant. The GC-invalidates-refs model is easy to reason about. Rust's borrow checker gives us the single-writer discipline for free. No locks, no atomics, no TLS.

The overall question — "is this worth pursuing as a real BDD engine?" — depends on what you want.

For a **compute engine racing OxiDD/CUDD on speed alone**: no, but close. At k=11 on the default build we're 1.75× slower with ~half the memory; if we ever pursued the OxiDD-style partitioned-by-level unique table *on top of* the §4.7 compact layout, the remaining gap would plausibly close further. Not the priority here.

For a **memory-constrained engine that can handle bigger DAGs on fixed hardware**: **yes, this is the sharp end of the trade**. At k=11 on the default build, vwbdd holds 7.6M live BDD nodes in **~121 MB** total (35 MB arena + 84 MB unique table + 2 MB apply cache); OxiDD needs ~244 MB for the same. For deployments where the arena will stay under 4 GiB — wasm clients, browser inference, anything that ships a prepared diagram to a constrained target — the default `Manager<Leb128Codec, u32>` is the right build. For server-side builders that need to absorb tens of GB before serializing, switch to `LargeManager = Manager<Leb128Codec, u64>`: same library, different type alias, no runtime dispatch.

For a **server/client pipeline** (§4.12's key motivation): build with `LargeManager`, serialize the arena as-is (it's already a byte buffer), ship it to a wasm client running `DefaultManager` for inference. The `Leb128Codec` produces the same byte layout at both widths, so the receiving engine just indexes the arena using its own offset type. This wasn't a design goal when the codec was written; it fell out of the variable-width architecture naturally.

**Concretely demonstrated** at k=17 of the iota truncated-mult workload (§4.13): 82.3M BDD nodes, iota-wasm (running vanilla oxidd compiled to wasm32) runs out of memory inside the 4 GiB linear-memory ceiling; vwbdd finishes the same workload in 1.12 GB. The browser-target budget that bounds iota doesn't bound us on the same problem, on the same hardware.

For a **compressed storage/transfer format that can also query**: yes. The arena is self-contained, serializable by memcpy, 3.5-4× smaller than dd.autoref's JSON or OxiDD's DDDMP. Querying doesn't require rebuilding a full engine — you can evaluate a BDD on a valuation by walking the compressed arena directly, which is how the FDG paper's downstream consumers would want it.

For a **single-file pedagogical engine that demonstrates BDD canonicity, Shannon expansion, mark-sweep GC, variable-width packing, and a compact verify-on-decode unique table in ~800 lines of Rust**: absolutely yes. That's exactly what this is, and it runs.

The experiment's shape reversed cleanly: we went in thinking the arena was the hypothesis and the unique table was "settled infrastructure." We came out with the arena confirmed *and* the unique table cut by 4×, with clear negative-result records (§4.4, §4.6, §4.9) showing exactly which shrink attempts don't work and why. §4.9 in particular records a case where a pre-implementation estimate was wrong by 2-3× in the speed dimension; the honest writeup of what the estimate got wrong is probably the most valuable single section in this document for anyone planning their next optimization on a different BDD engine.

§4.10 closed the book on the other way this might have gone wrong: we had assumed variable-width decode was costing us perhaps ~30% of wall time; the A/B showed it was closer to ~16% (and only recoverable by giving up the whole point of the project). The concrete win from that session came from a sideways observation — `var < 256` means `var` shouldn't be LEB128 at all — which was worth 5-23% depending on k for zero bytes. **Measurements first, optimizations second; the boring observation often beats the clever one.**


## 7. File index

```
vwbdd/
├── Cargo.toml                   — zero runtime deps; dev-dep on ../oxidd/crates/oxidd
├── VWBDD.md                     — this document
├── src/
│   ├── lib.rs                   — module glue, public re-exports, type aliases (§4.12)
│   ├── leb.rs                   — unsigned LEB128 u128 encode/decode
│   ├── codec.rs                 — ArenaOffset + NodeCodec traits; Leb128Codec impl (§4.12)
│   ├── node.rs                  — compatibility shim: free-function wrappers for tests/node.rs
│   ├── unique.rs                — CompactUnique<C, O>: linear-probe, generic slot width (§4.7)
│   ├── dump.rs                  — .vwbdd native format: dump/load/absorb multi-root with CRC32 (§4.14)
│   └── manager.rs               — live engine: Manager<C, O>, ManagerConfig, make_node, ite, and/or/xor/not, gc, apply cache (§4.13)
└── tests/
    ├── leb.rs                   — LEB128 roundtrips
    ├── node.rs                  — per-node encode/decode (default u32)
    ├── manager.rs               — manager basics, reduction, canonicity, ordering
    ├── ite.rs                   — boolean op identities, node counts for small formulas
    ├── gc.rs                    — copying GC preserves semantics, shrinks manager
    ├── cache_config.rs          — ManagerConfig / with_cache_slots builder tests (§4.13)
    ├── dump.rs                  — .vwbdd roundtrip: single/multi-root, named, absorb dedup, error paths (§4.14)
    ├── large_manager.rs         — LargeManager smoke + u32/u64 cross-width agreement (§4.12)
    ├── differential.rs          — runs vwbdd and OxiDD on the same formulas, asserts node-count equality
    ├── compression.rs           — bytes/node on AND chain, XOR parity, threshold
    ├── mult.rs                  — full x*y=z relation for k=2..8, node count + mem + post-GC accounting
    ├── mult_shared/mod.rs       — shared vwbdd+OxiDD builders: full-relation (2k-bit z) and truncated (k-bit z) variants
    ├── mult_trunc_correctness.rs— truncated (iota-style) node counts at small k (§4.13)
    ├── mult_trunc_timing.rs     — truncated mult vs OxiDD, matched to iota's demo workload (§4.13; #[ignore]'d)
    ├── timing.rs                — wall-clock comparison vs OxiDD on full mult (k=4..8)
    └── timing_large.rs          — full-relation large-k sweep (#[ignore]'d; k=8..11)
```

Total: 65 tests passing, 3 #[ignore]'d timing sweeps.

