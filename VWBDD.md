# VWBDD: a variable-width BDD engine experiment

A design note. How and why we built a live BDD engine where nodes are stored in an append-only byte buffer using LEB128-based variable-width encoding, with children referenced by backward byte offsets.

Audience: the author, future collaborators, and anyone curious whether variable-width node layouts can win on modern hardware. This document records the shape of the decisions we made and the measurements that justify or refute them.

---

## TL;DR

We built a single-file Rust BDD engine where each node is encoded as two LEB128 varints (`var`, then an interleave of backward-offset child references) into an append-only byte buffer. Canonicity is enforced by a unique table keyed on a cheap hash of `(var, lo, hi)`, with collisions resolved by decoding and verifying.

Across the mult-relation `x*y=z` from k=2 to k=8 (reachable nodes 29 to 122k):

- **Correctness**: node counts match OxiDD exactly at every scale, verified differentially via a shared bitblaster running both engines in the same test.
- **Arena compression**: post-GC, we store BDD nodes at **2.6 to 4.6 bytes each** (k=2 to k=11) vs OxiDD's fixed 16 B/node. Dominant factor is the byte-offset distance to child nodes, which LEB128 compresses well.
- **Unique-table compression** (§4.7): struct-of-arrays compact table — `Vec<u32>` offsets + parallel `Vec<u8>` hash tags, 5 B/slot at 0.75 load. Exploits the observation that node offsets always fit in 32 bits (arena < 4 GB at any practical k). **~6.7 B/node**, down from 32 B/node for hashbrown.
- **Apply-cache compression** (§4.8): a 4-byte `PackedRef` lets each direct-mapped cache entry fit in 16 B (exactly one cache line), down from 72 B. Cache total shrinks from 9 MB to **2 MB**, and every probe reads exactly one cache line.
- **Total live memory**: **~15 B/node** (arena + unique table), plus a 2 MB fixed apply cache. Grand total at k=11: **~15.9 B/live-node**. **Half of OxiDD's ~30-32 B/live-node.**
- **Wall-clock time**: **~2.7× slower than OxiDD** at k=8, **2.2× at k=11**. Down from 10.8× in v0.

**Design trade-off, accepted explicitly**: within ~3× of OxiDD on runtime in exchange for **~2× smaller total engine footprint**. For the intended workload — compressed BDDs for FDG reachability on fixed laptop hardware, where working-set size determines whether a problem fits at all — this is the right trade. At k=11, vwbdd holds 7.6M nodes in ~121 MB total; OxiDD needs ~244 MB. A problem that OOMs in OxiDD on a 256 MB budget still runs here.

§4.6 documents four earlier unique-table shrink attempts that *regressed speed without winning memory* (hand-rolled open addressing variants that stored a separate hash tag or full inline key per slot). §4.7 shows what finally worked: **don't store the key at all** (verify-on-decode from the arena), and exploit that offsets fit in u32. §4.8 repeats that "exploit the u32 bound" pattern on the apply cache: pack Refs into u32 so each cache entry is one cache line.


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

Each node is two LEB128 varints in sequence:

```
LEB128(var)                                    // u32, level number (0 = top)
LEB128(interleave(lo_code, hi_code))           // u128
```

Child reference encoding (u64 before interleaving):

```
0                 -> false terminal
1                 -> true terminal
2 + delta         -> node at (current_offset - delta), delta >= 0
```

The interleave function (`src/pair.rs`) is a pure integer bit-interleave: `x` goes in even bits of the u128, `y` in odd bits. Inverse is bit-gather. Both encode and decode are straight-line integer math, no floating point or branches.

**Why two LEB128s, not one?** An earlier version packed `v_skip` (the var's distance above children's min-var) together with the interleaved children using Minsky pairing, so a single LEB128 covered everything. That saved ~1 byte per node but required the decode path to look up children's vars from a side HashMap — three times per node, at ~10 ns per lookup. The decode cost dominated the savings. Inline-`var` trades 1 byte/node for ~30 ns/decode. Easy call for a live engine; a storage codec would keep v_skip.

**Why interleave instead of concatenate?** Concatenation would require storing the bit-width of one child so decode knows where to split. Interleave is self-framing: the two LEB128 halves are extracted by bit-gather, no length prefix needed. And small child deltas (the common case: children are a few bytes back in the buffer) produce small interleaved values, which LEB128 encodes in 1-2 bytes.

**Why u128?** Two u64 child codes interleave into 128 bits. LEB128 of a small u128 is the same size as LEB128 of a small u64 — we only pay for the bits we use — so u128 costs nothing for small nodes and removes the arithmetic ceiling entirely (a u64 interleave maxed out at a 4 GiB buffer, which real workloads can exceed).


### 2.2 Append-only arena + unique table + apply cache

The manager (`src/manager.rs`) owns three pieces:

1. `buf: Vec<u8>` — the arena. Nodes are appended as they're built, never mutated.
2. `unique: HashMap<u64, u64>` — primary unique-table slot. Key is a hash of `(var, lo, hi)`; value is the canonical offset. On lookup, we decode the candidate and verify.
3. `unique_collisions: HashMap<u64, Vec<u64>>` — secondary for the rare hash collision.
4. `ite_cache: HashMap<(Ref, Ref, Ref), Ref>` — apply-cache for `ite`.

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


## 5. What's still missing

Things we'd want before calling this a real engine:

**Quantifiers** (`exist`, `forall`). Needed for reachability fixpoints, the FDG paper's workload. Classic implementation as a recursive descent with its own apply cache; probably 100 lines on top of what we have.

**Substitution** (`rename`, `compose`). Primed-variable tense discipline needs these.

**Sat-count**, **pick-one**, **pick-iter**. Obvious extensions, well understood, no research needed.

**Tunable / adaptive apply-cache size.** The fixed 2^17-slot direct-mapped cache (§4.5) is undersized at k ≥ 10 on mult (see §3.4's k=10 ratio peak). With §4.8's 16 B/entry, we could now cheaply double or quadruple the slot count without meaningful memory impact (4 MB or 8 MB total). `Manager::new_with_cache_size(n)` plus a sensible growth heuristic would flip the k=10 regime. Low risk, high payoff at large k.

**Decoded-node cache.** For very hot nodes (usually the ones near the roots of the current operation), cache the decoded `(var, lo, hi)` in a small fixed-width table to skip repeated LEB128 work. This is the hybrid architecture I mentioned earlier. Not yet attempted.

**(Resolved — see §4.7):** the unique-table HashMap is gone. Compact struct-of-arrays table (u32 offsets + u8 hash tags) achieves ~6.7 B/node and stays within 3× of OxiDD on speed. §4.6 documents four *unsuccessful* variants that came first; the §4.7 design is what finally worked.

**Dump/load.** The arena is already a serialized form; `dump` is `write(&self.buf)` plus metadata. `load` is `read` + rebuild unique table. Could be ~30 lines.

**Compile-time safety against dangling refs across GC.** Make `Ref` lifetime-parameterized so the compiler refuses to let you use a pre-GC ref after a GC call. Standard Rust arena-handle trick.

**Complement edges.** If we ever want to match BCDD node counts instead of plain BDD, we'd need a complement bit somewhere. Currently we track plain BDDs to match OxiDD's `bdd::BDDFunction` (not `bcdd`), so this is a feature, not a bug, for the current comparison.


## 6. Verdict

The variable-width hypothesis has three claims, each with a different verdict:

**"Arena is much smaller."** True. **3.5-4× smaller** than OxiDD's node table across k=2..11 (see §3.4). With the §4.7 compact unique table, the win now extends to the full engine footprint, not just the arena: **~15 B/node total** vs OxiDD's ~32, a 2.1-2.4× shrink sustained across all tested scales.

**"Smaller working set → faster compute."** Confirmed at the scale where it matters. We're 2.5-3.1× slower than OxiDD at k=8..11, but the ratio *improves* as k grows: at k=11 we're 2.55×, actually **faster** than the pre-§4.7 HashMap baseline (2.97×) at the same k. The smaller unique table fits L2 better at scale, and that cache-residency win shows up cleanly in the timing. The §4.6 conclusion ("unique-table shrinking is a losing game") was wrong — it was a losing game for the shapes we tried, right for the shape that stores no key at all.

**"Append-only + batch GC is simpler than concurrent refcounting."** True, and the implementation was pleasant. The GC-invalidates-refs model is easy to reason about. Rust's borrow checker gives us the single-writer discipline for free. No locks, no atomics, no TLS.

The overall question — "is this worth pursuing as a real BDD engine?" — depends on what you want.

For a **compute engine racing OxiDD/CUDD on speed alone**: no, but closer than earlier sessions suggested. At k=11 we're 2.55× slower with 2× less memory; if we ever pursued the OxiDD-style partitioned-by-level unique table *on top of* the §4.7 compact layout, the remaining gap would plausibly close further. Not the priority here.

For a **memory-constrained engine that can handle bigger DAGs on fixed hardware**: **yes, this is the sharp end of the trade**, and §4.7 doubled down on it. At k=11, vwbdd holds 7.6M live BDD nodes in **~119 MB** total (34 MB arena + 65 MB unique table + 5 MB apply cache + small overhead); OxiDD needs ~244 MB for the same. On a 256 MB memory budget, vwbdd can fit a 2× larger problem before OOM. For FDG reachability sweeps where the working set is the binding constraint, that's meaningful headroom.

For a **compressed storage/transfer format that can also query**: yes. The arena is self-contained, serializable by memcpy, 3.5-4× smaller than dd.autoref's JSON or OxiDD's DDDMP. Querying doesn't require rebuilding a full engine — you can evaluate a BDD on a valuation by walking the compressed arena directly, which is how the FDG paper's downstream consumers would want it.

For a **single-file pedagogical engine that demonstrates BDD canonicity, Shannon expansion, mark-sweep GC, variable-width packing, and a compact verify-on-decode unique table in ~800 lines of Rust**: absolutely yes. That's exactly what this is, and it runs.

The experiment's shape reversed cleanly: we went in thinking the arena was the hypothesis and the unique table was "settled infrastructure." We came out with the arena confirmed *and* the unique table cut by 4×, with clear negative-result records (§4.4, §4.6, §4.9) showing exactly which shrink attempts don't work and why. §4.9 in particular records a case where a pre-implementation estimate was wrong by 2-3× in the speed dimension; the honest writeup of what the estimate got wrong is probably the most valuable single section in this document for anyone planning their next optimization on a different BDD engine.


## 7. File index

```
vwbdd/
├── Cargo.toml                   — dev-dep on ../oxidd/crates/oxidd
├── VWBDD.md                     — this document
├── src/
│   ├── lib.rs                   — module glue, public re-exports
│   ├── leb.rs                   — unsigned LEB128 u128 encode/decode
│   ├── pair.rs                  — interleave (u64, u64) → u128, Minsky pair (unused in current node.rs but kept for reference)
│   ├── node.rs                  — single-node encode/decode: LEB128(var) + LEB128(interleave(lo,hi))
│   └── manager.rs               — live engine: make_node, ite, and/or/xor/not, gc
└── tests/
    ├── leb.rs                   — LEB128 roundtrips (5 tests)
    ├── pair.rs                  — interleave/Minsky roundtrips (7 tests)
    ├── node.rs                  — per-node encode/decode (3 tests)
    ├── manager.rs               — manager basics, reduction, canonicity, ordering (8 tests)
    ├── ite.rs                   — boolean op identities, node counts for small formulas (12 tests)
    ├── gc.rs                    — copying GC preserves semantics, shrinks manager (5 tests)
    ├── differential.rs          — shares tests across vwbdd and OxiDD, asserts node counts match (6 tests)
    ├── compression.rs           — bytes/node on AND chain, XOR parity, threshold (3 tests)
    ├── mult.rs                  — x*y=z relation for k=2..8, node count + mem + post-GC accounting (1 test, internal sweep)
    ├── mult_shared/mod.rs       — shared mult-relation builder, reused by mult.rs and edge_stats.rs
    ├── microbench.rs            — decode cost, construction cost (2 tests)
    ├── edge_stats.rs            — per-edge abs/rel/hybrid byte-cost measurement on mult k=2..8 (1 test, §4.4)
    ├── timing.rs                — wall-clock comparison vs OxiDD on mult (1 test, k=4..8)
    └── timing_large.rs          — large-k sweep, vwbdd vs OxiDD up to 30s budget (1 #[ignore]'d test, k=8..10+)
```

Total: 54 tests, all green, full suite runs in ~3 s release.

