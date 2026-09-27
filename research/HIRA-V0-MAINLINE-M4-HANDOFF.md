# HIRA V0 MAINLINE M4 handoff — runtime / latency / RAM / packaging

Status: **M4-A READY / M4-B BOUNDED CACHE IMPLEMENTED PRE-STRESS**

Issue: #167  
Branch: `feat/hira-v0-mainline-m4-runtime`  
Base main: `29eaa8b4eb687efb18155930aebfe5f14dd3c851`

## 1. Mainline state entering M4

Frozen scientific maturity:
- typed runtime: available
- state-once: available
- high-K mechanics: available
- high-K semantic quality: provisional
- reliability/OOD/abstention: provisional/fail-closed
- multilingual: provisional
- production-ready: false

M4 is a deployment/runtime phase. It must not silently promote semantic maturity.

## 2. M4-A bottleneck

Exact M2 K=255 integration measured:
- schema compile: ~1482.21 ms
- decision core: ~2.66 ms
- schema tensors: ~4.48 MB

The decision engine is not the dominant runtime cost.

Code inspection found an O(K) encoder-call pattern inside `SchemaCompiler.compile()`:

Before M4-A:
1. encode question
2. for every option, encode criterion/aliases/exemplars separately
3. if token artifacts requested, encode question+criteria
4. encode all positive multi-view texts again

At K=255 this can require hundreds of semantic encoder forward calls.

## 3. M4-A optimization

Implemented in:
- `src/nmd/schema.py`

New compile path:

Without token artifacts:
1. question batch
2. one flattened positive-view batch

Total cold encoder calls:
- **2**

With token artifacts:
1. question batch
2. one flattened positive-view batch
3. question+criteria token-artifact batch

Total cold encoder calls:
- **3**

The positive-view batch is reused for:
- pooled logical option embeddings
- option multi-view token artifacts

No second positive-view encode is performed.

## 4. Semantic contract preserved

Unchanged:
- schema hash
- cache key
- option ordering
- option IDs as non-semantic routing keys
- question semantics
- positive prototype semantics
- criterion/alias/exemplar view set
- prototype aggregation formula:
  - normalize each prototype
  - average per option
  - normalize the mean
- full-K runtime
- relation refinement OFF
- adaptive budget OFF

No model/scorer parameter changes.

## 5. M4 manifest

Added:
- version `0.0-m4a`
- `HiraV0Manifest.m4_runtime_provisional()`
- `build_hira_v0_m4_runtime()`

The M4 manifest changes phase/version only.

It preserves:
- reliability provisional
- high-K semantic provisional
- multilingual provisional
- production_ready false

## 6. Unit authority

Implemented:
- `tests/test_schema_batching.py`
- `tests/test_mainline_m4_runtime.py`

Contracts:
- optimized option embeddings match legacy per-option aggregation
- K255 cold compile with token artifacts uses exactly 3 encoder calls
- K255 cold compile without token artifacts uses exactly 2 encoder calls
- warm schema cache adds zero encoder calls
- token artifacts remain present and correctly packed
- downstream probabilities remain stable
- M4 maturity does not overclaim readiness

## 7. Immediate next boundary

Before exact A13 performance claims:
1. dedicated M4 CI must pass;
2. full repo Python 3.10/3.12 CI must pass;
3. run exact A13/W28/W34 integration benchmark;
4. compare M4 K-ladder timings with frozen M2 mechanics receipt;
5. record call-count and latency improvement;
6. only then proceed to M4-B bounded cache/RAM discipline.

No performance improvement is claimed before the exact integration receipt freezes.


## 8. M4-A exact A13 integration — authoritative result

Run:

`36327904288`

Empirical marker head:

`144858eff949a00aec9da086a593b89d9ade301a`

Artifact:
- `hira-v0-mainline-m4-schema-benchmark`
- ID: `10934506746`
- digest: `sha256:447c394a5a6b2246a354b21d49e9ee848c2271a636eedd019ba2ae23496324eb`

Outcome:

`HIRA_V0_M4_SCHEMA_BATCHING_READY`

Every frozen gate passed:
- semantic tensor equivalence
- mechanics preservation
- cold encoder calls <= 3
- warm encoder calls = 0
- cold cache miss
- warm cache hit
- K255 optimized cold compile faster than same-runner legacy

### K=255 exact result

Legacy same-runner:
- semantic compile: **1392.831263 ms**
- encoder calls: **258**

M4-A:
- cold schema compile: **304.870836 ms**
- cold encoder calls: **3**
- warm schema compile: **0.812787 ms**
- warm encoder calls: **0**
- decision: **2.562596 ms**
- same-runner cold speedup: **4.5686x**

Historical M2:
- schema compile: 1482.206577 ms
- decision: 2.655830 ms

Semantic/mechanics integrity:
- option embedding max error: 1.3783574104309082e-07
- question embedding error: 0
- criterion token error: 0
- view token error: 0
- selected option: `m2-route-044`
- historical selected option: `m2-route-044`
- probability mass error: 5.960464477539063e-08
- relation delta: 0
- full-K: true
- finite: true

Schema tensor bytes:
- **4,477,609 bytes**

Benchmark process RSS peak delta across the K ladder:
- **19,492 KiB**

M4-A is therefore accepted as a runtime optimization.

It does not promote:
- semantic quality;
- reliability;
- multilingual;
- production readiness.

## 9. M4-B bounded schema cache

The original schema cache was an unbounded Python dict.

That is unsafe for long-running local inference because one K=255 token-artifact schema is approximately 4.48 MB in tensor residency.

Implemented M4-B policy:

Default bounds:
- max entries: **16**
- max tensor bytes: **64 MiB**

Eviction:
- true LRU
- a cache hit moves the entry to MRU
- insertion evicts oldest entries until both limits are satisfied
- an individual entry larger than the byte budget remains valid for the current query but is not retained

Cache receipt now reports:
- cache stored
- entry tensor bytes
- resident entries
- resident tensor bytes
- cumulative evictions

Runtime/mainline controls:
- inspect cache info
- clear schema cache
- reconfigure entry/byte limits
- M4 builder accepts deployment-specific cache limits

Changing cache limits never changes:
- schema hash
- semantic compilation
- selected option
- probability calculation
- maturity claims

## 10. M4-B frozen stress boundary

Before claiming RAM discipline, run a cache-stress authority that:
- uses exact A13/W28/W34;
- repeatedly compiles distinct dynamic schemas;
- includes multiple K=255 schemas;
- exceeds both the entry working set and byte working set;
- verifies resident entries <= 16;
- verifies resident tensor bytes <= 64 MiB;
- verifies evictions occur;
- verifies recently-hit entries survive LRU pressure;
- verifies an evicted schema recompiles to semantically/mechanically equivalent output;
- verifies state-once and full-K remain unchanged.

M4-B is runtime-only.

No model parameter, threshold, semantic authority, or scientific maturity may change.


## 11. M4-B stress v1 — harness failure, cache bounds validated

Authoritative run:

`36328918665`

Artifact:
- `hira-v0-mainline-m4-cache-stress`
- ID: `10934986918`
- digest: `sha256:e2624ce369a17c068cf8bb78fffb70615d86581a630dc49d31c3ae9d3cfc5ed4`

Outcome:

`HIRA_V0_M4_BOUNDED_CACHE_FAIL`

Observed after pressure:
- resident entries: **12**
- resident tensor bytes: **63,260,364**
- max entries: 16
- max bytes: 67,108,864
- evictions: **9**

After evicted-schema recompile:
- resident entries: 12
- resident bytes: 62,466,276
- evictions: 11

Passed gates:
- entry bound
- byte bound
- eviction observed
- early schema evicted
- early semantic equivalence
- early probability equivalence
- early selected option match
- full-K
- state-once
- relation delta zero

Exact recompile probability max error:
- **0.0**

Failed gate:
- `anchor_survived_lru`

Reason:

The v1 harness touched the anchor after 8 unique schemas and then inserted 13 additional distinct K=255 schemas.

The preregistration implicitly assumed the 64 MiB byte budget would retain at least 14 K=255 entries based on the M4-A 4,477,609-byte schema.

In this stress workload, longer question text increases padded token-artifact tensor width. The actual resident working set was only 12 entries under 64 MiB. Therefore 13 newer entries necessarily evicted the touched anchor even with correct LRU ordering.

This is a **stress-authority design error**, not evidence of incorrect eviction:
- byte accounting bounded correctly;
- entry accounting bounded correctly;
- eviction happened;
- evicted schema reproduced identical decision probabilities;
- no semantic/runtime invariant regressed.

M4-B v1 remains immutable failed evidence.

A single M4-B2 mechanical witness is authorized with:
- the same cache implementation;
- the same 16-entry / 64 MiB limits;
- no code change to LRU policy;
- a touch point chosen from an entry known to be resident immediately before bounded additional pressure;
- explicit proof that a hit moves that entry to MRU while an older resident entry is evicted.

M4-B2 must not alter cache limits or scientific maturity.
