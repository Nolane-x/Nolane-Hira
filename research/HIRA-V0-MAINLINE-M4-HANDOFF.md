# HIRA V0 MAINLINE M4 handoff — runtime / latency / RAM / packaging

Status: **M4-A BATCHED SCHEMA COMPILER IMPLEMENTED / PRE-INTEGRATION**

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
