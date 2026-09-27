# HIRA V0 MAINLINE M4 — runtime, latency, RAM and packaging closure

Status: **CLOSED — HIRA_V0_M4_PACKAGE_READY**

Issue: #167  
PR: #168  
Branch: `feat/hira-v0-mainline-m4-runtime`  
Base main: `29eaa8b4eb687efb18155930aebfe5f14dd3c851`

## 1. Scope

M4 closes deployment behavior only:

- batched semantic schema compilation;
- bounded schema-cache residency with real LRU eviction;
- deterministic local runtime bundle;
- exact checkpoint/evidence provenance;
- packaged Python wheel;
- CPU local-load path;
- package integrity verification;
- one real state-once/full-K decision smoke.

M4 does **not** promote the scientific maturity frozen in M1-M3.

The following remain provisional:

- semantic front-end quality;
- transfer/composition core;
- reliability/OOD/abstention qualification;
- high-K semantic quality;
- multilingual quality.

`production_ready=false` remains mandatory.

## 2. M4-A — batched schema compilation

Authority outcome:

`HIRA_V0_M4_SCHEMA_BATCHING_READY`

Run: `36327904288`  
Artifact: `10934506746`  
Digest: `sha256:447c394a5a6b2246a354b21d49e9ee848c2271a636eedd019ba2ae23496324eb`

Exact A13 K=255 evidence:

- encoder calls: **258 -> 3**;
- cold schema compile: **1392.83 ms -> 304.87 ms**;
- cold compile speedup: **~4.57x**;
- warm cached compile: **0.813 ms**;
- warm encoder calls: **0**;
- decision core: **2.56 ms**;
- selected option unchanged;
- maximum probability error: **5.96e-8**;
- relation delta: **0**;
- full-K: **PASS**.

M4-A therefore removes the dominant O(K) encoder-call bottleneck without changing the Hira decision contract.

## 3. M4-B — bounded RAM/cache discipline

Immutable failed stress-harness evidence:

Run: `36328918665`  
Artifact: `10934986918`  
Digest: `sha256:e2624ce369a17c068cf8bb78fffb70615d86581a630dc49d31c3ae9d3cfc5ed4`  
Outcome: `HIRA_V0_M4_BOUNDED_CACHE_FAIL`

That failure is retained intentionally because it records the first invalid witness design rather than hiding it.

Qualified witness:

Outcome:

`HIRA_V0_M4_BOUNDED_CACHE_READY`

Run: `36329648932`  
Artifact: `10934788096`  
Digest: `sha256:15f2b3358338049fec79a5ce79049aa7e43d61c7ec8285203d8cba85314833ef`

Frozen runtime defaults:

- maximum schemas: **16**;
- maximum tensor residency: **64 MiB**;
- eviction: real **LRU**;
- inspection API: available;
- clear API: available;
- runtime limit configuration: available;
- oversized schemas may execute but cannot remain resident without bound.

Qualified witness gates:

- byte bound: PASS;
- entry bound: PASS;
- LRU hit -> MRU: PASS;
- older resident eviction: PASS;
- touched-entry survival: PASS;
- recompiled semantic equivalence: PASS;
- probability error after recompile: **0.0**;
- selected option unchanged;
- state-once: PASS;
- full-K: PASS;
- relation delta: **0**.

## 4. M4-C — deterministic runtime package

Authority outcome:

`HIRA_V0_M4_PACKAGE_READY`

Run: `36357825580`  
Artifact: `10944174953`  
Artifact digest: `sha256:5fa30c93aca4110bc8602f347c2992b7eb034056c01c432d26c84acbfd619f9a`

Authority head:

`7e61af0e0db1dcedd8280b84742c81fa9813e7a4`

The runtime artifact contains:

- exact W28 T0 checkpoint;
- exact W34 transfer/composition checkpoint;
- packaged `nolane-hira` Python wheel;
- M4-A qualified receipt;
- M4-B1 immutable failed-harness receipt;
- M4-B2 qualified receipt;
- deterministic `runtime-manifest.json`;
- `INTEGRITY.sha256`;
- pinned A13 model/revision/weight SHA identity;
- cache defaults;
- runtime README;
- package smoke receipt.

The A13 snapshot bytes are deliberately **not redistributed inside the bundle**. The manifest pins:

- model: `microsoft/xtremedistil-l6-h256-uncased`;
- revision: `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA256: `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`.

The loader accepts an exact local snapshot for offline execution, or fetches that exact pinned revision when a snapshot is not supplied.

## 5. Exact CPU package smoke

The authority workflow:

1. downloaded all frozen checkpoint/evidence authorities;
2. built one wheel;
3. created the deterministic runtime bundle;
4. verified every `INTEGRITY.sha256` entry;
5. installed the runtime from the wheel inside the bundle;
6. passed `tests/test_local_runtime_bundle.py` (**5 passed**);
7. fetched the exact pinned A13 revision;
8. verified the exact A13 `model.safetensors` SHA;
9. loaded the packaged W28/W34 runtime on CPU;
10. opened one state-once session;
11. executed one typed full-K choice query.

Receipt:

- schema: `hira-v0-mainline-m4-package-smoke-v1`;
- status: **PASS**;
- outcome: `HIRA_V0_M4_PACKAGE_READY`;
- CPU local load: **3072.45 ms** on the GitHub-hosted authority runner;
- state encode calls for session creation: **1**;
- query count: **1**;
- candidate budget: **2/2**;
- full-K: **true**;
- probability sum: **1.0**;
- trainable parameters: **0**;
- production-ready claimed: **false**.

Parameter report:

- resident total: **13,213,199**;
- semantic front-end resident: **12,750,080**;
- relation core resident: **422,159**;
- transfer scorer resident: **40,960**;
- W28 projection parameters: **32,768**;
- W34 transfer candidate parameters: **8,192**;
- trainable total: **0**.

Runner-process peak RSS observations:

- before runtime load: **695,956 KiB**;
- after runtime load: **784,812 KiB**;
- after one query: **814,068 KiB**.

These RSS values are process-level observations on that runner, not a standalone Hira-model memory claim. The observed peak increase from the pre-load process baseline was **88,856 KiB after load** and **118,112 KiB after the query**.

After the two-option query, schema cache accounting reported:

- entries: **1**;
- bytes: **67,786**;
- evictions: **0**;
- max entries: **16**;
- max bytes: **67,108,864**.

## 6. Frozen provenance

Projection:

- W28 T0;
- SHA256 `1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f`.

Transfer/composition:

- W34 co-evidence;
- candidate capacity: 8,192 parameters;
- SHA256 `d69fa11805291e6a06631d5bda941065f209ea96f5c46091187e984ff083834c`;
- scientific status remains `PROVISIONAL_TRANSFER_CORE`.

No M4 optimization changes either checkpoint identity.

## 7. M4 exit criteria

| Gate | Result |
|---|---|
| K255 schema encoder calls <= 3 | PASS |
| Schema decision equivalence | PASS |
| Warm cache O(1) encoder behavior | PASS |
| Bounded entry residency | PASS |
| Bounded tensor-byte residency | PASS |
| Real LRU witness | PASS |
| Evict/recompile semantic equivalence | PASS |
| Deterministic runtime manifest | PASS |
| Exact checkpoint provenance | PASS |
| Exact evidence provenance | PASS |
| Wheel packaged | PASS |
| Integrity manifest verified | PASS |
| Wheel install from bundle | PASS |
| Exact A13 identity verified | PASS |
| CPU runtime load | PASS |
| State-once smoke | PASS |
| Full-K smoke | PASS |
| Trainable parameters = 0 | PASS |
| No production-ready overclaim | PASS |

Therefore M4 satisfies its exit criterion.

## 8. Scientific maturity after M4

M4 changes deployment maturity, not scientific qualification.

Frozen status after closure:

- typed runtime: **available**;
- state-once: **available**;
- high-K mechanics: **available**;
- semantic front-end: **provisional**;
- transfer core: **provisional**;
- reliability/OOD/abstention: **provisional / fail-closed**;
- high-K semantic quality: **provisional**;
- multilingual: **provisional**;
- production-ready: **false**.

## 9. Next phase

The next and final Hira v0 mainline phase is:

**M5 — matched external benchmark**

M5 must compare Hira against external Laya/JEV authorities under a matched, reproducible contract. It must not convert unmatched historical table numbers into a superiority claim.

M5 should close only after:

- benchmark contract is frozen;
- external baselines are pinned;
- tasks/data/splits are matched;
- parameter/system measurements use explicit accounting;
- Hira is run from the M4 frozen runtime;
- raw receipts and integrity evidence are retained;
- final claims are limited to what matched evidence demonstrates.

M4 is closed. M5 is the active frontier.
