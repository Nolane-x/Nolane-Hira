# HIRA V0 MAINLINE M4 -> M5 HANDOFF

Status: **M4 CLOSED — NEXT FRONTIER M5 MATCHED EXTERNAL BENCHMARK**

Issue: #167  
PR: #168  
Branch: `feat/hira-v0-mainline-m4-runtime`

## Frozen M4 authority

Package outcome:

`HIRA_V0_M4_PACKAGE_READY`

Run: `36357825580`  
Artifact: `10944174953`  
Digest: `sha256:5fa30c93aca4110bc8602f347c2992b7eb034056c01c432d26c84acbfd619f9a`

M4-A authority:

- run `36327904288`;
- artifact `10934506746`;
- digest `sha256:447c394a5a6b2246a354b21d49e9ee848c2271a636eedd019ba2ae23496324eb`;
- outcome `HIRA_V0_M4_SCHEMA_BATCHING_READY`.

M4-B2 authority:

- run `36329648932`;
- artifact `10934788096`;
- digest `sha256:15f2b3358338049fec79a5ce79049aa7e43d61c7ec8285203d8cba85314833ef`;
- outcome `HIRA_V0_M4_BOUNDED_CACHE_READY`.

Preserve M4-B1 failed stress-harness evidence:

- run `36328918665`;
- artifact `10934986918`;
- digest `sha256:e2624ce369a17c068cf8bb78fffb70615d86581a630dc49d31c3ae9d3cfc5ed4`.

Canonical closure:

`research/HIRA-V0-MAINLINE-M4-CLOSURE.md`

## Runtime to benchmark

Use the exact M4 frozen runtime:

- W28 T0 SHA256 `1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f`;
- W34 SHA256 `d69fa11805291e6a06631d5bda941065f209ea96f5c46091187e984ff083834c`;
- A13 revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- A13 weight SHA256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- cache defaults: 16 schemas / 64 MiB tensor residency;
- full-K;
- state-once;
- relation refinement disabled;
- adaptive budget disabled;
- trainable parameters: 0.

Do not silently retrain or change checkpoints inside M5.

## Scientific status entering M5

Do not overclaim:

- semantic quality: provisional;
- reliability/OOD/abstention: provisional/fail-closed;
- high-K semantic quality: provisional;
- multilingual: provisional;
- production-ready: false.

## M5 purpose

M5 is not another internal ablation wave.

It must answer a stricter question:

**How does the frozen Hira v0 runtime compare with Laya/JEV under the same benchmark contract?**

Matched means:

- same examples;
- same split;
- same allowed input information;
- same output/scoring rule;
- same K where applicable;
- same retry policy;
- explicit model/checkpoint versions;
- explicit hardware/runtime accounting;
- no cherry-picking best historical rows from different conditions.

## Recommended M5 order

1. recover and pin authoritative Laya/JEV repositories/checkpoints;
2. freeze benchmark contract before seeing final Hira-vs-baseline results;
3. create adapters preserving each model's native inference path;
4. run matched correctness/quality benchmarks;
5. run dynamic-K/high-K only where the external baseline supports it;
6. measure latency, memory and parameter surfaces separately from quality;
7. retain raw per-example outputs;
8. generate matched receipts and integrity manifests;
9. write M5 closure with bounded claims;
10. only then create the final Hira v0 release package.

Any baseline that cannot execute a matched task must be reported as unsupported/not-comparable, not assigned a fabricated score.

## M4 package smoke reference

The exact CPU smoke from run `36357825580` passed:

- CPU local load: ~3072.45 ms on the authority runner;
- resident total: 13,213,199 parameters;
- trainable total: 0;
- state encode calls: 1;
- full-K: true;
- probability sum: 1.0;
- production_ready: false.

M5 is now the only Hira v0 mainline frontier.
