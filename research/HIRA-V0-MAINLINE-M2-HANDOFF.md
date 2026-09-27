# HIRA V0 MAINLINE M2 — dynamic high-K

Status: **M2-A MECHANICS IMPLEMENTED / EXACT INTEGRATION PRE-PROOF**

Issue: #163  
Branch: `feat/hira-v0-mainline-m2-high-k`  
Base main: `2920dc025c840245b77468f0b13626ed6dc57fac`

## 1. Goal

M2 moves high-cardinality dynamic schemas into the actual Hira v0 mainline product contract.

Historical R8 evidence already showed:
- tensor mechanics can remain healthy at K=128/K=255;
- older high-K failures were predominantly semantic rather than numerical.

M2 therefore separates:

1. **high-K mechanics**;
2. **high-K semantic quality**.

M2-A addresses only mechanics.

## 2. Frozen K contract

Hira v0 maximum supported schema cardinality:

`K <= 255`

Required mechanics ladder:

`4 / 8 / 16 / 32 / 64 / 128 / 255`

The existing low-K typed behavior remains valid.

## 3. Mainline API changes

Implemented in `src/nmd/mainline.py`:

- `HIRA_V0_MAX_K = 255`
- `HIRA_V0_MAINLINE_M2_VERSION = "0.0-m2a"`
- manifest field `high_k_mechanics`
- `HiraV0Manifest.m2_mechanics_provisional()`
- `HiraV0Session.decide_compiled()`
- `build_hira_v0_m2_mechanics()`

`decide_compiled()` permits schema compilation and decision execution to be timed separately while retaining all mainline invariants.

## 4. Mandatory query invariants

Every mainline query must preserve:

- no state re-encoding;
- candidate budget == K;
- all K candidates selected;
- no hidden pruning;
- relation delta exactly zero;
- finite logits;
- finite probabilities;
- probability mass error <= 1e-6.

Schemas above K=255 fail closed.

## 5. High-K capability suite

Implemented:

`src/nmd/mainline_high_k.py`

Frozen K ladder:

`(4, 8, 16, 32, 64, 128, 255)`

Per-K report:

- requested K;
- actual candidate budget;
- state encode delta;
- schema compile latency;
- decision latency;
- total query latency;
- approximate schema tensor bytes;
- probability mass error;
- finite status;
- full-K status;
- relation delta max;
- selected option ID.

## 6. Permutation contract

Each K executes twice inside the same state session:

1. canonical schema;
2. deterministic rotated schema.

Probabilities are restored by option ID.

Frozen gate:

- max probability error <= 2e-6;
- selected option ID invariant.

No new state encoding is allowed.

## 7. Whole-suite state-once contract

One call to `run_high_k_mechanics_suite()`:

- compiles state exactly once;
- executes 14 dynamic queries;
- 2 queries for each of seven K values;
- expected session query count: 14;
- expected state encode delta: 1.

## 8. Parameter and maturity policy

M2-A adds:

**0 trainable parameters**

M2-A manifest:

- semantic front-end: provisional;
- W28 projection: frozen research base;
- W34 transfer core: provisional;
- typed runtime: available;
- reliability/OOD/abstention: provisional;
- high-K: provisional;
- high-K mechanics: provisional;
- multilingual: pending;
- production-ready: false.

M1 reliability is not promoted by M2.

## 9. Exact integration proof required

Before high-K mechanics can be marked available, run the exact model:

- exact A13 revision;
- exact W28 T0;
- exact W34 provisional transfer;
- M1 fail-closed reliability default;
- M2 K ladder;
- one state encode;
- full-K throughout;
- permutation contract;
- timing and schema memory report.

The exact integration proof is mechanics-only.

It must make no claim that semantic accuracy at K=255 is good.

## 10. M2-B boundary

Only after M2-A code + exact integration + full CI pass:

- create wholly fresh high-K semantic authority;
- do not reuse previously exposed W4-W7 text rows;
- first frozen current-mainline measurement before any new semantic training.

Planned semantic ladder:

- TRAIN authority surface: K=4/8/16/32;
- DEV: K=64/128;
- SEALED CONFIRM: K=255.

No candidate pruning is allowed in the first M2-B authority.

## 11. Immediate next steps

1. dedicated M2 mechanics CI;
2. full repository CI;
3. exact A13/W28/W34 M2 integration runner;
4. freeze M2-A integration receipt;
5. only then design fresh M2-B semantic authority.

No M2 semantic evidence has been exposed yet.
