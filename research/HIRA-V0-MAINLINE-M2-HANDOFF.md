# HIRA V0 MAINLINE M2 — dynamic high-K

Status: **M2-A MECHANICS READY / M2-B SEMANTIC PRE-EXPOSURE**

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


## 12. Exact M2-A integration — authoritative result

Run:

`36310118240`

Outcome:

`HIRA_V0_M2_MECHANICS_READY`

Artifact:
- `hira-v0-mainline-m2-mechanics-integration`
- ID: `10928771833`
- digest: `sha256:99e8c02aeb48fa32910a591d71b92e3e6f96258c0fa8501f38682ea4a2b7f453`

Exact model:
- resident params: 13,213,199
- trainable params: 0
- semantic front-end resident: 12,750,080
- relation core resident: 422,159
- transfer scorer resident: 40,960
- W28 projection: 32,768
- W34 transfer candidate: 8,192

Whole-suite:
- K ladder: 4 / 8 / 16 / 32 / 64 / 128 / 255
- state encode delta: **1**
- total queries: **14**
- full-K: PASS at every K
- finite: PASS at every K
- relation delta: 0 at every K
- max probability-mass error: **1.1920928955078125e-07**
- max permutation probability error: **4.656612873077393e-10**
- selected-option permutation invariant: PASS at every K
- mechanics pass: **true**

### Exact CPU timing / schema tensor surface

K=4:
- schema compile: 41.46 ms
- decision: 1.62 ms
- schema tensors: 81,344 bytes

K=8:
- schema compile: 60.96 ms
- decision: 1.25 ms
- schema tensors: 151,404 bytes

K=16:
- schema compile: 107.88 ms
- decision: 1.35 ms
- schema tensors: 291,524 bytes

K=32:
- schema compile: 200.82 ms
- decision: 1.43 ms
- schema tensors: 571,764 bytes

K=64:
- schema compile: 397.30 ms
- decision: 1.60 ms
- schema tensors: 1,132,244 bytes

K=128:
- schema compile: 762.79 ms
- decision: 1.92 ms
- schema tensors: 2,253,204 bytes

K=255:
- schema compile: **1482.21 ms**
- decision: **2.66 ms**
- schema tensors: **4,477,609 bytes**

Scientific/product interpretation:

**Hira v0 decision mechanics themselves scale cleanly through K=255.**

The dominant high-K runtime cost is schema semantic compilation, not the typed decision core.

No semantic quality claim was made by this integration.

## 13. M2-A product maturity promotion

Exact mechanics authority:

`run:36310118240;artifact:10928771833;digest:sha256:99e8c02aeb48fa32910a591d71b92e3e6f96258c0fa8501f38682ea4a2b7f453`

M2 manifest now permits:

- `high_k_mechanics = available`
- `high_k_mechanics_authority = <exact authority above>`

It does **not** permit:

- `high_k = available`
- semantic K=255 quality claims
- candidate pruning claims
- production-ready claims

`high_k` remains `provisional` until M2-B fresh semantic authority.


## 14. M2-B fresh semantic authority — preregistered, unexposed

Implemented:
- `src/nmd/mainline_m2_authority.py`
- `src/nmd/mainline_m2_eval.py`
- `tests/test_mainline_m2_authority.py`
- `tests/test_mainline_m2_eval.py`

Fresh authority partitions:

Reserved TRAIN surface, **not exposed in the first frozen baseline**:
- HKA: K=4, 16 cases
- HKB: K=8, 16 cases
- HKC: K=16, 16 cases
- HKD: K=32, 16 cases

Fresh DEV:
- HKE: K=64, 16 cases
- HKF: K=128, 16 cases

Sealed CONFIRM:
- HKG: K=255, 24 cases

HKG fails closed unless explicitly opened after DEV qualification.

The first baseline evaluates only HKE/HKF.

HKA-HKD remain untouched and are reserved for a later rescue only if the frozen current-mainline DEV baseline fails.

## 15. Frozen M2-B baseline model

The first M2-B semantic baseline uses the exact current mainline with **zero new training**:

- exact A13 revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`
- exact W28 T0 checkpoint
- exact W34 provisional co-evidence transfer checkpoint
- relation refinement OFF
- adaptive budget OFF
- candidate pruning OFF
- trainable decision-core params: 0

No HKA-HKD row may be used in this first baseline.

## 16. Frozen M2-B evaluator

Every semantic case executes:

1. one state encode;
2. canonical full-K schema;
3. deterministic rotated full-K schema;
4. probabilities restored by option ID for permutation comparison.

Reported per K:
- top-1;
- top-5 recall;
- MRR;
- mean gold rank;
- mean gold probability;
- mean confidence;
- probability-mass max error;
- permutation max error;
- selected-option invariance rate;
- full-K rate;
- state-once rate;
- relation-delta max;
- finite-output rate;
- mean wall-clock case time.

Mechanics remain a mandatory gate even when semantic metrics pass.

## 17. Frozen DEV gates

### K=64 / HKE

Required:
- top-1 >= **0.70**
- top-5 >= **0.90**
- MRR >= **0.75**
- full-K = 1.0
- state-once = 1.0
- finite = 1.0
- selected-option permutation invariance = 1.0
- permutation max error <= 2e-6
- probability-mass max error <= 1e-6
- relation delta max = 0

### K=128 / HKF

Required:
- top-1 >= **0.60**
- top-5 >= **0.85**
- MRR >= **0.65**
- the same mechanics gates as K=64

Both K=64 and K=128 must pass.

Only then may HKG be exposed.

## 18. Frozen sealed K=255 gates

HKG may be exposed only after a frozen DEV PASS.

Required at K=255:
- top-1 >= **0.50**
- top-5 >= **0.80**
- MRR >= **0.58**
- full-K = 1.0
- state-once = 1.0
- finite = 1.0
- selected-option permutation invariance = 1.0
- permutation max error <= 2e-6
- probability-mass max error <= 1e-6
- relation delta max = 0

If HKE/HKF DEV fails:
- HKG remains sealed;
- HKE/HKF become exposed evidence and may never be used for fitting/selection;
- HKA-HKD may be used only under a new preregistered rescue;
- rescue requires a **new fresh DEV authority**, not HKE/HKF.

No M2-B semantic data has been exposed at the time of this preregistration.
