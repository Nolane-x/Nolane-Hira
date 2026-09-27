# HIRA V0 MAINLINE M0 closure — integrated typed decision model shell

Status: **CLOSED — M0 INTEGRATION READY**

Issue: #159  
PR: #160  
Branch: `feat/hira-v0-mainline-m0`  
Base main: `830bf0a98c232c63753eff319e8ae4ef269e1561`

## 1. Phase meaning

M0 is the first mainline model-construction phase after the R8 transfer-research program stopped blocking product integration.

M0 does not promote the W34 transfer core.

It integrates the strongest frozen research pieces into one explicit Hira v0 runtime while preserving their maturity status.

## 2. Mainline runtime

Implemented:

- `src/nmd/mainline.py`
- `HiraV0Mainline`
- `HiraV0Session`
- `HiraV0Manifest`
- `HiraV0ParameterReport`
- `build_hira_v0_mainline()`

Default inference:

- state encode once;
- dynamic schema per query;
- `choice / score / noul`;
- `coevidence_symmetric_semantic`;
- full-K;
- relation refinement OFF;
- adaptive budget OFF;
- no unqualified reliability calibrator.

## 3. Frozen provenance

W28 projection:

`1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f`

W34 provisional transfer checkpoint:

`d69fa11805291e6a06631d5bda941065f209ea96f5c46091187e984ff083834c`

Transfer maturity:

`PROVISIONAL_TRANSFER_CORE`

Production transfer promotion:

false

## 4. Unit/contract gate

Dedicated mainline run:

`36302372466`

Outcome:

PASS

Contracts include:

- machine-readable provisional manifest;
- `production_ready=false`;
- one state encode serving multiple typed queries;
- dynamic schema including K=3 architectural smoke;
- choice/score/noul;
- option-order semantic invariance;
- full-K;
- relation delta zero;
- probability mass integrity;
- parameter-surface accounting;
- wrong-checkpoint fail-closed behavior.

Full repository Python 3.10/3.12 CI also passed on the initial integrated M0 head.

## 5. Exact-provenance integration

Authoritative integration run:

`36302647323`

Outcome:

PASS

This run loaded:

- exact frozen A13 revision;
- exact W28 T0 artifact;
- exact W34 provisional transfer artifact.

It then built `HiraV0Mainline`, opened one state session, and executed:

- choice K=2;
- score K=3;
- noul K=2.

Observed:

- state encode count: 1
- session query count: 3
- candidate budgets: [2, 3, 2]
- relation delta max abs: 0.0
- probability mass max error <= 1e-6
- quality claim made: false

Integration artifact:

- name: `hira-v0-mainline-m0-integration`
- ID: `10925958503`
- digest: `sha256:872966ce40a56aa1ae39b2ea2cbb67e1e1c999e2a82f3abc425934829be7ba4a`

## 6. Actual resident parameter census

Exact integrated runtime:

- total resident: **13,213,199**
- semantic front-end: **12,750,080**
- relation core: **422,159**
- transfer scorer total: **40,960**
- W28 projection: **32,768**
- W34 candidate: **8,192**
- packaged trainable parameters: **0**

This puts the current integrated Hira v0 shell at approximately **13.2M resident parameters**.

Resident parameters are not the same as trained W34 candidate parameters.

## 7. Machine-readable maturity

M0 manifest freezes:

- semantic front-end: `provisional`
- projection: `frozen_research_base`
- transfer core: `provisional`
- typed runtime: `available`
- reliability/OOD/abstention: `pending`
- high-K qualification: `pending`
- multilingual: `pending`
- transfer core promoted: false
- production ready: false

M0 cannot silently claim production readiness.

## 8. What M0 proves

M0 proves the existing research components can be assembled into one real state-once typed model runtime with exact frozen provenance.

It proves the integrated code path works.

It does not prove production semantic quality beyond the already-frozen W34 evidence.

## 9. What M0 does not claim

No claim is made yet for:

- calibrated confidence;
- OOD detection;
- abstention;
- production transfer-core readiness;
- high-K quality;
- multilingual quality;
- adaptive-budget safety;
- Laya/JEV superiority.

## 10. Next mainline phase

Next:

**HIRA V0 MAINLINE M1 — reliability / OOD / abstention**

M1 must build on the M0 exact-provenance runtime.

Existing W6c calibration research may inform mechanism design, but its exposed rows and promoted competitive scorer are not reusable authority for M1.

M1 must use fresh evidence appropriate to the W34/M0 semantic core.
