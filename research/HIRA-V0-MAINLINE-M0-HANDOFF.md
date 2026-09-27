# HIRA V0 MAINLINE M0 — integrated typed decision model shell

Status: **ACTIVE — MODEL CONSTRUCTION STARTED**

Issue: #159  
Branch: `feat/hira-v0-mainline-m0`  
Base main: `830bf0a98c232c63753eff319e8ae4ef269e1561`

## 1. Phase transition

R8-W34 closed the blocking pre-mainline transfer-research phase.

Frozen W34 outcome:

`W34_COEVIDENCE_COMPOSITION_FAIL`

This outcome did not promote the transfer core to production readiness, but the sealed relative-transfer gate passed strongly:

- composed severity delta: +0.5416666666666667
- worst-factor top-1 delta: +0.35416666666666663
- F2 top-1 delta: +0.15625
- no factor regression
- TH/TI runtime gates: PASS
- invalid-vector rate: 0.0

The selected W34 checkpoint is therefore designated:

`PROVISIONAL_TRANSFER_CORE`

Checkpoint SHA256:

`d69fa11805291e6a06631d5bda941065f209ea96f5c46091187e984ff083834c`

It is not `HIRA_V0_TRANSFER_CORE_READY`.

No W35 blocks mainline construction.

## 2. Mainline product contract

Hira v0 is a compact typed decision model, not an autoregressive text generator.

Core execution:

```
state text
  -> semantic front-end
  -> StateMemory                         [encode once]
        |
        +-> dynamic schema compiler
        |     question + logical options
        |
        +-> provisional transfer/composition core
        |
        +-> typed decision
              choice | score | noul
```

A session must compile state once and answer multiple independently compiled typed queries without re-encoding the state.

## 3. M0 architecture

### HiraV0Mainline

Stable facade around the integrated runtime.

Responsibilities:
- expose one machine-readable manifest;
- enforce provisional/qualified module states;
- open state-once sessions;
- provide a stable default inference policy;
- report parameter surfaces;
- fail closed on invalid core configuration.

### HiraV0Session

Owns exactly one compiled `StateMemory`.

Each `decide()`:
- compiles a fresh dynamic schema;
- uses the same state memory;
- defaults to `coevidence_symmetric_semantic`;
- uses full-K;
- disables relation refinement;
- returns existing typed `DecisionOutput`;
- verifies that no state re-encoding occurred.

### Mainline manifest

Must make maturity explicit.

Initial M0 statuses:

- semantic front-end: `provisional`
- W28 projection: `frozen_research_base`
- W34 transfer/composition: `provisional`
- typed decision runtime: `available`
- reliability/calibration/OOD: `pending`
- high-K qualification: `pending`
- multilingual qualification: `pending`
- production-ready: `false`

A research checkpoint must never be representable as production-ready by omission.

## 4. Exact provisional provenance

Projection:

- W28 T0
- SHA256 `1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f`

Transfer/composition:

- W34 co-evidence
- 8,192 candidate parameters
- SHA256 `d69fa11805291e6a06631d5bda941065f209ea96f5c46091187e984ff083834c`
- status `PROVISIONAL_TRANSFER_CORE`

The mainline builder must bind these exact identities by default.

## 5. M0 inference invariants

Default path:

- coarse mode: `coevidence_symmetric_semantic`
- relation refinement: false
- adaptive budget: false
- full-K: true
- state-once: true
- schema cache: allowed
- reliability calibrator: absent/unqualified

Every query must verify:

- candidate budget equals K;
- selected mask covers all options;
- relation delta is exactly zero;
- probability mass is finite and sums to one;
- runtime did not call `compile_state` again.

## 6. Parameter accounting

M0 must report separately:

- semantic front-end resident parameters;
- W28 projection parameters;
- W34 transfer candidate parameters;
- relation-core resident parameters;
- total resident parameters;
- total trainable parameters.

The 8,192 W34 candidate parameters remain frozen in packaged M0 inference.

The report must not confuse resident parameters with actively trained parameters.

## 7. M0 tests

Required:

1. manifest is machine-readable and `production_ready=false`;
2. one state encode serves multiple typed decisions;
3. choice works;
4. score works;
5. noul works;
6. schema can change between queries;
7. option ordering preserves semantic selection;
8. full-K invariant holds;
9. relation delta remains zero;
10. parameter report exposes W28 projection and W34 candidate surfaces;
11. exact builder provenance fails closed on wrong checkpoint bytes.

## 8. What M0 intentionally does not solve

M0 does not claim:

- calibrated confidence;
- OOD detection;
- abstention;
- high-K qualification;
- multilingual transfer;
- adaptive budget readiness;
- production transfer-core readiness;
- matched Laya/JEV superiority.

Those are explicit later mainline phases.

## 9. After M0

M1: reliability / OOD / abstention  
M2: dynamic high-K qualification  
M3: EN/VI multilingual  
M4: package, CPU latency, RAM, CLI/API  
M5: matched external benchmark

Transfer-core improvement may continue in parallel, but it is no longer allowed to block these phases.
