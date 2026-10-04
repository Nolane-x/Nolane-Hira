# HIRA V1 S49 treatment-only continuation staging receipt

Status: **STAGED / CONTINUATION NOT AUTHORIZED**

Issue: #281  
PR: #282

## Prior partial scientific run

Run: `37178974852`  
Scientific head: `7ece9b16fd5ccfd5b93d9ff9eb1cb73ff15105ab`

Completed:
- pre-TRAIN gates PASS
- reference arm completed 24/24 epochs
- reference DEV exposed for all epochs
- frozen selector chose reference epoch **6**

Did not complete:
- treatment arm never reached TRAIN_BEGIN
- treatment DEV exposure = **0**

Failure:
`TypeError: QueryFreeIdentityPrivateCorrectionFork.__init__() got an unexpected keyword argument 'native_dimension'`

Frozen evidence:
- `research/HIRA-V1-S49-RECOVERED-REFERENCE.json`
- `research/HIRA-V1-S49-PARTIAL-POSTREFERENCE-MECHANICAL-FAILURE-37178974852.md`

## Mechanical repair only

`QueryFreeIdentityPrivateCorrectionFork` now accepts the exact frozen S44/S45 constructor contract:
- native_dimension 256
- hidden_dimension 64
- query_norm_epsilon 1e-12
- private_norm_epsilon 1e-12
- residual_scale 1.0
- adapter_seed 65044
- train_correction

The parent class still rejects any non-frozen values.

No identity equation, parameter shape, seed, loss, optimizer, data, selector, or pair temperature changed.

## Continuation design

The continuation script:
- **never calls reference training**
- runs treatment only
- uses exact S49 seed 70001
- uses exact frozen S49 TRAIN/DEV rows
- compares treatment selected DEV against the frozen recovered reference selected DEV
- requires exact equality of all **24 native runtime hashes** to the frozen reference trajectory
- keeps correction capacity 114,688
- keeps total trainable 163,840
- keeps identity added params 0
- keeps one encoder pass/state-once.

Files:
- `scripts/hira_v1_s49_treatment_continuation.py`
- `tests/test_v1_s49_treatment_continuation.py`
- `.github/workflows/hira-v1-s49-treatment-only-continuation.yml`

## Authorization rule

Continuation is marker-gated on:

`research/HIRA-V1-S49-ENABLE-TREATMENT-CONTINUATION`

The marker MUST remain absent until the exact final continuation staging head passes generic CI on Python 3.10 and 3.12.

After authorization:
- one treatment-only continuation;
- no reference rerun;
- no semantic edits;
- no retry for scientific weakness;
- no second treatment continuation after treatment TRAIN_BEGIN;
- no second full S49 DEV run.
