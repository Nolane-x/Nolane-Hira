# HIRA V1 S65 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #317

Parent S64 merged main:
`c5063da0365d49dc4f9dc2c64f94bc309caef0ba`

Parent fresh S64:
- run `37319541951`
- artifact `11348594503`
- digest `sha256:f3d04d452a51b7358a23a9391579bca6ce1ab2c16fa107dc9f7f79740dba90ea`
- verdict **Case B**.

## Frozen controlled variable

Both arms:
- exact S64 contextual gate
- **60 trainable params**
- same context path
- bit-identical initialization
- exact S62 pair-level reliability target
- same correction/head trajectory
- single-view inference.

Reference:
`0.5 * [BCE(z_c,y) + BCE(z_p,y)]`

Treatment:
`BCE(min(z_c,z_p),y)`

Exact soft-AND:
`0.5*(z_c+z_p-|z_c-z_p|)`

No extra params, temperature, objective mixing coefficient, new target, or paired inference path.

## A0 staged

Core:
`src/nmd/v1_cross_view_soft_and_reliability.py`

Tests:
- `tests/test_v1_cross_view_soft_and_reliability.py`
- `tests/test_v1_s65_a0_harness.py`
- `tests/test_v1_s65_a0_workflow.py`

Court:
`scripts/hira_v1_s65_a0_cross_view_soft_and_reliability.py`

Workflow:
`.github/workflows/hira-v1-s65-a0-cross-view-soft-and-reliability.yml`

A0 must prove:
- 60 vs 60 params
- treatment parameter advantage 0
- bit-identical gate initialization/context path
- exact S62 target
- soft-AND exact min
- pair symmetry
- lower-view gradient routing
- no extra trainable objective tensors
- single-view inference unchanged
- staged gradient path
- upstream gradient zero
- K=3/7/255
- bounded residual
- alpha=0 identity
- probability mass <=1e-6
- exact checkpoint replay
- one encoder/state-once
- no fresh S65 TRAIN/DEV.

## Authorization rule

Marker:
`research/HIRA-V1-S65-ENABLE-A0`

The marker MUST remain absent until the exact final pre-A0 staging head passes generic CI on Python 3.10 and 3.12.
