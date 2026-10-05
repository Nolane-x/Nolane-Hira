# HIRA V1 S66 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #319

Parent S65 merged main:
`6bae27159fc2f9577f72540c7070da3e41ef3d15`

Parent fresh S65:
- run `37325995163`
- artifact `11352920320`
- digest `sha256:a44c560d304537bb7acaa1b5d569160ec744f897ae728edf95ceca7701be9dfc`
- verdict **Case B**.

## Frozen S66 variable

Reference:
- exact S64 contextual single-view gate, 60 params
- shared S62 pair target applied to both views
- independent-view BCE.

Treatment:
- exact same S64 contextual single-view gate, 60 params
- per-view counterfactual responsibility targets
- independent-view BCE.

Treatment parameter advantage: **0**.

## Responsibility semantics

For each view independently:
- fixed alpha probe = **0.35**
- correctness safe iff own probe CE <= own baseline CE + **1e-8**
- stability better iff probing only that view lowers paired JS by > **1e-8**
- positive iff both conditions hold.

This permits:
- 00
- 01
- 10
- 11.

No teacher, no DEV target, no smoothing, no coefficient.

## A0 staged

Core:
`src/nmd/v1_per_view_responsibility.py`

Unit mechanics:
`tests/test_v1_per_view_responsibility.py`

A0 court:
`scripts/hira_v1_s66_a0_per_view_responsibility.py`

Guards:
- `tests/test_v1_s66_a0_harness.py`
- `tests/test_v1_s66_a0_workflow.py`

Workflow:
`.github/workflows/hira-v1-s66-a0-per-view-responsibility.yml`

A0 must prove:
- 60/60 params
- added treatment params 0
- bit-identical init/context projection
- all four target states observed in deterministic probe bank
- target disagreement >0
- view-swap equivariance
- treatment target differs mechanically from shared pair target
- correctness and stability vetoes active for both views
- staged output/phi gradient path
- upstream gradients zero
- K=3/7/255
- bounded residual
- alpha0 identity
- probability mass <=1e-6
- checkpoint replay exact
- no fresh S66 TRAIN/DEV.

## Authorization rule

Marker:
`research/HIRA-V1-S66-ENABLE-A0`

The marker MUST remain absent until the exact final pre-A0 staging head passes generic CI on Python 3.10 and 3.12.

A0 is mechanical only.
