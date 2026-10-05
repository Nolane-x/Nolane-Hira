# HIRA V1 S67 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #321

Parent S66 merged main:
`e898314b51995db7e6fa18b06256a59551d0797e`

Parent fresh S66:
- run `37331468679`
- artifact `11354228532`
- digest `sha256:f3c6322eec8cce02a698807037a4c05e48c32b2addc5709a116a4384338379b1`
- verdict **Case B**.

## Frozen S67 variable

Reference:
- exact S66 binary per-view responsibility BCE
- exact S64 contextual single-view gate, 60 params.

Treatment:
- same gate, 60 params
- safe oracle-alpha continuous BCE target.

Treatment parameter advantage: **0**.

## Frozen alpha lattice

`[0, 0.0875, 0.175, 0.2625, 0.35]`

Each view is evaluated independently with the other view held at baseline.

Selected nonzero alpha must:
- keep own-view CE non-worse within **1e-8**
- strictly improve paired JS by > **1e-8**.

Among safe candidates choose lowest JS; ties retain smaller alpha.

Target:
`alpha*/0.35` in `[0,.25,.5,.75,1]`.

## A0 staged

Core:
`src/nmd/v1_safe_oracle_alpha_responsibility.py`

Unit mechanics:
`tests/test_v1_safe_oracle_alpha_responsibility.py`

A0 court:
`scripts/hira_v1_s67_a0_safe_oracle_alpha.py`

Guards:
- `tests/test_v1_s67_a0_harness.py`
- `tests/test_v1_s67_a0_workflow.py`

Workflow:
`.github/workflows/hira-v1-s67-a0-safe-oracle-alpha.yml`

A0 must prove:
- 60/60 params
- added treatment params 0
- bit-identical init/context path
- frozen lattice exact
- target levels exact
- zero + at least three nonzero levels observed
- view-swap equivariance
- oracle target differs from S66 binary
- smaller-alpha tie break exact
- gradient path live
- upstream gradients zero
- K=3/7/255
- bounded residual
- alpha0 identity
- probability mass <=1e-6
- checkpoint replay exact
- no fresh S67 TRAIN/DEV.

## Authorization rule

Marker:
`research/HIRA-V1-S67-ENABLE-A0`

The marker MUST remain absent until exact final pre-A0 staging head passes generic CI on Python 3.10 and 3.12.

A0 is mechanical only.
