# HIRA V1 S69 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #325

Parent S68 merged main:
`4cfdbc6fe100715bd950c9a0e2f504f71c486e9c`

Parent fresh S68:
- run `37389861004`
- artifact `11381470793`
- digest `sha256:0156bb57125f6b8c4f380649505803ed76cf49b8edf1c5bd4774fd8790386eaa`
- verdict **Case C**.

## Frozen S69 variable

Reference:
- exact S59 representation `normalize([identity, joint_context])`
- exact S59 pairwise head, 32,832 params.

Treatment:
- zero-parameter query-gated interaction:
  `x = joint_context + 16 * (identity ⊙ joint_context)`
  then normalize;
- representation `normalize([identity,x])`
- same exact S59 pairwise head, 32,832 params.

Treatment parameter advantage: **0**.

Representation trainable params:
- reference 0
- treatment 0.

## A0 staged

Core:
`src/nmd/v1_query_gated_identity_interaction.py`

Unit mechanics:
`tests/test_v1_query_gated_identity_interaction.py`

A0 court:
`scripts/hira_v1_s69_a0_query_gated_interaction.py`

Guards:
- `tests/test_v1_s69_a0_harness.py`
- `tests/test_v1_s69_a0_workflow.py`

Workflow:
`.github/workflows/hira-v1-s69-a0-query-gated-interaction.yml`

A0 must prove:
- reference exact S59 identity
- treatment 512D / zero-parameter / detached
- interaction scale exactly 16
- real and synthetic representation difference
- zero-identity neutralization
- option permutation equivariance
- matched S59 head initialization
- 32,832 vs 32,832 params
- antisymmetry/zero diagonal
- K=3/7/255
- live A/u gradients
- no upstream gradients
- checkpoint replay exact
- no fresh S69 TRAIN/DEV.

## Authorization rule

Marker:
`research/HIRA-V1-S69-ENABLE-A0`

Marker must remain absent until this exact pre-A0 staging head passes generic CI on Python 3.10 and 3.12.

A0 is mechanical only.
