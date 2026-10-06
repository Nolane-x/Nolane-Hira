# HIRA V1 S70 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #327

Parent S69 merged main:
`1ada877bcd9381486395599519363e87ef3dd831`

Parent fresh S69:
- run `37420104460`
- artifact `11392672918`
- digest `sha256:c24c00e9b79d23b3e7e3fe4d17013270e0410d822b85c62230bb85e972a2f94d`
- verdict **Case B**.

## Frozen S70 variable

Reference aggregation:
`mean_{k!=j} P[j,k]`.

Treatment aggregation:
- `w=softmax(fused)`, temperature **1.0**
- remove self weight
- renormalize opponents
- weighted sum of pairwise comparisons.

Both aggregators:
- trainable params **0**
- detached pair/fused inputs.

Both arms downstream:
- exact S64 contextual gate
- **60 trainable params**
- bit-identical initialization
- exact S66 responsibility objective.

Treatment parameter advantage: **0**.

## A0 staged

Core:
`src/nmd/v1_fused_anchored_pairwise_aggregation.py`

Unit mechanics:
`tests/test_v1_fused_anchored_pairwise_aggregation.py`

A0 court:
`scripts/hira_v1_s70_a0_fused_anchored_pairwise.py`

Guards:
- `tests/test_v1_s70_a0_harness.py`
- `tests/test_v1_s70_a0_workflow.py`

Workflow:
`.github/workflows/hira-v1-s70-a0-fused-anchored-pairwise.yml`

A0 must prove:
- reference exact row mean
- uniform prior collapse
- permutation equivariance
- diagonal isolation
- opponent weights normalized/nonnegative
- K=3/7/255
- no aggregation parameters
- 60/60 gate params
- gate init bit-identical
- bounded residual
- alpha0 identity
- gradient isolation
- checkpoint replay
- no fresh S70 TRAIN/DEV.

## Authorization rule

Marker:
`research/HIRA-V1-S70-ENABLE-A0`

The marker MUST remain absent until exact final pre-A0 head passes generic CI on Python 3.10 and 3.12.

A0 is mechanical only.
