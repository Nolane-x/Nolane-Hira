# HIRA V1 S59 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #301  
PR: #303

Parent S58 merged main:
`decc5a744f69e78e83bc57bf28c3283d230f9283`

Parent verdict:
**Case B**

## Frozen S59 family

Reference:
- matched learned **pointwise** low-rank decision head.

Treatment:
- matched learned **antisymmetric pairwise** low-rank decision head.

Both:
- exact S54/S58 correction shell **114,688 params**
- decision head **16,384 params**
- total private **131,072 params**
- head rank **32**
- A seed **80590**
- B zero initialization
- head scale **1.0**
- no teacher
- no pseudo-target
- no self-anchor
- one encoder/state-once.

Only controlled structural variable:
**pointwise vs antisymmetric pairwise aggregation.**

## A0 staged

Core:
`src/nmd/v1_learned_pairwise_decision_head.py`

A0:
`scripts/hira_v1_s59_a0_matched_pairwise_decision_head.py`

Workflow:
`.github/workflows/hira-v1-s59-a0-matched-pairwise-decision-head.yml`

Marker:
`research/HIRA-V1-S59-ENABLE-A0`

A0 must establish:
- exact matched capacity/init
- exact zero initial head residual
- initial relation/fused output identity
- pairwise antisymmetry
- zero diagonal
- permutation equivariance
- K=3/7/255
- finite degenerate context
- B-first warm-start gradient
- A gradient live after controlled B update
- legacy correction gradient live
- cache/native isolation
- no teacher dependency
- no second encoder.

## Authorization rule

The A0 marker MUST remain absent until the exact final pre-A0 staging head passes generic CI on Python 3.10 and 3.12.

A0 is mechanical only:
- no fresh S59 TRAIN/DEV
- no S59 model selection
- no external Laya/Jev evaluation.
