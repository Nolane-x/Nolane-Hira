# HIRA V1 S59 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #302

Parent S58 merged main:
`decc5a744f69e78e83bc57bf28c3283d230f9283`

Parent S58 verdict:
**Case B**

Fresh S58 evidence:
- run `37276076841`
- artifact `11330721014`
- digest `sha256:cea8203e98899feda6c1ca1558b9941a8900b987124dfb43958c90e5d0285e50`.

## Frozen S59 head

Representation:
- detached concat(identity 256D, joint state-query-option context 256D)
- total **512D**
- no native/fused/corrected logits admitted.

Head:
- A **64x512**
- u **64**
- no bias
- total **32,832 trainable params**
- seed **80059**
- explicit antisymmetrization.

Gold supervision:
- only gold-vs-distractor pairs
- distractor-vs-distractor target count **0**
- no teacher
- no pseudo-target
- no self-anchor.

Ownership:
- correction params **114,688**
- pairwise head cannot update correction/native/cache
- one encoder/state-once.

## A0 staged

Core:
`src/nmd/v1_explicit_pairwise_decision_head.py`

A0:
`scripts/hira_v1_s59_a0_explicit_learned_pairwise_decision_head.py`

Workflow:
`.github/workflows/hira-v1-s59-a0-explicit-learned-pairwise-decision-head.yml`

Marker:
`research/HIRA-V1-S59-ENABLE-A0`

A0 proves:
- exact 32,832 parameter surface
- no bias
- deterministic same initialization
- anti-symmetry + diagonal zero
- option permutation equivariance
- K=3/7/255
- full-K
- gold-only pair supervision
- A/u gradients live
- detached representation
- correction/native/cache isolation
- no teacher/logit bypass
- deterministic representation tiebreak
- checkpoint replay exact
- probability mass
- one encoder/state-once.

## Authorization rule

The A0 marker MUST remain absent until this exact final staging head passes generic CI on Python 3.10 and 3.12.

A0 is mechanical only:
- no fresh S59 TRAIN/DEV
- no S59 model selection
- no external Laya/Jev evaluation.
