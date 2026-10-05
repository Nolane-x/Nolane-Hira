# HIRA V1 S68 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #323

Parent S67 merged main:
`942871bb44bbef3e28e74698f161b02c6796cdcb`

Unique S67 scientific run:
- `37336779415`
- verdict **Case B**
- Actions artifact unavailable because upload was skipped after a stale post-DEV verifier key;
- unique raw receipt is frozen in main.

## Frozen S68 controlled variable

Both arms:
- exact S59 detached 512D representation
- A [64,512]
- u [64]
- G [64,8]
- exact **33,344 trainable params**
- same initialization
- same gold-pair objective
- same dynamic-K antisymmetric path.

Reference:
- set context retains identity half only;
- joint-context half forced to zero.

Treatment:
- set context uses full identity + joint state/query/option representation.

Treatment parameter advantage: **0**.

## Fixed mechanics

Context projection:
- [8,512]
- seed **68068**
- non-trainable.

Modulation:
`gamma = 1 + 0.5*tanh(G z)`

G starts exact zero, therefore:
- gamma=1 exact
- reference exact S59 at init
- treatment exact S59 at init.

## A0 staged

Core:
`src/nmd/v1_context_modulated_pairwise_head.py`

Unit mechanics:
`tests/test_v1_context_modulated_pairwise_head.py`

A0 court:
`scripts/hira_v1_s68_a0_context_modulated_pairwise.py`

Guards:
- `tests/test_v1_s68_a0_harness.py`
- `tests/test_v1_s68_a0_workflow.py`

Workflow:
`.github/workflows/hira-v1-s68-a0-context-modulated-pairwise.yml`

A0 uses only already-exposed S67 TRAIN rows plus synthetic mechanics.

Required:
- 33,344 / 33,344 params
- added treatment params 0
- tensors A,G,u exactly
- arm parameter initialization bit-identical
- A/u exact S59 init
- G exact zero
- fixed projection identical and non-trainable
- initial S59 pair/aggregate/choice identity exact
- reference/treatment context separation
- joint-context ablation equality
- context option-permutation invariance
- pair permutation equivariance
- antisymmetry and diagonal exact
- K=3/7/255
- finite probability mass
- A/u/G gradients live
- reference/treatment G gradients differ
- no representation/correction/native/cache gradient leakage
- checkpoint replay exact
- no fresh S68 TRAIN/DEV.

## Authorization rule

Marker:
`research/HIRA-V1-S68-ENABLE-A0`

Marker MUST remain absent until exact final pre-A0 head passes generic CI on Python 3.10 and 3.12.

A0 is mechanical only.
