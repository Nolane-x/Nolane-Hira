# HIRA V1 S64 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #313  
PR: #314

Parent S63:
- merged main `4fcb58f0c35e2feca841b08f6f560c092ae600fd`
- scientific run `37314015283`
- artifact `11347706059`
- digest `sha256:dcb69358dfa97f3644198534faa1c2f5f214b4f1773d79008eec13cafaf2a97e`
- verdict **Case B**.

## Frozen S64 variable

Reference:
- exact S63 decision-surface learned set gate
- 61 trainable params.

Treatment:
- context-projected reliability gate
- detached S59 [B,K,512] representation
- fixed Rademacher projection [4,512], seed 64064
- exact same learned 61-param architecture and initialization.

Added treatment trainable params: **0**.

Shared:
- exact S62 reliability target
- alpha probe 0.35
- tolerance 1e-8
- bounded residual / alpha max 0.35
- correction/head/native/cache ownership
- optimizer/LR/weight decay
- selector.

## Staged mechanical authority

Core:
`src/nmd/v1_context_projected_reliability_gate.py`

Court:
`scripts/hira_v1_s64_a0_context_projected_reliability_gate.py`

Tests:
- `tests/test_v1_context_projected_reliability_gate.py`
- `tests/test_v1_s64_a0_harness.py`
- `tests/test_v1_s64_a0_workflow.py`

Workflow:
`.github/workflows/hira-v1-s64-a0-context-projected-reliability-gate.yml`

A0 must prove:
- 61 vs 61 trainable params
- learned initialization bit-identical
- fixed projection exact/deterministic/non-trainable
- context affine invariance
- option permutation invariance
- exact initial alpha 0.10
- exact S62 target inheritance
- staged phi gradient path
- zero reliability gradient to fused/pairwise/context/upstream
- real S59 context shape/ownership
- K=3/7/255
- residual bound/probability mass
- checkpoint replay exact
- one encoder/state-once.

A0 is mechanical only:
- no fresh S64 TRAIN/DEV
- no model selection
- no threshold interpretation
- no external Laya/Jev evaluation.

## Authorization rule

Marker:
`research/HIRA-V1-S64-ENABLE-A0`

It MUST remain absent until the exact final pre-A0 staging head passes generic CI on Python 3.10 and 3.12 plus preflight.
