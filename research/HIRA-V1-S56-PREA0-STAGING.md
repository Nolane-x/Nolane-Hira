# HIRA V1 S56 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #295  
PR: #296

## Parent

S55 merged main:
`4457d70b419364931f95d15e726c8179ebc6b4e9`

S55 fresh scientific court:
- run `37212962160`
- artifact `11307227880`
- digest `sha256:7059cf8477eb4bd775e89badb0c734a2d351afe2e1a1d23ceaf4f7189c7744e4`
- verdict **Case C**.

Scientific consequence:
S51–S55 correction/factorization family is exhausted. S56 changes the objective at the final decision level, not representation capacity.

## Frozen S56 design

Both arms:
- `JointStateQueryOptionPrivateCorrectionFork`
- correction params **114,688**
- interaction trainable params **0**
- identity params **0**
- total private trainable **114,688**
- bit-identical initialization
- same immutable cache
- same optimizer/selector
- no new trainable parameters.

Reference:
- decision consistency coefficient **0.0**
- ordering consistency coefficient **0.0**

Treatment:
- standardized full-K JS coefficient **0.10**
- pairwise ordering coefficient **0.05**

Frozen mechanics:
- standardization epsilon **1e-6**
- ordering active threshold **0.25**
- ordering margin floor **0.05**
- no entropy auxiliary.

## Parent native authority

Exact S51 persisted native artifact:
- run `37192490832`
- artifact `11299783210`
- runtime/native digest `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`
- native trainable params 0.

## Core staged

- `src/nmd/v1_cross_view_decision_consistency.py`
- `tests/test_v1_cross_view_decision_consistency.py`

Core invariants:
- identical logits -> zero JS
- controlled disagreement -> positive JS
- offset invariance
- positive-scale invariance
- flat-logit finiteness
- matched-permutation invariance
- ordering sign-flip sensitivity
- inactive low-confidence path
- exact-zero reference auxiliary
- treatment gradients live
- K=3/7/255 finite.

## A0 staged

- `scripts/hira_v1_s56_a0_cross_view_decision_consistency.py`
- `tests/test_v1_s56_a0_harness.py`
- `.github/workflows/hira-v1-s56-a0-cross-view-decision-consistency.yml`

A0 proves:
- real cached evidence path
- equal 114,688 private params
- added params 0
- bit-identical initialization
- treatment auxiliary gradient reaches private correction
- cache/native ownership frozen
- decision offset/scale invariance
- ordering consistency mechanics
- uniform-collapse diagnostics
- K=3/7/255
- full-K probability mass
- no second encoder.

## Authorization rule

A0 marker:
`research/HIRA-V1-S56-ENABLE-A0`

The marker MUST remain absent until the exact final staging head passes generic CI on Python 3.10 and 3.12.

A0 is mechanical only:
- no model selection
- no fresh S56 TRAIN/DEV exposure
- no external Laya/Jev evaluation.
