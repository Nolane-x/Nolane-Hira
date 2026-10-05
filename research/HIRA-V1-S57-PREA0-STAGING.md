# HIRA V1 S57 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #297  
PR: #298

## Parent

S56 merged main:
`e58262a22a14187e61cd12cfa6f5233bdb1a1ea6`

S56 scientific result:
- run `37265241846`
- artifact `11326048054`
- Case **B**

Key evidence:
- fused agreement **+6.51 pp**
- relation agreement **+16.93 pp**
- fused canonical **-13.54 pp**
- fused paraphrase **-9.90 pp**

Whole-distribution consistency moves stability but collapses correctness.

## Frozen S57 design

Architecture:
- exact `JointStateQueryOptionPrivateCorrectionFork`
- correction params **114,688 / arm**
- added trainable params **0**
- identity params **0**
- same S51 persisted native authority
- same immutable cache
- bit-identical init.

Reference:
- ordinal coefficient **0.0**

Treatment:
- ordinal coefficient **0.05**

Frozen ordinal mechanics:
- standardized full-K logits epsilon **1e-6**
- activation threshold **0.25**
- preserved margin floor **0.05**
- detached directional anchor signs
- wrong gold-vs-distractor anchors filtered
- correct gold anchors retained
- non-gold active pairs remain eligible
- no S56 distribution-JS auxiliary.

## A0 staged

Core:
- `src/nmd/v1_pairwise_ranking_consistency.py`

Tests:
- `tests/test_v1_pairwise_ranking_consistency.py`

A0:
- `scripts/hira_v1_s57_a0_discrete_pairwise_ranking_consistency.py`
- `tests/test_v1_s57_a0_harness.py`
- `.github/workflows/hira-v1-s57-a0-discrete-pairwise-ranking-consistency.yml`

A0 proves:
- matched capacity/init
- anchor detach
- sign-flip sensitivity
- offset/positive-scale invariance
- gold-anchor filtering
- non-gold eligibility
- flat/uniform anti-collapse
- option permutation
- K=3/7/255
- probability mass
- real-cache gradient liveness
- native/cache isolation
- no second encoder.

## Parent persisted native

- run `37192490832`
- artifact `11299783210`
- runtime/native digest `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`.

## Authorization rule

Marker:
`research/HIRA-V1-S57-ENABLE-A0`

The marker MUST remain absent until the exact final staging head passes generic CI on Python 3.10 and 3.12.

A0 is mechanical only:
- no fresh S57 TRAIN/DEV exposure
- no model selection
- no external Laya/Jev evaluation.
