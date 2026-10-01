# HIRA V1 S28 interpretation plan — frozen before S28-A0/DEV exposure

Status: **FROZEN**

Issue: #239  
PR: #240

## Fixed hypothesis

S28 tests exactly one claim:

> With S26/S27 production inference, capacity and gradient ownership frozen, directly transporting query-conditioned state role/value anchors across equivalent wording views — while penalizing cross-query anchor collapse — improves fresh relation discrimination and wording transport more reliably than downstream output-signature canonicalization.

## Frozen architecture

Inherited exactly:
- shared A13 LoRA **16,384**
- primary-private projection **32,768**
- relation-private projection **32,768**
- total trainable surface **81,920**
- original A13 frozen
- HIRACore frozen
- S21 primary
- S26 factorized relation inference
- S14 equal standardized full-K fusion
- S25 shared/private gradient ownership
- neutral-bisector only on shared LoRA
- relation detach from fused-primary objective

S28 adds zero learned inference/training parameters.

## Frozen S28 anchor objective

Training-only state anchors:
- query-conditioned role anchor
- role-orthogonal value/content anchor

Cross-view transport per factor:
- same-pair cosine alignment
- hardest wrong cross-query separation
- separation margin **0.20**

Fixed factor weights:
- role **0.50**
- value **0.50**

Outer relation-block coefficient:
- anchor transport **0.15**

S28 replaces S27 output blockwise canonicalization with anchor transport. They are not stacked.

Other loss shell:
- fused decision both wording views
- swap coefficient **0.25**
- swap margin **0.20**
- option alignment **0.05**
- option temperature **0.10**
- fused cross-view JS **0.25**
- relation CE **0.10**

## Frozen fresh TRAIN/DEV

- one fresh English authority
- seed **49001**
- TRAIN **768**
- DEV **192**
- **12 wholly fresh S28 domains**
- K **4**
- two state views
- two question views per semantic query
- two option views
- epochs **24**
- batch size **16**
- AdamW lr **2e-4**
- weight decay **0.01**
- grad clip **1.0**

## Frozen DEV selector

Lexicographic:
1. fused paired-both-correct
2. fused canonical accuracy
3. relation canonical accuracy
4. relation canonical signed margin
5. fused canonical signed margin
6. question-swap choice-change
7. fused cross-view selected-choice agreement
8. whole-signature same-option cosine
9. whole-signature discrimination margin
10. lower canonical decision loss
11. earlier epoch

Anchor diagnostics cannot alter selector after DEV exposure.

## DEV_READY gates

All existing frontier gates remain required:
- fused canonical >= **0.85**
- paired >= **0.75**
- question-swap >= **0.80**
- fused agreement >= **0.95**
- fused JS <= **0.05**
- fused canonical margin >= **0.15**
- relation canonical >= **0.80**
- relation canonical margin >= **0.15**
- whole-signature same-option cosine >= **0.90**
- whole-signature discrimination margin >= **0.15**
- option-order / mass / full-K / state-once / ownership / capacity gates PASS

S28 additionally records, but does not gate-tune on:
- role-anchor cross-view cosine
- value-anchor cross-view cosine
- role-anchor same-vs-wrong margin
- value-anchor same-vs-wrong margin

## Preregistered interpretation

### A — DEV_READY
Anchor-level transport solves enough of the fresh binding/transport bottleneck to advance. Freeze candidate before any confirmatory/multilingual/Laya-Jev work.

### B — anchor metrics improve but relation discrimination remains weak
Latent transport is real but insufficient. Close S28; do not strengthen coefficient/margin after DEV.

### C — role anchor improves, value anchor remains weak
Residual localizes to value extraction. Close S28 and preregister a value-structure hypothesis.

### D — value anchor improves, role anchor remains weak
Residual localizes to query/role extraction. Close S28 and preregister a role-structure hypothesis.

### E — anchors align but output relation remains weak
The mapping from anchors to option relation is the bottleneck. Move away from transport-only losses.

### F — no material improvement/regression
Reject anchor transport as sufficient.

## Stop rule

Exactly one fresh S28 TRAIN/DEV authority.

After DEV exposure:
- no role/value anchor weight change
- no margin change
- no coefficient change
- no anchor formula change
- no seed/LR/epoch/batch retry
- no inference/capacity/fusion change
- no selector/gate change
- no second DEV run

Scientific FAIL is valid.

No sealed confirmation, multilingual transfer, or Laya/Jev benchmark unless DEV_READY.
