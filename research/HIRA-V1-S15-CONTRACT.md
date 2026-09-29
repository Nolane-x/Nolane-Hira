# HIRA V1 S15 contract — Gradient-Isolated Evidence Fusion

Status: **OPEN / PREREGISTERED BEFORE S15-A0 EXPOSURE**

Issue: #207

Base main:

`8d750596505a915f5eb0771bd7b1b283d329d1d4`

S14 is frozen as `HIRA_V1_S14_EVIDENCE_FUSION_DEV_FAIL`.

## 1. Motivation

S14 established genuine zero-parameter fusion synergy on fresh DEV:

- fused canonical accuracy: **0.6354166667**
- relation canonical accuracy: **0.5833333333**
- raw triadic canonical accuracy: **0.5260416667**
- paired both-correct: **0.375**
- question-swap choice-change: **0.6979166667**
- fused canonical signed margin: **0.1034617118**
- fused cross-view selected-choice agreement: **0.5572916667**

But semantic relation stability remained weaker:
- same-option signature cosine: **0.7023841192**
- same-vs-strongest-wrong signature margin: **0.0355867463**
- relation canonical signed margin: **0.0116562198**

The controlled S15 hypothesis is that direct fused-primary gradients into the relation expert interfere with the dedicated relation/canonicalization objectives.

## 2. Frozen physical surface

Trainable:
- A13 final-attention LoRA: **16,384 params**
- shared bias-free 256->128 projection: **32,768 params**

Total: **49,152 trainable parameters**.

Frozen:
- every original A13 parameter
- HIRACore
- reliability/calibration
- adaptive budget
- relation refinement

Excluded:
- learned fusion head/gate
- fitted mixture coefficient
- learned gradient router
- task/domain/language-specific head
- W34

S15 adds **0 learned parameters**.

## 3. Frozen inference

S15 inference is numerically identical to S14.

For full-K raw triadic logits `t` and canonical-relation logits `r`:
- independently center each expert over K;
- divide non-flat centered vectors by own RMS magnitude;
- fixed epsilon: **1e-6**;
- flat expert -> all-zero neutral evidence;
- fuse with exact **0.5 / 0.5** equal weight.

`fused = 0.5 * standardized(t) + 0.5 * standardized(r)`

No inference coefficient, temperature, scale, or gate changes from S14.

## 4. Frozen training gradient route

Only the direct fused-primary autograd route changes:

`protected_fused = fusion(triadic_logits, stop_gradient(relation_logits))`

Therefore:
- fused primary CE: gradient directly reaches triadic logits only;
- fused paired swap-margin: direct gradient reaches triadic logits only;
- fused cross-view JS: direct gradient reaches triadic logits only;
- relation CE remains fully differentiable through relation logits;
- relation-signature canonicalization remains fully differentiable;
- option-view InfoNCE remains unchanged;
- relation and triadic experts still share the same underlying LoRA/projection physical parameters.

This is **gradient-route isolation**, not parameter isolation.

## 5. A0 requirements

Fresh English A0:
- 16 wholly fresh semantic cases
- two state wording views
- two question wording views per semantic query
- K=4
- two semantic option views

Required:
- exact A13 token identity
- exact A13 pooled identity
- exact inherited raw-triadic logit identity
- exact inherited raw-triadic selected-choice identity
- S15 fused forward exactly equals S14 fused forward for identical expert logits
- primary fused CE gives nonzero gradient to triadic logits
- primary fused CE gives zero direct gradient to relation logits
- dedicated relation CE gives nonzero gradient to relation logits
- physical candidate surface = 49,152
- runtime trainable during A0 = 0
- fusion added params = 0
- state-once
- full-K
- fused/raw option permutation equivariance
- relation delta = 0
- probability-mass error <= 1e-6
- A0 is diagnostic only and never used for model selection

Only qualified A0 may authorize TRAIN/DEV.

## 6. Fresh TRAIN / DEV

Frozen partitions:
- TRAIN: **768 wholly fresh semantic cases**
- DEV: **192 wholly fresh semantic cases**
- 12 fresh domains
- fresh lexical banks
- unseen DEV wording families
- K=4
- two state views
- two question wording views per semantic query
- two semantic views per option

Forbidden:
- every S0-S14 localization/A0/TRAIN/DEV row
- M5 final/confirmatory rows
- W29-W34 sealed rows

## 7. Frozen optimizer and objective

- seed: **22001**
- AdamW
- 24 epochs
- batch size: 16 semantic cases
- lr: 2e-4
- weight decay: 0.01
- grad clip: 1.0

Objective coefficients remain numerically identical to S14:
- protected fused canonical + paraphrase primary CE
- protected fused paired swap-margin coefficient **0.25**, margin **0.20**
- option-view InfoNCE coefficient **0.05**, temperature **0.10**
- canonical-relation CE coefficient **0.10**
- cross-view relation-signature canonicalization coefficient **0.15**
- signature separation margin **0.20**
- protected fused cross-view symmetric JS coefficient **0.25**

Raw triadic CE is not separately optimized.
No coefficient differs from S14.

## 8. Frozen DEV selection order

1. fused canonical paired both-correct
2. fused canonical accuracy
3. fused question-swap choice-change
4. fused cross-view selected-choice agreement
5. canonical relation-binding accuracy
6. fused signed gold-vs-max-wrong margin
7. canonical relation-binding signed margin
8. mean same-option relation-signature cosine
9. mean signature same-vs-strongest-wrong margin
10. lower fused canonical decision loss
11. earlier epoch

## 9. Frozen DEV gate

`HIRA_V1_S15_GRADIENT_ISOLATED_FUSION_DEV_READY` requires all:
- fused canonical accuracy >= **0.85**
- fused canonical paired both-correct >= **0.75**
- fused question-swap choice-change >= **0.80**
- fused cross-view selected-choice agreement >= **0.95**
- fused cross-view mean JS <= **0.05**
- fused canonical signed margin >= **0.15**
- canonical relation-binding accuracy >= **0.80**
- canonical relation-binding signed margin >= **0.15**
- mean same-option signature cosine >= **0.90**
- mean signature same-vs-strongest-wrong margin >= **0.15**
- fused option-order flip <= **0.02**
- fused probability-mass error <= **1e-6**
- full-K
- relation delta = 0
- state-once
- exact 49,152 trainable params
- original A13 trainable = 0
- HIRACore trainable = 0
- fusion/canonicalizer/downstream learned params = 0

Scientific FAIL is valid and must be frozen.

No post-DEV:
- detach-route changes
- loss-weight tuning
- fusion coefficient/epsilon tuning
- temperature tuning
- seed/LR/template retry
- gate weakening

## 10. Sealed boundary

Only DEV READY may open:
1. one-shot sealed English confirmation
2. separately preregistered zero-training Vietnamese transfer

S15 alone cannot claim Laya/Jev parity, multilingual qualification, reliability/OOD readiness, or production readiness.
