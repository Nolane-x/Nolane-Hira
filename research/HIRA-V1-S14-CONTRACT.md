# HIRA V1 S14 contract — Canonical Relation Evidence Fusion

Status: **OPEN / PREREGISTERED BEFORE S14-A0 EXPOSURE**

Issue: #204

Base main:

`cc388a85c6c6a797a95661e99cbfb1a8d69fc5d5`

S13 is frozen as `HIRA_V1_S13_CANONICAL_RELATION_DEV_FAIL`.

## 1. Motivation

S13 established a clear gap between the unchanged primary triadic decision path and the parameter-free relation path.

Selected S13 DEV:
- primary canonical accuracy: **0.2942708333**
- primary paraphrase accuracy: **0.2786458333**
- canonical relation-binding accuracy: **0.4921875**
- paraphrase relation-binding accuracy: **0.5338541667**
- canonical relation margin: **-0.1725222593**
- same-option signature cosine: **0.7653450121**
- signature margin: **0.0656896873**

S13 TRAIN:
- relation-binding loss: **1.39382457 -> 0.57022986**
- canonicalization loss: **0.31337504 -> 0.12005968**

The strongest current hypothesis is that usable semantic relation evidence exists but the legacy primary operator does not exploit it.

## 2. Frozen physical surface

Trainable:
- A13 final-attention LoRA: **16,384 params**
- shared bias-free 256->128 projection: **32,768 params**

Total: **49,152 trainable parameters**.

Frozen:
- all original A13 parameters
- HIRACore
- reliability/calibration
- adaptive budget
- relation refinement

Excluded:
- learned fusion head
- learned gate
- fitted mixture coefficient
- option-specific fusion parameter
- task/domain/language-specific head
- W34

S14 fusion adds **0 learned parameters**.

## 3. Frozen full-K fusion rule

For raw triadic logits `t` and S13 canonical-relation logits `r`, each [B,K]:

For each expert independently:
1. subtract its mean over K;
2. compute RMS of the centered vector;
3. if RMS > epsilon, divide by RMS;
4. otherwise emit the all-zero neutral vector.

Frozen epsilon:

`1e-6`

Then:

`fused = 0.5 * standardized(t) + 0.5 * standardized(r)`

No learned or fitted coefficient.
No expert-specific temperature.
No DEV-selected scaling.

The fused logits become the S14 primary decision surface.

Raw triadic and raw relation logits remain diagnostics.

## 4. Fusion invariants

Required:
- exact expert-swap symmetry
- option permutation equivariance
- per-expert positive affine scale/shift invariance
- a flat expert contributes exactly zero standardized evidence
- full-K
- finite logits
- added learned params = 0

## 5. S14-A0

Fresh English identity/localization:
- 16 wholly fresh semantic cases
- two state wording views
- two question wording views per semantic query
- K=4
- two semantic option views

Required:
- exact A13 token identity
- exact A13 pooled identity
- exact inherited raw triadic-logit identity
- exact inherited raw triadic selected-choice identity
- physical candidate surface = 49,152
- runtime trainable during A0 = 0
- fusion added params = 0
- state-once
- raw/fused full-K
- raw primary option-order flip = 0
- fused option-order flip = 0
- relation delta = 0
- probability mass error <=1e-6 for fused softmax
- record raw triadic, raw relation and fused accuracy/margin/agreement
- A0 is never used for model selection

Only a qualified A0 may authorize TRAIN/DEV.

## 6. Fresh TRAIN / DEV

Frozen intended partitions:
- TRAIN: **768 wholly fresh semantic cases**
- DEV: **192 wholly fresh semantic cases**
- 12 fresh domains
- fresh lexical banks
- unseen DEV wording families
- K=4
- two state views
- two question wording views per semantic query
- two semantic option views

Forbidden:
- every S0-S13 localization/A0/TRAIN/DEV row
- M5 final/confirmatory rows
- W29-W34 sealed rows

## 7. Frozen optimizer and objective

- seed: **20001**
- AdamW
- 24 epochs
- batch size: 16 semantic cases
- lr: 2e-4
- weight decay: 0.01
- grad clip: 1.0

Objective:
- fused canonical + paraphrase primary CE
- fused paired swap-margin coefficient **0.25**, margin **0.20**
- option-view InfoNCE coefficient **0.05**, temperature **0.10**
- canonical-relation CE coefficient **0.10**
- cross-view relation-signature canonicalization coefficient **0.15**
- signature separation margin **0.20**
- fused cross-view symmetric JS coefficient **0.25**

Important:
- raw triadic CE is not separately optimized in S14;
- the fused primary objective sends gradients through both the triadic and relation paths;
- S13 relation/canonicalization auxiliaries are retained without changing coefficients.

## 8. Frozen DEV selection order

1. fused canonical paired both-correct
2. fused canonical accuracy
3. fused question-swap choice-change
4. fused cross-view selected-choice agreement
5. canonical relation-binding accuracy
6. fused signed gold-vs-max-wrong margin
7. mean same-option relation-signature cosine
8. mean signature same-vs-strongest-wrong margin
9. lower fused canonical decision loss
10. earlier epoch

No raw-triadic metric is allowed to override fused-primary selection.

## 9. Frozen DEV gate

`HIRA_V1_S14_EVIDENCE_FUSION_DEV_READY` requires all:
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
- fused probability mass error <= **1e-6**
- full-K
- relation delta = 0
- state-once
- exact 49,152 trainable params
- original A13 trainable = 0
- HIRACore trainable = 0
- fusion/canonicalizer/downstream learned params = 0

A scientific FAIL is valid and must be frozen.

No post-DEV:
- fusion coefficient tuning
- epsilon tuning
- temperature tuning
- loss-weight tuning
- seed/LR/template retry
- gate weakening

## 10. Sealed boundary

Only DEV READY may open:
1. one-shot sealed English confirmation
2. separately preregistered zero-training Vietnamese transfer

S14 alone cannot claim Laya/Jev parity, multilingual qualification, reliability/OOD readiness, or production readiness.
