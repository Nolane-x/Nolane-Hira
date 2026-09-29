# HIRA V1 S19 contract — Expert-Separated Triadic View Consistency

Status: **OPEN / PREREGISTERED BEFORE S19-A0 EXPOSURE**

Issue: #221

Base main:

`b7e71b5bfbbb4402feca3fe38ccdefa801908ff0`

S18 is frozen as `HIRA_V1_S18_PAIRED_VIEW_MARGIN_DEV_FAIL`.

## 1. Motivation

S17 remains the strongest scientific frontier:
- fused canonical accuracy: **0.7213541667**
- fused paraphrase accuracy: **0.5546875**
- paired both-correct: **0.5104166667**
- relation canonical/paraphrase: **0.640625 / 0.6380208333**
- raw triadic canonical/paraphrase: **0.5911458333 / 0.4427083333**
- relation cross-view agreement: **0.6380208333**
- raw triadic cross-view agreement: **0.5208333333**

The relation expert was markedly more view-stable than the triadic expert.

S18's fused paired-margin term learned on TRAIN but degraded fresh paraphrase/relation generalization. S19 does not retain it.

## 2. Frozen physical / inference surface

Trainable:
- A13 final-attention LoRA: **16,384**
- shared bias-free 256->128 projection: **32,768**
- total: **49,152**

Frozen:
- original A13
- HIRACore
- relation refinement
- reliability/calibration
- adaptive budget

No learned consistency/head/router is added.

Inference remains exactly S17/S18:

`fused = 0.5 * standardized(triadic) + 0.5 * standardized(relation)`

Fusion epsilon: **1e-6**.

## 3. Frozen optimizer

Retain S17 norm-balanced shared-gradient optimization exactly:
- unit-normalize primary/relation shared gradients
- relation-priority projection if normalized directions conflict
- equal directional combination
- reference scale = arithmetic mean raw norm
- balance epsilon **1e-12**
- AdamW
- 24 epochs
- batch size 16
- lr 2e-4
- weight decay 0.01
- final grad clip 1.0

No moving average, learned router, per-module weighting or history state.

## 4. Frozen objective

Return to S17 objective and make one controlled consistency change.

S17 primary fused CE/swap remain unchanged.

Option alignment:
- coefficient **0.05**
- temperature **0.10**

Relation block unchanged:
- relation CE coefficient **0.10**
- relation-signature canonicalization coefficient **0.15**
- signature separation margin **0.20**
- role/pair/relation temperatures **0.10**

S19 cross-view consistency:
- compute symmetric JS on **raw triadic canonical/paraphrase logits**
- coefficient **0.25**

The S17 fused-output JS term is removed rather than duplicated.

S18 paired-margin term is absent.

Thus:

`primary = fused_decision + 0.05*option_alignment + 0.25*raw_triadic_JS`

`relation = 0.10*relation_CE + 0.15*signature_canonicalization`

## 5. Triadic consistency invariants

Required:
- adds 0 learned parameters/state
- JS is finite and non-negative
- JS = 0 for identical distributions
- exact symmetry under canonical/paraphrase swap
- exact option permutation invariance
- mismatched finite logits give nonzero finite gradient
- no relation logits participate in this JS term
- inference unchanged

## 6. S19-A0

16 wholly fresh English semantic cases:
- two state views
- two question wording views per semantic query
- K=4
- two semantic option views

Must prove:
- exact A13 token/pooled identity
- exact inherited raw triadic logits/choices
- S19 inference == S17/S18
- exact 49,152 physical surface
- runtime trainable during A0 = 0
- consistency operator params/state = 0
- triadic JS invariants above
- full-K
- state-once
- option permutation equivariance
- probability-mass error <=1e-6

A0 is diagnostic only.

Only qualified A0 may authorize TRAIN/DEV.

## 7. Fresh TRAIN / DEV

- TRAIN **768 wholly fresh semantic cases**
- DEV **192 wholly fresh semantic cases**
- 12 fresh domains
- fresh lexical banks
- unseen DEV wording families
- K=4
- two state views
- two question views per semantic query
- two semantic option views

Forbidden:
- S0-S18 A0/TRAIN/DEV rows
- M5 final/confirmatory rows
- W29-W34 sealed rows

## 8. Seed and selection

Seed: **32001**.

Frozen DEV selection:
1. fused paired both-correct
2. fused canonical accuracy
3. fused paraphrase accuracy
4. fused cross-view selected-choice agreement
5. raw triadic cross-view agreement
6. relation canonical accuracy
7. relation signed margin
8. fused signed margin
9. question-swap
10. signature cosine
11. signature margin
12. lower canonical decision loss
13. earlier epoch

## 9. DEV_READY gate

All required:
- fused canonical >= **0.85**
- paired >= **0.75**
- question-swap >= **0.80**
- fused cross-view choice agreement >= **0.95**
- fused cross-view mean JS <= **0.05**
- fused canonical signed margin >= **0.15**
- relation canonical >= **0.80**
- relation signed margin >= **0.15**
- signature cosine >= **0.90**
- signature margin >= **0.15**
- option-order flip <= **0.02**
- mass error <= **1e-6**
- full-K
- state-once
- relation delta = 0
- exact 49,152 trainable params
- original A13/HIRACore frozen
- no learned consistency/balancing/fusion/canonicalizer/downstream params

Scientific FAIL is valid.

No post-DEV:
- JS coefficient/location tuning
- balancing changes
- fusion/epsilon/temperature tuning
- seed/LR/template retry
- gate weakening

Only DEV_READY may open sealed English confirmation and separately preregistered zero-training Vietnamese transfer.
