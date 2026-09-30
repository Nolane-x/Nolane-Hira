# HIRA V1 S20 contract — Standardized Triadic Evidence Consistency

Status: **OPEN / PREREGISTERED BEFORE S20-A0 EXPOSURE**

Issue: #223

Base main:

`02d368380730e3144216a654edbef8b3022a4bd2`

S19 is frozen as `HIRA_V1_S19_TRIADIC_VIEW_CONSISTENCY_DEV_FAIL`.

## 1. Motivation

S19 directly regularized raw-triadic canonical/paraphrase logits with symmetric softmax JS.

The intervention was effectively inert:
- TRAIN raw-triadic JS remained only ~4.6e-9 to ~1.15e-8 across all 24 epochs;
- selected DEV raw-triadic top-1 agreement was only 0.5494791667;
- DEV top-1 agreement moved widely despite near-zero raw JS.

The triadic logits are sufficiently flat that softmax probability vectors can be almost identical while tiny relative logit changes alter ranking.

## 2. Frozen physical / inference surface

Trainable:
- A13 final-attention LoRA: **16,384**
- shared bias-free 256->128 projection: **32,768**
- total: **49,152**

Frozen:
- original A13
- HIRACore
- reliability/calibration
- adaptive budget
- relation refinement

No learned consistency/head/router is added.

Inference remains exactly S14-S19:

`fused = 0.5 * standardized(triadic) + 0.5 * standardized(relation)`

Fusion epsilon: **1e-6**.

## 3. Frozen optimizer

Retain S17 norm-balanced shared-gradient optimization exactly:
- unit-normalize primary/relation shared gradients;
- relation-priority projection if normalized directions conflict;
- equal directional combination;
- reference scale = arithmetic mean raw norm;
- balance epsilon **1e-12**;
- AdamW;
- 24 epochs;
- batch size 16 semantic cases;
- lr **2e-4**;
- weight decay **0.01**;
- final grad clip **1.0**.

No moving average, learned router, per-module weighting or history state.

## 4. Frozen standardized triadic consistency

For each raw triadic full-K vector `x`:

`c = x - mean_K(x)`

`rms = sqrt(mean_K(c^2))`

Frozen standardization epsilon:

**1e-6**

If `rms <= epsilon`:
- `z = 0` (neutral flat evidence)

Otherwise:
- `z = c / rms`

The implementation must reuse `standardize_full_k_evidence` from S14 evidence fusion.

For canonical/paraphrase standardized triadic vectors `z_c, z_p`:

`L_std = mean((z_c - z_p)^2)`

Frozen coefficient:

**0.25**

No temperature.
No fitted scale.
No learned projection.

## 5. Frozen objective

Return to S17 objective except for the consistency target.

Fused decision:
- canonical/paraphrase CE
- swap coefficient **0.25**
- swap margin **0.20**

Option alignment:
- coefficient **0.05**
- temperature **0.10**

Relation block:
- relation CE coefficient **0.10**
- relation-signature canonicalization coefficient **0.15**
- signature separation margin **0.20**
- role/pair/relation temperatures **0.10**

S20 primary:

`primary = fused_decision + 0.05*option_alignment + 0.25*standardized_triadic_consistency`

S20 relation:

`relation = 0.10*relation_CE + 0.15*signature_canonicalization`

S17 fused-output JS is absent.
S18 paired-margin term is absent.
S19 raw-softmax triadic JS is absent.

## 6. Standardized-consistency invariants

Required:
- 0 learned parameters/state;
- finite input validation inherited from S14 standardization;
- exact zero for identical evidence vectors;
- exact symmetry under canonical/paraphrase swap;
- exact option permutation invariance;
- positive affine invariance within tolerance: for `a > 0`, standardization of `a*x+b` equals standardization of `x` for non-flat evidence;
- flat evidence becomes all-zero neutral evidence with no NaN;
- mismatched relative evidence gives finite nonzero gradients to both views;
- common positive scaling cannot suppress the consistency signal.

## 7. S20-A0

Fresh English A0:
- 16 wholly fresh semantic cases;
- two state views;
- two question wording views per semantic query;
- K=4;
- two semantic option views.

Must prove:
- exact A13 token/pooled identity;
- exact inherited raw-triadic logits/choices;
- S20 inference == S17-S19;
- exact 49,152 physical surface;
- runtime trainable during A0 = 0;
- standardized consistency learned params/state = 0;
- all invariants in §6;
- full-K;
- state-once;
- option permutation equivariance;
- fused mass error <=1e-6.

A0 is diagnostic only.

Only qualified A0 may authorize TRAIN/DEV.

## 8. Fresh TRAIN / DEV

- TRAIN: **768 wholly fresh semantic cases**
- DEV: **192 wholly fresh semantic cases**
- 12 fresh domains
- fresh lexical banks
- unseen DEV wording families
- K=4
- two state views
- two question views per semantic query
- two semantic option views per option

Forbidden:
- S0-S19 A0/TRAIN/DEV rows
- M5 final/confirmatory rows
- W29-W34 sealed rows

## 9. Seed and selection

Seed:

**34001**

Frozen DEV selection order:
1. fused paired both-correct
2. fused canonical accuracy
3. fused paraphrase accuracy
4. fused cross-view selected-choice agreement
5. raw-triadic cross-view agreement
6. relation canonical accuracy
7. relation signed margin
8. fused signed margin
9. question-swap
10. same-option signature cosine
11. signature same-vs-wrong margin
12. lower canonical decision loss
13. earlier epoch

## 10. DEV_READY gate

All required:
- fused canonical >= **0.85**
- paired >= **0.75**
- question-swap >= **0.80**
- fused cross-view selected-choice agreement >= **0.95**
- fused cross-view mean JS <= **0.05**
- fused canonical signed margin >= **0.15**
- relation canonical >= **0.80**
- relation signed margin >= **0.15**
- same-option signature cosine >= **0.90**
- signature same-vs-wrong margin >= **0.15**
- option-order flip <= **0.02**
- probability-mass error <= **1e-6**
- full-K
- state-once
- relation delta = 0
- exact 49,152 trainable params
- original A13/HIRACore frozen
- no learned consistency/balancing/fusion/canonicalizer/downstream params

Scientific FAIL is valid.

No post-DEV:
- standardized-loss coefficient/epsilon/location tuning
- balancing changes
- fusion/temperature tuning
- seed/LR/template retry
- gate weakening

Only DEV_READY may open sealed English confirmation and separately preregistered zero-training Vietnamese transfer.
