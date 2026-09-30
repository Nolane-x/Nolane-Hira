# HIRA V1 S23 contract — Neutral Bisector Norm-Balanced Optimization

Status: **OPEN / PREREGISTERED BEFORE S23-A0 EXPOSURE**

Issue: #229

Base main:

`94ba32fb7b997ba1d3f7d279656f9c9c52599609`

S22 is frozen as `HIRA_V1_S22_PRIMARY_PRIORITY_DEV_FAIL`.

## 1. Motivation

S21 relation-priority retained useful relation separation but underperformed S17:
- fused canonical **0.6145833333**
- paired **0.3697916667**
- relation margin **+0.2356135895**
- mean conflict **0.5998263889**

S22 primary-priority regressed further:
- fused canonical **0.5729166667**
- paired **0.28125**
- relation margin **+0.0338486681**
- mean conflict **0.5**

S23 is the final optimizer-priority ablation: remove asymmetric conflict projection entirely after norm equalization.

## 2. Frozen model / physical surface

Exactly S21/S22:
- role-gated content primary scorer
- role temperature **0.10**
- role weight **0.50**
- content weight **0.50**
- A13 LoRA **16,384**
- shared bias-free 256->128 projection **32,768**
- total trainable physical params **49,152**
- original A13 frozen
- HIRACore frozen
- relation refinement off
- learned downstream/router params **0**

Inference must be numerically identical to S21/S22.

## 3. Controlled optimizer change

For nonzero shared gradients:

`u_p = g_p / ||g_p||`

`u_r = g_r / ||g_r||`

No conflict projection is applied, regardless of the sign of:

`c = dot(u_p, u_r)`.

Then:

`d = normalize(u_p + u_r)`

`s = 0.5 * (||g_p|| + ||g_r||)`

`g = s * d`

Frozen epsilon: **1e-12**.

Zero-gradient cases:
- both zero -> zero
- relation zero -> raw primary
- primary zero -> raw relation

No learned state/parameters are introduced.

## 4. Frozen training objective

Exactly S21/S22:
- canonical + paraphrase fused CE
- swap coefficient **0.25**
- swap margin **0.20**
- option-view InfoNCE coefficient **0.05**
- option-view temperature **0.10**
- fused cross-view symmetric JS coefficient **0.25**
- relation CE coefficient **0.10**
- relation-signature canonicalization coefficient **0.15**
- signature separation margin **0.20**
- relation role/pair/contrastive temperatures **0.10**
- S14 equal-weight standardized fusion, epsilon **1e-6**

No S18/S19/S20 intervention.

## 5. Frozen optimizer shell

- AdamW
- lr **2e-4**
- weight decay **0.01**
- batch size **16 semantic cases**
- epochs **24**
- grad clip **1.0**

Only conflict handling changes.

## 6. Required operator invariants

A0 must prove:
- 0 added params/state;
- exact analytical neutral-bisector direction on a controlled conflict tensor;
- primary/relation exchange symmetry;
- conflict does not project either expert;
- projection coefficient is exactly zero;
- no-conflict output matches S17/S21 normalized sum within **1e-7**;
- both-zero / primary-zero / relation-zero exact behavior;
- positive joint scale equivariance;
- finite near-opposite behavior;
- exact **49,152** physical surface.

## 7. S23-A0

Fresh English A0:
- **16 wholly fresh semantic cases**
- two state views
- two question views per semantic query
- K=4
- two option semantic views

Must prove:
- S21/S22 inference identity: logits and choices exact;
- S21 role/content synthetic court preserved;
- A13 token/pooled identity;
- exact 49,152 physical surface;
- runtime trainable during A0 = 0;
- factorization added params = 0;
- full-K/state-once;
- option-order equivariance;
- fusion mass error <= 1e-6;
- relation delta = 0;
- real shared-gradient neutral-bisector diagnostic finite.

A0 is diagnostic only.

## 8. Fresh TRAIN / DEV

Use wholly fresh S23 semantic authority:
- TRAIN **768 cases**
- DEV **192 cases**
- 12 domains not used by S22
- fresh lexical banks / wording families
- K=4
- two state views
- two question views
- two option views

Seed: **40001**

Forbidden:
- S0-S22 A0/TRAIN/DEV exact rows
- M5 final/confirmatory rows
- W29-W34 sealed rows

## 9. Frozen DEV selection

Lexicographic exactly S21/S22:
1. fused paired both-correct
2. fused canonical accuracy
3. fused paraphrase accuracy
4. question-swap choice-change
5. fused cross-view selected-choice agreement
6. primary canonical accuracy
7. relation canonical accuracy
8. relation signed margin
9. fused signed margin
10. same-option signature cosine
11. signature same-vs-wrong margin
12. lower canonical decision loss
13. earlier epoch

## 10. DEV_READY gate

Exactly S21/S22:
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
- no added optimizer/model parameters

Scientific FAIL is valid.

## 11. Stop rule

If S23 does not materially improve the S17/S21 frontier:
- close the optimizer-priority family;
- do not create S24 as another priority/projection variant;
- return to representation/fusion hypotheses.

No post-DEV:
- optimizer retry
- role temperature/weight tuning
- seed/LR/template retry
- gate weakening

No Laya/Jev, sealed confirmation or multilingual transfer unless DEV_READY.
