# HIRA V1 S22 contract — Primary-Priority Norm-Balanced Optimization

Status: **OPEN / PREREGISTERED BEFORE S22-A0 EXPOSURE**

Issue: #227

Base main:

`251f35eb42d01936c04f8559508dac66cd8f9998`

S21 is frozen as `HIRA_V1_S21_ROLE_GATED_CONTENT_DEV_FAIL`.

## 1. Motivation

S21 preserved useful role/content factorization evidence but exposed a strong optimizer-priority mismatch.

S21 selected:
- fused canonical **0.6145833333**
- fused paraphrase **0.5651041667**
- paired **0.3697916667**
- question-swap **0.8333333333**
- fused agreement **0.6588541667**
- primary canonical/paraphrase **0.5807291667 / 0.5182291667**
- relation canonical/paraphrase **0.6484375 / 0.5963541667**
- relation canonical margin **+0.2356135895**

S21 used S17 relation-priority normalized conflict projection.

Gradient conflict:
- S17 mean: ~**0.3220**
- S21 mean: **0.5998263889**
- S21 selected epoch 13: **0.7291666667**
- S21 epoch 24: **0.8125**

Therefore S22 tests one optimizer variable only: which expert direction is protected during conflict.

## 2. Frozen model / physical surface

Exactly S21:
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

Inference is numerically identical to S21.

## 3. Controlled optimizer change

Let raw shared gradients be `g_p` and `g_r`.

For nonzero gradients:

`u_p = g_p / ||g_p||`

`u_r = g_r / ||g_r||`

Let:

`c = dot(u_p, u_r)`

If `c < 0`:

`u_r' = u_r - c * u_p`

and **u_p is left unchanged**.

Otherwise:

`u_r' = u_r`.

Then:

`d = normalize(u_p + u_r')`

`s = 0.5 * (||g_p|| + ||g_r||)`

`g = s * d`

Frozen epsilon:

**1e-12**

Zero-gradient cases are exactly inherited:
- both zero -> zero
- relation zero -> raw primary
- primary zero -> raw relation

No-conflict behavior must be numerically identical to S17/S21.

## 4. Frozen training objective

Exactly S21 / S17:
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

No S18 paired-margin term.
No S19 raw-triadic JS replacement.
No S20 standardized-triadic consistency replacement.

## 5. Frozen optimizer shell

- AdamW
- lr **2e-4**
- weight decay **0.01**
- batch size **16 semantic cases**
- epochs **24**
- grad clip **1.0**

Only the normalized conflict-priority rule changes.

## 6. Required operator invariants

A0 must prove:
- primary-priority operator adds 0 parameters/state;
- exact analytical result on a controlled conflict tensor;
- conflict leaves normalized primary direction unchanged;
- conflict changes relation direction;
- post-projection primary↔relation dot >= **-1e-7**;
- no-conflict output matches S17 norm-balanced update within **1e-7**;
- both-zero / primary-zero / relation-zero behavior exact;
- joint positive scale equivariance;
- finite outputs.

## 7. S22-A0

Fresh English A0:
- **16 wholly fresh semantic cases**
- two state views
- two question views per semantic query
- K=4
- two option semantic views

Must prove:
- S21 inference identity: logits and choices exact between S21 and S22 frozen cores;
- S21 role/content synthetic court preserved;
- A13 token/pooled identity;
- exact 49,152 physical surface;
- runtime trainable during A0 = 0;
- factorization added params = 0;
- full-K/state-once;
- option-order equivariance;
- fusion mass error <=1e-6;
- relation delta = 0;
- real shared-gradient primary-priority diagnostic is finite.

A0 is diagnostic only.

## 8. Fresh TRAIN / DEV

Use wholly fresh S22 semantic authority:
- TRAIN **768 cases**
- DEV **192 cases**
- 12 domains not used by S21
- fresh lexical banks / wording families
- K=4
- two state views
- two question views
- two option views

Seed:

**38001**

Forbidden:
- S0-S21 A0/TRAIN/DEV exact rows
- M5 final/confirmatory rows
- W29-W34 sealed rows

## 9. Frozen DEV selection

Lexicographic exactly S21:
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

Exactly S21:
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

No post-DEV:
- optimizer-priority retry
- role temperature/weight tuning
- seed/LR/template retry
- gate weakening

No Laya/Jev, sealed confirmation or multilingual transfer unless DEV_READY.
