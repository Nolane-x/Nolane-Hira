# HIRA V1 S21 contract — Role-Gated Content Triadic Scoring

Status: **OPEN / PREREGISTERED BEFORE S21-A0 EXPOSURE**

Issue: #225

Base main:

`e822648da309b07f989e112809d1db3ea81008bb`

S20 is frozen as `HIRA_V1_S20_STANDARDIZED_TRIADIC_CONSISTENCY_DEV_FAIL`.

## 1. Motivation

S17 remains the strongest semantic frontier.

S18-S20 all tested stronger cross-view consistency pressure:
- S18 output-level paired margin;
- S19 raw-softmax triadic JS;
- S20 standardized triadic evidence MSE.

S20 proved that the consistency intervention itself was active and differentiable, yet fresh DEV regressed. Therefore S21 does **not** add another consistency auxiliary.

The current primary triadic operator directly mixes state/question/option coordinates and only separates them through symmetric aggregation. It does not explicitly represent:
- the queried semantic role/field;
- the bound content/value.

S21 tests whether primary scoring improves when those two factors are separated structurally.

## 2. Frozen physical surface

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

S21 role/content factorization adds:
- learned params: **0**
- learned state: **0**
- router params: **0**

## 3. S21 role-gated content scorer

All projected token vectors are L2-normalized through the inherited shared projection.

For state tokens `s_i`, question tokens `q_j`, and option-view tokens `o_t`:

### 3.1 Role selection

`state_role_score(i) = max_j q_j · s_i`

`option_role_score(t) = max_j q_j · o_t`

Masked softmax temperature:

**0.10**

produces one role distribution over state tokens and one role distribution over tokens of each active option view.

### 3.2 Role anchors

`a_s = normalize(sum_i w_s(i) s_i)`

`a_o = normalize(sum_t w_o(t) o_t)`

Role compatibility per option view:

`R = a_s · a_o`

### 3.3 Content residualization

Remove the selected role direction by orthogonal projection:

`c_s(i) = normalize(s_i - (s_i·a_s)a_s)`

`c_o(t) = normalize(o_t - (o_t·a_o)a_o)`

Masked/degenerate residuals must remain finite.

### 3.4 Symmetric content binding

On `c_s` and `c_o`, reuse W28-style bidirectional max/mean aggregation:
- option token -> strongest valid state token, then mean option tokens;
- state token -> strongest valid option token, then mean state tokens;
- equal 0.5 / 0.5 average.

Call this `C`.

### 3.5 Per-view / per-option score

Frozen equal factor weights:

`view_score = 0.5 * R + 0.5 * C`

Average active semantic views arithmetically.

No learned gate or temperature beyond the fixed role temperature.

## 4. Frozen surrounding inference

Relation expert remains the S13 canonical relation expert.

Fusion remains S14 equal-weight standardized fusion:

`fused = 0.5 * standardized(S21_primary) + 0.5 * standardized(relation)`

Fusion epsilon: **1e-6**.

S21 intentionally changes only the primary triadic scorer. This is a preregistered inference change justified by the structural hypothesis.

## 5. Frozen training objective

Return exactly to the S17 training objective around the new primary scorer.

Primary:
- canonical + paraphrase fused CE
- swap coefficient **0.25**
- swap margin **0.20**
- option-view InfoNCE coefficient **0.05**
- option-view InfoNCE temperature **0.10**
- fused cross-view symmetric JS coefficient **0.25**

Relation:
- relation CE coefficient **0.10**
- relation-signature canonicalization coefficient **0.15**
- signature separation margin **0.20**
- role/pair/contrastive temperatures **0.10**

Do not use:
- S18 paired-margin term
- S19 raw-triadic JS
- S20 standardized-triadic consistency

## 6. Frozen optimizer

Retain S17 norm-balanced shared-gradient optimization exactly:
- unit-normalize primary and relation shared gradients;
- relation-priority projection on directional conflict;
- equal-direction combination;
- reference scale = arithmetic mean raw norm;
- epsilon **1e-12**;
- AdamW;
- lr **2e-4**;
- weight decay **0.01**;
- batch size **16 semantic cases**;
- epochs **24**;
- grad clip **1.0**.

## 7. Operator invariants

Required:
- factorization added params/state = 0;
- finite input/mask validation;
- option permutation equivariance;
- semantic-view permutation invariance;
- identical duplicated option views preserve score;
- inactive views do not affect score;
- role softmax respects state/question/option masks;
- degenerate residual geometry has no NaN/Inf;
- role temperature fixed at 0.10;
- role/content weights fixed 0.5 / 0.5.

Synthetic factorization court must prove:
1. same-role + same-content option > same-role + wrong-content;
2. same-role + same-content > wrong-role + same-content;
3. changing only the question role changes role selection and selected option on a controlled tensor court.

## 8. S21-A0

Fresh English A0:
- 16 wholly fresh semantic cases;
- two state views;
- two question wording views per query;
- K=4;
- two semantic option views.

Must prove:
- A13 token/pooled identity;
- exact physical 49,152 surface;
- runtime trainable during A0 = 0;
- factorized scorer added params = 0;
- all §7 operator invariants;
- full-K;
- state-once;
- option-order equivariance;
- fused mass error <=1e-6;
- relation delta = 0;
- S14 equal-weight fusion mechanics unchanged.

A0 is diagnostic only and cannot select/tune the model.

## 9. Fresh TRAIN / DEV

- TRAIN: **768 wholly fresh semantic cases**
- DEV: **192 wholly fresh semantic cases**
- 12 fresh domains
- fresh lexical banks and DEV wording families
- K=4
- two state views
- two question views per semantic query
- two semantic option views per option

Forbidden:
- all S0-S20 A0/TRAIN/DEV rows
- M5 final/confirmatory rows
- W29-W34 sealed rows

Seed:

**36001**

## 10. Frozen DEV selection

Lexicographic:
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

## 11. DEV_READY gate

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
- factorized scorer adds no learned params beyond the shared projection

Scientific FAIL is valid.

No post-DEV:
- role temperature tuning
- role/content weight tuning
- operator formula changes
- seed/LR/template retries
- gate weakening

Only DEV_READY may open sealed English confirmation and separately preregistered zero-training Vietnamese transfer.
