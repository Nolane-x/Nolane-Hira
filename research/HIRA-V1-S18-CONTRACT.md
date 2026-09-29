# HIRA V1 S18 contract — Paired Both-View Margin Consistency

Status: **OPEN / PREREGISTERED BEFORE S18-A0 EXPOSURE**

Issue: #217

Base main:

`6bef8d935ccdc71c9f3bb2d0899d05b8c7347714`

S17 is frozen as `HIRA_V1_S17_NORM_BALANCED_GRADIENT_DEV_FAIL`.

## 1. Motivation

S17 produced the strongest fresh DEV semantic result in v1:

- fused canonical accuracy: **0.7213541667**
- fused paraphrase accuracy: **0.5546875**
- paired both-correct: **0.5104166667**
- question-swap choice-change: **0.984375**
- fused signed margin: **0.3138313380**
- relation canonical accuracy: **0.640625**
- relation signed margin: **0.2073315941**

But:
- fused cross-view selected-choice agreement: **0.5286458333**
- paired both-correct remains **0.5104166667**

The remaining bottleneck is paired wording stability, not basic query sensitivity or gold-vs-wrong separation.

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

No learned consensus head/router is added.

Inference remains exactly S17:
`fused = 0.5 * standardized(triadic) + 0.5 * standardized(relation)`

Fusion epsilon remains **1e-6**.

## 3. Frozen optimizer

Retain S17 norm-balanced shared-gradient optimization exactly:
- unit-normalize primary/relation shared gradients
- relation-priority projection on normalized directional conflict
- equal directional combination
- rescale by arithmetic mean raw gradient norm
- balance epsilon **1e-12**
- AdamW
- 24 epochs
- batch size 16 semantic cases
- lr 2e-4
- weight decay 0.01
- final global grad clip 1.0

No moving average, learned router, per-module coefficient, or gradient-history state.

## 4. Frozen paired margin objective

For fused canonical logits `c`, fused paraphrase logits `p`, and gold option `y`:

For each view:
`m_c = c_y - max_{j != y}(c_j)`
`m_p = p_y - max_{j != y}(p_j)`

Frozen margin:
`M = 0.20`

Per-view hinge:
`h_c = relu(M - m_c)`
`h_p = relu(M - m_p)`

Paired loss:
`L_pair = 0.5 * (h_c + h_p)`

Frozen coefficient:
`0.25`

The S18 primary block becomes:

`primary = S17_primary + 0.25 * L_pair`

Relation block is unchanged from S17.

This coefficient is frozen before A0 and is not selected from S17 DEV.

## 5. Paired operator invariants

Required:
- adds 0 learned parameters/state
- finite input validation
- zero iff both views satisfy margin >=0.20
- only violating view logits receive hinge gradient
- if both violate, both receive gradient
- exact option permutation equivariance
- identical-view inputs reduce to ordinary gold-vs-hardest-wrong margin hinge
- batch mean semantics fixed

## 6. S18-A0

Fresh English A0:
- 16 wholly fresh semantic cases
- two state views
- two question wording views per semantic query
- K=4
- two semantic option views

Must prove:
- exact A13 token and pooled identity
- exact inherited raw triadic logit/choice identity
- S18 inference exactly equals S17
- exact 49,152 physical candidate surface
- runtime trainable during A0 = 0
- paired operator learned params/state = 0
- norm balancer learned params/state = 0
- paired-loss zero/nonzero cases
- paired-loss gradient routing
- full-K
- state-once
- option permutation equivariance
- probability-mass error <=1e-6

A0 is diagnostic only.

Only qualified A0 may authorize TRAIN/DEV.

## 7. Fresh TRAIN / DEV

- TRAIN: **768 wholly fresh semantic cases**
- DEV: **192 wholly fresh semantic cases**
- 12 fresh domains
- fresh lexical banks
- unseen DEV wording families
- K=4
- two state views
- two question wording views per semantic query
- two semantic option views per option

Forbidden:
- S0-S17 A0/TRAIN/DEV rows
- M5 final/confirmatory rows
- W29-W34 sealed rows

## 8. Frozen loss partition

S17 terms unchanged:
- fused canonical + paraphrase CE
- swap coefficient **0.25**, margin **0.20**
- option-view InfoNCE coefficient **0.05**, temperature **0.10**
- relation CE coefficient **0.10**
- relation-signature canonicalization coefficient **0.15**
- signature separation margin **0.20**
- fused cross-view JS coefficient **0.25**
- role/pair/relation temperatures **0.10**

S18 addition only:
- paired both-view margin coefficient **0.25**
- paired margin **0.20**

## 9. Seed and selection

Seed: **30001**.

Frozen DEV selection order:
1. fused paired both-correct
2. fused canonical accuracy
3. fused paraphrase accuracy
4. fused cross-view selected-choice agreement
5. relation canonical accuracy
6. relation signed margin
7. fused signed margin
8. question-swap
9. same-option signature cosine
10. signature same-vs-wrong margin
11. lower canonical decision loss
12. earlier epoch

## 10. DEV_READY gate

All required:
- fused canonical >= **0.85**
- fused paired both-correct >= **0.75**
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
- no learned paired/balancing/fusion/canonicalizer/downstream params

Scientific FAIL is valid.

No post-DEV:
- paired coefficient/margin tuning
- balancing changes
- epsilon/temperature tuning
- seed/LR/template retry
- gate weakening

Only DEV_READY may open sealed English confirmation and separately preregistered zero-training Vietnamese transfer.
