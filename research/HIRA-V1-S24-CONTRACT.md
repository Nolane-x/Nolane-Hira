# HIRA V1 S24 contract — Reliability-Weighted Full-K Fusion

Status: **OPEN / PREREGISTERED BEFORE S24-A0 EXPOSURE**

Issue: #231

Base main:
`f7228c1cff5440504644a6f03f76698fa79f7af9`

S23 is frozen as `HIRA_V1_S23_NEUTRAL_BISECTOR_DEV_FAIL`.

The optimizer-priority family is CLOSED. S24 changes fusion only.

## 1. Motivation

S23 selected DEV:
- primary canonical **0.5651041667**
- relation canonical **0.6875**
- equal-weight fused canonical **0.6432291667**
- primary paraphrase **0.5625**
- relation paraphrase **0.6276041667**
- equal-weight fused paraphrase **0.609375**

The equal S14 0.5/0.5 fusion can suppress the stronger expert.

## 2. Frozen model / optimizer

Exactly S23:
- role-gated content primary scorer
- S13 relation expert/canonicalizer
- role temperature **0.10**
- role/content weights **0.50 / 0.50**
- S23 neutral-bisector norm-balanced optimizer
- no asymmetric gradient projection
- A13 LoRA **16,384**
- shared projection **32,768**
- exact trainable surface **49,152**
- original A13/HIRACore frozen
- relation refinement off
- learned downstream/router params **0**

Primary and relation logits must remain numerically identical to S23 for the same frozen state.

## 3. Controlled fusion change

Let `z_p, z_r` be S14 center/RMS-standardized full-K logits.

Frozen epsilon:
`eps = 1e-6`.

Reliability:
- `gap_p = top1(z_p) - top2(z_p)`
- `gap_r = top1(z_r) - top2(z_r)`
- `rho_p = gap_p + eps`
- `rho_r = gap_r + eps`

Weights:
- `w_p = rho_p / (rho_p + rho_r)`
- `w_r = rho_r / (rho_p + rho_r)`

Fusion:
`fused = w_p * z_p + w_r * z_r`

Relation logits are detached from the fused-primary objective exactly as in S15-S23.

No learned gate/calibrator/head/temperature/coefficient.

## 4. Required properties

A0 must prove:
- zero learned params/state;
- expert-swap numerical symmetry;
- option-permutation equivariance;
- equal reliability -> exact S14 equal fusion;
- flat/flat -> zero neutral fused vector with 0.5/0.5 weights;
- flat/nonflat -> finite and nonflat expert weight > 0.999;
- positive independent affine invariance within **2e-6**;
- weights finite/nonnegative and sum to 1 within **1e-7**;
- relation direct fused-primary gradient = 0;
- primary fused-primary gradient nonzero;
- full-K retained.

## 5. Frozen training objective

Exactly S23 except fusion implementation:
- fused canonical + paraphrase CE
- swap coefficient **0.25**
- swap margin **0.20**
- option alignment **0.05**
- option temperature **0.10**
- fused cross-view symmetric JS **0.25**
- relation CE **0.10**
- signature canonicalization **0.15**
- signature separation margin **0.20**

## 6. Frozen optimizer shell

Exactly S23:
- neutral-bisector norm-balanced gradient
- epsilon **1e-12**
- AdamW lr **2e-4**
- weight decay **0.01**
- batch size **16**
- epochs **24**
- grad clip **1.0**

## 7. A0

Use **16 wholly fresh English cases**.

Must prove:
- A13 token/pooled identity;
- S23 primary logits/choices identity **1.0/1.0**;
- S23 relation logits identity;
- only fusion output changes;
- reliability-fusion operator properties above;
- role/content synthetic court preserved;
- exact 49,152 physical surface;
- runtime trainable during A0 = 0;
- full-K/state-once;
- option permutation;
- mass error <=1e-6.

A0 is diagnostic only.

## 8. Fresh TRAIN / DEV

- TRAIN **768**
- DEV **192**
- 12 wholly fresh S24 domains
- K=4
- two state views
- two question views
- two option views
- seed **42001**

Forbidden:
- exact S0-S23 exposed rows
- M5 final/confirmatory rows
- W29-W34 sealed rows

## 9. DEV selection / gates

Use exactly the S23 lexicographic selection and DEV_READY gates.

Scientific FAIL is valid.

No post-DEV:
- reliability formula change
- epsilon change
- learned/fitted calibration
- optimizer change
- role weight/temperature change
- seed/LR/template retry
- gate weakening

No Laya/Jev, sealed confirmation or multilingual transfer unless DEV_READY.
