# HIRA V1 S13 contract — Cross-View Relation Canonicalization

Status: **OPEN / PREREGISTERED BEFORE S13-A0 EXPOSURE**

Issue: #201

Base main:

`77a52f551628b22f08d219a6561806604e1513e9`

S12 is frozen as `HIRA_V1_S12_RELATION_BINDING_DEV_FAIL`.

## 1. Motivation

S12 learned explicit role-relative relation geometry without a token-distance prior, yet fresh DEV remained strongly wording dependent.

Selected S12 DEV:
- canonical accuracy: **0.28125**
- paired both-correct: **0.109375**
- question-swap choice-change: **0.6302083333**
- cross-view decision agreement: **0.3802083333**
- canonical relation-binding accuracy: **0.4010416667**
- paraphrase relation-binding accuracy: **0.5390625**
- canonical signed relation margin: **-0.7074106541**
- paraphrase signed relation margin: **-0.1640277983**
- relation-binding cross-view agreement: **0.4036458333**

The canonical/paraphrase asymmetry is the main target. S13 tests whether explicitly aligning relation identity across equivalent wording views can make the learned geometry wording-stable.

## 2. Frozen physical model surface

Trainable:
- A13 final-attention LoRA: **16,384**
- shared bias-free 256->128 projection: **32,768**

Total: **49,152 trainable parameters**.

Frozen:
- all original A13 parameters
- HIRACore
- reliability/calibration
- adaptive budget
- relation refinement

Excluded:
- W34
- learned canonicalization head
- learned downstream scorer
- task/domain/language-specific head

S13 canonicalizer adds **0 learned parameters**.

## 3. Frozen S13 canonical relation signature

For each state/question/option view:

1. project state/question/option tokens through the existing shared projection;
2. derive a query-conditioned state-role anchor;
3. derive a query-conditioned option-role anchor;
4. express all state and option tokens as normalized residuals relative to their role anchors;
5. score all valid state-token <-> option-token pairs by equal-weight direct semantic cosine + role-relative cosine;
6. use a fixed soft pair distribution to keep the operator differentiable;
7. aggregate supported state residual, supported option residual, and role-anchor delta;
8. normalize to one relation signature per option view;
9. average only active semantic views and renormalize to one relation signature per logical option.

Frozen temperatures:
- role: **0.10**
- pair: **0.10**
- contrastive: **0.10**

## 4. Cross-view canonicalization objective

For semantically equivalent canonical/paraphrase wording views of the same case/query:
- align the same logical option signature by cosine distance;
- repel every same-case wrong option with fixed margin **0.20**.

This objective must operate only through existing LoRA/projection weights.

Primary Hira decision path remains unchanged.

## 5. S13-A0

Fresh English localization authority:
- 16 wholly fresh semantic cases
- two semantically equivalent state wording views
- two question wording views per semantic query
- K=4
- two semantic option views

Required:
- exact A13 token identity
- exact A13 pooled identity
- exact inherited primary-decision logit identity
- exact inherited primary selected-choice identity
- candidate physical capacity = 49,152
- runtime trainable during A0 = 0
- canonicalizer added params = 0
- state-once
- full-K
- primary option-order invariance
- relation delta = 0
- probability-mass error <=1e-6
- signature option-permutation equivariance
- record same-option cross-view signature cosine and wrong-option separation diagnostics
- A0 cannot be used for model selection

Only qualified A0 may authorize TRAIN/DEV.

## 6. Fresh TRAIN / DEV

Frozen intended partitions:
- TRAIN: 768 wholly fresh semantic cases
- DEV: 192 wholly fresh semantic cases
- 12 fresh domains
- fresh lexical banks
- unseen DEV wording families
- K=4
- two state views
- two question wording views per semantic query
- two semantic views per option

Forbidden:
- every S0-S12 localization/A0/TRAIN/DEV row
- M5 final/confirmatory rows
- W29-W34 sealed rows

## 7. Frozen optimizer and objective

- seed: **19001**
- AdamW
- 24 epochs
- batch size: 16 semantic cases
- lr: 2e-4
- weight decay: 0.01
- grad clip: 1.0

Objective:
- canonical + paraphrase primary decision CE
- paired swap-margin coefficient 0.25 / margin 0.20
- option-view InfoNCE coefficient 0.05 / temperature 0.10
- S12 relation-binding CE coefficient **0.10**
- S13 cross-view relation-signature canonicalization coefficient **0.15**
- S13 signature separation margin **0.20**
- primary cross-view symmetric JS coefficient 0.25

No S10 pooled-grounding loss or S11 fixed-window loss is used.

## 8. Frozen DEV selection order

1. canonical paired both-correct
2. canonical accuracy
3. canonical relation-binding accuracy
4. mean same-option relation-signature cosine
5. cross-view selected-choice agreement
6. canonical question-swap choice-change
7. larger canonical relation-binding signed margin
8. lower canonical decision loss
9. earlier epoch

## 9. Frozen DEV gate

`HIRA_V1_S13_CANONICAL_RELATION_DEV_READY` requires all:
- canonical accuracy >=0.85
- canonical paired both-correct >=0.75
- canonical question-swap choice-change >=0.80
- cross-view selected-choice agreement >=0.95
- cross-view mean JS <=0.05
- canonical relation-binding accuracy >=0.80
- canonical signed relation margin >=0.15
- mean same-option signature cosine >=0.90
- mean same-option-vs-strongest-wrong signature margin >=0.15
- option-order flip <=0.02
- probability-mass error <=1e-6
- full-K
- relation delta = 0
- state-once
- exactly 49,152 trainable params
- original A13 trainable = 0
- HIRACore trainable = 0
- canonicalizer/downstream learned params = 0

A scientific FAIL is valid and must be frozen.
No post-DEV coefficient/temperature/seed/template retry or gate weakening is authorized.

## 10. Sealed boundary

Only DEV READY may open:
1. one-shot sealed English confirmation
2. separately preregistered zero-training Vietnamese transfer

S13 alone cannot claim Laya/Jev parity, multilingual qualification, reliability/OOD readiness, or production readiness.
