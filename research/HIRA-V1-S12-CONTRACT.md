# HIRA V1 S12 contract — Relation-Structured Role/Value Binding

Status: **OPEN / PREREGISTERED BEFORE S12-A0 EXPOSURE**

Issue: #198

Base main:

`9669fba4f4db805cf14ce1fb56ecbe8945348a04`

S11 is frozen as `HIRA_V1_S11_ROLE_BINDING_DEV_FAIL`.

## 1. Motivation

S11 improved query-role localization but failed to bind the selected role to the correct semantic value across fresh wording.

Selected S11 DEV:
- canonical accuracy: 0.3489583333;
- paired both-correct: 0.125;
- question-swap choice-change: 0.6458333333;
- canonical binding accuracy: 0.4192708333;
- canonical signed binding margin: -0.9078732127;
- binding cross-view agreement: 0.2057291667.

TRAIN binding loss fell from 1.37438 to 0.39169, while fresh DEV separation deteriorated after an early peak.

The failed structural assumption was the fixed positional value window. S12 tests whether **role-relative semantic relation geometry** can bind values independently of token distance and word order.

## 2. Frozen model surface

Trainable:
- A13 final-layer attention LoRA: **16,384 params**;
- shared bias-free 256->128 projection: **32,768 params**.

Total: **49,152 trainable parameters**.

Frozen:
- every original A13 parameter;
- HIRACore;
- reliability/calibration;
- adaptive budget;
- relation refinement.

Excluded:
- W34;
- learned downstream scorer;
- learned relation/binding head;
- task/domain/language-specific head.

S12 adds **0 learned parameters**.

## 3. Parameter-free relation binding

Frozen operator:

1. project state, question, and option-view tokens with the existing shared projection;
2. compute query-conditioned role weights over state tokens;
3. compute query-conditioned role weights over each option view;
4. form one state role anchor and one option-view role anchor;
5. express every state token as a normalized residual relative to the state role anchor;
6. express every option token as a normalized residual relative to its option-view role anchor;
7. score every valid state-token <-> option-token pair using the arithmetic mean of:
   - direct semantic cosine;
   - role-relative residual cosine;
8. retain the best explicit pair per option view;
9. combine pair evidence with state-role <-> option-role anchor compatibility;
10. average only over active semantic option views.

No token-distance/window feature is permitted.

Frozen temperatures:
- role temperature: **0.10**;
- contrastive temperature: **0.10**.

## 4. Invariants

Required at every authority:
- added learned parameters = 0;
- exact physical trainable surface = 49,152;
- original A13 trainable = 0;
- HIRACore trainable = 0;
- state-once;
- full-K;
- option permutation equivariance;
- relation refinement off;
- probability-mass error <= 1e-6;
- no prior exposed rows used for fitting or selection.

## 5. S12-A0

Fresh English identity/localization:
- 16 wholly fresh semantic cases;
- two semantically equivalent state wording views;
- two question wording views per semantic query;
- K=4;
- two semantic views per option.

Required:
- exact A13 token identity;
- exact A13 pooled identity;
- exact inherited final primary-decision logit identity;
- exact inherited selected-choice identity;
- candidate capacity 49,152;
- runtime trainable during A0 = 0;
- relation-binding added params = 0;
- state-once/full-K/order invariance;
- relation delta = 0;
- probability-mass error <= 1e-6;
- fresh relation-binding accuracy, signed margin, state-role concentration, option-role concentration, and best-pair diagnostics captured;
- A0 is never used for model selection.

## 6. Fresh TRAIN / DEV after A0

Only a qualified A0 may authorize TRAIN/DEV.

Frozen intended partitions:
- TRAIN: 768 wholly fresh semantic cases;
- DEV: 192 wholly fresh semantic cases;
- 12 fresh domains;
- fresh lexical banks;
- unseen DEV template families;
- K=4;
- two state wording views;
- two question wording views per semantic query;
- two semantic views per option.

Forbidden:
- every S0-S11 localization/A0/TRAIN/DEV row;
- M5 final/confirmatory rows;
- W29-W34 sealed rows.

## 7. Frozen optimizer and objective

- seed: **18001**;
- AdamW;
- 24 epochs;
- batch size: 16 semantic cases;
- lr: 2e-4;
- weight decay: 0.01;
- grad clip: 1.0.

Objective:
- CE on canonical + paraphrase primary decision views;
- paired swap-margin coefficient 0.25 / margin 0.20;
- option-view InfoNCE coefficient 0.05 / temperature 0.10;
- relation-structured binding CE coefficient **0.10**;
- cross-view symmetric JS coefficient 0.25.

S10 pooled grounding and S11 fixed-window binding losses are not used.

## 8. Frozen DEV selection order

1. canonical paired both-correct;
2. canonical accuracy;
3. canonical relation-binding accuracy;
4. cross-view selected-choice agreement;
5. canonical question-swap choice-change;
6. larger canonical relation-binding signed margin;
7. lower canonical decision loss;
8. earlier epoch.

## 9. Frozen DEV gate

`HIRA_V1_S12_RELATION_BINDING_DEV_READY` requires all:
- canonical accuracy >= 0.85;
- canonical paired both-correct >= 0.75;
- canonical question-swap choice-change >= 0.80;
- cross-view selected-choice agreement >= 0.95;
- cross-view mean JS <= 0.05;
- canonical relation-binding accuracy >= 0.80;
- canonical relation-binding gold-vs-max-wrong margin >= 0.15;
- option-order flip <= 0.02;
- probability-mass error <= 1e-6;
- full-K;
- relation delta = 0;
- state-once per wording view;
- total trainable exactly 49,152;
- original A13 trainable = 0;
- HIRACore trainable = 0;
- learned downstream/relation-binding params = 0.

A scientific FAIL is valid and must be frozen.
No post-DEV retry, temperature change, objective change, seed retry, or gate weakening is authorized.

## 10. Sealed boundary

Only DEV READY may open:
1. one-shot sealed English confirmation;
2. separately preregistered zero-training Vietnamese transfer.

S12 alone cannot claim Laya/Jev parity, multilingual qualification, reliability/OOD readiness or production readiness.
