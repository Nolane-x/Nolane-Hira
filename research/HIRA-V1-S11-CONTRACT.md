# HIRA V1 S11 contract — Role-Preserving Evidence Binding

Status: **OPEN / PREREGISTERED BEFORE S11-A0 EXPOSURE**

Issue: #195

Base main:

`f76a3c2cbe4a95435e10c7120dbc8a657cc95462`

S10 is frozen as `HIRA_V1_S10_GROUNDING_DEV_FAIL`.

## 1. Motivation

S10 made the state-grounding auxiliary objective learnable on TRAIN but failed fresh DEV:
- TRAIN grounding loss ~1.380856 -> ~0.373064;
- selected DEV canonical grounding accuracy 0.4036458333;
- selected DEV signed grounding margin -0.8021047898;
- selected DEV primary accuracy 0.2682291667;
- selected DEV paired both-correct 0.0572916667;
- selected DEV question-change 0.359375.

Attention sharpened while correct grounding did not generalize. The preregistered interpretation is primarily S10 Outcome C + E.

S11 tests whether the missing structure is explicit **role -> nearby value -> option** binding before pooling.

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
- learned binding/attention head;
- task/domain/language-specific head.

S11 adds **0 learned parameters**.

## 3. Role-preserving binding operator

Fixed operator:

1. project state, question and option tokens through the existing shared projection;
2. compute question-token <-> state-token cosine relevance;
3. reduce over valid question tokens to a role score for every state position;
4. softmax role scores over valid state positions;
5. spread each role position through a fixed symmetric local kernel over nearby non-identical state positions;
6. compute state-token <-> option-token cosine similarity for every option and semantic view;
7. retain per-state-position MaxSim over valid option tokens;
8. combine local value-position weights with option MaxSim **before state pooling**;
9. average only across valid semantic option views.

Frozen hyperparameters:
- role temperature: **0.10**;
- option contrastive temperature: **0.10**;
- symmetric value window: **4 content-token positions**.

No parameter may be introduced by the binding operator.

## 4. Why this differs from S10

S10:

`question -> pooled state evidence -> pooled option -> score`

S11:

`question -> role positions -> local value positions <-> option tokens -> score`

The S11 score cannot be formed until role and value evidence have interacted at token level.

## 5. S11-A0

Fresh English identity/localization suite:
- 16 wholly fresh semantic cases;
- two semantically equivalent state wording views;
- two question wording views per semantic query;
- K=4;
- two semantic views per option;
- no S0-S10 row may be reused.

Required:
- exact A13 token identity;
- exact A13 pooled identity;
- exact inherited final primary-decision logit identity;
- exact inherited selected-choice identity;
- candidate physical capacity 49,152;
- runtime trainable during A0 = 0;
- role-binding added params = 0;
- state-once;
- full-K;
- primary option-order invariance;
- relation delta = 0;
- probability mass error <=1e-6;
- record fresh binding accuracy, signed gold margin and role/value concentration diagnostics;
- A0 cannot be used for model selection.

## 6. Fresh TRAIN / DEV authority after A0

Only a qualified A0 may authorize TRAIN/DEV.

Frozen intended authority:
- 768 wholly fresh TRAIN semantic cases;
- 192 wholly fresh DEV semantic cases;
- 12 domain families not used as S10 rows;
- fresh lexical banks;
- unseen DEV template families;
- K=4;
- two state views;
- two question views per semantic query;
- two semantic views per option.

Forbidden:
- every S0-S10 localization/A0/TRAIN/DEV row;
- M5 final/confirmatory rows;
- W29-W34 sealed rows.

## 7. Frozen optimizer and objective

- seed: **17001**;
- AdamW;
- 24 epochs;
- batch size: 16 semantic cases;
- lr: 2e-4;
- weight decay: 0.01;
- grad clip: 1.0.

Return to the strongest invariant decision frame and replace the failed S10 pooled-grounding auxiliary with S11 binding:
- CE on canonical + paraphrase decision views;
- paired swap-margin coefficient 0.25 / margin 0.20;
- option-view InfoNCE coefficient 0.05 / temperature 0.10;
- role-preserving binding CE coefficient **0.10**;
- cross-view symmetric JS coefficient 0.25.

No S10 pooled-grounding loss is used.

## 8. Frozen DEV selection order

1. canonical paired both-correct;
2. canonical accuracy;
3. canonical role-binding accuracy;
4. cross-view selected-choice agreement;
5. canonical question-swap choice-change;
6. larger canonical role-binding signed margin;
7. lower canonical decision loss;
8. earlier epoch.

## 9. Frozen DEV gate

`HIRA_V1_S11_ROLE_BINDING_DEV_READY` requires all:
- canonical accuracy >=0.85;
- canonical paired both-correct >=0.75;
- canonical question-swap choice-change >=0.80;
- cross-view selected-choice agreement >=0.95;
- cross-view mean JS <=0.05;
- canonical role-binding accuracy >=0.80;
- canonical role-binding gold-vs-max-wrong margin >=0.15;
- option-order flip <=0.02;
- probability mass error <=1e-6;
- full-K;
- relation delta = 0;
- state-once per wording view;
- total trainable exactly 49,152;
- original A13 trainable = 0;
- HIRACore trainable = 0;
- learned downstream/binding params = 0.

A scientific FAIL is valid and must be frozen.
No post-DEV retry or gate weakening is authorized.

## 10. Sealed boundary

Only DEV READY may open:
1. one-shot sealed English confirmation;
2. separately preregistered zero-training Vietnamese transfer.

S11 alone cannot claim Laya/Jev parity, multilingual qualification, reliability/OOD readiness or production readiness.
