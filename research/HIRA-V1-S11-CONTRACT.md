# HIRA V1 S11 contract — Role-Value Factorized Evidence Binding

Status: **OPEN / PREREGISTERED BEFORE S11-A0 EXPOSURE**

Issue: #193

Base main:
`f76a3c2cbe4a95435e10c7120dbc8a657cc95462`

S10 is frozen as `HIRA_V1_S10_GROUNDING_DEV_FAIL`.

## Motivation

S10 made pooled grounding learnable on TRAIN but not semantically correct on fresh DEV. Attention sharpened while the signed gold margin remained negative and worsened.

S11 tests whether the failure comes from collapsing two distinct facts into one evidence vector:
- which role/field the query asks for;
- which value in the state fills that role.

## Physical model surface

Trainable:
- A13 final-layer attention LoRA: **16,384**
- shared bias-free 256->128 projection: **32,768**

Total: **49,152 trainable parameters**.

Frozen:
- all original A13 parameters;
- HIRACore;
- W34 excluded;
- no learned downstream scorer;
- no task/domain/language-specific head.

S11 binding operator adds **0 learned parameters**.

## Parameter-free binding operator

Use the existing shared projected token space.

Role expert:
- for each valid question token, strongest cosine match against valid option tokens;
- mean over question tokens;
- mean across active semantic option views.

State-presence expert:
- for each valid option token, strongest cosine match against valid state tokens;
- mean over option tokens;
- mean across active semantic views.

Frozen temperatures:
- role expert: **0.10**
- state expert: **0.10**

Fusion:
`binding_logits = log_softmax(role / 0.10) + log_softmax(state / 0.10)`

This product-of-experts keeps query-role and state-value evidence separate until final K-way conjunction.

## Controlled training objective

Remove S10 query->state pooled grounding and its loss.

Train S11 binding decisions with:
- CE on canonical + paraphrase views;
- paired swap-margin coefficient 0.25 / margin 0.20;
- option-view InfoNCE coefficient 0.05 / temperature 0.10;
- symmetric cross-view JS coefficient 0.25.

No post-DEV coefficient or temperature tuning.

## Evidence isolation

Forbidden:
- every S0-S10 localization/A0/TRAIN/DEV row;
- M5 final/confirmatory rows;
- W29-W34 sealed rows.

S11 uses 12 wholly fresh domain families.

## A0

Fresh English:
- 16 semantic cases;
- two state views;
- two question wording views per semantic query;
- K=4;
- two option semantic views.

Required:
- total candidate capacity 49,152;
- binding-added params 0;
- runtime trainable during A0 0;
- full-K;
- state-once;
- option-order invariance;
- probability mass error <=1e-6;
- factorized diagnostics captured;
- A0 not used for model selection.

## Fresh TRAIN / DEV

TRAIN:
- 768 semantic cases;
- 12 wholly fresh domain families;
- two state views;
- two question views per semantic query;
- K=4;
- two option views.

DEV:
- 192 semantic cases;
- fresh lexical banks;
- unseen template families;
- no exact TRAIN/prior-track overlap.

## Frozen optimizer

- seed 17001;
- AdamW;
- 24 epochs;
- batch size 16 semantic cases;
- lr 2e-4;
- weight decay 0.01;
- grad clip 1.0.

## DEV selection order

1. canonical paired both-correct;
2. canonical binding accuracy;
3. canonical signed binding margin;
4. cross-view selected-choice agreement;
5. canonical question-swap choice-change;
6. paired-gold state-support top-2 containment;
7. lower canonical binding loss;
8. earlier epoch.

## Frozen DEV gate

`HIRA_V1_S11_ROLE_VALUE_DEV_READY` requires:
- canonical binding accuracy >=0.85;
- paired both-correct >=0.75;
- question-swap choice-change >=0.80;
- cross-view selected-choice agreement >=0.95;
- cross-view mean JS <=0.05;
- canonical signed binding margin >=0.15;
- paired-gold state-support top-2 containment >=0.90;
- option-order flip <=0.02;
- probability mass error <=1e-6;
- full-K;
- state-once;
- total trainable exactly 49,152;
- original A13 trainable 0;
- HIRACore trainable 0;
- learned downstream scorer params 0;
- binding-added params 0.

No post-DEV retry is authorized.

Only DEV READY may open sealed English confirmation and a separately preregistered zero-training Vietnamese transfer.
