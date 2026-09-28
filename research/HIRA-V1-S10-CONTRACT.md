# HIRA V1 S10 contract — Query-Conditioned State-to-Option Grounding

Status: **OPEN / PREREGISTERED BEFORE S10-A0 EXPOSURE**

Issue: #191

Base main:

`99167cd6ca286ae6ccbe34ddaf29b63d07458bb1`

S9 is frozen as `HIRA_V1_S9_CLOSED_DEV_FAIL`.

## 1. Motivation

S9 is the first Hira v1 track to pass the frozen question-swap sensitivity gate:
- selected question-change: 0.8229166667.

But:
- selected canonical accuracy: 0.3619791667;
- signed gold-vs-max-wrong margin: -0.0191702538;
- margin satisfaction >=0.20: 0.0.

The model reacts to which fact is queried, yet it does not reliably bind the query-selected state fact to the correct option.

The previous question-option auxiliary InfoNCE is state-blind. S10 replaces it with a state-grounded semantic target.

## 2. Architecture

S10 keeps the exact S8/S9 physical model surface.

Trainable:
- A13 final-layer attention LoRA: 16,384;
- shared 256->128 relation projection: 32,768.

Total: **49,152 trainable parameters**.

Frozen:
- all original A13 parameters;
- HIRACore;
- reliability/calibration;
- adaptive budget;
- relation refinement.

Excluded:
- W34;
- learned downstream scorer;
- task/domain/language-specific head.

Primary decision operator remains parameter-free triadic state × question × option scoring.

## 3. Controlled objective

S10 returns to the strongest S8 decision objective:
- CE on canonical/paraphrase decision views;
- paired swap-margin coefficient 0.25 / margin 0.20;
- option-view InfoNCE coefficient 0.05 / temperature 0.10;
- cross-view symmetric JS coefficient 0.25.

Remove:
- state-blind question-option InfoNCE.

Replace with parameter-free Query-Conditioned State-to-Option Grounding:
- relation-space question/state cosine relevance;
- softmax over valid state tokens per valid question token;
- attention-weighted state evidence;
- valid-question mean;
- relation-space option pooling across token + semantic views;
- K-way grounding logits by evidence/option cosine similarity.

Frozen:
- state-attention temperature 0.10;
- grounding contrastive temperature 0.10;
- grounding loss coefficient 0.10.

No learned grounding head or attention head is allowed.

## 4. Grounding diagnostics

A0 and DEV must record:
- canonical grounding accuracy;
- paraphrase grounding accuracy;
- canonical/paraphrase mean grounding CE;
- canonical/paraphrase mean grounding gold-vs-max-wrong margin;
- grounding cross-view selected-choice agreement;
- normalized state-attention entropy;
- max state-attention weight.

Primary decision diagnostics remain unchanged.

## 5. Evidence isolation

Forbidden:
- every S0-S9 localization/A0/TRAIN/DEV row;
- M5 final/confirmatory rows;
- W29-W34 sealed rows.

S10 A0 cannot be used for fitting or selection.

## 6. S10-A0

Fresh English:
- 16 semantic cases;
- 2 state wording views;
- 2 question wording views per semantic query;
- K=4;
- two semantic option views.

Required:
- exact zero-init A13 token/pooled identity;
- exact final triadic logit identity;
- exact selected-choice identity;
- candidate capacity 49,152;
- runtime trainable = 0;
- state-once;
- full-K;
- option-order invariance;
- relation delta = 0;
- probability mass error <=1e-6;
- fresh grounding diagnostics captured.

## 7. Fresh TRAIN / DEV

TRAIN:
- 768 semantic cases;
- 12 wholly fresh domains;
- two state views;
- two question views per semantic query;
- K=4;
- two semantic option views.

DEV:
- 192 semantic cases;
- same domain families with fresh lexical banks and unseen template families;
- no exact TRAIN or prior-track state/question overlap.

## 8. Frozen optimizer

- seed 16001;
- AdamW;
- 24 epochs;
- batch size 16 semantic cases;
- lr 2e-4;
- weight decay 0.01;
- grad clip 1.0.

Loss:
- CE on both views;
- paired swap-margin coefficient 0.25 / margin 0.20;
- option-view InfoNCE coefficient 0.05 / temperature 0.10;
- query-conditioned state-option grounding coefficient 0.10;
- grounding attention temperature 0.10;
- grounding contrastive temperature 0.10;
- cross-view JS coefficient 0.25.

## 9. DEV selection order

1. canonical paired both-correct;
2. canonical accuracy;
3. canonical grounding accuracy;
4. cross-view selected-choice agreement;
5. canonical question-swap choice-change;
6. larger canonical grounding gold margin;
7. lower canonical decision loss;
8. earlier epoch.

## 10. Frozen DEV gate

`HIRA_V1_S10_GROUNDING_DEV_READY` requires all:
- canonical accuracy >=0.85;
- canonical paired both-correct >=0.75;
- canonical question-swap choice-change >=0.80;
- cross-view selected-choice agreement >=0.95;
- cross-view mean JS <=0.05;
- canonical grounding accuracy >=0.80;
- canonical grounding gold-vs-max-wrong margin >=0.15;
- option-order flip <=0.02;
- probability mass error <=1e-6;
- full-K;
- relation delta = 0;
- state-once per wording view;
- total trainable exactly 49,152;
- original A13 trainable = 0;
- HIRACore trainable = 0;
- learned downstream scorer parameters = 0.

A scientific FAIL is valid and must be frozen.
No post-DEV retry is authorized.

## 11. Sealed boundary

Only DEV READY may open:
1. one-shot sealed English confirmation;
2. separately preregistered zero-training Vietnamese transfer.

S10 alone cannot claim Laya/Jev parity, multilingual qualification, reliability/OOD readiness or production readiness.
