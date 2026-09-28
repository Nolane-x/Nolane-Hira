# HIRA V1 S5 contract — Semantic Projection Relearning

Status: **OPEN / PREREGISTERED BEFORE S5-A EXPOSURE**

Issue: #181

Base main:

`d928e0683c6d73b488f44c760c4512b8caff9472`

## 1. Motivation

S4 is frozen as `HIRA_V1_S4_CLOSED_DEV_FAIL`.

S4 trained a shared 16,384-parameter residual adapter before frozen W28, yet fresh DEV remained near chance. Every v1 track through S4 inherited the same W28 256->128 relation projection.

S5 directly relearns that shared projection before any A13 layer unfreezing.

## 2. S5-A architecture

Frozen:
- A13 encoder;
- HIRACore;
- reliability/calibration;
- relation refinement;
- adaptive budget.

Trainable:
- one shared bias-free `256 -> 128` semantic projection;
- used identically for state, question and option-view tokens.

Trainable parameters:

**32,768**

Initialization:
- exact W28 T0 projection;
- epoch-0 functional behavior equals the parameter-free W28 triadic baseline.

Downstream:
- normalized relation vectors;
- parameter-free coordinate triadic scoring;
- no W34;
- no learned decision head;
- no dataset/domain/language-specific head.

## 3. Representation alignment objective

Every S5 TRAIN option exposes:
- canonical criterion view;
- alias/paraphrase view.

Auxiliary symmetric InfoNCE:
- mean-pool projected tokens per view;
- normalize pooled view vectors;
- canonical view of option i identifies alias view of option i among K options;
- symmetric alias->canonical direction;
- temperature 0.10;
- coefficient 0.10.

This objective trains representation geometry without a learned task head.

## 4. S5-A0 identity authority

Wholly fresh English suite:
- 16 base states / 32 paired queries;
- K=4;
- two semantic views per option;
- projection initialized from exact W28 and frozen;
- exact logit identity with parameter-free triadic scorer;
- exact selected-choice identity;
- state-once/full-K/permutation invariance;
- not used for model selection.

## 5. Evidence isolation

Forbidden for S5 fitting/selection:
- all S0-S4 A0/localization/TRAIN/DEV rows;
- S5-A0 rows;
- M5 final/confirmatory rows;
- W29-W34 sealed rows.

Only aggregate prior conclusions may motivate S5.

## 6. Fresh TRAIN / DEV

TRAIN:
- 512 English base states / 1,024 paired queries;
- eight wholly fresh domains;
- K=4;
- two semantic views per option.

DEV:
- 128 English base states / 256 paired queries;
- fresh templates and lexical banks;
- no exact TRAIN/prior-track state or question overlap.

## 7. Frozen optimizer

- seed 10501;
- AdamW;
- 30 epochs;
- lr 3e-4;
- weight decay 0.01;
- grad clip 1.0;
- decision CE;
- paired swap-margin coefficient 0.25;
- swap margin 0.20;
- option-view InfoNCE coefficient 0.10;
- InfoNCE temperature 0.10.

DEV selection order:
1. paired both-correct;
2. accuracy;
3. question-swap choice-change;
4. lower DEV decision loss;
5. earlier epoch.

No post-DEV retry is authorized inside S5.

## 8. Frozen DEV gate

`HIRA_V1_S5_PROJECTION_DEV_READY` requires:
- accuracy >= 0.85;
- paired both-correct >= 0.75;
- question-swap choice-change >= 0.80;
- option-order flip <= 0.02;
- max probability mass error <= 1e-6;
- full-K;
- relation delta = 0;
- state encoder once per base state;
- trainable projection params exactly 32,768;
- A13 trainable = 0;
- HIRACore trainable = 0.

A scientific FAIL is valid and must remain frozen.

Only DEV READY may open sealed English confirmation and then zero-training Vietnamese transfer.

S5 alone does not authorize Laya/Jev parity, multilingual qualification, reliability/OOD readiness or production readiness.
