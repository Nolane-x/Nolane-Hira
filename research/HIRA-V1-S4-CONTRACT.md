# HIRA V1 S4 contract — Compact Semantic Representation Adaptation

Status: **OPEN / PREREGISTERED BEFORE S4-A EXPOSURE**

Issue: #179

Base main:

`040bbcc0f182b517011ebb848e3baebd233465c2`

## 1. Motivation

S3 is frozen as `HIRA_V1_S3_CLOSED_DEV_FAIL`.

S0-S3 progressively removed question routing and downstream scorer explanations. S3 removed W34 entirely and learned direct state × question × option factors, yet TRAIN fit improved strongly while fresh DEV degraded.

The strongest common frozen substrate is now the A13 token geometry plus frozen W28 relation projection.

S4 tests whether a very small shared representation adapter can reshape that geometry in a transferable way.

## 2. S4-A architecture

Name:

**Shared Residual Semantic Adapter (SRSA)**

Frozen:
- A13 semantic encoder;
- W28 T0 projection;
- HIRACore;
- reliability/calibration;
- relation refinement;
- adaptive budget.

Trainable shared adapter applied identically to state/question/option-view A13 token embeddings before W28:

- `down: 256 -> 32` = 8,192 params;
- GELU;
- `up: 32 -> 256` = 8,192 params;
- residual `x + up(gelu(down(x)))`.

Total trainable parameters:

**16,384**

The up matrix initializes to zero. Therefore the initial adapted representation equals the original A13 token representation exactly.

Downstream:
- frozen W28 projection;
- normalized relation vectors;
- parameter-free coordinate triadic state × question × option evidence;
- no W34;
- no learned decision head;
- no task/domain/language-specific head.

This isolates representation learning from downstream scorer learning.

## 3. S4-A0 identity/localization authority

Use a wholly fresh English suite:
- 16 base states / 32 paired queries;
- K=4;
- opaque IDs;
- no prior S0-S3 state/question text;
- exact frozen A13/W28;
- SRSA up matrix = 0;
- adapter frozen.

Required:
- S4 logits exactly equal parameter-free triadic logits;
- S4 selected choices equal parameter-free triadic choices;
- adapter params = 16,384 but trainable = 0;
- W28 trainable = 0;
- state encoder once per base state;
- full-K;
- option-order invariance;
- relation delta = 0;
- probability mass error <= 1e-6.

A0 is localization/identity only and cannot be used for model selection.

## 4. Evidence isolation

Forbidden for S4 fitting/DEV selection:
- every S0 localization/TRAIN/DEV row;
- every S1 A0/TRAIN/DEV row;
- every S2 A0/TRAIN/DEV row;
- every S3 A0/TRAIN/DEV row;
- S4-A0 rows;
- M5 final/confirmatory rows;
- W29-W34 sealed rows.

Only aggregate prior conclusions may motivate S4.

## 5. Fresh S4-A TRAIN / DEV

TRAIN:
- 384 English base states;
- 768 paired queries;
- six wholly fresh domains;
- K=4;
- opaque option IDs;
- deterministic option permutations.

DEV:
- 96 English base states;
- 192 paired queries;
- fresh templates and lexical banks;
- no exact TRAIN state/question overlap;
- no exact prior-track state/question overlap.

## 6. Frozen optimizer

- seed: 9401;
- AdamW;
- epochs: 24;
- lr: 4e-4;
- weight decay: 0.01;
- grad clip: 1.0;
- CE + 0.25 paired swap-margin;
- swap margin: 0.20.

DEV selection order:
1. paired both-correct;
2. overall accuracy;
3. question-swap choice-change;
4. lower DEV loss;
5. earlier epoch.

No post-DEV hyperparameter/architecture retry is authorized inside S4.

## 7. Frozen DEV gate

Outcome `HIRA_V1_S4_ADAPTER_DEV_READY` requires all:
- accuracy >= 0.85;
- paired both-correct >= 0.75;
- question-swap choice-change >= 0.80;
- option-order flip <= 0.02;
- max probability mass error <= 1e-6;
- full-K every query;
- relation delta = 0;
- state encoder exactly once per base state;
- trainable adapter params exactly 16,384;
- W28 trainable = 0;
- encoder trainable = 0;
- HIRACore trainable = 0.

A scientific `HIRA_V1_S4_ADAPTER_DEV_FAIL` is acceptable and must be frozen.

## 8. Sealed / multilingual boundary

Only DEV READY authorizes:
1. a new one-shot English sealed confirmatory suite;
2. then a separately preregistered zero-training Vietnamese transfer probe.

S4 alone does not authorize:
- Laya parity;
- Jev parity;
- reliability/OOD readiness;
- multilingual qualification;
- production readiness.
