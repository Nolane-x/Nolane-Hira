# HIRA V1 S3 contract — Triadic State–Question–Option Semantic Scoring

Status: **OPEN / PREREGISTERED BEFORE S3-A EXPOSURE**

Issue: #177

Base main:
`b42dbaa0790cef519da0e6b00a26e305564b50b6`

## 1. Scientific motivation

S2 is permanently closed as `HIRA_V1_S2_CLOSED_DEV_FAIL`.

Canonical S2:
- closure run `36392181915`
- closure artifact `10956742043`
- closure digest `sha256:e906f0c6ddc46982453fbaee99854e8edff19bcec4020796308d7ef32d4fdec5`
- selected DEV accuracy 0.2578125
- paired both-correct 0.09375
- question-swap choice-change 0.546875

S2 established that query conditioning can materially move decisions while accuracy remains near chance.

S0-S2 all retained frozen W34 option grounding. S3 therefore tests the distinct hypothesis that the remaining bottleneck is the frozen state-option semantic geometry itself.

## 2. S3-A0 — parameter-free triadic baseline

Exact frozen:
- A13 semantic encoder;
- W28 T0 projection.

W34 is not in this semantic path.

For normalized W28 relation vectors:
- state token `s`;
- question token `q`;
- option-view token `o`;

compute:

`T(s,q,o) = sum_d s_d * q_d * o_d / sqrt(128)`

The tensor is aggregated symmetrically:
- each option token takes strongest state/question support then token mean;
- each state token takes strongest option/question support then state mean;
- each question token takes strongest state/option support then question mean;
- the three directional means are averaged per semantic view;
- active views are averaged per option.

Added parameters: **0**.
Trainable parameters: **0**.

A0 is fresh English localization only and is forbidden for model selection.

## 3. S3-A learned Triadic CP scorer

Frozen:
- A13 semantic encoder;
- W28 T0 projection;
- HIRACore;
- reliability/calibration;
- relation refinement;
- adaptive budget.

W34 is not in the S3 semantic scoring path.

Shared trainable maps:
- `state_factor: 128 -> 32` = 4,096 params;
- `question_factor: 128 -> 32` = 4,096 params;
- `option_factor: 128 -> 32` = 4,096 params.

Width 32 is reshaped into:
- 2 heads;
- rank 16 per head.

New trainable parameters:

**12,288**

Initialization:
- all three factor maps use Xavier uniform;
- global TRAIN authority seed is fixed before model construction.

For each state/question/option token triple:

`score = sum_(head,rank) S_factor * Q_factor * O_factor / sqrt(32)`

The same symmetric triadic aggregation as A0 produces one logit per dynamic option.

No factor-, task-, domain-, language- or option-ID-specific learned head is permitted.

## 4. Runtime invariants

Required:
- semantic state encoder exactly once per base state;
- question/options compiled as schema token artifacts;
- dynamic K;
- full-K, no candidate truncation;
- opaque option IDs;
- option permutation equivariance;
- relation refinement OFF;
- adaptive budget OFF;
- finite logits;
- probability mass max error <= 1e-6.

## 5. Evidence isolation

Forbidden for S3 fitting, selection, architecture ranking, thresholding and hyperparameter tuning:
- all S0 localization/TRAIN/DEV rows;
- all S1 A0/TRAIN/DEV rows;
- all S2 A0/TRAIN/DEV rows;
- M5 final rows;
- M5 confirmatory rows;
- W29-W34 exposed sealed rows.

Only aggregate conclusions from prior tracks may motivate S3.

S3-A0 rows are also forbidden for learned S3 selection.

## 6. S3-A fresh English TRAIN / DEV

TRAIN:
- 256 English base states;
- 512 paired queries;
- four wholly fresh domains;
- K=4;
- opaque option IDs;
- deterministic option permutations.

DEV:
- 64 English base states;
- 128 paired queries;
- fresh templates and lexical banks;
- no exact TRAIN state/question overlap;
- no exact S0/S1/S2 state/question overlap.

## 7. Frozen optimizer

- seed: 8301
- AdamW
- 20 epochs
- learning rate: 5e-4
- weight decay: 0.01
- gradient clip: 1.0
- cross entropy
- paired swap-margin coefficient: 0.25
- swap margin: 0.20

DEV selection order:
1. paired both-correct;
2. overall accuracy;
3. question-swap choice-change;
4. lower DEV loss;
5. earlier epoch.

## 8. Frozen S3-A DEV gate

Outcome `HIRA_V1_S3_TRIADIC_DEV_READY` requires all:
- accuracy >= 0.85;
- paired both-correct >= 0.75;
- question-swap choice-change >= 0.80;
- option-order flip <= 0.02;
- max probability mass error <= 1e-6;
- full-K every query;
- relation delta = 0;
- state encode exactly once per base state;
- trainable parameter count = 12,288;
- encoder trainable count = 0;
- HIRACore trainable count = 0.

A scientific `HIRA_V1_S3_TRIADIC_DEV_FAIL` is acceptable and must remain frozen.

## 9. Sealed / multilingual boundary

Only DEV READY authorizes:
1. a newly preregistered one-shot English sealed confirmation;
2. then a separately preregistered zero-training Vietnamese transfer probe.

S3 alone cannot authorize:
- Laya parity;
- Jev parity;
- multilingual qualification;
- reliability/OOD readiness;
- production readiness.
