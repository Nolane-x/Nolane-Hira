# HIRA V1 S9 contract — Invariant Semantic Margin Separation

Status: **OPEN / PREREGISTERED BEFORE S9-A0 EXPOSURE**

Issue: #189

Base main:

`b3d95676745bae7afb74820f1f901e3480fce8fe`

S8 is frozen as:

`HIRA_V1_S8_INVARIANT_DEV_FAIL`

Canonical S8:
- A0 run `36416813621`;
- A0 artifact `10968037685`;
- TRAIN/DEV run `36417365478`;
- TRAIN/DEV artifact `10969085984`;
- selected epoch 23;
- selected canonical accuracy 0.4192708333;
- canonical paired both-correct 0.1822916667;
- canonical question-swap choice-change 0.7239583333;
- cross-view selected-choice agreement 0.6223958333;
- cross-view mean JS 0.0008056818.

S8 closure:
- run `36420446608`;
- artifact `10969078358`;
- archive SHA256 `670d570d577a7038cb89a51f1b705699a235af67cc97f816a127cba1713bd175`.

## 1. Motivation

S8 materially improves fresh semantic decisions without adding capacity, but it exposes a mismatch between probability-distribution similarity and stable decisions.

Selected S8 DEV:
- cross-view mean JS: ~0.000806;
- cross-view selected-choice agreement: ~0.6224.

S8-A0 shows the same phenomenon before training:
- mean JS ~9.75e-09;
- selected-choice agreement 0.6875.

Therefore low JS divergence can coexist with unstable argmax decisions when the K-way distribution is nearly flat.

S9 tests whether explicit gold-vs-all-wrong separation can convert S8's higher question sensitivity into stable correct decisions.

## 2. Architecture

S9 inherits the S8/S7 model surface exactly.

Trainable:
- A13 final-layer attention LoRA:
  - layer 5 only;
  - query/key/value/attention-output dense;
  - rank 8;
  - alpha 8;
  - dropout 0;
  - 16,384 params.
- shared bias-free semantic projection:
  - 256 -> 128;
  - initialized from exact W28 T0;
  - 32,768 params.

Total trainable:

**49,152**

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

Decision operator:
- parameter-free triadic state × question × option scoring.

## 3. Controlled loss change

Preserve from S8:
- CE on canonical/paraphrase views;
- option-view InfoNCE coefficient 0.05 / temperature 0.10;
- question-option InfoNCE coefficient 0.10 / temperature 0.10;
- symmetric cross-view JS coefficient 0.25;
- optimizer scale;
- 24-epoch authority structure.

Replace only the old pair-specific swap margin with:

# all-negative semantic margin

For each decision row:

[
L_{margin} = \frac{1}{K-1}\sum_{j\neq y}
\max(0, m - (z_y-z_j))
]

Frozen:
- margin (m=0.20);
- coefficient 0.25.

Apply identically to canonical and paraphrase decision views.

The margin compares gold against **all three wrong options** at K=4.

No learned margin head is allowed.

## 4. Margin diagnostics

Every DEV evaluation must record separately for canonical/paraphrase:
- mean signed gold-minus-max-wrong margin;
- fraction of decisions with signed gold-minus-max-wrong >= 0.20;
- mean absolute top1-top2 margin.

Also retain:
- accuracy;
- paired both-correct;
- question-swap selected-choice change;
- cross-view selected-choice agreement;
- cross-view mean JS;
- option-order flip;
- probability mass error.

This distinguishes:
1. correct semantic separation;
2. confident-but-wrong separation;
3. flat-but-invariant outputs.

## 5. Evidence structure

Same two-view semantic structure as S8:
- semantically equivalent state A/state B;
- question A1/A2;
- question B1/B2;
- shared K=4 dynamic options;
- two semantic text views per option.

All S9 text must be fresh.

Forbidden:
- every S0-S8 localization/A0/TRAIN/DEV row;
- M5 final/confirmatory rows;
- W29-W34 sealed rows.

Only aggregate prior conclusions may motivate S9.

## 6. S9-A0

Fresh English identity/localization suite:
- 16 semantic cases;
- 2 state wording views per case;
- 2 question wording views per semantic query;
- K=4;
- two option semantic views.

Required:
- zero-init exact A13 token/pooled identity;
- exact final logit identity;
- exact selected-choice identity;
- candidate physical capacity = 49,152;
- runtime trainable during A0 = 0;
- full-K;
- state-once per wording view;
- option-order invariance;
- relation delta = 0;
- probability mass error <= 1e-6;
- record fresh margin diagnostics.

A0 is never eligible for model selection.

## 7. Fresh TRAIN / DEV

TRAIN:
- 768 fresh underlying semantic cases;
- 12 wholly fresh domains;
- 2 state views per case;
- 2 question wording views per semantic query;
- K=4;
- two option semantic views.

DEV:
- 192 fresh underlying semantic cases;
- same domain families but fresh lexical banks;
- unseen template families;
- no exact TRAIN or prior-track state/question overlap.

## 8. Frozen optimizer

- seed: 14901;
- AdamW;
- epochs: 24;
- batch size: 16 semantic cases;
- lr: 2e-4;
- weight decay: 0.01;
- grad clip: 1.0.

Loss:
- CE on canonical/paraphrase;
- all-negative margin coefficient 0.25 / margin 0.20;
- option-view InfoNCE 0.05 / temperature 0.10;
- question-option InfoNCE 0.10 / temperature 0.10;
- cross-view JS 0.25.

## 9. DEV selection order

1. canonical paired both-correct;
2. canonical accuracy;
3. canonical fraction gold-vs-max-wrong margin >= 0.20;
4. cross-view selected-choice agreement;
5. canonical question-swap choice-change;
6. larger canonical mean gold-vs-max-wrong margin;
7. lower canonical decision loss;
8. earlier epoch.

## 10. Frozen DEV gate

`HIRA_V1_S9_MARGIN_DEV_READY` requires all:
- canonical accuracy >= 0.85;
- canonical paired both-correct >= 0.75;
- canonical question-swap choice-change >= 0.80;
- canonical/paraphrase selected-choice agreement >= 0.95;
- cross-view mean JS <= 0.05;
- canonical fraction gold-vs-max-wrong margin >= 0.20 >= 0.80;
- canonical mean gold-vs-max-wrong margin >= 0.15;
- option-order flip <= 0.02;
- probability mass error <= 1e-6;
- full-K;
- relation delta = 0;
- state-once per wording view;
- total trainable exactly 49,152;
- original A13 trainable = 0;
- HIRACore trainable = 0;
- learned downstream scorer params = 0.

A scientific FAIL is valid and must be frozen.

No post-DEV retry is authorized.

## 11. Sealed / multilingual boundary

Only DEV READY may open:
1. a one-shot fresh sealed English confirmation;
2. then a separately preregistered zero-training Vietnamese transfer probe.

S9 alone does not authorize:
- Laya parity;
- Jev parity;
- multilingual qualification;
- reliability/OOD readiness;
- production readiness.
