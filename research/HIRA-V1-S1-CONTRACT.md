# HIRA V1 S1 contract — Query-Keyed Evidence Extraction

Status: **OPEN / PREREGISTERED BEFORE S1-A EXPOSURE**

Issue: #173

Base main:
`5cfa41b297b389ebb37422028c7aab7bafcef1d8`

## 1. Scientific motivation

S0 is permanently closed as `HIRA_V1_S0_CLOSED_DEV_FAIL`.

S0 established:
- frozen v0 W34 coarse scoring is question-blind;
- zero-parameter multiplicative query relevance changes logits but not decisions reliably;
- learned 3,072-parameter multiplicative QCCE reaches only 0.3828125 fresh DEV accuracy.

S1 changes the architectural hypothesis. The question no longer rescales an already-formed state-option similarity matrix. Instead it determines which state evidence is presented to the frozen W34 option scorer.

## 2. S1-A0 — zero-parameter evidence extraction

Exact frozen base:
- A13 model/revision/weight identity unchanged from M4;
- W28 T0 SHA256 `1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f`;
- W34 SHA256 `d69fa11805291e6a06631d5bda941065f209ea96f5c46091187e984ff083834c`.

Mechanism:
1. project state/question tokens with frozen W28;
2. mean-pool normalized question relation tokens;
3. cosine-rank state content tokens;
4. select top-2 original state tokens;
5. score only those two evidence slots with frozen W34.

New parameters: **0**.

A0 is localization only. It is never a promotion or model-selection set.

## 3. S1-A learned QKEE

Frozen:
- semantic encoder;
- HIRACore;
- W28 projection;
- all W34 adapter/interaction/composition weights.

Trainable:
- `question_heads: 128 -> 32`: 4,096 params;
- `state_keys: 128 -> 32`: 4,096 params.

The width 32 is reshaped into 2 heads × rank 16.

Per query:
- compile question once as part of schema;
- project question/state through frozen W28;
- form two query heads and two state-key heads;
- masked attention over already encoded state content tokens;
- weighted sum produces two evidence slots in original d_model space;
- exact W34 co-evidence scorer consumes those slots.

Trainable S1 surface: **8,192 params**.
Complete W34+QKEE candidate surface: **16,384 params**.

## 4. Runtime invariants

Required throughout:
- state semantic encoder exactly once per base state;
- question does not re-encode state;
- dynamic schema cache remains valid;
- full-K, no candidate truncation;
- option IDs are non-semantic;
- option permutation equivariance;
- relation refinement OFF;
- adaptive budget OFF;
- finite logits and probability mass error <= 1e-6.

## 5. Evidence isolation

Forbidden for S1 fitting/selection:
- S0-A rows;
- S0 TRAIN/DEV rows;
- M5 final held-out rows;
- M5 fresh confirmatory rows;
- W29-W34 sealed rows.

S1-A0 uses a fresh English localization suite and is not eligible for later selection.

S1-A TRAIN/DEV must use new English rows distinct from A0 and all above evidence.

## 6. Frozen S1-A optimization

TRAIN:
- 256 English base states / 512 paired queries;
- four fresh domains;
- K=4;
- opaque IDs;
- deterministic option permutations.

DEV:
- 64 English base states / 128 paired queries;
- fresh templates/lexical values.

Optimizer:
- seed 6101;
- AdamW;
- 12 epochs;
- lr 5e-4;
- weight decay 0.01;
- grad clip 1.0;
- CE + 0.20 paired swap-margin;
- margin 0.15.

DEV selection:
1. paired both-correct;
2. accuracy;
3. question-swap choice-change;
4. lower DEV loss;
5. earlier epoch.

## 7. Frozen S1-A DEV gate

Outcome `HIRA_V1_S1_QKEE_DEV_READY` requires:
- accuracy >= 0.85;
- paired both-correct >= 0.75;
- question-swap choice-change >= 0.80;
- option-order flip <= 0.02;
- probability mass max error <= 1e-6;
- full-K every query;
- relation delta = 0;
- state exactly once per base state;
- trainable params = 8,192.

A scientific FAIL is valid and must be frozen without post-DEV tuning.

## 8. Next evidence boundary

Only DEV READY authorizes a one-shot fresh English sealed confirmatory suite.

Vietnamese is a separate zero-training S1-B transfer probe and cannot be used to rescue S1-A.

No S1 result alone authorizes Laya/Jev parity, reliability/OOD readiness, multilingual qualification or production readiness.
