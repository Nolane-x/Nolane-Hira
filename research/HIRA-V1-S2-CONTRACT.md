# HIRA V1 S2 contract — Query-Token Residual Fusion

Status: **OPEN / PREREGISTERED BEFORE S2-A EXPOSURE**

Issue: #175

Base main:
`7d2f71dbf9745862ddafaedd6ebe5ba1d8b49c05`

## 1. Scientific motivation

S1 is permanently closed as:

`HIRA_V1_S1_CLOSED_DEV_FAIL`

Frozen S1 closure:
- run `36388732248`
- artifact `10955099362`
- digest `sha256:d93fe7e87a6083f9a243c8c709d9ec534dddbd4df0c34a137d7f0685a8f1fd86`
- archive SHA256 `9f237152cd0239e5bf4a27a23c6687126bdd702042c988e39586b865f5131bfa`

S1 selected DEV:
- accuracy 0.265625
- paired both-correct 0.0
- question-swap choice-change 0.0
- option-order flip 0.0

S1 therefore did not establish useful query routing. No post-DEV S1 tuning is authorized.

## 2. New S2 hypothesis

S2 changes the architecture rather than tuning S1.

S1 compressed a full state into two weighted evidence slots. S2 preserves **all state tokens** and makes the question modify each state token's relation-space representation before frozen W34 option scoring.

This hypothesis may be motivated by aggregate S1 behavior only. S1 rows remain forbidden.

## 3. S2-A0 — direct question-as-evidence baseline

Zero-new-parameter diagnostic:

1. exact frozen M4 A13/W28/W34 base;
2. compile state exactly once;
3. compile question as normal schema token artifacts;
4. append question content tokens to the state evidence sequence;
5. run exact frozen W34 co-evidence scoring on
   `[state content tokens ; question content tokens]`.

New parameters: **0**.
Trainable parameters: **0**.

A0 uses a wholly fresh English localization suite and is never eligible for model selection or promotion.

## 4. S2-A learned candidate — QTRF

Name:

**Query-Token Residual Fusion (QTRF)**

Frozen:
- A13 semantic encoder;
- W28 T0 projection;
- all W34 residual adapters;
- all W34 interaction maps;
- all W34 composition maps;
- HIRACore;
- reliability/calibration;
- relation refinement;
- adaptive budget.

Trainable maps:
- `state_query: 128 -> 16` = 2,048 params;
- `question_key: 128 -> 16` = 2,048 params;
- `question_value: 128 -> 16` = 2,048 params;
- `fusion_up: 16 -> 128` = 2,048 params.

New trainable parameters:

**8,192**

Complete W34 + QTRF candidate surface:

**16,384 parameters**

### Forward path

For each already encoded state content token:
1. project through exact frozen W28 + W34 state adapter;
2. map to a rank-16 state query;
3. map every question relation token to rank-16 key/value;
4. masked softmax attention over question content tokens;
5. aggregate rank-16 question value;
6. project back to 128-d relation space;
7. residual-fuse into the state relation token;
8. normalize;
9. run frozen W34 interaction/composition scoring against the dynamic option views.

All original state-token positions survive.

The low-rank query/key tensors are **not normalized before attention**, so learned magnitude can alter attention contrast.

`fusion_up` initializes to zero, giving an exact functional W34 starting point modulo floating-point tolerance.

## 5. Runtime invariants

Required:
- semantic state encoder exactly once per base state;
- question never re-encodes the state;
- schema compiler/cache continues to own question artifacts;
- full-K, no candidate truncation;
- dynamic option count;
- option IDs non-semantic;
- option permutation equivariance;
- relation refinement OFF;
- adaptive budget OFF;
- finite logits;
- probability mass error <= 1e-6.

## 6. Evidence isolation

Forbidden for S2 fitting, DEV selection, hyperparameter tuning and architecture ranking:
- all S0 localization rows;
- all S0 TRAIN/DEV rows;
- all S1-A0 rows;
- all S1 TRAIN/DEV rows;
- M5 final held-out rows;
- M5 fresh confirmatory rows;
- W29-W34 exposed sealed rows.

A0 uses separate fresh rows and is also forbidden for learned S2 selection.

## 7. S2-A fresh English TRAIN / DEV

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
- no exact S0/S1 state/question overlap.

## 8. Frozen optimizer

- seed: 7201
- AdamW
- epochs: 16
- learning rate: 3e-4
- weight decay: 0.01
- gradient clip: 1.0
- cross entropy
- paired swap-margin coefficient: 0.25
- swap margin: 0.20

DEV selection order:
1. paired both-correct rate;
2. overall accuracy;
3. question-swap choice-change rate;
4. lower DEV loss;
5. earlier epoch.

## 9. Frozen S2-A DEV gate

Outcome `HIRA_V1_S2_QTRF_DEV_READY` requires all:
- accuracy >= 0.85;
- paired both-correct >= 0.75;
- question-swap choice-change >= 0.80;
- option-order flip <= 0.02;
- max probability mass error <= 1e-6;
- full-K every query;
- relation delta = 0;
- state encode exactly once per base state;
- trainable parameter count = 8,192;
- W34/encoder/HIRACore trainable count = 0.

A scientific `HIRA_V1_S2_QTRF_DEV_FAIL` is acceptable and must remain frozen without post-DEV rescue.

## 10. Sealed / multilingual boundary

Only DEV READY authorizes:
1. a newly preregistered one-shot English sealed confirmation;
2. after that, a separate zero-training Vietnamese transfer probe.

Do not combine mechanism qualification with multilingual rescue.

## 11. Claim boundary

S2 does not by itself authorize:
- Laya parity;
- Jev parity;
- multilingual qualification;
- reliability/OOD readiness;
- production readiness.
