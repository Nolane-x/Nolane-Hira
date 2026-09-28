# HIRA V1 S2 closure — Query-Token Residual Fusion

Status: **CLOSED — HIRA_V1_S2_QTRF_DEV_FAIL**

Issue: #175  
PR: #176  
Branch: `feat/hira-v1-s2-query-token-residual-fusion`  
Base main: `7d2f71dbf9745862ddafaedd6ebe5ba1d8b49c05`

## 1. Purpose

S2 followed the frozen S1 result `HIRA_V1_S1_CLOSED_DEV_FAIL`.

S1 showed that hard or learned two-slot query-keyed evidence compression did not yield useful question-dependent decisions. S2 therefore preserved all state content tokens and tested whether compiled question tokens could modify each state token's relation-space representation before frozen W34 option scoring.

No S0/S1/M5/sealed rows were used for S2 fitting or selection.

## 2. S2-A0 — direct question-as-evidence

Authority:
- run: `36390086785`
- artifact: `10955893093`
- digest: `sha256:e17ccf529e503450686895c983d77df945b21e90570a8945da8c48e95c57c471`
- outcome: `HIRA_V1_S2_A0_QUESTION_AS_EVIDENCE_READY`

Fresh English localization:
- 16 base states / 32 queries;
- K=4;
- added params: 0;
- trainable params: 0;
- all state tokens preserved;
- question tokens appended to W34 evidence;
- state-once/full-K/relation-delta-zero: PASS.

Observed:
- accuracy: **0.25**
- paired both-correct: **0.0**
- question changes logits: **1.0**
- question changes final choice: **0.0625**

Fresh frozen-v0 control:
- accuracy: **0.25**
- paired both-correct: **0.0**
- question changes choice: **0.0**

A0 established that direct question exposure removes strict logit blindness but does not materially rescue decision quality.

## 3. S2-A learned QTRF candidate

Architecture:
- exact frozen A13 semantic encoder;
- exact frozen W28 T0 projection;
- exact frozen W34 residual/interaction/composition core;
- frozen HIRACore;
- all original state tokens preserved.

New shared trainable maps:
- `state_query: 128 -> 16` = 2,048 params;
- `question_key: 128 -> 16` = 2,048 params;
- `question_value: 128 -> 16` = 2,048 params;
- `fusion_up: 16 -> 128` = 2,048 params.

New trainable parameters: **8,192**  
Complete W34 + QTRF candidate surface: **16,384**

`fusion_up` initializes at zero, giving the exact frozen W34 functional boundary before optimization.

## 4. TRAIN/DEV authority provenance

First attempted TRAIN/DEV run:

`36390503180`

It failed in **pre-exposure unit freshness checks** before:
- A0 authority verification;
- M4 integrity verification;
- any optimizer step;
- any DEV exposure.

Cause:
- the TRAIN sentinel string `ion` was tested as a raw substring;
- it accidentally matched the unrelated DEV word `calibration`.

The qualified retry changed **only the unit-test sentinel wording** to full lexical phrases. Model architecture, generated TRAIN/DEV rows, seed, optimizer, epochs, loss, selection order and frozen gates were unchanged.

Qualified authority:
- run: `36391742553`
- artifact: `10957020098`
- digest: `sha256:0e665e01074eada113ef481fe980fd78076692337344f6312d7022ef4d170472`
- outcome: `HIRA_V1_S2_QTRF_DEV_FAIL`

Frozen setup:
- seed: 7201;
- AdamW;
- 16 epochs;
- lr: 3e-4;
- weight decay: 0.01;
- grad clip: 1.0;
- CE + 0.25 paired swap-margin;
- swap margin: 0.20;
- 256 fresh English TRAIN states / 512 queries;
- 64 fresh English DEV states / 128 queries;
- K=4;
- four new domains: sensor_registry, recipe_card, flight_clearance, museum_loan.

## 5. Selected DEV result

Selected epoch:

**2**

Selected checkpoint SHA256:

`dc2cff637771aea70ab7f7011d108cfc08a63048e8ef2502767099126eea4af2`

Selected DEV:
- accuracy: **0.2578125**
- paired both-correct: **0.09375**
- question-swap choice-change: **0.546875**
- option-order flip: **0.0**
- mean loss: **1.4502812698483467**
- max probability mass error: **1.1920928955078125e-07**
- full-K: PASS
- relation delta: 0
- TRAIN state encode: 256/256
- DEV state encode: 64/64

Mechanical isolation:
- QTRF trainable params: 8,192 exact
- W34 base trainable: 0
- encoder trainable: 0
- HIRACore trainable: 0

## 6. Frozen DEV gate

Failed quality gates:
- accuracy >= 0.85 — **FAIL**
- paired both-correct >= 0.75 — **FAIL**
- question-swap choice-change >= 0.80 — **FAIL**

Passed mechanical gates:
- option-order flip <= 0.02 — PASS
- probability mass error <= 1e-6 — PASS
- full-K — PASS
- relation delta = 0 — PASS
- state-once — PASS
- trainable parameters exactly 8,192 — PASS

No threshold or hyperparameter was weakened after DEV exposure.

## 7. Learning dynamics

S2 differs materially from S1.

S1 selected DEV:
- accuracy 0.265625;
- paired both-correct 0.0;
- question-swap choice-change 0.0.

S2 selected DEV:
- accuracy 0.2578125;
- paired both-correct 0.09375;
- question-swap choice-change 0.546875.

Across S2 epochs, question-swap choice-change rose as high as **0.671875** at epoch 3, while accuracy remained poor.

This shows QTRF successfully makes decisions much more question-dependent than S1, but the changed decisions are not reliably the correct option.

The evidence therefore moves the bottleneck downstream: question conditioning is now materially active, while option grounding / triadic state-question-option discrimination remains weak.

## 8. Evidence boundary

Because DEV failed:
- no post-DEV S2 tuning is authorized;
- no learning-rate retry;
- no fusion-rank retry;
- no attention-temperature retry;
- no threshold weakening;
- no sealed confirm;
- no Vietnamese probe;
- no M5/Laya/Jev reopening.

S2 DEV is permanently exposed and forbidden for future fitting, selection, architecture ranking or hyperparameter tuning.

## 9. Next research direction — S3

The next track must use wholly fresh evidence and a distinct mechanism.

Recommended hypothesis:

# Triadic State–Question–Option Semantic Scoring

S0–S2 kept the W34 option scorer frozen while progressively improving question conditioning. S2 finally made decisions question-sensitive, but accuracy remained near chance.

S3 should therefore attack the remaining frozen bottleneck directly:
- keep the compact A13 semantic frontend and state-once runtime;
- preserve dynamic full-K schemas;
- construct shared low-rank interactions among **state tokens, question tokens and each option view jointly**;
- allow a compact trainable option-grounding scorer instead of forcing every candidate through frozen W34's state-option geometry;
- no dataset/domain-specific head;
- strict small parameter budget;
- fresh TRAIN/DEV only.

This is a new architecture hypothesis, not an S2 rescue.

## 10. Closure decision

S2-A0: **CLOSED — localization evidence only**  
S2-A TRAIN/DEV: **CLOSED — DEV FAIL**  
S2 sealed confirm: **NOT OPENED**  
S2 multilingual probe: **NOT OPENED**  
Production-ready: **false**  
Laya/Jev parity: **not claimed**

S2 is complete as a scientifically useful negative result with one positive localization result: meaningful question-dependent decision movement was achieved, but correct semantic grounding was not.
