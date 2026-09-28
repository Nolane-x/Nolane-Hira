# HIRA V1 S3 closure — Triadic State–Question–Option Semantic Scoring

Status: **CLOSED — HIRA_V1_S3_TRIADIC_DEV_FAIL**

Issue: #177  
PR: #178  
Branch: `feat/hira-v1-s3-triadic-semantic`  
Base main: `b42dbaa0790cef519da0e6b00a26e305564b50b6`

## 1. Purpose

S3 followed `HIRA_V1_S2_CLOSED_DEV_FAIL`.

S2 showed that question conditioning could materially change decisions while accuracy remained near chance, suggesting that the frozen W34 state-option geometry might be the remaining bottleneck.

S3 therefore removed W34 from the semantic scoring path and tested a compact shared triadic state × question × option scorer while keeping A13/W28 and HIRACore frozen.

No S0/S1/S2/M5/sealed rows were used for S3 fitting or selection.

## 2. S3-A0 — parameter-free triadic baseline

Authority:
- run: `36393304057`
- artifact: `10956484697`
- digest: `sha256:614c1b48421392e53906075c34670b879fc2033141f7e824c31cc848bf41e0e6`
- outcome: `HIRA_V1_S3_A0_PARAMETER_FREE_TRIADIC_READY`

Fresh English localization:
- 16 base states / 32 queries;
- K=4;
- exact frozen A13/W28;
- W34 excluded;
- added parameters: 0;
- trainable parameters: 0;
- state-once/full-K/relation-delta-zero: PASS.

Observed S3-A0:
- accuracy: **0.3125**
- paired both-correct: **0.0**
- question changes logits: **1.0**
- question changes final choice: **0.25**

Fresh frozen-v0 control on the same memory/schema:
- accuracy: **0.21875**
- paired both-correct: **0.0**
- question changes choice: **0.0**

A0 established that direct triadic geometry contains more useful localization signal than frozen v0/W34 on this fresh suite, but it was never eligible for model selection.

## 3. S3-A learned Triadic CP candidate

Frozen:
- A13 semantic encoder;
- W28 T0 projection;
- HIRACore;
- reliability/calibration;
- relation refinement;
- adaptive budget.

W34 was excluded from the S3 semantic scoring path.

Shared trainable maps:
- `state_factor: 128 -> 32` = 4,096 params;
- `question_factor: 128 -> 32` = 4,096 params;
- `option_factor: 128 -> 32` = 4,096 params.

Width:
- 2 heads;
- rank 16/head.

New trainable parameters: **12,288**.

The three factor maps used Xavier-uniform initialization under the preregistered seed set before model construction.

## 4. Fresh TRAIN / DEV authority

Authority:
- run: `36393893293`
- artifact: `10957677382`
- digest: `sha256:6ef1e9dc5188814ea77a2b3ecfbc66787e93cf8b2ca754ce91a186e42d5f9863`
- outcome: `HIRA_V1_S3_TRIADIC_DEV_FAIL`

Frozen setup:
- seed: 8301;
- AdamW;
- 20 epochs;
- lr: 5e-4;
- weight decay: 0.01;
- grad clip: 1.0;
- CE + 0.25 paired swap-margin;
- swap margin: 0.20;
- 256 fresh English TRAIN states / 512 queries;
- 64 fresh English DEV states / 128 queries;
- K=4;
- four fresh domains:
  - geology_sample
  - broadcast_station
  - robot_inventory
  - water_treatment.

No exact S0/S1/S2/S3-A0/M5/W29-W34 sealed row was used.

## 5. Selected DEV result

Selected epoch:

**1**

Selected checkpoint SHA256:

`ddcb2e99049aa3c321030cd93b499d4dcbb2c6f1fd3550de8e07710f67d3efd5`

Selected DEV:
- accuracy: **0.3125**
- paired both-correct: **0.09375**
- question-swap choice-change: **0.25**
- option-order flip: **0.0**
- mean loss: **1.4236116148531437**
- max probability mass error: **1.1920928955078125e-07**
- full-K: PASS
- relation delta: 0
- TRAIN state encode: 256/256
- DEV state encode: 64/64

Mechanical isolation:
- triadic factor trainable params: 12,288 exact
- W28 projection trainable: 0
- encoder trainable: 0
- HIRACore trainable: 0
- W34 in semantic path: false

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
- trainable parameters exactly 12,288 — PASS

No threshold or hyperparameter was weakened after DEV exposure.

## 7. Learning dynamics

The strongest S3 signal is the TRAIN/DEV divergence.

TRAIN mean CE:
- epoch 1: ~1.3586
- epoch 2: ~1.0776
- epoch 3: ~0.8418
- epoch 20: ~0.7062

DEV:
- epoch 1 accuracy: **0.3125**
- epoch 2 accuracy: 0.28125
- epochs 3–17: mostly 0.25
- epoch 18: 0.234375
- epoch 20: 0.25

DEV mean loss increased from **1.4236** at epoch 1 to values above **2.7** later, while TRAIN loss continued to fall.

Question-swap choice-change also declined from **0.25** at selected epoch 1 and remained mostly 0.0625–0.125 later.

This is strong evidence of fitting the fresh TRAIN lexical/template distribution without learning a transferable semantic relation representation.

## 8. Scientific interpretation

S3 is important because it removes a major alternative explanation.

S0–S2 could still blame frozen W34 option grounding. S3 removed W34 entirely and directly trained joint state-question-option factors, yet fresh DEV remained weak and deteriorated as TRAIN fit improved.

Across v1:
- S0 proved the v0 coarse path was question-blind;
- S1 showed evidence routing alone was insufficient;
- S2 made decisions materially question-sensitive but not correct;
- S3 removed W34 and still failed to generalize.

The remaining common frozen substrate is now primarily:
- the A13 semantic frontend;
- the frozen W28 relation projection / token geometry.

Therefore the next defensible hypothesis is no longer another small scorer on top of frozen relation vectors. It is a **semantic representation adaptation** track.

This does not prove A13/W28 is the only bottleneck, but it is now the strongest common architectural suspect supported by S0–S3.

## 9. Evidence boundary

Because DEV failed:
- no post-DEV S3 tuning is authorized;
- no learning-rate retry;
- no factor-rank/head retry;
- no aggregation change;
- no threshold weakening;
- no sealed confirm;
- no Vietnamese probe;
- no M5/Laya/Jev reopening.

S3 A0 and DEV rows are permanently exposed and forbidden for future fitting, selection, architecture ranking or hyperparameter tuning.

## 10. Next research direction — S4

S4 must use wholly fresh evidence and a distinct mechanism.

Recommended hypothesis:

# Compact Semantic Representation Adaptation

Instead of adding another decision scorer over frozen W28 relation vectors:
- preserve state-once;
- preserve dynamic full-K schemas and opaque option IDs;
- keep the compact A13 backbone frozen initially;
- insert a shared low-rank semantic adapter on A13 token embeddings **before** state/question/option relation comparison;
- train the same adapter for state, question and option tokens so the geometry itself becomes task-transferable;
- use a simple triadic or symmetric downstream scorer with no domain-specific head;
- enforce a strict compact parameter budget;
- preregister wholly fresh TRAIN/DEV.

If adapter-only representation learning is insufficient, only then should a later track consider limited A13 layer unfreezing.

## 11. Closure decision

S3-A0: **CLOSED — localization evidence only**  
S3-A TRAIN/DEV: **CLOSED — DEV FAIL**  
S3 sealed confirm: **NOT OPENED**  
S3 multilingual probe: **NOT OPENED**  
Production-ready: **false**  
Laya/Jev parity: **not claimed**

S3 is complete as a scientifically useful negative result that shifts the primary bottleneck hypothesis from question routing / W34 grounding toward the frozen semantic representation itself.
