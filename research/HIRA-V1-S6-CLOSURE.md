# HIRA V1 S6 closure — Limited A13 Semantic Encoder Adaptation

Status: **CLOSED — HIRA_V1_S6_A13_LORA_DEV_FAIL**

Issue: #183  
PR: #184  
Branch: `feat/hira-v1-s6-a13-lora`  
Base main: `60c0aaf9e08619f44a7b2bdb52d635b8017ef194`

## 1. Purpose

S6 tested the first direct adaptation inside the frozen A13 semantic frontend after S0-S5 progressively failed outside it.

Only the final BERT encoder layer attention linears were adapted with zero-initialized rank-8 LoRA:
- query;
- key;
- value;
- attention output dense.

All original A13 weights, W28 and HIRACore remained frozen.

## 2. S6-A0 identity authority

Authority:
- run: `36406203754`
- artifact: `10962577478`
- digest: `sha256:13ad5013cd2e4873c5801e4441602de6997e50305ccac39284886647268c4976`
- outcome: `HIRA_V1_S6_A0_IDENTITY_READY`

Observed:
- A13 token output identity: true;
- A13 pooled output identity: true;
- exact final triadic logit identity: 1.0;
- exact selected-choice identity: 1.0;
- LoRA parameter count: 16,384;
- original A13 trainable parameters: 0;
- state-once/full-K/order invariance: PASS.

A0 was identity/localization only and was never eligible for model selection.

## 3. Fresh TRAIN / DEV authority

Authority:
- run: `36407493234`
- artifact: `10963225985`
- artifact digest: `sha256:bfbc2cbc63c5f06cf1fc6dfdbe2f86a4a84cb05ec80f4cb98d3683401c6ae49f`
- outcome: `HIRA_V1_S6_A13_LORA_DEV_FAIL`

Frozen setup:
- seed 11601;
- 24 epochs;
- batch size 32 base states;
- AdamW lr 2e-4;
- weight decay 0.01;
- grad clip 1.0;
- CE + 0.25 swap-margin;
- option-view InfoNCE coefficient 0.05;
- question->correct-option InfoNCE coefficient 0.10;
- 768 fresh English TRAIN states / 1,536 queries;
- 192 fresh English DEV states / 384 queries;
- 12 fresh domains;
- K=4;
- two semantic views per option;
- no S0-S5/S6-A0/M5/W29-W34 sealed rows used for fitting or selection.

## 4. Selected candidate

Selected epoch:

**20**

Selected checkpoint SHA256:

`e4ed8bafc7ef85d1b2ed2ad4fd46be0d7982758a708f1683c7969b2281d99e3b`

Selected DEV:
- accuracy: **0.3333333333333333**
- paired both-correct: **0.08854166666666667**
- question-swap choice-change: **0.390625**
- option-order flip: **0.0**
- mean decision loss: **1.4330843488375347**
- mean option-view alignment loss: **0.572310209274292**
- mean question-option loss: **2.59052973985672**
- max probability mass error: **1.1920928955078125e-07**
- full-K: PASS
- relation delta: 0
- DEV state encodes: 192/192.

Mechanical isolation:
- LoRA trainable params: 16,384 exact;
- original A13 trainable: 0;
- W28 trainable: 0;
- HIRACore trainable: 0;
- learned downstream scorer params: 0.

## 5. Frozen DEV gate

Failed quality gates:
- accuracy >= 0.85 — **FAIL**
- paired both-correct >= 0.75 — **FAIL**
- question-swap choice-change >= 0.80 — **FAIL**

Passed mechanical gates:
- option-order flip <= 0.02 — PASS
- probability mass error <= 1e-6 — PASS
- full-K — PASS
- relation delta = 0 — PASS
- state-once per model snapshot — PASS
- exact LoRA parameter count — PASS
- original A13/W28/HIRACore frozen — PASS.

No threshold or hyperparameter was weakened after DEV exposure.

## 6. Learning dynamics and scientific interpretation

The S6 intervention clearly changed representation geometry:
- TRAIN question-option contrastive loss fell from about 1.39 to about 0.74;
- TRAIN option-view alignment fell from about 1.32 to about 0.55;
- TRAIN decision loss also declined gradually.

However the learned geometry did not transfer:
- selected DEV question-option loss was 2.59;
- later DEV question-option loss approached or exceeded 2.9;
- selected DEV accuracy remained 0.3333;
- selected paired both-correct remained 0.0885.

Therefore limited last-layer A13 attention adaptation is insufficient in isolation.

Together with S5, the evidence now distinguishes a stronger hypothesis:
- adapting W28 alone is insufficient;
- adapting A13 alone while W28 remains frozen is insufficient;
- the semantic frontend and relation projection may require **joint co-adaptation** so that improvements in A13 geometry are not projected through a coordinate system optimized for the frozen encoder.

This is a hypothesis, not yet a conclusion.

## 7. Evidence boundary

Because DEV failed:
- no post-DEV S6 tuning;
- no rank/alpha/layer/loss/lr retry;
- no threshold weakening;
- no sealed confirm;
- no Vietnamese probe;
- no M5/Laya/Jev reopening.

S6 A0 and DEV rows are permanently exposed and forbidden for future fitting, selection, architecture ranking or hyperparameter tuning.

## 8. Next direction — S7

Recommended hypothesis:

# Joint A13–W28 Semantic Co-Adaptation

Train together:
- the same compact final-layer A13 LoRA surface from S6;
- the shared 256->128 relation projection from S5.

Keep:
- all original A13 weights frozen;
- HIRACore frozen;
- no W34;
- no learned downstream decision head;
- parameter-free triadic scoring;
- state-once/full-K/opaque IDs.

Use wholly fresh authority data and preregister all optimization choices before exposure.

## 9. Closure decision

S6-A0: **CLOSED — identity/localization only**  
S6 TRAIN/DEV: **CLOSED — DEV FAIL**  
S6 sealed confirm: **NOT OPENED**  
S6 multilingual probe: **NOT OPENED**  
Production-ready: **false**  
Laya/Jev parity: **not claimed**
