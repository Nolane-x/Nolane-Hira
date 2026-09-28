# HIRA V1 S7 closure — Joint A13-W28 Semantic Co-Adaptation

Status: **CLOSED — HIRA_V1_S7_COADAPT_DEV_FAIL**

Issue: #185  
PR: #186  
Branch: `feat/hira-v1-s7-a13-w28-coadapt`  
Base main: `5b980cef925cac524385b18ea47bbccc63e0857a`

## 1. Purpose

S7 tested the clean co-adaptation hypothesis created by S5 and S6:
- S5 adapted W28 while A13 was frozen and failed DEV;
- S6 adapted A13 while W28 was frozen and failed DEV;
- S7 jointly adapted those exact two surfaces while leaving the downstream decision operator parameter-free.

## 2. S7-A0 authority

Authority:
- run: `36411385031`
- artifact: `10964497583`
- digest: `sha256:38c9a23606053c4131143817a555677bdbe0e09dac645cbd725572a1d6cd4bdc`
- outcome: `HIRA_V1_S7_A0_IDENTITY_READY`

Observed:
- A13 token output identity: true;
- A13 pooled output identity: true;
- exact final logit identity: 1.0;
- exact selected-choice identity: 1.0;
- LoRA physical params: 16,384;
- projection physical params: 32,768;
- candidate physical params: 49,152;
- runtime trainable during A0: 0;
- state-once/full-K/order invariance: PASS.

A0 was identity/localization only and was never used for model selection.

## 3. Fresh TRAIN / DEV authority

Authority:
- run: `36411829108`
- artifact: `10966140203`
- digest: `sha256:4ba4cf0b0e3a95c85994df3d329ddb66a0039bea6d6e0b895ed034e266fa1c3a`
- outcome: `HIRA_V1_S7_COADAPT_DEV_FAIL`

Frozen setup:
- seed 12701;
- 24 epochs;
- batch size 32 base states;
- AdamW lr 2e-4;
- weight decay 0.01;
- grad clip 1.0;
- exact S6 loss coefficients;
- 768 fresh English TRAIN states / 1,536 queries;
- 192 fresh English DEV states / 384 queries;
- 12 fresh domains;
- K=4;
- two semantic views per option;
- no S0-S6/S7-A0/M5/W29-W34 sealed rows used for fitting or selection.

Trainable:
- A13 last-attention LoRA: 16,384;
- shared 256->128 projection: 32,768;
- total: 49,152.

Frozen:
- all original A13 weights;
- HIRACore;
- learned downstream scorer surface = 0.

## 4. Selected candidate

Selected epoch:

**5**

Selected checkpoint SHA256:

`5aa9e5b9cdb90d8dac1d76cdd46021085011177ff9fa010add8437e68c46dbac`

Selected DEV:
- accuracy: **0.3463541666666667**
- paired both-correct: **0.125**
- question-swap choice-change: **0.578125**
- option-order flip: **0.0**
- mean decision loss: **1.4351223309834797**
- mean option-view alignment loss: **0.6357965568701426**
- mean question-option loss: **1.9249513149261475**
- max probability mass error: **1.1920928955078125e-07**
- full-K/state-once/relation-delta-zero: PASS.

The selection order prioritized paired both-correct. Raw DEV accuracy temporarily reached **0.375** at epoch 7, but that epoch did not beat the selected paired-both-correct criterion.

## 5. Learning dynamics

S7 clearly optimized both joint surfaces.

TRAIN:
- CE: about 1.3862 -> 1.3069;
- decision loss: about 1.4362 -> 1.3154;
- swap term: about 0.1998 -> 0.0340;
- question-option loss: about 1.3819 -> 0.7414;
- option-view alignment: about 1.3172 -> 0.5550.

Surface diagnostics:
- LoRA B norm: 0.356 after epoch 1 -> 3.888 after epoch 24;
- projection weight norm: 8.430 after epoch 1 -> 6.257 after epoch 24.

Thus the joint optimizer did not remain near initialization.

DEV behaved differently:
- early question-swap sensitivity rose as high as ~0.66;
- early accuracy reached 0.375;
- later accuracy regressed toward 0.25-0.30 as TRAIN continued improving;
- DEV question-option loss rose above 2.5 while TRAIN question-option loss fell below 0.75.

This is a stronger train/fresh-template divergence than a simple under-capacity explanation.

## 6. Frozen DEV gate

Failed:
- accuracy >= 0.85;
- paired both-correct >= 0.75;
- question-swap choice-change >= 0.80.

Passed:
- option-order flip <= 0.02;
- probability mass error <= 1e-6;
- full-K;
- relation delta = 0;
- state-once;
- total trainable exactly 49,152;
- LoRA exactly 16,384;
- projection exactly 32,768;
- original A13 frozen;
- HIRACore frozen.

No post-DEV tuning or threshold weakening was performed.

## 7. Scientific interpretation

S7 improves on S6 in paired correctness and question sensitivity, and it proves that A13/W28 co-adaptation can fit the fresh TRAIN distribution substantially better than either surface alone.

However, continued TRAIN improvement accompanies DEV degradation.

The next defensible hypothesis is therefore not simply "more capacity." The dominant failure now looks like **semantic binding invariance across wording/template/lexical changes**.

This is still a hypothesis. S7 does not prove that invariance is the only remaining bottleneck.

## 8. Evidence boundary

Because DEV failed:
- no S7 hyperparameter retry;
- no early-stop rule invented from exposed S7 DEV;
- no loss coefficient retry;
- no capacity widening;
- no sealed confirm;
- no Vietnamese probe;
- no M5/Laya/Jev reopening.

S7 A0 and DEV are permanently exposed and forbidden for future fitting or selection.

## 9. Next direction — S8

Recommended hypothesis:

# Paraphrase-Invariant Semantic Binding

Keep the S7 49,152-parameter architecture unchanged.

Change the training evidence structure rather than model capacity:
- multiple semantically equivalent state wording views;
- multiple question paraphrase views;
- consistency constraints across views;
- same dynamic options and gold semantics;
- new fresh TRAIN/DEV only.

The experiment should ask whether explicit invariance training prevents the late TRAIN/DEV divergence observed in S7.

## 10. Closure

S7-A0: **CLOSED — identity/localization only**  
S7 TRAIN/DEV: **CLOSED — DEV FAIL**  
S7 sealed confirm: **NOT OPENED**  
S7 multilingual probe: **NOT OPENED**  
Production-ready: **false**  
Laya/Jev parity: **not claimed**
