# HIRA V1 S8 closure — Paraphrase-Invariant Semantic Binding

Status: **CLOSED — HIRA_V1_S8_INVARIANT_DEV_FAIL**

Issue: #187  
PR: #188  
Branch: `feat/hira-v1-s8-paraphrase-invariance`  
Base main: `e654f1001cd5c29e2aef97cb66129b238e29655c`

## 1. Purpose

S8 kept the S7 model capacity fixed at 49,152 trainable parameters and changed the evidence/objective structure instead of widening the model.

Each semantic case carried two semantically equivalent state wording views and two question wording views per semantic query. S8 added symmetric Jensen-Shannon consistency between canonical and paraphrase decision distributions.

The hypothesis was that explicit wording/template invariance would reduce the train/fresh-template divergence observed in S7.

## 2. S8-A0 identity authority

Authority:
- run: `36416813621`
- artifact: `10968037685`
- digest: `sha256:cd7b84cf7de183a6bd263ab8923764d810ff97f44a56b930190893e213b2804a`
- outcome: `HIRA_V1_S8_A0_IDENTITY_READY`

Observed:
- semantic cases: 16;
- decisions: 64 across canonical/paraphrase views;
- A13 token output identity: true;
- A13 pooled output identity: true;
- exact final logit identity: 1.0;
- exact selected-choice identity: 1.0;
- candidate capacity: 49,152 params;
- LoRA: 16,384;
- projection: 32,768;
- runtime trainable during A0: 0;
- option-order flip: 0.0;
- state-once/full-K/relation-delta-zero: PASS;
- canonical paired both-correct: 0.0625;
- cross-view mean JS: 9.746536022703367e-09;
- cross-view selected-choice agreement: 0.6875.

A0 was identity/localization only and was not used for model selection.

## 3. Fresh TRAIN / DEV authority

Authority:
- run: `36417365478`
- artifact: `10969085984`
- digest: `sha256:40a0f1ae053f03ae59b87f784c5aaeeada3fb36646e7009eb6781592e8c6b49b`
- outcome: `HIRA_V1_S8_INVARIANT_DEV_FAIL`

Frozen setup:
- seed 13801;
- 24 epochs;
- batch size 16 semantic cases;
- AdamW lr 2e-4;
- weight decay 0.01;
- grad clip 1.0;
- exact S7 49,152-parameter trainable surface;
- CE on both wording views;
- paired swap-margin coefficient 0.25 / margin 0.20;
- option-view InfoNCE coefficient 0.05 / temperature 0.10;
- question-option InfoNCE coefficient 0.10 / temperature 0.10;
- cross-view symmetric JS coefficient 0.25;
- 768 fresh TRAIN semantic cases;
- 192 fresh DEV semantic cases;
- 12 wholly fresh domain families;
- two state wording views per case;
- two question wording views per semantic query;
- K=4;
- two semantic views per option;
- no S0-S7/S8-A0/M5/W29-W34 sealed rows used for fitting or selection.

Trainable:
- A13 last-attention LoRA: 16,384;
- shared 256->128 projection: 32,768;
- total: 49,152.

Frozen:
- all original A13 weights;
- HIRACore;
- learned downstream scorer surface = 0;
- W34 excluded.

## 4. Selected candidate

Selected epoch:

**23**

Selected checkpoint SHA256:

`32e3eec6ab3e1c1403284a60b24f13780783222fb237df2ae47eeaf977c4c2a7`

Selected DEV:
- canonical accuracy: **0.4192708333333333**
- canonical paired both-correct: **0.18229166666666666**
- canonical question-swap choice-change: **0.7239583333333334**
- paraphrase accuracy: **0.3880208333333333**
- cross-view selected-choice agreement: **0.6223958333333334**
- cross-view mean JS: **0.0008056817847072276**
- option-order flip: **0.0078125**
- mean canonical decision loss: **1.3772865037123363**
- mean paraphrase decision loss: **1.3873663544654846**
- mean option-view alignment loss: **0.6154157370328903**
- mean question-option loss: **1.4962438146273296**
- max probability mass error: **1.1920928955078125e-07**
- full-K: PASS
- relation delta: 0
- DEV state-view encodes: 384/384.

S8 is the strongest Hira v1 fresh-DEV result so far:
- S7 selected accuracy: 0.3463541667;
- S8 selected canonical accuracy: 0.4192708333;
- S7 paired both-correct: 0.125;
- S8 paired both-correct: 0.1822916667;
- S7 question-change: 0.578125;
- S8 question-change: 0.7239583333.

## 5. Frozen DEV gate

Failed:
- canonical accuracy >= 0.85;
- canonical paired both-correct >= 0.75;
- question-swap choice-change >= 0.80;
- canonical/paraphrase selected-choice agreement >= 0.95.

Passed:
- cross-view mean JS <= 0.05;
- option-order flip <= 0.02;
- probability mass error <= 1e-6;
- full-K;
- relation delta = 0;
- state-once per wording view;
- total trainable exactly 49,152;
- LoRA exactly 16,384;
- projection exactly 32,768;
- original A13 frozen;
- HIRACore frozen.

No post-DEV tuning or threshold weakening was performed.

## 6. Scientific interpretation

S8 materially improves semantic decision quality and question sensitivity without increasing model capacity.

However, the invariance objective reveals a critical distinction:
- selected DEV mean JS divergence is extremely small: 0.00080568;
- selected-choice agreement is only 0.6224.

The same phenomenon is visible before training in S8-A0:
- mean JS approximately 9.75e-09;
- selected-choice agreement only 0.6875.

Therefore low distribution divergence alone is not sufficient evidence of robust semantic invariance. Nearly flat K-way distributions can be numerically almost identical while tiny perturbations change the argmax.

S8 improves semantic binding, but the remaining bottleneck is now better described as **insufficient decision separation / semantic margin**, not simply insufficient probability-distribution consistency.

This is a hypothesis for the next experiment, not a production conclusion.

## 7. Evidence boundary

Because DEV failed:
- no S8 coefficient retry;
- no architecture or capacity retry;
- no early-stop rule invented from exposed DEV;
- no JS-weight retry;
- no sealed confirm;
- no Vietnamese probe;
- no M5/Laya/Jev reopening.

S8 A0 and DEV rows are permanently exposed and forbidden for future fitting, selection, architecture ranking or hyperparameter tuning.

## 8. Next direction — S9

Recommended hypothesis:

# Invariant Semantic Margin Separation

Keep the S8/S7 49,152-parameter architecture unchanged.

Replace the reliance on distribution-level consistency as the primary new signal with an explicit all-negative semantic decision margin:
- for every canonical and paraphrase query view;
- gold logit must exceed every incorrect option logit by a preregistered margin;
- preserve cross-view consistency as a secondary invariant;
- use fresh authority only.

The key question is whether explicit separation can convert S8's higher question sensitivity into stable correct choices instead of near-tied argmax decisions.

## 9. Closure

S8-A0: **CLOSED — identity/localization only**  
S8 TRAIN/DEV: **CLOSED — DEV FAIL**  
S8 sealed confirm: **NOT OPENED**  
S8 multilingual probe: **NOT OPENED**  
Production-ready: **false**  
Laya/Jev parity: **not claimed**
