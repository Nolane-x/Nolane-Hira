# HIRA V1 S9 closure — Invariant Semantic Margin Separation

Status: **CLOSED — HIRA_V1_S9_MARGIN_DEV_FAIL**

Issue: #189  
PR: #190  
Branch: `feat/hira-v1-s9-margin-separation`  
Base main: `b3d95676745bae7afb74820f1f901e3480fce8fe`

## 1. Purpose

S9 kept the S8/S7 architecture fixed at 49,152 trainable parameters and replaced only the pair-specific swap-margin with an all-negative gold-vs-every-wrong-option margin.

The goal was to determine whether S8's low cross-view JS but unstable argmax decisions were primarily caused by insufficient semantic decision separation.

## 2. S9-A0 authority

Authority:
- run: `36421792484`
- artifact: `10969906709`
- digest: `sha256:e0ff0ad9907bf452fe0771822c085962b3fdda05ca95b3cf8855f7054d5dbb29`
- outcome: `HIRA_V1_S9_A0_IDENTITY_READY`

Observed:
- semantic cases: 16;
- decisions: 64;
- A13 token output identity: true;
- A13 pooled output identity: true;
- exact final logit identity: 1.0;
- exact selected-choice identity: 1.0;
- candidate physical capacity: 49,152 params;
- LoRA: 16,384;
- projection: 32,768;
- runtime trainable during A0: 0;
- canonical mean gold-vs-max-wrong margin: -8.804556273389608e-05;
- canonical margin satisfaction >= 0.20: 0.0;
- canonical mean top1-top2 margin: 7.738269050605595e-05;
- cross-view selected-choice agreement: 0.46875;
- cross-view mean JS: 1.1452669923528447e-08;
- option-order flip: 0.0;
- state-once/full-K/relation-delta-zero: PASS.

A0 was identity/localization only and was never eligible for model selection.

## 3. Fresh TRAIN / DEV authority

Authority:
- run: `36422317628`
- artifact: `10971005811`
- digest: `sha256:9c4a3e03a7964af6552ce8dc87608ca42074ccd8fd66d89b4dcaedca132c3094`
- outcome: `HIRA_V1_S9_MARGIN_DEV_FAIL`

Frozen setup:
- seed 14901;
- 24 epochs;
- batch size 16 semantic cases;
- AdamW lr 2e-4;
- weight decay 0.01;
- grad clip 1.0;
- exact S8 49,152-parameter model surface;
- CE on canonical/paraphrase views;
- all-negative margin coefficient 0.25 / margin 0.20;
- option-view InfoNCE coefficient 0.05 / temperature 0.10;
- question-option InfoNCE coefficient 0.10 / temperature 0.10;
- cross-view symmetric JS coefficient 0.25;
- 768 fresh TRAIN semantic cases;
- 192 fresh DEV semantic cases;
- 12 wholly fresh domains;
- two state wording views per case;
- two question wording views per semantic query;
- K=4;
- two semantic views per option;
- no S0-S8/S9-A0/M5/W29-W34 sealed rows used for fitting or selection.

Trainable:
- A13 last-attention LoRA: 16,384;
- shared 256->128 projection: 32,768;
- total: 49,152.

Frozen:
- all original A13 parameters;
- HIRACore;
- learned downstream scorer surface = 0;
- W34 excluded.

## 4. Selected candidate

Selected epoch:

**16**

Selected checkpoint SHA256:

`532ee4b0339c8861e4851aaa21de9a051333e1cf21ffe2d1799375df1d5bc3c7`

Selected DEV:
- canonical accuracy: **0.3619791666666667**
- canonical paired both-correct: **0.203125**
- canonical question-swap choice-change: **0.8229166666666666**
- paraphrase accuracy: **0.3854166666666667**
- cross-view selected-choice agreement: **0.703125**
- cross-view mean JS: **0.00041781535160793454**
- canonical mean gold-vs-max-wrong margin: **-0.019170253774063895**
- canonical margin satisfaction >= 0.20: **0.0**
- canonical mean top1-top2 margin: **0.006291117770160781**
- paraphrase mean gold-vs-max-wrong margin: **-0.01959060490480624**
- paraphrase margin satisfaction >= 0.20: **0.0**
- paraphrase mean top1-top2 margin: **0.0020990623694766932**
- option-order flip: **0.0**
- mean canonical decision loss: **1.3950619498888652**
- mean paraphrase decision loss: **1.390383630990982**
- mean option-view alignment loss: **0.6695548097292582**
- mean question-option loss: **1.3298651625712712**
- max probability mass error: **1.1920928955078125e-07**
- full-K/state-once/relation-delta-zero: PASS.

## 5. Frozen DEV gate

Passed semantic/mechanical gate:
- canonical question-swap choice-change >= 0.80 — **PASS at 0.8229166667**
- cross-view mean JS <= 0.05 — PASS
- option-order flip <= 0.02 — PASS
- probability mass error <= 1e-6 — PASS
- full-K — PASS
- relation delta = 0 — PASS
- state-once per wording view — PASS
- total trainable exactly 49,152 — PASS
- original A13 frozen — PASS
- HIRACore frozen — PASS.

Failed:
- canonical accuracy >= 0.85 — FAIL
- canonical paired both-correct >= 0.75 — FAIL
- cross-view selected-choice agreement >= 0.95 — FAIL
- canonical margin satisfaction >= 0.80 — FAIL
- canonical mean gold margin >= 0.15 — FAIL.

No post-DEV tuning or threshold weakening was performed.

## 6. Learning dynamics

S9 optimized its TRAIN objective:
- TRAIN all-negative margin loss fell from about 0.1996 to about 0.1222;
- TRAIN CE fell from about 1.3860 to about 1.3305;
- TRAIN question-option loss fell from about 1.3768 to about 0.7224;
- both LoRA and projection moved materially.

Fresh DEV did not acquire correct semantic separation:
- canonical margin satisfaction remained exactly 0.0 throughout;
- selected signed gold-vs-max-wrong margin was negative (-0.01917);
- selected absolute top1-top2 margin was positive but small (~0.00629);
- accuracy stayed below S8's selected 0.41927.

At the same time, question-swap choice-change rose above the 0.80 gate.

This is not simple question blindness and is not evidence that larger margin coefficients should be tried.

## 7. Preregistered interpretation

The S9 interpretation plan was frozen before TRAIN/DEV result exposure.

Observed behavior matches primarily **Outcome E — TRAIN separation but fresh DEV collapse**, with an additional useful signal:
- query sensitivity generalizes;
- correct option grounding does not.

Therefore the strongest remaining bottleneck hypothesis is now:

**question-conditioned state-to-option semantic grounding**

The model changes decisions when the question changes, but does not reliably bind the queried fact in state memory to the matching option under fresh lexical/template variation.

This is a hypothesis for S10, not a production conclusion.

## 8. Evidence boundary

Because DEV failed:
- no S9 margin retry;
- no coefficient/margin/LR/seed retry;
- no reinterpretation of top1-top2 confidence as semantic success;
- no sealed confirm;
- no Vietnamese probe;
- no M5/Laya/Jev reopening.

S9 A0 and DEV rows are permanently exposed and forbidden for future fitting, selection, architecture ranking or hyperparameter tuning.

## 9. Next direction — S10

Recommended hypothesis:

# Query-Conditioned State-to-Option Grounding

Keep the 49,152-parameter S9/S8 architecture unchanged.

Add no learned downstream head.

Use a parameter-free auxiliary grounding objective that explicitly combines:
1. question-conditioned state evidence;
2. the correct dynamic option;
3. all wrong options.

The objective must train the existing A13 LoRA + projection surface to make the query-selected state evidence align with the correct option and repel all distractors.

Use fresh two-view semantic authority only.

## 10. Closure

S9-A0: **CLOSED — identity/localization only**  
S9 TRAIN/DEV: **CLOSED — DEV FAIL**  
S9 sealed confirm: **NOT OPENED**  
S9 multilingual probe: **NOT OPENED**  
Production-ready: **false**  
Laya/Jev parity: **not claimed**
