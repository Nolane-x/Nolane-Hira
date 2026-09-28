# HIRA V1 S7 contract — Joint A13-W28 Semantic Co-Adaptation

Status: **OPEN / PREREGISTERED BEFORE S7-A EXPOSURE**

Issue: #185

Base main:

`5b980cef925cac524385b18ea47bbccc63e0857a`

## 1. Motivation

S5 is frozen as `HIRA_V1_S5_PROJECTION_DEV_FAIL`.

S6 is frozen as `HIRA_V1_S6_A13_LORA_DEV_FAIL`.

S5 changed W28 while A13 stayed frozen. S6 changed A13 while W28 stayed frozen. Both learned TRAIN representation signals without sufficient fresh DEV decision generalization.

S7 tests whether the semantic encoder and relation projection require joint coordinate adaptation.

## 2. Controlled causal intervention

S7 intentionally preserves the S6 architecture and optimizer choices except for one major change:

**W28-style shared projection becomes trainable jointly with S6 A13 LoRA.**

This makes S7 a direct test of co-adaptation rather than a generic capacity increase experiment.

## 3. Trainable surfaces

### A13 surface

Exact S6 final-layer attention LoRA:
- final encoder layer index 5 only;
- query;
- key;
- value;
- attention output dense;
- rank 8;
- alpha 8;
- dropout 0.

Trainable A13 LoRA params:

**16,384**

All original A13 parameters remain frozen.

### Relation surface

One shared bias-free:
- `256 -> 128` projection;
- initialized from exact W28 T0.

Trainable projection params:

**32,768**

### Total

**49,152 trainable parameters**

## 4. Frozen downstream

Frozen:
- HIRACore;
- reliability/calibration;
- adaptive budget;
- relation refinement.

Excluded:
- W34;
- learned downstream decision heads;
- task/domain/language-specific heads.

Scoring:
- normalized jointly adapted relation tokens;
- parameter-free coordinate triadic state × question × option operator.

## 5. Identity initialization

At epoch 0:
- all LoRA B matrices are zero;
- trainable projection equals exact W28 T0.

Therefore S7 must exactly equal the frozen A13 + W28 parameter-free triadic baseline before training.

A0 must verify:
- A13 token output identity;
- A13 pooled output identity;
- final logit identity;
- selected-choice identity;
- projection identity;
- zero trainable parameters when A0 candidate is frozen.

## 6. Evidence isolation

Forbidden for S7 fitting or selection:
- all S0-S6 localization/A0/TRAIN/DEV rows;
- S7-A0 rows;
- M5 final/confirmatory rows;
- W29-W34 sealed rows.

Only aggregate prior conclusions may motivate S7.

## 7. S7-A0 identity authority

Fresh English:
- 16 base states / 32 paired queries;
- K=4;
- two semantic views per option;
- opaque IDs.

A0 is identity/localization only and cannot be used for model selection.

## 8. Fresh TRAIN / DEV

TRAIN:
- 768 English base states / 1,536 paired queries;
- 12 wholly fresh domains;
- K=4;
- two semantic views per option.

DEV:
- 192 English base states / 384 paired queries;
- fresh templates and lexical banks;
- no exact TRAIN/prior-track state or question overlap.

## 9. Frozen optimizer and objectives

To isolate co-adaptation, preserve S6 settings:

- seed: 12701;
- AdamW;
- 24 epochs;
- mini-batch: 32 base states;
- lr: 2e-4;
- weight decay: 0.01;
- grad clip: 1.0;
- decision CE;
- paired swap-margin coefficient 0.25;
- swap margin 0.20;
- option-view InfoNCE coefficient 0.05;
- option-view InfoNCE temperature 0.10;
- question-option InfoNCE coefficient 0.10;
- question-option InfoNCE temperature 0.10.

DEV selection order:
1. paired both-correct;
2. accuracy;
3. question-swap choice-change;
4. lower DEV decision loss;
5. earlier epoch.

No post-DEV retry is authorized inside S7.

## 10. State-once with trainable encoder

At every model snapshot:
- each base state is encoded exactly once;
- the same state representation is reused for its paired questions;
- no stale state/schema representation crosses an optimizer update.

Every full DEV evaluation must encode exactly 192 DEV states once at that snapshot.

## 11. Frozen DEV gate

`HIRA_V1_S7_COADAPT_DEV_READY` requires:
- accuracy >= 0.85;
- paired both-correct >= 0.75;
- question-swap choice-change >= 0.80;
- option-order flip <= 0.02;
- max probability mass error <= 1e-6;
- full-K;
- relation delta = 0;
- state-once;
- total trainable exactly 49,152;
- A13 LoRA trainable exactly 16,384;
- projection trainable exactly 32,768;
- original A13 trainable = 0;
- HIRACore trainable = 0;
- learned downstream scorer params = 0.

A scientific FAIL is valid and must remain frozen.

## 12. Sealed / multilingual boundary

Only DEV READY authorizes:
1. a new one-shot English sealed confirmation;
2. then a separately preregistered zero-training Vietnamese transfer probe.

S7 alone does not authorize Laya/Jev parity, multilingual qualification, reliability/OOD readiness or production readiness.
