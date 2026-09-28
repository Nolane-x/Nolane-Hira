# HIRA V1 S6 contract — Limited A13 Semantic Encoder Adaptation

Status: **OPEN / PREREGISTERED BEFORE S6-A EXPOSURE**

Issue: #183

Base main:

`60c0aaf9e08619f44a7b2bdb52d635b8017ef194`

## 1. Motivation

S5 is frozen as `HIRA_V1_S5_CLOSED_DEV_FAIL`.

S5 directly relearned the shared W28-style 256->128 projection and strongly reduced the option-view alignment objective on TRAIN, but fresh DEV decision accuracy remained 0.3125.

S0-S5 have now tested query routing, evidence routing, W34 grounding, learned triadic scoring, a shared pre-W28 adapter and direct W28 projection relearning.

S6 therefore changes the remaining common frozen semantic substrate: A13 itself.

## 2. Pinned A13 architecture boundary

A13:
- `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- BERT;
- hidden size 256;
- six encoder layers;
- eight attention heads.

S6 must fail closed if the loaded model does not match this pinned structure.

## 3. S6-A candidate — last-layer attention LoRA

Only final encoder layer index 5 is adapted.

Target linears:
- `attention.self.query`;
- `attention.self.key`;
- `attention.self.value`;
- `attention.output.dense`.

Per target:
- base 256->256 linear remains frozen;
- LoRA rank = 8;
- alpha = 8;
- dropout = 0;
- A matrix = 8x256;
- B matrix = 256x8;
- B initializes to zero.

Per target trainable params: 4,096.

Total S6 trainable params:

**16,384**

Every original A13 parameter remains frozen.

Zero-initialized LoRA must give exact epoch-0 A13 output identity in eval mode.

## 4. Frozen downstream

Frozen:
- exact W28 T0 projection;
- parameter-free coordinate triadic scorer;
- HIRACore;
- reliability/calibration;
- relation refinement;
- adaptive budget.

W34 is not in the S6 semantic path.

No learned downstream scorer or domain/language-specific head is allowed.

## 5. Training-mode determinism

The frozen A13 base remains in `eval()` during LoRA optimization:
- base dropout is disabled;
- LoRA parameters still receive gradients;
- S6 authority therefore measures only LoRA updates, not stochastic base-dropout changes.

Schema cache is disabled during optimization and cleared whenever model state changes.

## 6. S6 representation objectives

Each option has two semantic views.

Loss:
- decision CE;
- paired swap-margin coefficient 0.25, margin 0.20;
- option-view InfoNCE coefficient 0.05, temperature 0.10;
- question-to-correct-option InfoNCE coefficient 0.10, temperature 0.10.

Question-option contrastive evidence is computed within the same K=4 dynamic schema and does not introduce a learned task head.

## 7. Evidence isolation

Forbidden for S6 fitting/selection:
- all S0-S5 localization/A0/TRAIN/DEV rows;
- S6-A0 rows;
- M5 final/confirmatory rows;
- W29-W34 sealed rows.

Only aggregate prior conclusions may motivate S6.

## 8. S6-A0 identity authority

Fresh English suite:
- 16 base states / 32 paired queries;
- K=4;
- two semantic views per option;
- opaque IDs.

Required:
- exact A13 token-output identity before/after zero-init LoRA in eval mode;
- exact final triadic logit identity;
- exact selected-choice identity;
- LoRA params = 16,384;
- original A13 trainable = 0;
- W28/HIRACore trainable = 0;
- state-once/full-K;
- option-order flip = 0;
- relation delta = 0;
- probability mass error <= 1e-6.

A0 is identity/localization only and cannot select the S6 candidate.

## 9. Fresh TRAIN / DEV

TRAIN:
- 768 English base states / 1,536 paired queries;
- twelve wholly fresh domains;
- K=4;
- two views per option.

DEV:
- 192 English base states / 384 paired queries;
- fresh templates and lexical banks;
- no exact TRAIN/prior-track state or question overlap.

## 10. Frozen optimizer

- seed: 11601;
- AdamW;
- 24 epochs;
- mini-batch size: 32 base states;
- lr: 2e-4;
- weight decay: 0.01;
- grad clip: 1.0.

Because A13 now changes after each optimizer step, S6 cannot reuse a single
state/schema embedding snapshot across the full training run.  The state-once
contract therefore means:

- within each model snapshot / mini-batch evaluation, every base state is
  encoded exactly once and reused for its paired questions;
- every full DEV evaluation encodes each of the 192 DEV base states exactly
  once at that selected model snapshot;
- stale state/schema tensors are never reused after an optimizer update.

TRAIN uses batched A13 encoding for efficiency but preserves this semantic
state-once rule.

DEV selection order:
1. paired both-correct;
2. accuracy;
3. question-swap choice-change;
4. lower DEV decision loss;
5. earlier epoch.

No post-DEV retry is authorized inside S6.

## 11. Frozen DEV gate

`HIRA_V1_S6_A13_LORA_DEV_READY` requires:
- accuracy >= 0.85;
- paired both-correct >= 0.75;
- question-swap choice-change >= 0.80;
- option-order flip <= 0.02;
- max probability mass error <= 1e-6;
- full-K;
- relation delta = 0;
- state encoder once per base state;
- LoRA trainable params exactly 16,384;
- original A13 trainable = 0;
- W28 trainable = 0;
- HIRACore trainable = 0.

A scientific FAIL is valid and must remain frozen.

## 12. Sealed / multilingual boundary

Only DEV READY authorizes:
1. a new one-shot English sealed confirmation;
2. then a separately preregistered zero-training Vietnamese transfer probe.

S6 alone does not authorize Laya/Jev parity, multilingual qualification, reliability/OOD readiness or production readiness.
