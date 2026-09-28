# HIRA V1 S8 contract — Paraphrase-Invariant Semantic Binding

Status: **OPEN / PREREGISTERED BEFORE S8-A EXPOSURE**

Issue: #187

Base main:

`e654f1001cd5c29e2aef97cb66129b238e29655c`

## 1. Motivation

S7 is frozen as `HIRA_V1_S7_CLOSED_DEV_FAIL`.

S7 jointly adapted A13 and W28, learned the TRAIN distribution substantially, and improved early fresh DEV sensitivity, but later TRAIN improvement accompanied DEV degradation.

The immediate S8 hypothesis is that the remaining failure is semantic-binding invariance across equivalent wording/template changes rather than trainable capacity.

## 2. Model architecture

S8 inherits S7 exactly.

Trainable:
- final-layer A13 attention LoRA, rank 8: 16,384 params;
- shared 256->128 projection: 32,768 params.

Total trainable:

**49,152**

Frozen:
- every original A13 parameter;
- HIRACore;
- reliability/calibration;
- relation refinement;
- adaptive budget.

Excluded:
- W34;
- learned downstream decision heads;
- task/domain/language-specific heads.

Scoring remains parameter-free triadic state × question × option.

## 3. Invariance evidence

Each semantic case contains:
- state view A;
- semantically equivalent state view B;
- semantic question A with wording views A1/A2;
- semantic question B with wording views B1/B2;
- one shared K=4 dynamic option set.

The two state views and paired question views must use separately frozen templates and preserve identical facts/golds.

No external model generates paraphrases at authority runtime.

## 4. Cross-view consistency objective

For every semantic question:
- canonical path uses state A + question view 1;
- paraphrase path uses state B + question view 2;
- both produce K-way decision logits.

Add symmetric Jensen-Shannon divergence between the two predicted K-way distributions.

Coefficient:

**0.25**

No learned teacher/head is introduced.

Supervised CE and paired swap-margin remain applied to both wording views.

## 5. Existing S7 objectives preserved

- option-view InfoNCE coefficient 0.05, temperature 0.10;
- question-option InfoNCE coefficient 0.10, temperature 0.10;
- swap-margin coefficient 0.25, margin 0.20.

## 6. Evidence isolation

Forbidden for S8 fitting/selection:
- all S0-S7 localization/A0/TRAIN/DEV rows;
- S8-A0 rows;
- M5 final/confirmatory rows;
- W29-W34 sealed rows.

Only aggregate prior conclusions may motivate S8.

## 7. Authority partitions

S8-A0:
- 16 fresh semantic cases;
- identity/localization only;
- never used for selection.

TRAIN:
- 768 fresh underlying semantic cases;
- 12 fresh domains;
- two state wording views per case;
- two wording views per semantic question;
- K=4;
- two semantic views per option.

DEV:
- 192 fresh underlying semantic cases;
- same 12 domain families with disjoint lexical banks;
- entirely unseen canonical/paraphrase template families;
- no exact TRAIN/prior-track state/question overlap.

## 8. Frozen optimizer

- seed: 13801;
- AdamW;
- 24 epochs;
- mini-batch: **16 underlying semantic cases**;
- lr: 2e-4;
- weight decay: 0.01;
- grad clip: 1.0;
- invariance coefficient: 0.25.

Because two state views and four question views are encoded per semantic case, the S8 batch is frozen at 16 underlying cases per optimizer step before any S8-A0/TRAIN exposure. The semantic-case order and loss definitions are deterministic and frozen.

## 9. State-once

At each model snapshot:
- state view A encoded exactly once and reused for its paired canonical questions;
- state view B encoded exactly once and reused for its paired paraphrased questions;
- no stale representation crosses an optimizer update.

## 10. Selection

DEV selection order:
1. canonical paired both-correct;
2. canonical accuracy;
3. canonical/paraphrase selected-choice agreement;
4. canonical question-swap choice-change;
5. lower canonical decision loss;
6. earlier epoch.

## 11. Frozen DEV gate

`HIRA_V1_S8_INVARIANT_DEV_READY` requires:
- canonical accuracy >= 0.85;
- canonical paired both-correct >= 0.75;
- canonical question-swap choice-change >= 0.80;
- cross-view selected-choice agreement >= 0.95;
- cross-view mean JS divergence <= 0.05;
- option-order flip <= 0.02;
- probability mass error <= 1e-6;
- full-K;
- relation delta = 0;
- state-once for both state views;
- total trainable exactly 49,152;
- LoRA exactly 16,384;
- projection exactly 32,768;
- original A13 trainable = 0;
- HIRACore trainable = 0;
- learned downstream scorer params = 0.

No post-DEV retry is authorized.

## 12. Sealed / multilingual boundary

Only DEV READY authorizes:
1. a new one-shot English sealed confirmation;
2. a separately preregistered zero-training Vietnamese transfer probe.

S8 alone does not authorize Laya/Jev parity, multilingual qualification, reliability/OOD readiness or production readiness.
