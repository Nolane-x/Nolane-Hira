# HIRA V1 S13 closure — Cross-View Relation Canonicalization

Status: **CLOSED — DEV FAIL / SCIENTIFIC NEGATIVE RESULT**

Issue: #201  
PR: #202  
Branch: `feat/hira-v1-s13-cross-view-canonicalization`

## 1. Authority

A0:
- run: `36524047158`
- artifact: `11013757416`
- artifact digest: `sha256:d2aeeec672dd7787aab3f3fe68799e83d303a99b74b4ebf8926ed75e44fbf8ba`
- outcome: `HIRA_V1_S13_A0_IDENTITY_READY`

Fresh TRAIN/DEV:
- run: `36524706099`
- artifact: `11014078172`
- artifact digest: `sha256:0abe4623eef73dddda3aa49a2725731388dc02bb2ab2305262595f4c501e037a`
- outcome: `HIRA_V1_S13_CANONICAL_RELATION_DEV_FAIL`
- selected epoch: **24**
- selected checkpoint SHA256: `1ede44bc65a275941c8313c8ee78d82c8759eaf3985eb62080e13d48aa9f7360`

One harness-only retry occurred before TRAIN exposure:
- the first TRAIN/DEV attempt stopped in checkpoint-contract tests;
- only the frozen checkpoint-kind test string was corrected;
- architecture, data, optimizer, objective, selection order and gates were unchanged;
- the successful authority above is the first run that exposed S13 TRAIN/DEV.

No post-DEV tuning was performed.
No sealed confirmation was opened.
No multilingual probe was opened.
No Laya/Jev benchmark was reopened.

## 2. Frozen model surface

S13 kept the exact S8-S12 optimization surface:
- A13 final-layer attention LoRA: **16,384 params**
- shared bias-free 256->128 projection: **32,768 params**
- total trainable: **49,152 params**
- original A13 trainable: **0**
- HIRACore trainable: **0**
- learned downstream scorer params: **0**
- canonicalizer-added params: **0**

The S13 structural change was parameter-free:
- query-conditioned state and option role anchors;
- role-relative state/option residuals;
- differentiable token-pair support;
- one normalized relation signature per logical option;
- same-option cross-view signature alignment;
- fixed wrong-option signature separation margin.

## 3. A0 baseline

Fresh A0 showed a partially stable but non-discriminative relation signal:
- mean same-option cross-view signature cosine: **0.7203133106**
- minimum same-option cosine: **0.3168613315**
- mean same-option vs strongest-wrong signature margin: **-0.0024688528**
- canonical relation-binding accuracy: **0.28125**
- paraphrase relation-binding accuracy: **0.4375**
- canonical signed relation margin: **-0.3119711280**
- primary cross-view selected-choice agreement: **0.6875**

Mechanical identity/invariants all passed.

## 4. Selected DEV result

Primary decision:
- canonical accuracy: **0.2942708333**
- paraphrase accuracy: **0.2786458333**
- canonical paired both-correct: **0.046875**
- canonical question-swap choice-change: **0.3489583333**
- cross-view selected-choice agreement: **0.3359375**
- cross-view mean JS: **9.6776463e-09**

Canonical relation evidence:
- canonical relation-binding accuracy: **0.4921875**
- paraphrase relation-binding accuracy: **0.5338541667**
- canonical signed gold-vs-max-wrong margin: **-0.1725222593**
- paraphrase signed margin: **-0.0383211312**
- relation-choice cross-view agreement: **0.5546875**
- canonical mean binding loss: **1.2433884839**
- paraphrase mean binding loss: **1.1273744355**

Relation signatures:
- mean same-option signature cosine: **0.7653450121**
- mean same-option vs strongest-wrong margin: **0.0656896873**
- mean canonicalization loss: **0.3709057843**

Mechanical invariants:
- option-order flip: **0.0**
- max probability mass error: **1.7881393433e-07**
- full-K: PASS
- state-once: PASS
- relation delta: **0.0**

## 5. Training dynamics

S13 mechanisms were learnable on TRAIN:
- total loss epoch 1: **1.6894008567**
- total loss epoch 24: **1.5273685679**
- relation-binding loss epoch 1: **1.3938245699**
- relation-binding loss epoch 24: **0.5702298631** (~59% reduction)
- canonicalization loss epoch 1: **0.3133750400**
- canonicalization loss epoch 24: **0.1200596813** (~62% reduction)

Fresh DEV showed transient relation structure but no stable primary improvement.

Fresh-DEV extrema:
- best canonical primary accuracy: **0.2942708333** at epochs 20 and 24;
- best paired both-correct: **0.046875** at epoch 24;
- best canonical relation-binding accuracy: **0.4921875** at epoch 24;
- best canonical relation margin: **-0.0872558256** at epoch 2;
- best same-option signature cosine: **0.8581791768** at epoch 4;
- best signature same-vs-wrong margin: **0.1783718355** at epoch 12;
- best primary cross-view selected-choice agreement: **0.8229166667** at epoch 9.

The important temporal pattern is decoupling:
- signature invariance becomes strong early;
- signature discrimination becomes positive and briefly exceeds 0.15;
- relation-binding accuracy continues to improve later;
- primary correctness never becomes strong;
- primary cross-view agreement eventually collapses to **0.3359375** at the selected checkpoint.

## 6. Frozen gate result

Passed:
- cross-view mean JS <= 0.05;
- option-order flip <= 0.02;
- probability-mass error <= 1e-6;
- full-K;
- relation delta = 0;
- state-once TRAIN/DEV;
- exact 49,152 trainable surface;
- A13 frozen;
- HIRACore frozen.

Failed:
- canonical accuracy >= 0.85;
- paired both-correct >= 0.75;
- question-swap change >= 0.80;
- cross-view selected-choice agreement >= 0.95;
- canonical relation-binding accuracy >= 0.80;
- canonical relation-binding signed margin >= 0.15;
- mean same-option signature cosine >= 0.90;
- mean signature same-vs-strongest-wrong margin >= 0.15.

Therefore `HIRA_V1_S13_CANONICAL_RELATION_DEV_READY` is not authorized.

## 7. Preregistered interpretation

S13 matches primarily **Outcome E**, with a secondary **Outcome C** pattern.

### Outcome E — TRAIN canonicalization improves but fresh DEV does not stay canonical

TRAIN canonicalization and binding losses improve strongly. Fresh DEV same-option cosine reaches **0.8582** early and signature margin reaches **0.1784** mid-training, but both regress by the selected checkpoint. Wording stability is therefore not robust under continued adaptation.

### Secondary Outcome C — invariance without enough semantic discrimination

Cross-view signatures become substantially more similar than A0, and the signature margin becomes positive, but not strongly or stably enough to satisfy the frozen gate. High relation similarity alone does not yield correct primary choices.

## 8. Scientific conclusion

S13 rejects the hypothesis that explicit cross-view relation-signature alignment, by itself, is sufficient to make the unchanged primary triadic decision rule semantically reliable under fresh wording.

However S13 also changes the diagnosis in an important way:

**the parameter-free relation evidence is now materially stronger than the primary decision path.**

At selected DEV:
- relation-binding accuracy: **49.22% canonical / 53.39% paraphrase**
- primary accuracy: **29.43% canonical / 27.86% paraphrase**

Therefore the next bottleneck is not only representation learning. The unchanged primary decision operator is failing to exploit the stronger relation evidence that already exists.

The evidence does **not** justify:
- tuning S13 canonicalization coefficient;
- tuning role/pair/contrastive temperatures;
- changing signature margin from exposed DEV;
- retrying seed/LR/templates;
- increasing capacity;
- reopening Laya/Jev.

The strongest next controlled hypothesis is parameter-free **canonical relation evidence fusion into the primary decision rule**, using a fixed non-tuned combination rule and wholly fresh evidence.

Production-ready remains false.
Laya/Jev parity remains unestablished.
