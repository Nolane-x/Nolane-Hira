# HIRA V1 S15 A0 receipt — Gradient-Isolated Evidence Fusion

Status: **QUALIFIED**

Issue: #207  
A0 run: `36556776351`  
Artifact: `11027644171`  
Artifact digest: `sha256:2f9025fc14c5bc6ca8c816834d7c5f0cf19c858cff61be93a8edc4c260145e5a`  
Head exposed by A0: `de8c31b3c5c3a8cf81beb2abf60bd548a06fd100`

Outcome:

`HIRA_V1_S15_A0_IDENTITY_READY`

## Identity / capacity

- A13 token identity: PASS
- A13 pooled identity: PASS
- exact inherited raw-triadic logits: **1.0**
- exact inherited raw-triadic choices: **1.0**
- candidate physical surface: **49,152**
- runtime trainable during A0: **0**
- original A13 trainable: **0**
- canonicalizer-added params: **0**
- fusion-added params: **0**
- full-K: PASS
- raw option-order flip: **0.0**
- fused option-order flip: **0.0**
- max fused probability-mass error: **1.1920928955078125e-07**

## Gradient-route proof

S15 protected fusion versus S14 fusion:
- forward max abs difference: **0.0**

Direct fused-primary CE:
- triadic-logit gradient L1: **1.1707811356**
- relation-logit direct gradient L1: **0.0**

Dedicated relation CE:
- relation-logit gradient L1: **1.6726777554**

Therefore:
- S15 inference is numerically identical to S14;
- fused-primary directly trains triadic evidence;
- fused-primary does not directly train relation logits;
- relation auxiliary still trains relation evidence.

## Fresh semantic baseline

- raw triadic canonical accuracy: **0.21875**
- raw triadic paraphrase accuracy: **0.375**
- relation canonical accuracy: **0.34375**
- relation paraphrase accuracy: **0.5**
- fused canonical accuracy: **0.34375**
- fused paraphrase accuracy: **0.40625**
- fused cross-view agreement: **0.71875**
- fused canonical signed margin: **-0.5820423961**
- same-option signature cosine: **0.7403765321**
- signature same-vs-strongest-wrong margin: **0.0084915329**

A0 is diagnostic only and was not used for model selection.

A later generic-CI checkpoint-loader naming mismatch did not affect A0. It was corrected before TRAIN/DEV authorization; no TRAIN/DEV row had been exposed.
