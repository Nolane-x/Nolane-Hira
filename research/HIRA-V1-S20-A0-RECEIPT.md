# HIRA V1 S20 A0 receipt — Standardized Triadic Evidence Consistency

Status: **QUALIFIED**

Issue: #223  
A0 run: `36677124443`  
Artifact: `11080820457`  
Artifact digest: `sha256:39769042b9620b5444f5a478a1ef12d63acb43c7abbd5332b8978b5177be4b06`  
Head exposed by A0: `705539dddb8316f4bcfd5cf6a08d7885aba29d1a`

Outcome:

`HIRA_V1_S20_A0_IDENTITY_READY`

## Identity / capacity

- A13 token identity: PASS
- A13 pooled identity: PASS
- inherited raw-triadic logit identity: **1.0**
- inherited raw-triadic choice identity: **1.0**
- LoRA params: **16,384**
- projection params: **32,768**
- physical candidate surface: **49,152**
- runtime trainable during A0: **0**
- original A13 trainable: **0**
- canonicalizer-added params: **0**
- fusion-added params: **0**
- full-K/state-once: PASS
- option-order flip: **0.0**
- fused option-order flip: **0.0**
- max probability-mass error: **1.1920928955e-07**

## Inference identity

S20 protected fusion versus inherited S14 family:
- forward max abs difference: **0.0**
- expert-swap max abs: **0.0**

Direct fused-primary gradient routing remains unchanged:
- triadic-logit gradient L1: **1.1707811356**
- relation-logit direct gradient L1: **0.0**
- relation auxiliary gradient L1: **1.6726777554**

## Standardized-consistency proof

Frozen coefficient: **0.25**  
Frozen epsilon: **1e-6**

- identical evidence loss: **0.0**
- mismatched relative-evidence loss: **0.4353144765**
- view-swap max abs: **0.0**
- option-permutation max abs: **0.0**
- positive-affine standardized-evidence max abs: **2.3841857910e-07**
- common positive-scale loss max abs: **8.9406967163e-08**
- canonical gradient L1: **0.8270463347**
- paraphrase gradient L1: **0.7869513631**
- gradients nonzero on both views: PASS
- flat evidence loss: **0.0**
- flat evidence neutral zero vector: PASS
- flat evidence finite: PASS
- flat rate probe: **1.0**
- added learned parameters/state: **0**

## Why S20 is materially different from S19

Fresh A0 raw triadic evidence:
- mean raw triadic RMS: **0.0001218886**
- raw-triadic canonical accuracy: **0.3125**
- raw-triadic paraphrase accuracy: **0.125**
- raw-triadic cross-view agreement: **0.46875**
- raw-triadic softmax JS: **7.4053794e-09**

Yet the standardized consistency probe on relative evidence produces:
- mismatch loss **0.4353144765**
- strong finite gradients on both views.

This directly validates the S20 motivation: center/RMS evidence geometry exposes relative ranking differences that raw-softmax JS can almost entirely hide.

## Fresh semantic baseline

Fused:
- canonical accuracy: **0.375**
- paraphrase accuracy: **0.21875**
- paired both-correct: **0.0**
- cross-view agreement: **0.78125**
- cross-view mean JS: **0.0175077878**
- canonical signed margin: **-0.7609439492**

Relation:
- canonical accuracy: **0.375**
- paraphrase accuracy: **0.40625**
- cross-view agreement: **0.6875**
- canonical signed margin: **-0.3802713752**

Relation signatures:
- same-option cosine: **0.6742902398**
- same-vs-strongest-wrong margin: **-0.0127258804**

A0 is diagnostic only and was not used for model selection.
