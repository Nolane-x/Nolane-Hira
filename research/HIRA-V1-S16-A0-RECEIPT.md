# HIRA V1 S16 A0 receipt — Shared-Surface Gradient Surgery

Status: **QUALIFIED**

Issue: #210  
A0 run: `36564603152`  
Artifact: `11031676959`  
Artifact digest: `sha256:cefd83bc1decb2b6e7ebbbfe94ff8fab37d68724f9d3bced095373b4843774b1`  
Head exposed by A0: `f3fe91b23159e2f41caa227b729b48587aab4c8c`

Outcome:

`HIRA_V1_S16_A0_IDENTITY_READY`

## Identity and inference

- A13 token identity: PASS
- A13 pooled identity: PASS
- exact inherited raw-triadic logits: **1.0**
- exact inherited raw-triadic selected choices: **1.0**
- S16 vs S14 fused forward max abs: **0.0**
- candidate physical surface: **49,152**
- runtime trainable during identity A0: **0**
- original A13 trainable: **0**
- fusion/canonicalizer added params: **0**
- full-K: PASS
- raw/fused option-order flip: **0.0**
- max fused probability-mass error: **1.7881393433e-07**

## Real 49,152-parameter shared-gradient probe

Fresh A0 batch:
- semantic cases: **16**
- primary gradient norm: **10.8336782455**
- relation gradient norm: **0.4236361384**
- pre-surgery dot: **+0.5877634883**
- gradient cosine: **+0.1280659485**
- conflict: **false**
- projection coefficient: **0.0**
- projected primary norm: **10.8336782455**
- post-projection primary/relation dot: **+0.5877634883**
- combined gradient norm: **10.8960361481**

This is a qualified no-op case: the real primary and relation gradients were already aligned, so the preregistered surgery correctly left the primary gradient unchanged.

This does not establish that S16 TRAIN will have zero conflicts. A0 is one fresh diagnostic batch only; conflict rate across the full fresh TRAIN trajectory is preregistered as a diagnostic.

## Fresh semantic baseline

- fused canonical accuracy: **0.34375**
- fused paraphrase accuracy: **0.5**
- raw triadic canonical accuracy: **0.28125**
- relation canonical accuracy: **0.4375**
- relation paraphrase accuracy: **0.4375**
- signature cosine: **0.7297229767**
- signature margin: **0.0012217909**

A0 is never used for model selection.
No TRAIN/DEV row was exposed by A0.
