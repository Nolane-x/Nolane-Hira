# HIRA V1 S17 A0 receipt — Norm-Balanced Shared Gradient Optimization

Status: **QUALIFIED**

Issue: #214
A0 run: `36574252783`
Artifact: `11034934999`
Artifact digest: `sha256:4a29beb1cf410f3f98585367d6a0dc0dc6f73be37aee64b91e4030d814f0f0fe`
Head exposed by A0: `27343f2c500629051e5365db88f45fe91dd71c86`

Outcome:
`HIRA_V1_S17_A0_IDENTITY_READY`

## Identity / mechanics

- exact A13 token identity: PASS
- exact A13 pooled identity: PASS
- exact inherited raw-triadic logits: 1.0
- exact inherited raw-triadic choices: 1.0
- S17 fused inference vs S14: max abs **0.0**
- physical candidate surface: **49,152**
- runtime trainable during A0 identity: **0**
- full-K/state-once/option permutation: PASS
- fusion/canonicalizer/balancer learned params: **0**

## Real shared-gradient probe

Fresh 16-case A0:
- primary norm: **16.3065052032**
- relation norm: **0.3229722381**
- raw norm ratio primary/relation: **50.4888757653**
- normalized pre-dot / cosine: **+0.0113130677**
- conflict: **false**
- reference scale: **8.3147382736**
- direction norm: **1.4221906662**
- final combined norm: **8.3147382736**
- projection coefficient: **0.0**
- epsilon: **1e-12**

The A0 batch demonstrates the exact bottleneck S17 targets: very large raw norm imbalance despite nearly orthogonal normalized objective directions.

The fixed balancing operator gives equal directional influence and preserves current-batch scale using the arithmetic mean raw norm.

## Fresh semantic baseline

- fused canonical accuracy: **0.375**
- relation canonical accuracy: **0.46875**
- same-option signature cosine: **0.7370818853**
- signature margin: **0.0065694200**

A0 is diagnostic only and was not used for model selection.

A later checkpoint replay symbol mismatch was harness-only and was corrected before TRAIN/DEV authorization. No S17 TRAIN/DEV row was exposed.
