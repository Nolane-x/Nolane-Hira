# HIRA V1 S23 A0 receipt — Neutral Bisector Norm-Balanced Optimization

Status: **QUALIFIED / DIAGNOSTIC ONLY**

Issue: #229  
PR: #230

## Authority

Canonical A0:
- run `36721891033`
- artifact `11101960494`
- digest `sha256:bb4fed5838c5c2ad0ad10e3c20bcd89d47f1e38317cabecf427b2c0321f9798a`
- exact head `67bb65ce3416a6985f624768f9196c007a214a05`
- outcome `HIRA_V1_S23_A0_NEUTRAL_BISECTOR_READY`

Pre-authority abort:
- run `36720789774`
- failed inside diagnostic probe because `math` was not imported;
- no A0 receipt/artifact was produced;
- not scientific evidence.

## Frozen model / surface

- S21 role-gated content primary retained
- role temperature **0.10**
- role/content weights **0.50 / 0.50**
- LoRA **16,384**
- shared projection **32,768**
- physical surface **49,152**
- runtime trainable during A0 **0**
- factorization added params **0**
- canonicalizer added params **0**
- fusion added params **0**

## Inference identity / mechanics

- A13 token identity: PASS
- A13 pooled identity: PASS
- S21 logit identity: **1.0**
- S21 choice identity: **1.0**
- option-order flip: **0.0**
- fused option-order flip: **0.0**
- max probability-mass error: **1.7881393433e-07**
- fused mass error: **1.1920928955e-07**
- fusion expert-swap max abs: **0.0**
- fusion vs S14 forward max abs: **0.0**
- full-K/state-once/relation-delta-zero: PASS

## Neutral-bisector analytical court

Controlled conflict:
- pre-dot: **-0.7999999523**
- post-dot: **-0.7999999523**
- projection coefficient: **0.0**
- expected direction max abs: **0.0**
- exchange symmetry max abs: **0.0**
- exchange conflict identity: PASS
- no-conflict vs S17 max abs: **0.0**
- joint-scale max abs: **0.0**
- near-opposite finite: PASS
- both-zero exact: PASS
- primary-only exact: PASS
- relation-only exact: PASS

This proves S23 performs no asymmetric projection.

## S21 structural mechanism preserved

- synthetic role/content court: PASS
- question A selected: **0**
- question B selected: **2**
- correct vs same-role/wrong-content margin: **+0.2051138282**
- correct vs wrong-role/same-content margin: **+0.5093060732**
- role concentration max: **0.9959511757**
- factorized option permutation max abs: **0.0**
- degenerate geometry finite: PASS

## Real shared-gradient probe

- primary norm: **3.2011847496**
- relation norm: **0.2983360291**
- raw norm ratio: **10.7301312542**
- normalized pre-dot: **+0.1540553570**
- normalized post-dot: **+0.1540553570**
- conflict: false
- projection coefficient: **0.0**
- combined norm: **1.7497602701**
- physical surface: **49,152**

## Diagnostic semantic metrics

A0 is not model-selection evidence.

- primary canonical: **0.4375**
- primary paraphrase: **0.40625**
- raw-primary agreement: **0.71875**
- relation canonical: **0.28125**
- relation paraphrase: **0.40625**
- fused canonical: **0.375**
- fused paraphrase: **0.4375**
- fused paired: **0.0**
- fused agreement: **0.75**
- fused JS: **0.0201435089**
- fused canonical margin: **-0.5849592686**
- signature cosine: **0.7352106571**
- signature margin: **0.0018393844**

No tuning may be based on A0 metrics.

## Authorization consequence

S23-A0 is QUALIFIED.

Fresh S23 TRAIN/DEV may open only after:
1. interpretation plan is frozen;
2. TRAIN/DEV workflow is staged;
3. exact-head generic CI passes.

No optimizer-form change is authorized after this A0.
