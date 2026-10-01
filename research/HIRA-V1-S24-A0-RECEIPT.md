# HIRA V1 S24 A0 receipt — Reliability-Weighted Full-K Expert Fusion

Status: **QUALIFIED / DIAGNOSTIC ONLY**

Issue: #231  
PR: #232

## Authority

Canonical A0:
- run `36793018851`
- artifact `11133405764`
- digest `sha256:4eed0771f2e2037a9cccbee547ffc7831140e53781256ff2b4dd9ce1c67810b3`
- exact head `970768c70d3b558a03258863b4c6f3ae09dfe02d`
- outcome `HIRA_V1_S24_A0_RELIABILITY_FUSION_READY`

Earlier A0 attempts `36790788592` and `36792147081` are harness aborts only. Neither produced qualified scientific authority.

## Frozen surface

- S23 primary/relation experts retained
- A13 LoRA **16,384**
- shared projection **32,768**
- physical trainable surface **49,152**
- runtime trainable during A0 **0**
- role/content factorization added params **0**
- canonicalizer added params **0**
- reliability fusion added params **0**
- relation refinement: **off**

## Identity and isolation

- A13 token output identity: PASS
- A13 pooled output identity: PASS
- S23 primary logits identity: **1.0**
- S23 primary choices identity: **1.0**
- S23 relation logits identity: **1.0**
- S23 relation choices identity: **1.0**
- primary fused-path gradient L1: **0.9414215088**
- relation direct gradient from fused-primary objective: **0**
- relation auxiliary gradient remains nonzero
- full-K/state-once preserved
- state encode calls: **32**

Only the fusion operator changes.

## Reliability-fusion analytical court

- expert-swap max abs: **0**
- option-permutation max abs: **0**
- equal-gap vs S14 max abs: **0**
- equal reliability weights: **0.5 / 0.5**
- flat/flat fused max abs: **0**
- flat/flat weights: **0.5 / 0.5**
- flat/nonflat primary weight: **7.3950895e-7**
- flat/nonflat relation weight: **0.9999992847**
- independent positive-affine invariance max abs: **5.9604645e-8**
- weight-sum error: **5.9604645e-8**
- reliability parameter count: **0**
- `fusion_vs_s14_forward_max_abs`: **0.7367611527**

The last value is expected: S24 is intentionally not fixed 50/50 when expert reliabilities differ.

## Preserved mechanics

- synthetic role/content court: PASS
- correct vs same-role/wrong-content margin: **+0.2051138282**
- correct vs wrong-role/same-content margin: **+0.5093060732**
- role concentration max: **0.9959511757**
- factorized option permutation max abs: **0**
- fused option-order flip: **0**
- fused probability-mass error: **1.1920929e-7**
- neutral-bisector projection coefficient: **0**
- shared-surface gradient cosine: **+0.4512995780**
- shared-surface primary/relation raw norm ratio: **8.2063540198**

## Diagnostic semantic metrics

A0 is not model-selection evidence.

- primary canonical: **0.375**
- primary paraphrase: **0.25**
- relation canonical: **0.375**
- relation paraphrase: **0.3125**
- fused canonical: **0.375**
- fused paraphrase: **0.34375**
- fused paired: **0**
- fused agreement: **0.625**
- fused JS: **0.0199398845**
- fused canonical margin: **-0.9124498367**
- expert top1 agreement: **0.796875**
- same-option signature cosine: **0.7683802843**
- signature margin: **-0.0010229198**

These numbers are diagnostic only and MUST NOT be used to tune the fusion rule, epsilon, optimizer, seed, objective, or thresholds.

## Authorization consequence

S24-A0 is QUALIFIED.

Fresh S24 TRAIN/DEV may open only after:
1. this receipt is frozen;
2. the S24 interpretation plan is frozen;
3. the S24 TRAIN/DEV workflow is staged;
4. exact-head generic CI passes;
5. a dedicated `HIRA-V1-S24-ENABLE-TRAIN-DEV` authorization is committed.

No post-A0 fusion formula or epsilon change is authorized.
