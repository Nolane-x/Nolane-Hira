# HIRA V1 S27 A0 receipt — Blockwise Cross-View Factor Canonicalization

Status: **QUALIFIED / DIAGNOSTIC SEMANTICS ONLY**

Issue: #237  
PR: #238

## Canonical authority

- run `36859567980`
- artifact `11161461301`
- artifact digest `sha256:2c9f61c218679f8ed24f557fda56436a63c2109520385b48c6583e77d3cb06d0`
- authority head `93fc8ebef17ca936872c5c72899bacfd232cfd67`
- outcome `HIRA_V1_S27_A0_BLOCKWISE_CANONICALIZATION_READY`

Earlier run:
- `36858674182`: non-result harness abort caused by direct-script import path.

That run encoded no S27 case, produced no qualified receipt, and uploaded no artifact.

## Exact inference identity vs S26

- shared encoder identity: PASS
- primary inference max abs: **0**
- relation inference max abs: **0**
- factorized signature inference max abs: **0**
- fused inference max abs: **0**

S27 therefore changes no inference path.

## Frozen capacity

- shared A13 LoRA: **16,384**
- primary-private projection: **32,768**
- relation-private projection: **32,768**
- exact physical surface: **81,920**
- canonicalization-added params: **0**
- A0 runtime trainable params: **0**
- original A13 trainable params: **0**
- HIRACore trainable params: **0**
- factorized signature width: **256**
- role block: **128**
- value block: **128**

## Block-localization court

Identical well-separated views:
- total: **0**
- role: **0**
- value: **0**

Role-only mismatch:
- role total: **0.35**
- value total: **0**

Value-only mismatch:
- role total: **0**
- value total: **0.35**

Both-factor mismatch:
- role total: **0.35**
- value total: **0.35**

Option permutation:
- total error: **0**
- role error: **0**
- value error: **0**

Thus the blockwise objective activates only the intended factor in the synthetic localization court.

## Real A0 loss diagnostics

Diagnostic only:
- total blockwise canonicalization: **0.6128444672**
- role-block total: **0.5250456929**
- value-block total: **0.7006432414**

These values are not tuning evidence.

## Gradient ownership

- primary → relation-private leakage: **0**
- relation → primary-private leakage: **0**
- primary-private gradient L1: **130.7865600586**
- relation-private gradient L1: **58.4642715454**
- primary shared-LoRA gradient L1: **3.9276845455**
- relation shared-LoRA gradient L1: **1.6988719702**
- neutral-bisector projection coefficient: **0**
- combined shared norm: **0.0689116195**

S25/S26 ownership semantics remain intact.

## Full-K / numerical court

- primary option permutation max abs: **0**
- relation option permutation max abs: **0**
- signature option permutation max abs: **0**
- fused option-order flip: **0**
- probability-mass error: **1.1920929e-7**
- full-K: PASS
- state-once views: **32**

## A0 semantic diagnostics

Not model-selection evidence:
- primary canonical: **21.875%**
- relation canonical: **21.875%**
- fused canonical: **15.625%**

A0 semantic accuracy MUST NOT tune block weights, separation margin, canonicalization coefficient, seed, optimizer, capacity, selector, or DEV gates.

## Consequence

S27-A0 is QUALIFIED.

Fresh S27 TRAIN/DEV may open only after:
1. this receipt is frozen;
2. the frozen interpretation plan remains unchanged;
3. fresh S27 authority generator remains frozen;
4. TRAIN/DEV workflow is staged;
5. exact-head generic CI passes;
6. a separate one-shot TRAIN/DEV authorization is committed.

No further S27-A0 run is authorized.
