# HIRA V1 S45 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #273  
PR: #274

## Parent

S44 merged main:
`271511ed449be87d1ba5d826a1883a807efe9a2f`

S44 fresh scientific court:
- run `37123003224`
- artifact `11274687323`
- interpretation **Case A**
- treatment DEV_READY false

## Qualified S45-A0

Run: `37128007831`  
Artifact: `11275950524`  
Digest: `sha256:90ece8fe112a14c44b29e03d217efa372f857b47d1bfa52cd58a208e75ed924e`  
Authorization head: `d9245996126cfe668131393e4ebab2f38cb55d8a`

Outcome:
`HIRA_V1_S45_A0_CROSS_VIEW_CONSISTENT_PRIVATE_CORRECTION_READY`

A0 proves:
- native trainable 49,152
- correction-only 114,688
- treatment total 163,840
- one encoder pass / state-once
- identical-logit JS = 0
- deterministic non-identical JS > 0
- JS-only gradients reach A/B/W
- JS-only native gradient = 0
- correction/native ownership remains exact
- W -> B -> A warm-start remains live
- arbitrary-K, permutation, padding, checkpoint and full-K courts PASS

## Frozen S45 variable

Architecture is exactly S44.

Only correction objective changes:

`L_corr = 0.10 * CE_corr + 0.25 * JS_corr_cross_view`

No:
- capacity change
- temperature
- alternate divergence
- learned fusion gate/weight
- native-gradient leakage
- second encoder
- coefficient sweep.

## Fresh matched authority

Frozen:
- seed **66001**
- TRAIN **768**
- DEV **192**
- 12 fresh S45 domains
- K=4
- 24 epochs
- batch 16
- identical rows/order across arms
- one DEV only

Freshness guards reject:
- exact S0-S44 exposed rows
- S44 TRAIN/DEV
- S44-A0
- S45-A0
- M5 final/confirm
- W29-W34 sealed rows.

## Staged files

- `research/HIRA-V1-S45-CONTRACT.md`
- `research/HIRA-V1-S45-INTERPRETATION-PLAN.md`
- `src/nmd/v1_s45_authority.py`
- `scripts/hira_v1_s45_train_dev.py`
- `.github/workflows/hira-v1-s45-matched-cross-view-consistent-private-correction-train-dev.yml`
- `tests/test_v1_s45_matched_harness.py`

## Authorization rule

The matched workflow is marker-gated on:

`research/HIRA-V1-S45-ENABLE-TRAIN-DEV`

That marker MUST NOT exist until the **exact staging head** passes generic CI on Python 3.10 and 3.12.

After authorization:
- exactly one fresh matched TRAIN/DEV run;
- no retry for scientific weakness;
- no post-DEV tuning;
- no second S45 DEV;
- no external Laya/Jev evaluation unless a separately confirmed DEV_READY candidate exists.
