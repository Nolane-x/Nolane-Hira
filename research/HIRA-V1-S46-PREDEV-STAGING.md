# HIRA V1 S46 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #275  
PR: #276

## Parent

S45 merged main:
`ec2988391f6e73b99ed5d1057f44e9c2073fae67`

S45 fresh scientific court:
- run `37128945285`
- artifact `11276882786`
- interpretation **Case C**
- treatment DEV_READY false

## Qualified S46-A0

Run: `37134236894`  
Artifact: `11278770039`  
Digest: `sha256:076a4aa432d3ace98cac0e1836ceaa547bc76e6b377af01812e6f32d2d7a54a3`  
Authorization head: `84c8287ad9ce6099a2355d2c2946b86df85e467f`

Outcome:
`HIRA_V1_S46_A0_ROBUST_THREE_EXPERT_CONSENSUS_READY`

A0 proves:
- fusion parameter count 0
- K=3/K=7/K=255 PASS
- option permutation equivariance
- independent per-expert shift/positive-scale invariance
- identical-three-expert identity
- one-extreme-outlier containment
- flat-expert finiteness
- probability mass PASS
- actual robust shell differs materially from legacy S45 shell
- native/correction ownership remains exact
- one encoder batch / state-once

## Frozen S46 variable

Training mechanics remain exact S45:
- native 49,152
- correction-only 114,688
- treatment total 163,840
- correction objective `0.10 CE + 0.25 JS`
- seed changes only for fresh authority: **67001**

Decision comparison at the exact same selected treatment checkpoint:

Legacy:
`0.5 * (std(primary) + std(corrected_relation))`

S46:
`median(std(primary), std(native_relation), std(corrected_relation))`

No learned fusion parameter, gate, threshold or temperature.

## Fresh authority

Frozen:
- seed **67001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S46 domains
- K=4
- 24 epochs
- batch 16
- identical rows/order across native training arms
- one DEV authority only.

Freshness guards reject:
- prior exact exposed surfaces through S45
- S45 TRAIN/DEV exact rows
- S46-A0 exact rows.

## Staged files

- `research/HIRA-V1-S46-CONTRACT.md`
- `research/HIRA-V1-S46-INTERPRETATION-PLAN.md`
- `research/HIRA-V1-S46-A0-RECEIPT.json`
- `research/HIRA-V1-S46-A0-RECEIPT.md`
- `src/nmd/v1_s46_authority.py`
- `scripts/hira_v1_s46_train_dev.py`
- `tests/test_v1_s46_authority.py`
- `tests/test_v1_s46_matched_harness.py`
- `.github/workflows/hira-v1-s46-matched-robust-three-expert-consensus-train-dev.yml`

## Authorization rule

Matched workflow is marker-gated on:

`research/HIRA-V1-S46-ENABLE-TRAIN-DEV`

The marker MUST NOT exist until the exact final staging head passes generic CI on Python 3.10 and 3.12.

After authorization:
- exactly one fresh S46 TRAIN/DEV court;
- checkpoint selection remains exact S45 before robust-shell evaluation;
- no post-DEV shell/threshold/temperature tuning;
- no alternate aggregator;
- no second S46 DEV;
- no external Laya/Jev evaluation unless separately confirmed DEV_READY.
