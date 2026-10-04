# HIRA V1 S49 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #281  
PR: #282

## Parent

S48 merged main:
`721b506688044f75b142e7a0f41a63cf74663674`

S48 scientific verdict:
**DOMINATED NEGATIVE**.

## Qualified S49-A0

Run: `37178164136`  
Artifact: `11293039754`  
Digest: `sha256:ce07ed474c21185d5a4316550e116606d7c09b90c808aea028354f2f046678e0`  
Authorization head: `b793e54e8c969a110859a9966d99ed3f69ec7ffd`

Outcome:
`HIRA_V1_S49_A0_PRIVATE_QUERY_FREE_OPTION_IDENTITY_READY`

A0 proves:
- query-free identity params 0
- identity API has no question inputs
- K=3/7/255
- option/state/token/view permutation invariance
- masked padding invariance
- raw query remains live after identity
- same correction capacity 114,688
- total treatment 163,840
- one encoder batch
- exact native/private ownership
- W -> B -> A warm-start.

## Fresh authority

Frozen:
- seed **70001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S49 domains
- K=4
- 24 epochs
- batch 16
- one DEV authority.

Freshness guards reject:
- prior exposed surfaces through S45;
- S48 TRAIN/DEV exact rows;
- S49-A0 exact rows.

## Matched arms

Reference:
- question-conditioned native relation signature
- raw normalized query
- private A/B/W correction

Treatment:
- query-free state↔option identity
- same raw normalized query
- same A/B/W correction

Matched:
- correction initialization exactly equal
- correction params 114,688 per arm
- total trainable 163,840 per arm
- identical native initialization/updates
- identical optimizer/LR/weight decay/clip
- identical CE + cross-view JS coefficients
- identical checkpoint-selection rule
- identical fresh rows/order.

Treatment identity has no query input and adds zero parameters.

## Staged files

- `src/nmd/v1_s49_authority.py`
- `tests/test_v1_s49_authority.py`
- `scripts/hira_v1_s49_train_dev.py`
- `tests/test_v1_s49_matched_harness.py`
- `.github/workflows/hira-v1-s49-matched-query-free-option-identity-train-dev.yml`

## Authorization rule

Matched workflow is marker-gated on:

`research/HIRA-V1-S49-ENABLE-TRAIN-DEV`

The marker MUST remain absent until the exact final pre-DEV staging head passes generic CI on Python 3.10 and 3.12.

After authorization:
- exactly one fresh S49 TRAIN/DEV court;
- no post-DEV identity operator tuning;
- no pair-temperature/direct-relative sweep;
- no query leak;
- no identity blend/projector;
- no retry for scientific weakness;
- no second S49 DEV;
- no external Laya/Jev evaluation unless separately confirmed DEV_READY.
