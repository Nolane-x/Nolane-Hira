# HIRA V1 S47 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #277  
PR: #278

## Parent

S46 merged main:
`540694b63b47d5209c2a35730e2fffbbed783429`

S46 fresh scientific court:
- run `37163095749`
- artifact `11289416980`
- interpretation **Case C**
- robust DEV_READY false

## Qualified S47-A0

Run: `37167755900`  
Artifact: `11290671268`  
Digest: `sha256:b2505aeac631498e606025ceb5848ea7dd0faf738387d9920fc672230de671c3`  
Authorization head: `de7fe028abe8e931637d3a2d4ff5320b2819022e`

Outcome:
`HIRA_V1_S47_A0_ORDINAL_PAIRWISE_CONSENSUS_READY`

A0 proves:
- decision parameter count 0
- K=3/K=7/K=255 PASS
- exact option permutation equivariance
- exact independent positive-affine invariance
- exact strictly increasing nonlinear rank-transform invariance
- exact extreme-magnitude blow-up invariance
- two-identical-expert pairwise dominance
- strict Copeland dominance over private tie-break
- deterministic private ordinal tie-break
- one encoder batch / state-once
- exact native/private ownership

## Frozen S47 variable

Training mechanics remain exact S45:
- native 49,152
- correction-only 114,688
- treatment total 163,840
- correction objective `0.10 CE + 0.25 JS`
- fresh seed **68001**

Decision comparison at the exact same selected checkpoint:

Legacy:
`0.5 * (std(primary) + std(corrected_relation))`

S47:
1. pairwise ordinal vote from primary/native/corrected
2. 3-expert majority per option pair
3. Copeland score
4. corrected-private ordinal tie-break
5. exact K-derived lexicographic base `2K-1`

No raw score magnitude enters after ranking.
No learned parameter, scalar, threshold, temperature or gate.

## Fresh authority

Frozen:
- seed **68001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S47 domains
- K=4
- 24 epochs
- batch 16
- identical rows/order across native training arms
- one DEV authority only.

Freshness guards reject:
- prior exact exposed surfaces through S45
- S46 TRAIN/DEV exact rows
- S47-A0 exact rows.

## Staged files

- `research/HIRA-V1-S47-CONTRACT.md`
- `research/HIRA-V1-S47-INTERPRETATION-PLAN.md`
- `research/HIRA-V1-S47-A0-RECEIPT.json`
- `research/HIRA-V1-S47-A0-RECEIPT.md`
- `src/nmd/v1_s47_authority.py`
- `scripts/hira_v1_s47_train_dev.py`
- `tests/test_v1_s47_authority.py`
- `tests/test_v1_s47_matched_harness.py`
- `.github/workflows/hira-v1-s47-matched-ordinal-pairwise-consensus-train-dev.yml`

## Authorization rule

Matched workflow is marker-gated on:

`research/HIRA-V1-S47-ENABLE-TRAIN-DEV`

The marker MUST NOT exist until the exact final staging head passes generic CI on Python 3.10 and 3.12.

After authorization:
- exactly one fresh S47 TRAIN/DEV court;
- checkpoint selection remains exact S45 before ordinal-shell evaluation;
- no post-DEV rank/threshold/tie-break tuning;
- no alternate aggregator;
- no second S47 DEV;
- no external Laya/Jev evaluation unless separately confirmed DEV_READY.
