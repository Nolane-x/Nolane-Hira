# HIRA V1 S46 mechanical abort receipt

Status: **PRE-TRAIN MECHANICAL ABORT / DEV BUDGET NOT CONSUMED**

Attempted run: `37135134289`  
Authorization commit: `9bcea5d64ef099d193ed541332529d647a5204e8`

## Failure point

The workflow stopped at:

`Compile and run S46 contracts`

because the staging-era regression test
`test_s46_train_dev_marker_not_armed_during_staging`
continued asserting that the authorization marker must not exist even after the marker had intentionally been created to start the one-shot court.

This is a harness lifecycle error, not a scientific or model result.

## Exposure audit

Completed:
- checkout
- M4 artifact download
- S46-A0 artifact download
- Python setup
- dependency install

Failed:
- contract suite

Skipped:
- canonical S46-A0 verification
- M4 integrity verification
- `Run one fresh S46 TRAIN DEV court`
- matched receipt verification
- integrity freeze
- artifact upload

Log audit found:
- no `HIRA_V1_S46_ARM_EPOCH`
- no `HIRA_V1_S46_TRAIN_DEV_RECEIPT`
- no invocation of `hira_v1_s46_train_dev.py --bundle ...`
- no TRAIN epoch
- no DEV metric
- no selected checkpoint
- no scientific artifact

Therefore **the S46 fresh DEV authority remains unconsumed**.

## Mechanical repair

The harness is changed from a staging-only absence assertion to:
- allow the marker to be absent during staging;
- when present, verify the marker's frozen authority content.

No scientific variable changes:
- seed remains 67001
- TRAIN/DEV remain 768/192
- 12 domains remain fixed
- training remains exact S45
- S46 shell remains coordinate-wise three-expert median
- no threshold/temperature/gate/weight changes.

A re-arm requires a fresh exact-head generic CI PASS before the marker is updated.
