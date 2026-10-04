# HIRA V1 S55 mechanical replay staging receipt

Status: **STAGED / REPLAY NOT AUTHORIZED**

Issue: #293  
PR: #294

## Failed first court

Run: `37212252407`  
Job: `111465642356`  
Scientific head: `9fd608bd43af3c05a3a1992fda0f44f021da8fe2`

Failure:
`RuntimeError: S55 reference learned-joint gradient vanished`

Exposure audit:
- pre-court gates PASS
- reference TRAIN_BEGIN reached
- reference completed epochs **0**
- treatment TRAIN_BEGIN **not reached**
- private DEV scoring **0**
- model selection **0**
- scientific verdict **none**.

## Mechanical repair

The learned transform uses zero-init B by preregistration.

The first warm-start batch may therefore produce:
- live correction gradient
- zero learned-joint gradient.

The repaired trainer:
- still requires correction gradient on every batch;
- tracks learned-joint gradient liveness across the TRAIN epoch;
- requires learned-joint gradient to become live before the first DEV evaluation;
- aborts before DEV if it never becomes live.

No changes to:
- architecture
- 65,536 learned-joint capacity
- 114,688 correction capacity
- total 180,224 private capacity
- initialization
- seed 76001
- S55 TRAIN/DEV rows
- optimizer/LR/weight decay/grad clip
- CE/JS objective
- checkpoint selector
- parent native authority.

## Replay marker

`research/HIRA-V1-S55-ENABLE-MECHANICAL-REPLAY`

The workflow accepts this marker separately from the original scientific authorization marker.

Exactly one mechanical replay may be opened only after the exact repaired final head passes generic CI on Python 3.10 + 3.12.

After repaired private DEV is reached:
- no further mechanical retry
- no scientific retry
- no second S55 DEV.
