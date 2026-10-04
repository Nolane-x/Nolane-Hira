# HIRA V1 S55 mechanical pre-DEV abort receipt

Status: **MECHANICAL PRE-DEV ABORT / NO PRIVATE DEV EXPOSURE**

Failed run: `37212252407`  
Job: `111465642356`  
Scientific head: `9fd608bd43af3c05a3a1992fda0f44f021da8fe2`

## What completed

Pre-court gates all PASS:
- exact S55 stack
- S55 contracts
- canonical S55-A0
- sealed S51 native authority
- frozen M4 integrity.

Reference branch emitted:
`HIRA_V1_S55_REFERENCE_TRAIN_BEGIN`

Then it failed in the **first TRAIN batch of epoch 1**.

No:
- completed reference epoch
- treatment TRAIN_BEGIN
- private DEV score
- model selection
- A/B/C/D/E verdict.

Therefore S55 scientific evidence remains unexposed.

## Mechanical cause

S55 uses zero-initialized learned-transform B for warm-start preservation.

At the first optimizer step:
- correction gradients are live;
- learned-joint gradients can legitimately be zero because the neutral correction shell has not yet opened a gradient path back into the zero-init learned relation transform.

The original trainer incorrectly required a nonzero learned-joint gradient on **every individual batch**, including the first warm-start batch.

## Mechanical repair

No scientific variable changes.

Repair:
- correction gradient must remain live every batch;
- learned-joint gradient may be zero on initial warm-start batches;
- learned-joint gradient must become nonzero **within the TRAIN epoch**;
- this condition is checked **before any DEV metrics are evaluated**.

Thus the court still refuses a dead learned path, but no longer rejects the preregistered zero-init warm start itself.

## Replay governance

Exactly one mechanical replay may be authorized after:
- repair regression tests PASS;
- exact final repaired head passes CI on Python 3.10 + 3.12.

No:
- architecture change
- initialization change
- seed change
- data change
- optimizer/loss/selector change
- capacity change
- second scientific retry after private DEV exposure.

If the repaired court reaches private DEV, S55 DEV authority is consumed there.
