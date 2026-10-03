# HIRA V1 S44 mechanical abort receipt — pre-DEV import failure

Status: **MECHANICAL ABORT / NO SEMANTIC DEV EXPOSURE**

Issue: #271
PR: #272

## Aborted run

- workflow run `37122580868`
- authorization head `6810d6aa34c029e7e7eaf7e204f62b52bcc3cbcd`
- failed step: **10 — Run one fresh matched S44 TRAIN DEV court**

## Exact failure

The trainer exited during Python import before constructing TRAIN/DEV rows:

`ModuleNotFoundError: No module named 'nmd.v1_private_correction_representation_fusion'`

The intended import is:

`from nmd.v1_gradient_isolated_fusion import GradientIsolatedFullKEvidenceFusion`

This bad module name was introduced mechanically by a stage-name transform.

## Exposure audit

From the completed job log:

- epoch lines emitted: **0**
- S44 TRAIN/DEV receipt lines emitted: **0**
- fresh DEV metrics emitted: **0**
- selected DEV epoch: **none**
- checkpoints/artifact: **none**
- receipt verifier: skipped
- integrity freeze: skipped
- artifact upload: skipped

The process failed approximately immediately after invoking the trainer and before semantic data generation/evaluation.

Therefore the frozen S44 one-DEV scientific budget is **not consumed** by run `37122580868`.

## Repair

Mechanical-only repair:
- restore the existing canonical import `nmd.v1_gradient_isolated_fusion`;
- add a regression assertion that forbids the nonexistent transformed module name.

No scientific mechanism changed:
- architecture unchanged;
- A/B/W capacity unchanged;
- objective unchanged;
- optimizer unchanged;
- data authority unchanged;
- seed unchanged;
- epochs/batch unchanged;
- selector/gates unchanged.

A rerun may be authorized only after the repaired exact head passes generic CI.

No additional A0 is authorized or required.
