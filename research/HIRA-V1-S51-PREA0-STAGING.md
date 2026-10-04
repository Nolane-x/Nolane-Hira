# HIRA V1 S51 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #285  
PR: #286

## Parent

S50 merged main:
`61f0d596ef277a1469b78f2db90c393c7c859321`

S50 closure:
**INVALID REPRODUCIBILITY COURT / INCONCLUSIVE**

Key S50 evidence:
- fresh run `37185080959`: native 24/24 completed, private DEV unexposed, then cache inference-tensor abort;
- hash-locked replay `37188685173`: native hashes differed at all 24 epochs;
- no S50 private DEV result was accepted.

## Frozen S51 design

S51 has two structurally separate phases.

### Phase A — persisted native authority

- S51 TRAIN only
- seed **72001**
- 12 wholly fresh domains
- 768 TRAIN cases
- fixed native epoch **24**
- no S51 DEV generation/import
- no private correction construction
- native trainable surface **49,152**
  - LoRA **16,384**
  - projection **32,768** / shape **128x256**
- output:
  - `native-authority.pt`
  - `train-manifest.json`
  - `authority-receipt.json`
  - integrity receipt.

### Phase B — artifact-pinned private court

- loads exact Phase-A checkpoint bytes
- verifies file SHA + logical native tensor digest + T0/revision binding
- freezes all native parameters
- native optimizer count **0**
- no native training helper call
- generates S51 TRAIN/DEV only after artifact load
- materializes one shared immutable cache
- reference/treatment consume identical native evidence
- correction params **114,688** each
- identity params **0**
- private seed **72001**
- private epochs **24**
- one DEV authority.

## Persisted authority core

`src/nmd/v1_persisted_native_authority.py`

Mechanics:
- logical tensor digest independent of torch serialization metadata
- exact LoRA key validation
- projection shape **128x256**
- file SHA tamper detection
- semantic revision binding
- T0 authority binding
- runtime hash equality after load
- loaded runtime fully frozen.

## Fresh S51 authority

`src/nmd/v1_s51_authority.py`

- TRAIN **768**
- DEV **192**
- 12 new S51 domains
- no exact S50 TRAIN/DEV state/question/option overlap
- separate public TRAIN and DEV generation APIs.

## A0 staged

- `scripts/hira_v1_s51_a0_persisted_native_authority.py`
- `tests/test_v1_s51_a0_harness.py`
- `.github/workflows/hira-v1-s51-a0-persisted-native-authority.yml`

A0 proves:
- save/load runtime hash exact
- second-load/order identity
- file tamper rejection
- loaded native trainable/optimizer surface 0
- cache normal non-inference tensors
- cache detached/contiguous
- branch-order replay exact
- correction initialization bit-identical
- reference/treatment correction params 114,688
- identity params 0
- Phase-A DEV/private isolation
- Phase-B native-training isolation.

## Authorization rule

A0 workflow is marker-gated on:

`research/HIRA-V1-S51-ENABLE-A0`

The marker MUST remain absent until the exact final staging head passes generic CI on Python 3.10 and 3.12.

A0 is mechanical only:
- no model selection
- no Phase-A native authority exposure
- no S51 DEV exposure
- no external Laya/Jev evaluation.
