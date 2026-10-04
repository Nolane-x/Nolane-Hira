# HIRA V1 S55 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #293  
PR: #294

## Parent

S54 merged main:
`b95e4b8d1e510f9fa938685a915f92e894627110`

S54 fresh scientific court:
- run `37209555118`
- artifact `11305929426`
- digest `sha256:1b9787fe775493e53d7f7f0108295914cd974bfd8da9f13a8f03eb02f7109ae2`
- verdict **Case C**.

## Frozen S55 design

Reference and treatment both contain:
- correction **114,688**
- learned joint transform **65,536**
- identity params **0**
- total private trainable **180,224**
- identical initialization.

Learned transform:
- input 768D = query↔option context + explicit state-conditioned channel + option identity
- A 64x768
- B 256x64
- seed 75555
- B zero-init
- residual output preserves query context at initialization.

Controlled variable:
- reference explicit state channel = zeros
- treatment explicit state channel = S54 joint state-query-option context.

No direct S54 context bypass into correction logits.

## Parent native authority

Exact S51 persisted native artifact:
- run `37192490832`
- artifact `11299783210`
- runtime/native digest `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`
- native trainable params 0
- no native retraining.

## Core staged

- `src/nmd/v1_learned_joint_relation_interaction.py`
- `tests/test_v1_learned_joint_relation_interaction.py`

Core CI on head `db4bca40ee146670bdce6313c00bcd999c5ca6b9`:
- Python 3.10 PASS
- Python 3.12 PASS.

## A0 staged

- `scripts/hira_v1_s55_a0_learned_joint_relation_interaction.py`
- `tests/test_v1_s55_a0_harness.py`
- `.github/workflows/hira-v1-s55-a0-learned-joint-relation-interaction.yml`

A0 proves:
- equal 180,224 private params
- 65,536 learned joint params each
- bit-identical initialization
- zero-init warm start
- explicit state-channel isolation
- learned joint gradients live
- correction gradients live
- cache/native isolation
- K=3/7/255
- full-K probability mass
- normalized relation codes
- no direct S54 bypass
- no second encoder.

## Authorization rule

A0 marker:
`research/HIRA-V1-S55-ENABLE-A0`

The marker MUST remain absent until the exact final staging head passes generic CI on Python 3.10 and 3.12.

A0 is mechanical only:
- no model selection
- no fresh S55 TRAIN/DEV exposure
- no external Laya/Jev evaluation.
