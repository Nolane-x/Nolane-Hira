# HIRA V1 S55 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #293  
PR: #294

## Qualified S55-A0

Run: `37211474503`  
Artifact: `11306837784`  
Digest: `sha256:66ad8bbc2a2046ba00fdc17b60bef991dc246cd2182268edf3965ea91e29b812`  
Authorization head: `9a89d44413f46b986d3a484faf8bbb2dc8fdf04d`

Outcome:
`HIRA_V1_S55_A0_LEARNED_JOINT_RELATION_INTERACTION_READY`

Qualified mechanics:
- correction 114,688 each arm
- learned joint transform 65,536 each arm
- total private trainable 180,224 each arm
- identity params 0
- initialization bit-identical
- zero-init warm start exact within 3e-8
- reference zero-state-channel invariant error 0
- treatment state-channel sensitivity >0
- correction gradients live
- learned-joint gradients live
- K=3/7/255 PASS
- full-K probability mass PASS
- cache/native isolation PASS
- direct S54 context bypass absent
- second encoder pass false.

## Parent native authority

Reuse exact S51 Phase-A artifact:
- run `37192490832`
- artifact `11299783210`
- runtime/native digest:
  `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA:
  `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`
- native trainable params 0
- native retraining forbidden.

## Fresh S55 authority

- seed **76001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh domains
- K=4
- no exact S54 TRAIN/DEV state/question/option overlap.

Authority:
`src/nmd/v1_s55_authority.py`

## Matched court

Reference:
`learned_joint_transform(q_k, zero_state_channel, option_identity)`

Treatment:
`learned_joint_transform(q_k, s54_joint_state_query_option_context, option_identity)`

Both:
- correction **114,688**
- learned joint transform **65,536**
- identity params **0**
- total private trainable **180,224**
- bit-identical initialization
- same immutable native cache
- same TRAIN order
- same optimizer/LR/weight decay/grad clip
- same private CE + JS objective
- same checkpoint selector
- private epochs **24**.

Trainer:
`scripts/hira_v1_s55_train_dev.py`

Selected checkpoint persists/restores:
- correction state
- learned-joint state

Workflow:
`.github/workflows/hira-v1-s55-learned-joint-relation-train-dev.yml`

Marker:
`research/HIRA-V1-S55-ENABLE-TRAIN-DEV`

## Stop rule after DEV begins

No:
- hidden-dimension sweep
- transform-seed sweep
- neutral-channel variant
- state-channel scaling sweep
- direct S54-context bypass
- native retraining
- identity/correction capacity change
- alternate selector
- retry for scientific weakness
- gate weakening
- second S55 DEV
- external Laya/Jev evaluation.

## Authorization rule

`HIRA-V1-S55-ENABLE-TRAIN-DEV` MUST remain absent until the exact final staging head passes generic CI on Python 3.10 and Python 3.12.
