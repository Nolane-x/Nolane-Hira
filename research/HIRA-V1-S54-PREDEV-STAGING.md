# HIRA V1 S54 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #291  
PR: #292

## Qualified S54-A0

Run: `37208186642`  
Artifact: `11305528035`  
Digest: `sha256:b83d5a8a11ed656fa67ad3b3c9d0b0bbb8a4e858a4ba2300555d932a168effde`  
Authorization head: `e174d14f0b36f21d44ad800ed47ce0d75be3fc98`

Outcome:
`HIRA_V1_S54_A0_JOINT_STATE_QUERY_OPTION_INTERACTION_READY`

Qualified mechanics:
- reference correction **114,688**
- treatment correction **114,688**
- joint interaction params **0**
- identity params **0**
- bit-identical initialization
- K=3/7/255 PASS
- full-K probability mass PASS
- cache gradients/inference tensors 0
- S53 query↔option-only bypass absent
- state perturbation changes context + logits
- query perturbation changes context + logits
- option perturbation changes context + logits
- second encoder pass false.

## Parent native authority

Reuse exact S51 Phase-A artifact:
- run `37192490832`
- artifact `11299783210`
- runtime/native digest:
  `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA:
  `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`
- native trainable params **0**
- native retraining forbidden.

## Fresh S54 authority

- seed **75001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh domains
- K=4
- no exact S53 TRAIN/DEV overlap
- no exact S54-A0 overlap.

## Matched court

Reference:
`option_conditioned_token_level_query_context`

Treatment:
`joint_state_query_option_context`

Both:
- correction **114,688**
- identity params **0**
- interaction trainable params **0**
- total private trainable **114,688**
- temperature **0.10**
- bit-identical initialization
- same immutable native cache
- same TRAIN order
- same optimizer/LR/weight decay/grad clip
- same private CE + JS objective
- same checkpoint selector
- private epochs **24**.

Trainer:
`scripts/hira_v1_s54_train_dev.py`

Workflow:
`.github/workflows/hira-v1-s54-joint-state-query-option-train-dev.yml`

Marker:
`research/HIRA-V1-S54-ENABLE-TRAIN-DEV`

## Stop rule after DEV begins

No:
- temperature sweep
- aggregation sweep
- learned interaction projection
- native retraining
- identity change
- correction capacity change
- alternate selector
- retry for scientific weakness
- gate weakening
- second S54 DEV
- external Laya/Jev evaluation.

## Authorization rule

`HIRA-V1-S54-ENABLE-TRAIN-DEV` MUST remain absent until the exact final staging head passes generic CI on Python 3.10 and Python 3.12.
