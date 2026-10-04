# HIRA V1 S54 A0 receipt — Joint State–Query–Option Late Interaction

Status: **QUALIFIED**

Run: `37208186642`  
Artifact: `11305528035`  
Artifact digest: `sha256:b83d5a8a11ed656fa67ad3b3c9d0b0bbb8a4e858a4ba2300555d932a168effde`  
Authorization head: `e174d14f0b36f21d44ad800ed47ce0d75be3fc98`

Outcome:
`HIRA_V1_S54_A0_JOINT_STATE_QUERY_OPTION_INTERACTION_READY`

## Parent authority

- S51 native run `37192490832`
- runtime/native digest:
  `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA:
  `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`
- native trainable params **0**

## Matched private surface

Reference:
- exact S53 query↔option late interaction
- correction **114,688**
- private trainable **114,688**

Treatment:
- joint state+query+option interaction
- correction **114,688**
- interaction params **0**
- identity params **0**
- private trainable **114,688**

Initialization bit-identical: **true**.

## Actual cache mechanics

- cache digest `4a5e2b8b3ff8e37f0af18e04f40ea183ac6d7e69c24a62776ce607af0cfaf695`
- inference tensors **0**
- requires-grad tensors **0**
- context norm max error **1.192e-7**
- attention mass max error **2.384e-7**
- joint context differs from S53 by max abs **0.0105502**
- inherited S53 query↔option-only bypass absent **true**

Mean support:
- state **0.975443**
- option **0.958013**

## Activated path sensitivity

State:
- context sensitivity **0.000848413**
- logit sensitivity **0.000982285**

Query:
- context sensitivity **0.00292815**
- logit sensitivity **0.000133514**

Option:
- context sensitivity **0.00104670**
- logit sensitivity **0.0000743866**

Thus all three evidence sources are mechanically live in the documented triadic path.

## Full-K

- K=3 PASS
- K=7 PASS
- K=255 PASS
- max probability-mass error **1.192e-7**
- max context-norm error **1.192e-7**
- second encoder pass **false**

A0 is diagnostic only:
- model selection **false**
- production-ready claim **false**

Fresh S54 TRAIN/DEV remains separately gated.
