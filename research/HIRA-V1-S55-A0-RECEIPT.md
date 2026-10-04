# HIRA V1 S55 A0 receipt — Learned Joint Relation Interaction

Status: **QUALIFIED**

Run: `37211474503`  
Artifact: `11306837784`  
Digest: `sha256:66ad8bbc2a2046ba00fdc17b60bef991dc246cd2182268edf3965ea91e29b812`  
Authorization head: `9a89d44413f46b986d3a484faf8bbb2dc8fdf04d`

Outcome:
`HIRA_V1_S55_A0_LEARNED_JOINT_RELATION_INTERACTION_READY`

## Matched capacity

- correction params: **114,688 / arm**
- learned joint params: **65,536 / arm**
- total private trainable: **180,224 / arm**
- identity params: **0**
- initialization bit-identical: **true**

## Warm-start / controlled variable

- reference zero-init warm-start max abs: **2.98e-8**
- treatment zero-init warm-start max abs: **2.98e-8**
- reference zero-state-channel invariant error: **0**
- treatment explicit state-channel sensitivity: **2.816e-5**

The matched learned transform therefore preserves the S53 query context at initialization and only the treatment exposes real S54 state-conditioned evidence.

## Gradient ownership

- correction gradient L1: **55.6959**
- learned-joint gradient L1: **1.26415**
- native trainable params: **0**
- cache inference tensors: **0**
- cache requires_grad tensors: **0**
- direct S54-context bypass absent: **true**
- second encoder pass: **false**

## Arbitrary-K / normalization

- K=3 PASS
- K=7 PASS
- K=255 PASS
- max probability mass error: **1.19e-7**
- max relation-code norm error: **1.19e-7**

## Parent native authority

- runtime/native digest:
  `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA:
  `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`

A0 is mechanical only and was not used for model selection.
