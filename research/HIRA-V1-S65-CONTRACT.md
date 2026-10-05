# HIRA V1 S65 contract — Cross-View Soft-AND Reliability Objective

Status: **FROZEN / PRE-A0**

Issue: #317

Parent:
- S64 merged main `c5063da0365d49dc4f9dc2c64f94bc309caef0ba`
- fresh run `37319541951`
- artifact `11348594503`
- digest `sha256:f3d04d452a51b7358a23a9391579bca6ce1ab2c16fa107dc9f7f79740dba90ea`
- verdict **Case B**.

## Scientific question

Can a TRAIN-only cross-view soft-AND objective let the same single-view contextual gate learn which member of a pair is actually unsafe, rather than forcing both views to inherit the same negative pair label?

## Controlled variable

Both arms:
- exact S64 contextual gate
- 60 trainable params
- exact same context channels
- exact same parameter initialization
- exact same optimizer/LR/weight decay
- exact same S62 reliability target
- exact same correction/head trajectory
- exact same alpha bounds
- exact same frozen selector.

Reference:
`0.5 * [BCE(z_c,y) + BCE(z_p,y)]`.

Treatment:
`BCE(min(z_c,z_p),y)`.

Exact implementation:
`z_and = 0.5*(z_c+z_p-|z_c-z_p|)`.

No temperature.
No mixing coefficient.
No extra parameter.
No new target.
No paired inference path.

## Semantics

For a positive pair target:
- the lower-reliability view is the bottleneck and is pushed upward.

For a negative pair target:
- only one view needs to carry the negative pair-level veto;
- the other view is not automatically suppressed merely because its partner made the pair unsafe.

The gate remains single-view deployable at inference.

## Ownership

All fused, pairwise and context inputs are detached.

Cross-view loss may update only the same 60 gate parameters:
- W_phi
- b_phi
- w_out
- b_out.

It may not update native, cache, correction or pairwise head.

## Required A0

Must prove:
- 60 vs 60 params
- added treatment params 0
- bit-identical gate initialization
- identical context path
- exact S62 target values
- soft-AND equals torch.minimum <=1e-7
- pair-order symmetry exact
- gradient routes through the lower unequal view
- no extra trainable tensors
- finite treatment loss
- single-view inference unchanged
- initial output gradient live
- staged phi gradient live
- upstream gradient zero
- alpha override 0 exact fused identity
- bounded residual
- K=3/7/255
- probability mass <=1e-6
- deterministic checkpoint replay
- one encoder/state-once
- fresh TRAIN/DEV exposed=false.

## Fresh S65 court

Only after qualified A0 and exact pre-DEV CI:
- seed **86001**
- TRAIN **768**
- DEV **192**
- 12 fresh S65 domains
- exact S64 state/question/option overlap **0**
- K=4
- 24 epochs
- one DEV.

## Stop rule

No soft-min temperature, min/mean/max sweep, objective mixing coefficient, BCE weighting, target change, context/projection/width/pooling change, optimizer change, regularizer, gradient coupling, native retraining, selector change, retry, second DEV or external Laya/Jev evaluation after scientific exposure.
