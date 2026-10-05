# HIRA V1 S66 contract — Per-View Counterfactual Responsibility Supervision

Status: **FROZEN / PRE-A0**

Issue: #319

Parent:
- S65 merged main `6bae27159fc2f9577f72540c7070da3e41ef3d15`
- fresh run `37325995163`
- artifact `11352920320`
- digest `sha256:a44c560d304537bb7acaa1b5d569160ec744f897ae728edf95ceca7701be9dfc`
- verdict **Case B**.

## Scientific question

Can explicit TRAIN-only per-view counterfactual responsibility labels teach the existing contextual single-view gate which view should accept pairwise residual?

## Controlled variable

Both arms:
- exact S64 contextual single-view gate
- **60 trainable params**
- identical initialization
- identical context path
- identical optimizer
- identical correction/head trajectory
- identical single-view inference path.

Reference:
- exact S62 shared pair target for both views
- independent-view BCE.

Treatment:
- per-view counterfactual responsibility labels
- independent-view BCE against `y_c,y_p`.

Treatment parameter advantage: **0**.

## Per-view target

Fixed probe:
- alpha **0.35**
- tolerance **1e-8**.

For each view independently:
- correctness safe iff its own fixed probe CE is non-worse;
- stability better iff probing only that view lowers paired JS while the other view remains at baseline.

Canonical:
`y_c = correctness_safe_c AND stability_better_c`.

Paraphrase:
`y_p = correctness_safe_p AND stability_better_p`.

No target smoothing, coefficient, teacher, pseudo-label or DEV dependency.

## Ownership

Targets and all fused/pairwise/context inputs are detached.

Reliability gradient may update only the 60 gate params and may not enter native runtime, cache, correction or pairwise head.

## A0 requirements

Must prove:
- 60 vs 60 params
- added treatment params 0
- bit-identical initialization/context projection
- same single-view inference path
- reference exact S62 target retained
- treatment per-view targets detached
- all four responsibility states mechanically occur on frozen deterministic probes: 00,01,10,11
- target disagreement fraction >0
- view-swap equivariance
- correctness veto active
- stability veto active
- output gradient live
- staged phi gradient live
- upstream gradients zero
- alpha override 0 identity
- bounded residual
- K=3/7/255
- probability mass error <=1e-6
- deterministic checkpoint replay
- one encoder/state-once
- fresh S66 TRAIN/DEV exposed=false.

## Fresh authority

Only after A0 and exact pre-DEV CI:
- seed **87001**
- TRAIN **768**
- DEV **192**
- 12 fresh S66 domains
- exact S65 overlap 0
- K=4
- 24 epochs
- one DEV.

## Stop rule

After one S66 DEV there is no target/tolerance/probe sweep, smoothing, objective mixing, architecture change, optimizer change, regularizer, gradient coupling, retry, second DEV, native retrain, selector change or external Laya/Jev evaluation.
