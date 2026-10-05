# HIRA V1 S67 contract — Safe Oracle-Alpha Responsibility Supervision

Status: **FROZEN / PRE-A0**

Issue: #321

Parent:
- S66 merged main `e898314b51995db7e6fa18b06256a59551d0797e`
- fresh run `37331468679`
- artifact `11354228532`
- digest `sha256:f3c6322eec8cce02a698807037a4c05e48c32b2addc5709a116a4384338379b1`
- verdict **Case B**.

## Scientific question

Can TRAIN-only per-view safe oracle-alpha supervision teach the existing contextual gate how much pairwise residual is useful?

## Controlled variable

Both arms:
- exact S64 contextual single-view gate
- **60 trainable params**
- identical initialization/context path
- same optimizer
- same correction/head trajectory
- same single-view inference.

Reference:
- exact S66 binary per-view responsibility BCE.

Treatment:
- safe oracle-alpha continuous target BCE.

Treatment parameter advantage: **0**.

## Frozen lattice

`{0, 0.0875, 0.175, 0.2625, 0.35}`.

For each view independently, hold the other view baseline.

A candidate alpha is safe iff own-view CE is non-worse within **1e-8**.

Among safe candidates:
- choose lowest paired JS;
- require JS improvement > **1e-8**;
- ties retain the smaller alpha.

No improving safe candidate => alpha*=0.

Target:
`y=alpha*/0.35` in `{0,.25,.5,.75,1}`.

No coefficient, temperature, smoothing, interpolation, learned target model or inference-time lattice search.

## Ownership

All target evidence is TRAIN-only and detached.

Reliability gradients may update only gate parameters and may not enter native, cache, correction or pairwise head.

## A0 requirements

Must prove:
- 60 vs 60 params
- added treatment params 0
- bit-identical initialization/context path
- exact S66 binary objective retained for reference
- oracle target detached
- selected alpha always in frozen lattice
- target always in frozen five levels
- deterministic probe bank includes zero and at least three distinct nonzero target levels
- smaller-alpha tie break exact
- selected nonzero alpha always correctness-safe
- selected nonzero alpha strictly lowers one-sided paired JS
- view-swap equivariance
- target differs mechanically from S66 binary semantics
- staged gradient path live
- upstream gradients zero
- K=3/7/255
- bounded residual
- alpha0 exact identity
- probability mass <=1e-6
- checkpoint replay exact
- no fresh S67 TRAIN/DEV.

## Fresh court

Only after A0 and exact pre-DEV CI:
- seed **88001**
- TRAIN **768**
- DEV **192**
- 12 fresh S67 domains
- exact S66 overlap 0
- K=4
- 24 epochs
- one DEV.

## Stop rule

No lattice/tie/tolerance change, no smoothing/mixing, no architecture or optimizer changes, no retry, no second DEV, no native retraining, no selector change, no external Laya/Jev evaluation after scientific exposure.
