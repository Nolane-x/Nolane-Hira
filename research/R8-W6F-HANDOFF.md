# R8-W6f handoff — high-K rank-path localization

Status: **PRE-DIAGNOSTIC IMPLEMENTATION ACTIVE. No W6f localization result is valid yet.**

Issue: #87

## Frozen premise

W6e closed as `JOINT_GENERALIZATION_REPLICATION_FAIL`.

Fresh untouched W6e evidence:
- CONFIRM-L scorer-only overall 88.229%, K64 41.667%;
- CONFIRM-L joint-primary overall 89.479%, K64 39.583%;
- CONFIRM-M scorer-only overall 84.896%, K64 33.333%;
- CONFIRM-M joint-primary overall 87.604%, K64 31.250%;
- joint replica also failed K64 competence on both domains.

Thus overall typed behavior is strong while high-cardinality diagnosis remains unstable.

W6f is a diagnostic lane. It does not train a model and cannot promote a mechanism.

## Fresh diagnostic design

Domains:
- N maritime cargo systems;
- O municipal water treatment;
- P precision robotics manufacturing.

Seeds:
- N 201173;
- O 201181;
- P 201187.

Each domain:
- 96 base states;
- one fixed gold signature/base;
- one deterministic K64 master candidate set/base;
- nested K8/K16/K32/K64 views from the same master set.

Total:
- 288 base states;
- 1,152 diagnostic K-views.

Every view preserves:
- the exact same state text within a base;
- the exact same gold semantic signature;
- nested candidate membership;
- stable option identity;
- deterministic view-level permutation.

Hard-negative cumulative targets:
- K8: 5 one-field + 2 two-field;
- K16: 8 one-field + 5 two-field + 2 three-plus;
- K32: 12 one-field + 12 two-field + 7 three-plus;
- K64: 20 one-field + 25 two-field + 18 three-plus.

## Frozen checkpoints

No training.

Evaluate:
- W6e multi-source scorer-only;
- W6e joint-primary;
- W6e joint-replica.

Frozen W6e artifacts:
- scorer-only `10805971002`;
- joint-primary `10805567861`;
- joint-replica `10805871471`;
- frozen bundle `10805836044`.

A13 stays exact and frozen.

## Required localization

For native production scoring:
- coarse top1/top5/MRR/gold rank/margin;
- final top1/top5/MRR/gold rank/margin;
- relation rescue vs damage;
- gold and best-negative relation deltas;
- K8 -> K64 rank/margin/logit trajectories;
- mismatch-field anatomy of winning negatives.

For candidate-relative salience attribution:
- evaluate a no-training uniform-salience scorer ablation;
- keep projection, scale, common-mode subtraction, coverage aggregation and minimum-coverage term exact;
- only replace candidate-relative IDF with unit valid-token weights.

W6f output is a localization classification, not a rescue verdict.

Forbidden:
- W6e L/M rows;
- all earlier CONFIRM rows;
- training on W6f diagnostics;
- threshold changes after W6f results;
- public benchmark claims.
