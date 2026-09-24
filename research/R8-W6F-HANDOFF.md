# R8-W6f handoff — high-K rank-path localization

Status: **CLOSED DIAGNOSTIC. Pooled classification: `UNRESOLVED_HIGH_K_FAILURE` for all three frozen checkpoints.**

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


## Authoritative diagnostic closure

Exact diagnostic head:
- `327099c37a8a1a1eebb7b8e02f2591aaa28cff8c`.

Diagnostic run:
- `35998772622`;
- all jobs PASS.

Artifacts:
- localization artifact `10807488841`;
- digest `sha256:0d3ec9dd406ba5c615b4531746cf711cc696e2e5d262db40b1fc01864f94ce01`;
- cache artifact `10807367327`;
- frozen-W6e bundle artifact `10806619116`.

Scope:
- 288 fresh base states;
- 1,152 nested K8/K16/K32/K64 views;
- no training performed;
- W6b/W6c/W6d/W6e CONFIRM rows not used;
- typed final/test rows not used;
- campaign cells 0.

### Pooled classifications

All three frozen checkpoints classify as:
- `multi-source-scorer-only` -> `UNRESOLVED_HIGH_K_FAILURE`;
- `multi-source-joint-primary` -> `UNRESOLVED_HIGH_K_FAILURE`;
- `multi-source-joint-replica` -> `UNRESOLVED_HIGH_K_FAILURE`.

Per-domain disagreement must be preserved:
- domain O reaches `COARSE_BINDING_LIMIT` for scorer-only and joint-replica;
- N/P remain unresolved;
- joint-primary remains unresolved on N/O/P.

Therefore W6f does **not** authorize a production rescue mechanism.

### Joint-primary pooled rank path

Native production scoring:

K8:
- coarse top1 69.10%;
- coarse top5 99.65%;
- final top1 75.69%;
- final top5 99.65%.

K16:
- coarse top1 62.15%;
- coarse top5 98.26%;
- final top1 70.14%;
- final top5 99.31%.

K32:
- coarse top1 51.39%;
- coarse top5 95.83%;
- final top1 55.21%;
- final top5 97.22%.

K64:
- coarse top1 32.64%;
- coarse top5 85.76%;
- final top1 38.54%;
- final top5 91.67%;
- final MRR 0.6115;
- native coarse MRR 0.5523;
- final mean gold margin -0.1224;
- native coarse mean gold margin -0.2347.

Relation anatomy at K64:
- rescue rate 12.15%;
- damage rate 6.25%;
- final-minus-coarse top1 +5.90 pp.

Thus relation reranking is net helpful, not the dominant source of failure.

### Candidate-relative salience ablation

Uniform-salience is substantially worse.

Joint-primary pooled K64:
- native coarse top1 32.64%;
- uniform coarse top1 4.17%;
- delta -28.47 pp;
- native coarse top5 85.76%;
- uniform coarse top5 31.25%;
- delta -54.51 pp.

K8 uniform top1 also regresses by about 41.32 pp.

Native candidate-relative salience is therefore not implicated by the preregistered rule; it materially helps ranking.

### Cardinality trajectory

For joint-primary pooled:
- mean native coarse rank drift K8 -> K64: 1.708;
- mean final rank drift: 1.188;
- mean native margin collapse: 0.515;
- relation stage reduces rank drift relative to coarse.

The dominant visible pattern is not catastrophic retrieval failure. A large fraction of K64 final errors still have the gold candidate already inside coarse top5, while one-field hard negatives dominate many winning wrong candidates.

### Scientific interpretation

W6f falsifies two simple explanations:
1. candidate-relative IDF/salience is not the main high-K failure; removing it is much worse;
2. relation reranking is not the main high-K failure; it usually rescues more cases than it damages.

The remaining signal is finer:
- as K grows, gold margins collapse;
- gold often remains in the correct local neighborhood/top5;
- errors are dominated by near-neighbor candidates differing in one field;
- domain O sometimes crosses the coarse-binding-limit rule, but that pattern is not stable across checkpoints/domains.

Therefore the next lane must be a **second fresh localization probe**, not a rescue authority.

That probe should decompose hard-negative confusion by field role and candidate-pair structure, while preserving frozen checkpoints and forbidding any training or threshold tuning. It should ask whether one-field confusion is driven by:
- role-token alignment;
- option-option interference/common-mode competition;
- field-specific semantic collapse;
- or candidate-set density itself.

Only after that second localization produces a stable causal target should a W6 rescue mechanism be preregistered.
