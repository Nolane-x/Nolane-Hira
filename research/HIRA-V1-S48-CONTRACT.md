# HIRA V1 S48 contract — Query-Quotient Option Evidence

Status: **PREREGISTERED / NO S48-A0 EXPOSURE**

Issue: #279

Parent evidence:
- S47 fresh run `37168305687`
- artifact `11291161358`
- S47 interpretation **Case C**
- ordinal DEV_READY false

## Scientific question

> Can Hira remove paraphrase-specific query nuisance before option ordering by quotienting the query representation through the state-conditioned option-difference subspace, while preserving the same correction capacity, one-pass runtime, and exact native trajectory?

## Frozen reference

Reference correction is the existing S44/S45 `PrivateCorrectionRepresentationFork`.

Reference query path:
1. mean question tokens
2. normalize raw query summary
3. concatenate raw query to each native option signature
4. A/B private residual
5. bilinear W against raw query summary.

Correction-only parameter count:
**114,688**

## Frozen S48 treatment

S48 uses the exact same A/B/W parameter shapes and initialization:
- adapter A: 32,768
- adapter B: 16,384
- bilinear W: 65,536
- correction total: **114,688**
- native trainable: **49,152**
- treatment total: **163,840**

No trainable parameter is added.

For detached native signatures `S in R^{B x K x 256}` and raw normalized query summary `q in R^{B x 256}`:

`D_i = S_i - mean_j S_j`

`u = sum_i <q, D_i> D_i`

`q* = normalize(u)`

The implementation must compute `u` directly from centered signatures; it must not introduce a learned projection matrix.

If `u == 0`, `q*` is the deterministic zero vector.

Treatment private residual:
`A/B([S_i, q*])`

Treatment bilinear correction:
`private_i^T W q*`

The raw query summary must not bypass `q*` anywhere in the treatment correction path.

## Why this is a quotient

Only components of q that lie in the span induced by current option differences can survive.

Any nuisance vector orthogonal to every centered option signature is annihilated before normalization.

The quotient is:
- option-permutation invariant;
- state/option-set conditioned;
- zero-parameter;
- full-K;
- deterministic.

## Frozen training comparison

Reference arm:
- raw-query private correction.

Treatment arm:
- query-quotient private correction.

Both arms:
- identical native runtime initialization;
- identical fresh TRAIN rows/order;
- identical native optimizer/update trajectory;
- identical correction parameter count;
- identical correction optimizer;
- exact same LR/weight decay/grad clip;
- exact same `0.10 CE + 0.25 cross-view JS`;
- exact same checkpoint-selection rule, preregistered before DEV.

No treatment result may influence reference/treatment checkpoint rule.

## Required S48-A0

Mechanical:
- correction parameter count exactly 114,688;
- total treatment trainable exactly 163,840;
- no extra learned quotient parameters;
- K=3, K=7, K=255;
- option-permutation equivariance;
- quotient invariant to synthetic nuisance orthogonal to option-difference span;
- relation-relevant query component remains nonzero;
- distinct relation-relevant directions remain separable;
- zero option-difference subspace yields exact finite zero quotient;
- raw query cannot bypass treatment quotient;
- corrected logits full-K;
- probability mass error <= 1e-6.

Ownership/runtime:
- one encoder batch/state-once;
- no second encoder;
- correction gradients cannot update native runtime;
- native objective cannot update correction;
- matched native one-step parameter/output identity;
- W -> B -> A warm-start remains live.

A0 semantic values are diagnostic only.

## Fresh authority

Only after:
1. S47 merge;
2. contract + interpretation plan frozen;
3. S48 core tests pass;
4. qualified one-shot S48-A0;
5. frozen A0 receipt;
6. wholly fresh S48 TRAIN/DEV authority;
7. matched trainer/workflow frozen;
8. exact staged-head CI PASS;
9. separate one-shot TRAIN/DEV marker.

Intended:
- seed **69001**
- TRAIN 768
- DEV 192
- 12 wholly fresh S48 domains
- K=4
- 24 epochs
- batch 16
- one DEV only.

## Prohibited tuning

No:
- covariance power sweep;
- raw-query residual blend;
- learned projector;
- quotient dimension/rank sweep;
- quotient epsilon sweep;
- fallback-to-raw-query;
- extra normalization variant sweep;
- loss-weight sweep;
- capacity/optimizer change;
- second encoder;
- seed/LR/epoch retry;
- gate weakening;
- second S48 DEV.

Scientific failure is valid.
