# HIRA V1 S70 contract — Fused-Anchored Weighted Pairwise Aggregation

Status: **FROZEN / PRE-A0**

Issue: #327

Parent:
- S69 merged main `1ada877bcd9381486395599519363e87ef3dd831`
- fresh scientific run `37420104460`
- artifact `11392672918`
- digest `sha256:c24c00e9b79d23b3e7e3fe4d17013270e0410d822b85c62230bb85e972a2f94d`
- verdict **Case B**.

## Scientific question

Can a zero-parameter fused-anchored weighted pairwise aggregation convert the stronger frozen S69 pair matrix into more useful final decision evidence than uniform row-mean aggregation?

## Frozen upstream evidence

Both arms retain:
- S69 query-gated identity interaction representation;
- representation dim **512**;
- representation trainable params **0**;
- exact S59 pairwise head **32,832 params**;
- same pairwise head state/trajectory;
- same correction state/trajectory;
- no teacher or pseudo-target.

Pairwise evidence itself is not the controlled variable in S70.

## Aggregation reference

For antisymmetric pair matrix `P`:

`e_ref(j)=sum_{k!=j}P[j,k]/(K-1)`.

This is the exact current S59 aggregate on a valid zero-diagonal pair matrix.

## Aggregation treatment

Use detached fused baseline logits `f`.

Frozen temperature:
**1.0**

`w=softmax(f)`.

For each option j:
- remove self weight;
- renormalize all opponent weights;
- `w_jk = w_k / sum_{l!=j}w_l`.

Treatment evidence:

`e_treat(j)=sum_{k!=j}w_jk*P[j,k]`.

No top-k.
No threshold.
No learned parameters.

If `f` is uniform, treatment collapses to reference.

## Downstream composition

Both arms use:
- exact S64 contextual reliability gate;
- **60 trainable params**;
- same initialization;
- exact S66 per-view responsibility objective;
- same gate context source;
- alpha max **0.35**;
- same selector.

Treatment aggregation parameter advantage:
**0**.

## Ownership

At the aggregation boundary:
- pair matrix detached;
- fused baseline detached.

Aggregation may not send gradient into:
- pairwise head
- correction
- native runtime
- cache.

Only the existing gate may train downstream.

## Required A0

Must prove:
- aggregation trainable params 0
- reference exact current S59 row mean on valid pair matrix
- treatment permutation equivariance
- uniform prior collapse <=2e-7
- diagonal contributes exactly zero
- treatment opponent weights nonnegative
- treatment opponent weights sum to one
- diagonal opponent weights exact zero
- K=3/7/255 finite
- aggregate probability mass diagnostic <=1e-6
- pair/fused inputs detached
- two S64 gates exactly 60 params each
- gate initialization bit-identical
- alpha override 0 exact fused identity
- residual bound holds
- output probability mass <=1e-6
- gate output gradients live
- pair/fused/context upstream gradients zero
- deterministic checkpoint replay
- one encoder/state-once retained
- fresh S70 TRAIN/DEV exposed=false.

## Fresh S70 authority

Only after A0 + exact pre-DEV CI:
- seed **91001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S70 domains
- exact S69 state/question/option overlap **0**
- K=4
- 24 epochs
- one DEV only.

## Stop rule

After one S70 DEV:
- no temperature sweep
- no top-k opponent selection
- no weighting-function sweep
- no learned aggregation parameters
- no representation/head change
- no gate-target reopening
- no optimizer/LR/weight-decay change
- no retry
- no second DEV
- no external Laya/Jev evaluation.

Scientific failure is valid.
