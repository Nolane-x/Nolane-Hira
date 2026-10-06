# HIRA V1 S74 contract — Direct Set Arbitration Core

Status: **FROZEN BEFORE A0**

Parent: S73 **Case C**, merged main `45ff1bc301972fc8aa04c1456200283f65616937`.

## Scientific variable

S74 replaces the residual/gate interface with a compact full-K permutation-equivariant direct decision core.

Both arms use exactly the same **257 trainable parameters** and bit-identical initialization.

Reference sees fused-native channels plus eight mechanically zero relational channels.

Treatment sees the same fused-native channels plus eight live relational channels derived from the shared S69/S59 pairwise matrix.

Treatment parameter advantage: **0**.

## Per-option feature contract

Fused-native channels, live in both arms:
1. standardized fused logit
2. fused probability
3. standardized fused log-probability
4. permutation-equivariant fused rank fraction

Relational channels, zero in reference and live in treatment:
5. standardized pairwise row mean
6. standardized row max
7. standardized row min
8. standardized row RMS
9. signed win fraction
10. standardized strongest-loss magnitude
11. signed mean / absolute-evidence ratio
12. pairwise-vs-fused rank disagreement.

All evidence inputs are detached.

## Core

- input dim 12
- hidden dim 16
- option encoder Linear(12,16)+tanh
- global mean pool 16
- global max pool 16
- score head Linear(48,1)
- total params 257.

No K-specific learned tensors.

Final logits are direct DSAC scores. There is no fused residual/fallback path inside S74.

## TRAIN objective

Frozen:
`0.5*(CE_c + CE_p) + 0.10*JS(c,p)`.

No teacher, pseudo-target or DEV-derived target.

## A0

Mechanical only:
- 257 vs 257 params
- bit-identical init
- zero relational reference channels
- live relational treatment channels
- option permutation equivariance
- finite K=3/7/255
- probability mass error <=1e-6
- direct core semantics
- all DSAC gradients live
- zero gradient into fused/pairwise sources
- exact checkpoint replay
- no fresh S74 TRAIN/DEV.

## Stop rule after fresh DEV

No width/channel/loss/activation sweep, no residual fallback, no target change, no retry, no second DEV.
