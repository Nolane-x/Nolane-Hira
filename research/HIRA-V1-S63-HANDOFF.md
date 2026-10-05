# HIRA V1 S63 → S64 handoff

Parent verdict: **S63 Case B**

## What S59–S63 established

1. S59: pairwise signal can strongly stabilize decisions, but pairwise-only destroys correctness.
2. S60: global bounded alpha protects correctness but transfers almost no stability.
3. S61: per-query scalar confidence features are mechanically adaptive but gold CE does not teach reliability.
4. S62: explicit TRAIN-only reliability supervision changes policy, but four scalar features are insufficient.
5. S63: a 61-param learned permutation-invariant representation of the full detached decision surfaces is still insufficient.

Fresh S63 treatment vs reference:
- canonical accuracy **+0.26 pp**
- paraphrase accuracy **0.00 pp**
- paired both-correct **0.00 pp**
- question-swap **+0.52 pp**
- agreement **0.00 pp**
- JS **+0.00000927 worse**
- reference mean alpha **0.09659**
- treatment mean alpha **0.08899**.

## Required S64 question

> Can reliability be predicted from richer state/query/option-conditioned representations rather than decision-surface geometry alone, while preserving the exact S62 correctness veto and bounded residual?

## Handoff constraints

Retain:
- S51 persisted native authority;
- immutable cache / one encoder state-once;
- S59 explicit pairwise head;
- bounded residual with alpha <= 0.35;
- exact S62 TRAIN-only reliability target semantics;
- correctness veto;
- no pairwise-only final path;
- no teacher;
- no DEV-derived target;
- no reliability-gradient path into native/correction/pairwise surfaces;
- fresh S64 TRAIN/DEV;
- one DEV only;
- frozen S17 selector unless selector itself is studied in a separate family.

S64 must add **representation information**, not sweep S63 width/pooling/hyperparameters.

Do not reopen S63.
