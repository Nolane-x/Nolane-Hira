# HIRA V1 S64 → S65 handoff

Parent verdict: **S64 Case B**

## What S59–S64 established

1. S59: pairwise signal can strongly stabilize decisions, but pairwise-only destroys correctness.
2. S60: a global bounded alpha preserves more correctness but transfers almost none of the stability.
3. S61: per-query scalar gating is adaptive, but gold CE does not teach reliability.
4. S62: explicit TRAIN-only reliability labels alter policy, but four scalar features are insufficient.
5. S63: richer detached decision-surface geometry is still insufficient.
6. S64: parameter-matched per-view state/query/option context is mechanically active, but still does not materially improve stability.

Fresh S64 treatment vs reference:
- canonical accuracy **-0.26 pp**
- paraphrase accuracy **0.00 pp**
- paired both-correct **-0.52 pp**
- question-swap **0.00 pp**
- agreement **0.00 pp**
- JS **-0.00001873** better
- reference mean alpha **0.0779477**
- treatment mean alpha **0.0772873**.

## Required S65 question

> Can TRAIN-time cross-view/context interaction expose the reliability signal that independent per-view contextual summaries miss, while preserving a single-view deployable gate, the exact S62 correctness veto, and bounded residual?

## Handoff constraints

Retain:
- S51 persisted native authority;
- immutable cache / one encoder state-once;
- S59 explicit pairwise head;
- bounded residual with alpha <= 0.35;
- exact S62 TRAIN-only reliability target semantics;
- explicit correctness veto;
- no pairwise-only final path;
- no teacher;
- no DEV-derived target;
- no reliability-gradient path into native/correction/pairwise surfaces;
- single-view deployable inference path;
- fresh S65 TRAIN/DEV;
- one DEV only;
- frozen S17 selector unless selector itself is studied separately.

S65 must add **cross-view/context interaction information during TRAIN**, not sweep S64 projection size, projection seed, hidden width, pooling or optimization hyperparameters.

Do not reopen S64.
