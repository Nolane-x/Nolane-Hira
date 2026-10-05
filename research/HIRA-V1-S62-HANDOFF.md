# HIRA V1 S62 → S63 handoff

Parent verdict: **S62 Case B**

## What S62 established

1. S59: pairwise-only decision exposes a strong stability signal but destroys correctness.
2. S60: one global bounded alpha protects correctness but transfers almost none of that stability signal.
3. S61: a 5-parameter per-query gate learns nonconstant alpha, but gold CE does not teach reliability.
4. S62: explicit TRAIN-only reliability labels successfully change gate behavior, yet the four scalar S61 features are not expressive enough to predict when pairwise intervention helps.

Fresh S62 treatment vs reference:
- canonical accuracy: **0.00 pp**
- paraphrase accuracy: **-0.26 pp**
- paired both-correct: **0.00 pp**
- question-swap: **0.00 pp**
- agreement: **-0.26 pp**
- JS: **0.00024088782568770783**
- reference mean alpha: **0.105502400547266**
- treatment mean alpha: **0.08251461200416088**.

## Required S63 question

> Can a richer but still compact TRAIN-only reliability representation predict pairwise usefulness better than the four hand-crafted scalar features, while preserving the bounded residual and correctness frontier?

## Handoff constraints

Retain:
- S51 persisted native authority;
- immutable cache / one encoder state-once;
- S59 explicit pairwise head;
- bounded residual with alpha <= 0.35;
- S62 fixed TRAIN-only reliability target semantics as the treatment authority unless S63 preregisters a clean controlled comparison;
- explicit correctness veto;
- no pairwise-only final path;
- no teacher;
- no DEV-derived target;
- no gate gradient into native/correction/pairwise surfaces;
- fresh S63 TRAIN/DEV;
- one DEV only;
- frozen S17 selector unless S63 explicitly studies selector in a separate family.

S63 should increase **reliability representation expressivity**, not simply sweep S62 hyperparameters.

Do not reopen S62.
