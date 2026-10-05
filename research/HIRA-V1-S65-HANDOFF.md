# HIRA V1 S65 → S66 handoff

Parent verdict: **S65 Case B**

## What S59–S65 established

1. S59: pairwise signal can strongly stabilize decisions but pairwise-only collapses correctness.
2. S60: global bounded alpha preserves correctness but transfers little stability.
3. S61: adaptive scalar gate learns, but gold CE does not teach when pairwise helps.
4. S62: pair-level TRAIN-only reliability labels are principled but four scalar features are insufficient.
5. S63: richer decision-surface geometry is still insufficient.
6. S64: per-view state/query/option context is active but insufficient.
7. S65: cross-view exact-min soft-AND routing changes policy but still yields no material stability gain.

Fresh S65 treatment vs reference:
- canonical **0.00 pp**
- paraphrase **0.00 pp**
- both-correct **0.00 pp**
- question-swap **0.00 pp**
- agreement **0.00 pp**
- JS **+0.00006992 worse**
- reference mean alpha **0.0741745**
- treatment mean alpha **0.0751679**.

## Required S66 question

> Can explicit TRAIN-only per-view responsibility labels identify which view should accept or reject pairwise residual, while preserving correctness, bounded residual and single-view deployment?

## Required direction

Keep the current single-view contextual gate and infer separate canonical/paraphrase reliability targets from counterfactual TRAIN evidence.

A candidate responsibility construction must remain preregistered and coefficient-free. For each view independently, compare:
- baseline fused CE for that view;
- fixed alpha=0.35 pairwise-probe CE for that view;
- the paired-view JS effect caused by changing only that view while holding the other baseline view fixed.

A view may receive positive responsibility only if:
- its own probe CE is non-worse within the frozen tolerance; and
- changing that view alone improves paired JS.

This avoids:
- one shared pair label;
- min/mean aggregation;
- teacher or pseudo-labels;
- DEV-derived responsibility.

## Constraints to retain

- S51 persisted native authority
- immutable cache / one encoder state-once
- S59 pairwise head
- S64 single-view contextual gate
- bounded residual alpha <= 0.35
- fixed probe alpha 0.35
- tolerance 1e-8
- detached ownership
- no gradient into native/correction/pairwise/cache
- no pairwise-only final path
- no teacher
- no DEV target
- fresh S66 authority
- one DEV only
- frozen S17 selector unless separately studied.

S66 must study **responsibility supervision**, not sweep S65 aggregation or S64 architecture.
