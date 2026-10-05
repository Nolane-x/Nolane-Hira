# HIRA V1 S59 interpretation plan — frozen before S59-A0/DEV exposure

Status: **FROZEN**

Issue: #302

## Controlled scientific variable

Shared:
- exact S51 native authority;
- immutable cache;
- exact correction architecture/capacity;
- explicit learned pairwise head **32,832 params**;
- same head initialization and TRAIN-only gold supervision;
- same base correction training trajectory;
- same optimizer constants and checkpoint selector;
- no teacher.

Reference decision:
- existing fused full-K decision.

Treatment decision:
- explicit learned pairwise full-K aggregate.

Pairwise gradients are isolated from base correction/native paths.

## Evidence hierarchy

1. S51 authority exact;
2. immutable cache identity;
3. S59 fresh partition authority;
4. pairwise anti-symmetry/permutation/full-K A0;
5. pairwise representation detachment;
6. correction/head parameter accounting;
7. reference/treatment trajectory identity before decision shell;
8. correctness/discrimination;
9. selected-choice stability.

If items 1–7 fail, A–E classification is forbidden.

## Primary DEV reporting

Reference/treatment:
- selected epoch
- canonical/paraphrase accuracy
- paired both-correct
- question-swap
- selected-choice agreement
- cross-view JS
- canonical/paraphrase margins
- full-K probability mass
- state-view encodes.

Pairwise diagnostics:
- gold pair accuracy
- mean gold pair margin
- pairwise loss
- top-score tie fraction
- representation-tie fallback fraction
- anti-symmetry max error.

Base-trajectory proof:
- correction checkpoint hashes per epoch identical across arms
- head checkpoint hashes per epoch identical across arms before decision selection.

## Frozen interpretation

**A** — treatment improves stability and retains/improves correctness/discrimination:
learned pairwise decision is a viable v1 decision surface.

**B** — correctness improves but stability remains weak:
keep learned pairwise boundary; open a separate stability mechanism.

**C** — stability improves but correctness materially falls:
pairwise-only aggregation loses useful absolute evidence; move to calibrated hybrid composition.

**D** — both materially regress:
reject explicit pairwise family.

**E** — full DEV_READY:
freeze immediately and open a separate fresh confirmation before external Laya/Jev evaluation.

No scalar winner score.

## Stop rule

No post-DEV:
- width/seed/activation sweep
- alternate gold-pair loss
- distractor-pair supervision
- tiebreak change
- base-logit blending
- gradient-path change
- selector change
- retry
- second DEV.
