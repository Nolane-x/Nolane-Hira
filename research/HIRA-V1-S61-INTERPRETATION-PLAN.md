# HIRA V1 S61 interpretation plan — frozen before S61-A0/DEV exposure

Status: **FROZEN**

Issue: #307

## Controlled scientific variable

Shared:
- exact S51 native authority;
- immutable cache;
- correction **114,688** params;
- pairwise head **32,832** params;
- adaptive gate **5** params;
- same correction/pairwise/gate TRAIN trajectory;
- same optimizer constants;
- same frozen S17 selector;
- no teacher/pseudo-target/self-anchor.

Reference decision:
- exact existing fused surface.

Treatment decision:
- bounded pairwise residual with per-query 4-feature adaptive alpha.

The only scientific change relative to S60 is:
**one global alpha -> a per-query linear gate over four preregistered detached confidence/disagreement features.**

## Evidence hierarchy

1. S51 authority exact;
2. immutable cache identity;
3. S61 fresh partition authority;
4. feature invariance/detachment;
5. gate 5-parameter accounting;
6. residual bound and fused identity path;
7. correction/pairwise/gate ownership isolation;
8. shared trajectory proof;
9. correctness/discrimination;
10. selected-choice stability.

If items 1–8 fail, A–E classification is forbidden.

## Primary DEV reporting

Reference/treatment:
- selected epoch;
- canonical/paraphrase accuracy;
- paired both-correct;
- question-swap;
- selected-choice agreement;
- cross-view JS;
- canonical/paraphrase margins;
- full-K probability mass;
- state-view encodes.

Adaptive diagnostics:
- mean/min/max alpha;
- alpha standard deviation;
- alpha split by fused/pairwise agreement vs disagreement;
- feature means and ranges;
- residual max and bound max;
- pairwise gold-pair accuracy/margin.

## Frozen interpretation

**A** — materially better stability while correctness/discrimination retained or improved:
S61 becomes a v1 candidate and proceeds to a separate fresh confirmation court.

**B** — correctness remains but stability gain weak:
retain fused correctness anchor; reject this four-feature gold-CE gate and open a stronger TRAIN-only reliability-target family.

**C** — stability materially improves but correctness falls:
adaptive gate is too permissive; open a correctness-preserving veto/anchor family.

**D** — correctness and stability regress:
reject adaptive hybrid gating.

**E** — full DEV_READY:
freeze immediately and open separate fresh confirmation before external Laya/Jev evaluation.

No scalar winner score.

## Stop rule

No post-DEV:
- feature changes;
- hidden layers;
- alpha/feature sweeps;
- objective/optimizer variants;
- regularizer;
- gradient-path change;
- selector change;
- retry;
- second DEV.
