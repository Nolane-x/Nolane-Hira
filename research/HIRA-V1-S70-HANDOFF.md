# HIRA V1 S70 → S71 handoff

Parent verdict: **S70 Case C**

## What is now frozen

Keep the stronger upstream evidence path from S69:
- query-gated identity interaction representation
- exact S59 antisymmetric pairwise head
- pairwise training objective
- S69 representation formula/scale/normalization.

S70 showed that replacing uniform row mean with fused-anchored scalar weighting is not enough.

## Structural bottleneck

Current composer receives only one pairwise scalar per option.

But each pairwise row contains richer structure:
- mean win evidence
- strongest win
- strongest loss
- row dispersion
- disagreement between mean and extrema.

Collapsing this to one scalar before the 60-param gate can destroy useful structure even when the pair matrix itself is better.

## Required S71 question

> Can a parameter-matched bounded **multi-stat pairwise row composer** transfer the frozen S69 pairwise improvement into final correctness/stability better than the current scalar aggregate gate?

## Preregistered row statistics

For each option j from pair matrix P:
1. `mean_j` = exact uniform row mean over k!=j
2. `max_j` = strongest pairwise win over k!=j
3. `min_j` = strongest pairwise loss over k!=j
4. `rms_j` = root-mean-square of row values over k!=j

All statistics detached.
No learned aggregation parameters.

Normalize each statistic across options with centered RMS using the existing numerical epsilon.

## Matched composer design

Both arms must use exactly **80 trainable parameters** and bit-identical initialization.

Shared per-option inputs:
- four S61 scalar reliability features;
- exact S64 fixed 4D context projection;
- fused confidence/surface data already available.

Reference arm:
- consumes `mean_j` plus three zero channels.

Treatment arm:
- consumes `[mean_j, max_j, min_j, rms_j]`.

A compact permutation-equivariant per-option composer should map these fixed features to a bounded residual coefficient alpha in `[0,0.35]`.

Parameter count must be matched by construction; reference receives zeroed extra channels rather than fewer parameters.

## Correctness-preserving output

Final logits remain:
`h = fused + alpha_j * scale(fused) * bounded_pairwise_direction_j`.

Treatment may use the multi-stat features only to choose per-option alpha; it may not create an unbounded additive logit path.

Alpha override 0 must return exact fused identity.

## Training objective

Reference and treatment:
- same frozen S66 TRAIN-only per-view responsibility supervision;
- no new target family;
- no teacher;
- no DEV target;
- gradients isolated from native/correction/pairwise/representation.

## Fresh S71 court

- seed **92001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh domains
- exact S70 overlap **0**
- K=4
- 24 epochs
- one DEV only.

## Primary readouts

1. final canonical/paraphrase accuracy
2. paired both-correct
3. question-swap
4. agreement / JS
5. raw pairwise gold-pair accuracy/margin must remain matched
6. alpha distribution / per-option diversity
7. correctness-preservation gates.

## Frozen interpretation direction

**A** — multi-stat composer materially improves final correctness/stability with matched raw pairwise evidence:
composition bottleneck resolved; open fresh confirmation toward v1.0.

**B** — composer learns materially different per-option alpha but final transfer remains weak:
the pairwise residual direction itself is the bottleneck; next stage studies a bounded vector residual direction, not another gate feature set.

**C** — composer policy barely changes:
row statistics add little beyond mean; move directly to vector residual direction.

**D** — final correctness/stability materially regress:
reject multi-stat composer.

**E** — full DEV_READY:
freeze and confirm immediately.

No width/statistic/normalization/target sweep after DEV.
