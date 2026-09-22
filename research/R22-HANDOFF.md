# R22 handoff — functional-nullspace structural augmentation

Status: **protocol frozen in issue #38; implementation prepared before empirical evaluation**.

## Why R22 exists

R21 established a useful but insufficient mechanism:
- exact-R15 teacher-KL damage modelling passed its mechanism gate;
- rank-32 damage subspace overlapped 0.57529% of the frozen R16 relation-delta energy;
- at alpha 0.25, projection recovered matched accuracy from 0.5200 to 0.53067;
- the only eligible R21 candidate gained only +0.006 structural accuracy;
- stronger projected R21 candidates recovered structural score only by damaging retention.

R22 therefore does not extend the R21 interpolation grid.

It tests a new claim:

> structural task-descent signal can be projected into the train-only functional-damage nullspace and added under a frozen teacher-KL trust region.

## Frozen endpoints

R15 competence anchor:
- SHA-256: `007e24fff0e0e48a096de59276f7ab8d0e25bdb826a3f81a7838c5bd6151723f`.

R16 failure-analysis endpoint defining the fixed relation delta:
- SHA-256: `dbe0ddd8bf3811c98f5062bbf482f5d5b4f1f5991fa6f734f7ca88c24e88cc75`.

A13 semantic encoder:
- revision: `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- safetensors SHA-256: `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`.

HIRA parameter count must remain exactly 422,159.

## Prior-train authority

R22 reconstructs exact R19/R20/R21 scored-train authorities.

Their union must contain exactly **30,000 distinct MultiNLI train source indices**.

R22 then creates:
- structural train: strongest 2,000 neutral + 2,000 contradiction after excluding the entire 30,000-example prior union;
- retention train: seed 29, exactly 6,000 examples after excluding prior train + R22 structural train.

All scored train authorities must be pairwise disjoint.

## Fresh validation authority

Structural:
- validation_mismatched deterministic ranking;
- neutral/contradiction ranks 1001–1250;
- exactly 500 examples;
- disjoint from R18/R19/R20/R21 structural validation.

Matched:
- seed-14 validation_matched ordering;
- positions 6000:7500;
- exactly 1,500 examples;
- disjoint from all prior matched windows.

Baseline validity:
- each label count >=400;
- exact R15 matched accuracy >=0.53.

## Functional damage mechanism

On R22 retention train:
1. exact R15 teacher probabilities;
2. fixed probe `R15 + 0.25 * (R16-R15)`;
3. batch-mean `KL(teacher || probe)`;
4. relation-only gradient rows weighted by `sqrt(n_b/N)`;
5. checked Gram eigenspace;
6. exact rank 32 used for candidate construction.

Mechanism must pass:
- mean teacher KL >1e-4;
- finite gradient rows;
- numerical rank >=32;
- basis orthonormal error <=1e-7;
- rank-32 projection contains >=0.5% of frozen R16 relation-delta energy.

## Protected R16 component

With rank-32 damage basis Q and frozen relation delta d:

`p = d - 0.5 * Q(Q^T d)`.

The half-removal factor is frozen from the prior R21 result and is not tuned in R22.

## New structural nullspace direction

R22 computes the balanced structural **cross-entropy** relation gradient at exact R15 on R22 structural train.

Let it be `g_struct`.

Then:
- `g_null = (I - QQ^T) g_struct`;
- `n_raw = -g_null`;
- normalize once to `n = ||p|| * n_raw / ||n_raw||`.

Direction gates:
- finite gradient/direction;
- nonzero gradient/direction;
- normalized rank-32 subspace leakage <=1e-6;
- first-order structural benefit >0.

No validation data enters this construction.

## Train-only teacher-KL trust region

Frozen trust anchor:

`R15 + 0.25 * p`.

Its mean teacher-KL on R22 retention train defines `KL_budget`.

Every selectable R22 candidate must have train-retention teacher-KL <= this budget.

## Frozen candidate grid

`u(alpha, gamma) = alpha * p + gamma * n`.

Selectable:
- alpha = {0.0, 0.125, 0.25};
- gamma = {0.03125, 0.0625, 0.125, 0.25};
- exactly 12 candidates.

Diagnosis-only:
- protected-only alpha {0.125, 0.25};
- unprojected R16 alpha {0.125, 0.25};
- exact R15 baseline.

No post-validation grid expansion is authorized.

## Eligibility and primary gate

Eligibility requires:
- baseline validity;
- functional mechanism validity;
- nullspace direction integrity;
- train KL within the frozen budget;
- matched >=0.53;
- exact matched correct count no worse than 3 predictions below same-slice R15;
- positive train-only first-order structural predicted benefit;
- non-relation bit identity;
- parameter count unchanged.

Selection:
1. maximize ranked-structural non-entailment accuracy;
2. higher matched accuracy;
3. lower train KL / budget;
4. lower update L2;
5. smaller gamma;
6. smaller alpha.

Primary pass additionally requires:
- selected structural >=0.60;
- same-slice structural gain >=+0.020.

Workflow success is not a scientific pass.

## Benchmark firewall

Forbidden for R22 selection/tuning:
- HANS;
- Breaking NLI;
- XNLI;
- MASSIVE;
- Banking77;
- every Laya final benchmark cell;
- every Jev/JEV final benchmark cell;
- issue #8 53-cell scorecard.

Laya/Jev comparison remains a later separate benchmark campaign.

## Implementation

R22 adds:
- `src/nmd/nullspace_augmentation.py`;
- `tests/test_nullspace_augmentation.py`;
- `scripts/r22_functional_nullspace_structural.py`;
- `.github/workflows/r22-functional-nullspace-structural.yml`;
- this handoff.

R21 data/encoding/teacher-KL helpers are reused rather than duplicated.

A failed R22 must be preserved unchanged as a negative result.
