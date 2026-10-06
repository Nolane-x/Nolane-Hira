# HIRA V1 S71 → S72 handoff

Parent verdict: **S71 Case C**

## Evidence now established

S69 materially improved pairwise representation.

S70 showed opponent reweighting of the scalar aggregate was insufficient.

S71 showed mean/max/min/RMS evidence does not materially alter a scalar-alpha composer.

Therefore the remaining scalar interface:

`residual_j = alpha * fused_rms * tanh(center(row_mean)_j / rms(row_mean))`

is now the primary bottleneck.

## Required S72 question

> Can a bounded **per-option vector residual direction** transfer the frozen S69 pairwise matrix into final correctness/stability better than the scalar pairwise-mean direction, without adding inference parameters or reopening reliability supervision?

## Frozen upstream

Both arms must use:
- exact S69 query-gated identity interaction representation;
- exact S59 pairwise head and pairwise objective;
- one shared pairwise head state/trajectory;
- one shared correction trajectory;
- exact S66 TRAIN-only per-view responsibility target;
- no teacher/pseudo-target;
- no DEV-derived target;
- no native/correction/pairwise gradient from composer;
- dynamic K and one encoder/state-once.

## Controlled residual direction

Let `P[B,K,K]` be the detached antisymmetric pair matrix.

### Reference

Exact current S71 direction:

`d_ref = tanh(center(row_mean(P)) / rms(center(row_mean(P))))`.

Final:
`h_ref = fused + alpha * fused_rms * d_ref`.

### Treatment — bounded opponent-profile vector direction

For each option j:
1. take its off-diagonal pairwise row `P[j,k]`;
2. compute the fused prior over opponents `q=softmax(fused)`, temperature fixed at 1.0;
3. remove self and renormalize q over opponents;
4. compute two detached row moments:
   - weighted signed mean `m_j = sum q_{j,k} P[j,k]`;
   - weighted absolute strength `a_j = sum q_{j,k} |P[j,k]|`;
5. define confidence ratio:
   `c_j = m_j / (a_j + epsilon)`, guaranteed approximately in [-1,1];
6. center c across options and normalize by centered RMS;
7. treatment direction:
   `d_treat = tanh(center(c)/rms(center(c)))`.

Final:
`h_treat = fused + alpha * fused_rms * d_treat`.

No learned direction parameters.

The same scalar alpha composer is used in both arms.

## Why this is a stronger architectural test

The reference collapses each row to a uniform signed mean.

Treatment retains the **signed win/loss balance relative to plausible opponents**, normalized by total comparison strength. It changes direction geometry, not gate capacity or supervision.

If fused prior is uniform and all opponent magnitudes are equal, the treatment reduces to the sign/scale structure of the mean direction.

## Matched composer

Use exact same **80-param S71 reference composer architecture** in both arms:
- reference row channels `[mean,0,0,0]`
- same input, weights and initialization
- treatment must NOT receive S71 extra max/min/RMS channels.

Thus S72 isolates only residual direction.

## Required A0

Must prove:
- same 80 params vs 80 params
- bit-identical composer init
- same scalar alpha for reference/treatment under identical inputs
- direction trainable params **0 vs 0**
- treatment confidence ratio finite and bounded within tolerance
- diagonal never contributes
- permutation equivariance
- K=3/7/255
- alpha override 0 exact fused identity
- residual abs <= alpha*fused_rms
- probability mass error <=1e-6
- direction differs mechanically on nontrivial pair matrices
- neutral construction can collapse reference/treatment direction within tolerance
- no gradient into pairwise/fused/context/native
- checkpoint replay exact
- no fresh S72 DEV.

## Fresh S72 authority

- seed **93001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh domains
- exact S71 state/question/option overlap **0**
- K=4
- 24 epochs
- one DEV only.

## Primary readouts

1. final canonical/paraphrase accuracy
2. paired both-correct
3. question-swap
4. agreement / JS
5. raw pairwise gold-pair metrics must remain matched
6. composer alpha metrics must remain matched or near-identical
7. reference-vs-treatment direction cosine / max difference
8. correctness-preservation gates.

## Frozen interpretation direction

**A** — vector direction materially improves final correctness/stability with matched raw evidence and composer policy:
residual-direction bottleneck resolved; open fresh confirmation toward v1.0.

**B** — vector direction materially changes output but improvements are mixed/weak:
pairwise residual needs a hard correctness-preserving acceptance/veto operator; S73 studies that, not another direction sweep.

**C** — direction changes mechanically but final metrics remain essentially unchanged:
the bounded pairwise residual path itself has reached its limit; stop extending this family and return to decision-core architecture.

**D** — final correctness/stability materially regress:
reject opponent-profile direction.

**E** — full DEV_READY:
freeze immediately and open confirmation.

No temperature/profile/statistic/normalization sweep after DEV.
