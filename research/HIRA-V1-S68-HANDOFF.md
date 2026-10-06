# HIRA V1 S68 → S69 handoff

Parent verdict: **S68 Case C**

## What S68 ruled out

A matched-capacity comparator with live full-context modulation did not improve fresh pairwise discrimination:
- gold-pair accuracy delta **0**
- gold-pair margin delta **-0.000260**
- final accuracy/agreement deltas **0**.

Therefore simply changing the comparison metric after S59 representation construction is insufficient.

## Structural representation hypothesis

S59 representation is:
`r_j = normalize([identity_j, context_j])`.

Pairwise comparison then sees:
`r_i - r_j`.

If `context_i` and `context_j` share most of the same state/query frame, subtraction cancels that shared semantic frame before the comparator can use it.

A clean way to preserve the frame without adding trainable representation parameters is to create an option-specific **multiplicative interaction** before subtraction.

## Required S69 question

> Can a zero-parameter query-gated identity interaction representation expose state/query-conditioned option geometry that the exact same S59 pairwise head can learn on fresh data?

## Preregistered treatment representation

Let:
- `i_j in R^256` = detached identity signature;
- `q_j in R^256` = detached S54 joint state/query/option context;
- both finite.

Reference:
`r_ref = normalize([i_j, q_j])`
— exact S59 representation.

Treatment:
1. `x_j = q_j + sqrt(256) * (i_j ⊙ q_j)`;
2. `x_j = normalize(x_j)`;
3. `r_treat = normalize([i_j, x_j])`.

Since `sqrt(256)=16`, the Hadamard term is restored to the natural scale of unit 256D vectors without introducing a tunable coefficient.

If a shared query frame is approximately `q`, then:
`x_i - x_j` contains approximately
`16 * q ⊙ (i_i-i_j)`,
so shared query/state dimensions modulate option differences instead of disappearing under subtraction.

## Controlled variable

Both arms:
- representation dimension **512**
- representation trainable params **0**
- exact S59 pairwise head **32,832 params**
- exact same A/u initialization
- exact same gold-pair objective
- same rows/order/optimizer
- no teacher/pseudo-target
- no native/correction gradient.

Treatment parameter advantage: **0**.

For downstream diagnostics, keep the exact same S66 contextual reliability composition in both arms. Do not reopen S62-S67 target families.

## Required A0

Must prove:
- reference exact bit identity to S59 representation;
- treatment dimension exactly 512;
- representation trainable params 0;
- finite and detached;
- option permutation equivariance;
- no extra encoder/state calls;
- treatment differs from reference on non-degenerate identity/context inputs;
- if identity-context interaction is mechanically neutralized, treatment collapses to the reference context half;
- pairwise A/u states bit-identical between arms;
- pairwise parameter count exactly 32,832 each;
- antisymmetry exact;
- diagonal zero;
- K=3/7/255;
- pairwise gradients only into A/u;
- representation/upstream gradients zero;
- checkpoint replay exact.

## Fresh S69 court

Use wholly fresh rows:
- seed **90001**
- TRAIN **768**
- DEV **192**
- 12 fresh S69 domains
- exact S68 state/question/option overlap **0**
- K=4
- 24 epochs
- one DEV only.

Primary readout:
1. pairwise gold-pair accuracy
2. pairwise gold-pair margin
3. pairwise aggregate correctness
4. final fused correctness/stability.

## Frozen interpretation direction

A — pairwise evidence improves materially and transfers downstream: representation bottleneck resolved; open confirmation.

B — pairwise evidence improves materially but final decision does not: freeze representation; composition is bottleneck.

C — TRAIN representation learns but fresh pairwise evidence does not improve: S54 identity/context ingredients themselves lack needed semantics; move further upstream into token-level evidence construction.

D — pairwise/final regress materially: reject Hadamard interaction representation.

E — full DEV_READY: freeze and confirm before external Laya/Jev evaluation.

No coefficient/scaling/function sweep after DEV.
