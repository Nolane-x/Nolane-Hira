# HIRA V1 S59 contract — Matched Antisymmetric Pairwise Decision Head

Status: **PREREGISTERED / NO S59-A0 OR DEV EXPOSURE**

Issue: #301

## Parent evidence

S58 is scientifically closed as **Case B**.

Fresh S58 authority:
- run `37276076841`
- artifact `11330721014`
- digest `sha256:cea8203e98899feda6c1ca1558b9941a8900b987124dfb43958c90e5d0285e50`
- merged main `decc5a744f69e78e83bc57bf28c3283d230f9283`.

S58 showed that frozen teacher consensus can reduce fused/distribution disagreement but still damages correctness/discrimination. The next test therefore removes pseudo-targets entirely.

## Scientific question

> Does a learned antisymmetric pairwise decision parameterization outperform an equal-capacity learned pointwise decision parameterization on the joint correctness/stability frontier?

## Frozen shared surface

Both arms retain:
- persisted S51 native authority;
- immutable encoded cache;
- one encoder/state-once;
- full-K;
- exact `JointStateQueryOptionPrivateCorrectionFork`;
- base correction parameters **114,688**;
- no teacher;
- no pseudo-target;
- no self-anchor;
- identical optimizer/base objective/checkpoint selector.

Both arms add the same decision-head parameter budget.

## Matched decision-head capacity

Frozen rank: **32**.

Learned matrices:
- `A: [32,256]`
- `B: [32,256]`
- no bias
- decision-head parameters **16,384 / arm**.

Total private trainable parameters:
- correction **114,688**
- decision head **16,384**
- total **131,072 / arm**.

Initialization:
- A seed **80590**, deterministic Gaussian
- B exact zeros
- head residual initially exact zero
- reference/treatment A and B bit-identical
- both arms reproduce the same legacy relation logits at initialization.

Inputs:
- `u_i`: query-free state-option identity, 256D
- `c_i`: joint state-query-option context, 256D.

All are produced from the already materialized encoded cache; no second encoder is permitted.

## Reference — pointwise low-rank head

`a_i = A u_i`

`b_i = B c_i`

`h_i = <a_i,b_i> / sqrt(32)`

The full-K head residual is `h`.

## Treatment — antisymmetric pairwise low-rank head

For option pair i,j:

`d_ij = A(u_i-u_j)`

`g_ij = B(normalize(c_i+c_j))`

Raw pair score:
`r_ij = <d_ij,g_ij> / sqrt(32)`

The implementation explicitly antisymmetrizes:
`p_ij = 0.5 * (r_ij-r_ji)`

and zeros the diagonal.

Therefore:
- `p_ji = -p_ij`
- `p_ii = 0`.

Aggregate:
`h_i = sum_j p_ij / (K-1)`.

No option pair is re-encoded.

## Decision integration

Both arms:
1. compute exact existing S54/S58 relation logits;
2. add head residual at frozen scale **1.0**;
3. use unchanged symmetric full-K evidence fusion.

There is no S59 auxiliary loss.

The sole controlled structural variable is:
**pointwise decision head vs antisymmetric pairwise decision head at equal parameter count and identical initialization.**

## Required S59-A0

Capacity:
- base correction 114,688
- head 16,384
- total 131,072
- equal across arms.

Initialization:
- A bit-identical across arms
- B bit-identical across arms
- B exact zero
- head residual exact zero
- initial relation logits bit-identical.

Pairwise mechanics:
- antisymmetry within floating exactness
- diagonal exactly zero
- option permutation equivariance
- aggregate permutation equivariance
- finite zero/degenerate context
- K=3/7/255
- full-K probability mass.

Gradient liveness:
- with B=0, gold CE gives B nonzero gradient;
- A gradient is initially zero as expected from zero-B warm start;
- after one controlled B update, A gradient becomes nonzero;
- existing correction gradients remain live;
- cache/native gradients remain zero.

Runtime:
- one encoder/state-once
- no second encoder
- no teacher artifact
- no per-pair encoder invocation.

## Fresh S59 authority

Only after qualified S59-A0 and exact-head pre-DEV staging CI.

Intended:
- seed **80001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S59 domains
- exact S58 state/question/option overlap **0**
- K=4
- private epochs **24**
- one DEV only.

## Frozen interpretation

**A** — pairwise materially improves stability while correctness/discrimination is retained or improved:
antisymmetric learned pairwise structure is a major missing primitive.

**B** — stability improves but correctness/discrimination materially regress:
move to conditional sparse pair routing.

**C** — correctness/discrimination remain or improve but stability does not:
move to explicit cross-view latent relation alignment without teacher targets.

**D** — both regress:
reject explicit pairwise-head family and revisit private representation.

**E** — treatment reaches full DEV_READY:
freeze and open one separate confirmation before external Laya/Jev evaluation.

## Stop rule

After one S59 DEV:
- no rank sweep
- no initialization sweep
- no residual-scale sweep
- no pointwise/pairwise hybrid
- no pair-context formula variant
- no auxiliary pairwise loss
- no teacher reintroduction
- no capacity/optimizer/selector change
- no native retraining
- no retry
- no gate weakening
- no second S59 DEV
- no external Laya/Jev evaluation.

Scientific failure is valid.
