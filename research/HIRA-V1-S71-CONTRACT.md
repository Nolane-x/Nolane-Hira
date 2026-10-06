# HIRA V1 S71 contract — Multi-Stat Pairwise Row Composer

Status: **FROZEN / PRE-A0**

Issue: #329

Parent S70:
- merged main `751161746887b795149fcd924735ee2d043b43b3`
- scientific run `37423476142`
- artifact `11394610000`
- digest `sha256:8af711f789efb9717eaed0eff882fade7bab37b546f86a35502db82242ad221f`
- verdict **Case C**.

## Scientific question

Can a parameter-matched bounded composer use detached mean/max/min/RMS pairwise-row structure to transfer more of the frozen S69 pairwise evidence into final correctness/stability than a mean-only composer?

## Controlled variable

Both arms:
- exact S69 pairwise representation
- exact S59 pairwise head
- exact same pairwise/correction trajectory
- exact S66 view-level reliability target
- bounded scalar alpha residual
- exact same residual direction based on uniform row mean
- **80 trainable composer params**.

Reference row channels:
`[normalized_mean,0,0,0]`.

Treatment row channels:
`[normalized_mean,normalized_max,normalized_min,normalized_rms]`.

Treatment parameter advantage: **0**.

## Capacity

Option input:
- surface 4
- fixed S64 context 4
- row statistics 4
= **12**.

Hidden:
- W_phi 5x12 = 60
- b_phi 5 = 5.

Pooled representation:
- mean hidden 5
- max hidden 5
- exact S61 scalar features 4
= 14.

Output:
- w_out 14
- b_out 1.

Total = **80**.

## Output semantics

`alpha = 0.35 * sigmoid(logit)`.

Initial alpha = **0.10** exactly.

Residual direction remains:
`tanh(center(row_mean)/rms(row_mean))`.

Final:
`fused + alpha * fused_rms * direction`.

No unbounded additive path.

## A0 requirements

- 80 vs 80 params
- bit-identical parameter state
- exact same S64 fixed context projection
- diagonal excluded
- row stats detached
- reference extra three channels exactly zero
- treatment extra channels non-degenerate
- option permutation equivariance
- alpha permutation invariant
- K=3/7/255
- initial alpha 0.10
- alpha0 exact identity
- residual bound exact
- probability mass <=1e-6
- output gradient live
- staged phi gradient live
- no upstream gradient
- checkpoint replay exact
- fresh S71 TRAIN/DEV exposed=false.

## Fresh S71 authority

Only after A0 and exact pre-DEV CI:
- seed **92001**
- TRAIN **768**
- DEV **192**
- 12 fresh S71 domains
- exact S70 overlap 0
- K=4
- 24 epochs
- one DEV.

No post-DEV row-stat/width/normalization/target/residual-direction sweep.
