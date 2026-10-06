# HIRA V1 S72 contract — Opponent-Profile Vector Residual Direction

Status: **FROZEN / PRE-A0**

Issue: #331

Parent:
- S71 merged main `ef2e8a4b71ba9a46dd9caea07698df742abb6461`
- fresh run `37459447097`
- artifact `11412505173`
- digest `sha256:aa116237d5508131db18f88a23893d9107d05ae67f46ccffe638520fe8098a50`
- verdict **Case C**.

## Scientific question

Can a bounded opponent-profile vector residual direction transfer the frozen S69 pairwise evidence into final correctness/stability better than the current pairwise-row-mean direction?

## Frozen upstream

Both arms use exactly:
- S69 query-gated identity interaction representation;
- S59 pairwise head and objective;
- shared pairwise head state/training trajectory;
- shared correction state/training trajectory;
- S71 mean-only 80-param composer architecture;
- S66 TRAIN-only per-view responsibility objective;
- same composer optimizer/training rows/order;
- no teacher/pseudo-target/DEV target.

## Controlled variable

Reference:
`d_ref=tanh(center(row_mean(P))/rms(center(row_mean(P))))`.

Treatment:
- fused opponent prior `q=softmax(fused)`, temperature 1.0;
- remove self and renormalize opponents;
- signed evidence `s_j=sum q_jk P_jk`;
- absolute evidence `a_j=sum q_jk |P_jk|`;
- confidence `c_j=s_j/(a_j+1e-8)`;
- `d_treat=tanh(center(c)/rms(center(c)))`.

Final both:
`h=fused + alpha*fused_rms*d`.

Direction trainable params: **0 vs 0**.

Composer params: **80 vs 80**, bit-identical init and training semantics.

## Safety / ownership

- alpha in [0,0.35]
- direction in [-1,1]
- residual abs <= alpha*fused_rms
- alpha override 0 gives exact fused identity
- pair/fused/context inputs detached
- no composer gradient into pairwise/correction/native/cache
- one encoder/state-once
- dynamic K.

## A0 requirements

Must prove:
- 80 vs 80 composer params
- bit-identical parameter init/context projection
- exact alpha equality between arms
- direction params 0
- profile confidence finite and bounded
- opponent weights nonnegative, self=0, opponent mass=1
- diagonal never contributes
- permutation equivariance
- K=3/7/255
- reference direction exact S71
- treatment direction mechanically differs on nontrivial inputs
- neutral equal-magnitude/uniform-prior construction collapses to reference <=1e-6
- initial alpha exactly 0.10
- alpha0 exact identity
- residual bound
- probability mass <=1e-6
- output/staged composer gradient live
- upstream gradients zero
- checkpoint replay exact
- fresh S72 TRAIN/DEV exposed=false.

## Fresh authority

Only after A0 + exact pre-DEV CI:
- seed **93001**
- TRAIN **768**
- DEV **192**
- 12 fresh S72 domains
- exact S71 overlap 0
- K=4
- 24 epochs
- one DEV.

## Stop rule

No profile/temperature/epsilon/normalization/composer/target/head/representation sweep after DEV. No retry, second DEV or external Laya/Jev evaluation.
