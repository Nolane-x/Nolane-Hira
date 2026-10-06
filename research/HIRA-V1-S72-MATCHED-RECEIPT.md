# HIRA V1 S72 matched scientific receipt — Opponent-Profile Vector Residual Direction

Status: **FROZEN / CASE B**

Scientific run: `37470998139`  
Artifact: `11416962628`  
Artifact digest: `sha256:9a653a681e0cb3182819634fe24eb345ce82a48f2d078d15e510d1d3ebfb0834`  
Scientific head: `197d2df295e6b44a270b771ae38b8b159f969611`

Outcome:
`HIRA_V1_S72_OPPONENT_PROFILE_VECTOR_RESIDUAL_DEV_COMPLETE`

A0:
- run `37463175304`
- artifact `11413616350`
- digest `sha256:a4d8052a0e76160310d4a99e86ef71f07810a96e77681cb34d747b85c98ef52a`.

## Controlled variable

Both arms:
- exact S69 query-gated identity interaction representation
- exact S59 pairwise head, **32,832 params**
- one shared pairwise trajectory
- one shared correction trajectory
- exact S71 **80-param mean-only composer**
- bit-identical composer state/policy
- exact S66 TRAIN-only per-view responsibility supervision.

Reference residual direction:
uniform pairwise row mean, centered/RMS-normalized, tanh bounded.

Treatment residual direction:
fused-opponent-profile signed evidence divided by weighted absolute evidence, centered/RMS-normalized, tanh bounded.

Direction trainable params:
**0 vs 0**.

## Selected checkpoint

Reference epoch: **22**  
Treatment epoch: **22**

Raw pairwise evidence is exactly matched:
- gold-pair accuracy **0.5907118056**
- gold-pair margin **0.1832211731**.

Composer policy is exactly matched:
- mean alpha **0.0775707277**
- min alpha **0.0524569787**
- max alpha **0.1046675816**
- std alpha **0.0153645393**.

Direction geometry differs materially:
- mean cosine **0.9641074588**
- max absolute difference **0.8949548006**.

## Final decision

Reference:
- canonical accuracy **0.4609375000**
- paraphrase accuracy **0.3619791667**
- paired both-correct **0.2135416667**
- question-swap **0.8333333333**
- agreement **0.2473958333**
- JS **0.1044836173**.

Treatment:
- canonical accuracy **0.4557291667**
- paraphrase accuracy **0.3567708333**
- paired both-correct **0.1927083333**
- question-swap **0.8385416667**
- agreement **0.2578125000**
- JS **0.1047436564**.

Treatment minus reference:
- canonical **-0.520833 pp**
- paraphrase **-0.520833 pp**
- paired both-correct **-2.083333 pp**
- question-swap **+0.520833 pp**
- agreement **+1.041667 pp**
- JS **+0.000260039 worse**
- canonical margin **+0.000707543**
- paraphrase margin **+0.000468561**.

## Frozen interpretation

**Case B.**

The treatment materially changes output geometry under exactly matched raw pairwise evidence and composer policy, but correctness/stability gains are mixed and weak.

This means the next bottleneck is not another direction formula. The pairwise residual needs an explicit **TRAIN-only correctness-preserving acceptance/veto operator** that decides when a candidate residual is safe to apply.

No S72 temperature/profile/epsilon/normalization/composer/target sweep, retry, second DEV or external Laya/Jev evaluation is authorized.
