# HIRA V1 S59 matched scientific receipt — Explicit Learned Pairwise Decision Head

Status: **FROZEN / CASE C**

Scientific run: `37289522292`  
Artifact: `11335428630`  
Artifact digest: `sha256:72a348c27f95c1f7ea2dc0dbcba8db7afac56fa5cd4d284d0593f7321d603e4c`  
Scientific head: `363b8caa20ff490e3c702548d5d97226c2add66f`

Outcome:
`HIRA_V1_S59_EXPLICIT_LEARNED_PAIRWISE_DECISION_HEAD_DEV_COMPLETE`

## Matched authority

- parent S51 native run `37192490832`
- parent native artifact `11299783210`
- S59-A0 run `37278474963`
- S59-A0 artifact `11331042601`
- S59-A0 digest `sha256:5271081c4bcacbced9ca85c146ddd36b5dc9ecdc0c6dc9038c9b3fb6219a11b6`
- TRAIN **768**, DEV **192**, 12 fresh domains
- exact S58 state/question/option overlap **0**
- correction params **114,688**
- pairwise head params **32,832**
- pairwise representation **512D detached**
- no teacher / pseudo-target / self-anchor
- pairwise gradient into correction/native **false**
- one shared correction/head training trajectory
- same frozen S17 selector.

Controlled variable:
- reference decision shell: **existing fused**
- treatment decision shell: **explicit learned pairwise**.

## Selected checkpoints

Reference selected epoch: **22**  
Treatment selected epoch: **19**

Reference checkpoint SHA:
`1fc1cb980a850bb7e9fa9a0809cd27e64b7410eab6510813b7814dd1af1026a8`

Treatment checkpoint SHA:
`6c57e44a2baf404848219b5c67796552973d8439662f2f017314e4016cc82b9e`

## Reference

- fused canonical accuracy **0.5**
- fused paraphrase accuracy **0.3463541666666667**
- paired both-correct **0.2708333333333333**
- question-swap change **0.7291666666666666**
- selected-choice agreement **0.2708333333333333**
- cross-view JS **0.13348887860774994**
- canonical gold margin **-0.030322759722669918**
- paraphrase gold margin **-0.5103224193056425**

## Treatment

- pairwise canonical accuracy **0.3463541666666667**
- pairwise paraphrase accuracy **0.3723958333333333**
- paired both-correct **0.052083333333333336**
- question-swap change **0.06770833333333333**
- selected-choice agreement **0.5963541666666666**
- cross-view JS **0.004804993960230301**
- canonical gold margin **-0.11519392331441243**
- paraphrase gold margin **-0.1309604635462165**
- gold-pair accuracy **0.6024305555555556**
- mean gold-pair margin **0.056327266773829855**
- top-score tie fraction **0**
- representation-key fallback fraction **0**
- antisymmetry max error **0**.

## Treatment minus reference

Stability:
- selected-choice agreement **+32.55 pp**
- cross-view JS **-0.128684**

Correctness/discrimination:
- canonical accuracy **-15.36 pp**
- paraphrase accuracy **+2.60 pp**
- paired both-correct **-21.87 pp**
- question-swap change **-66.15 pp**
- canonical gold margin **-0.084871**
- paraphrase gold margin **+0.379362**

## Frozen interpretation

**Case C — explicit pairwise-only final aggregation strongly improves stability but materially destroys correctness/discrimination.**

The learned pairwise head proves several important points:
- a gold-supervised, teacher-free pairwise boundary can become substantially more cross-view stable;
- the head itself is mechanically healthy: anti-symmetry error **0**, no ties/fallbacks on selected DEV, gold-pair accuracy about **60.24%**;
- the failure is therefore not self-anchor or frozen-teacher leakage.

But pairwise-only aggregation discards too much absolute/fused evidence:
- canonical accuracy falls **15.36 pp**
- paired both-correct falls **21.87 pp**
- question-swap discrimination falls **66.15 pp**.

Per the preregistered S59 interpretation, the next family is **calibrated hybrid decision composition**: preserve the absolute fused evidence that carries correctness/discrimination while adding only the learned pairwise residual needed for stable ordering.

No S59 width/activation/loss/tiebreak/blend/gradient-path sweep, retry, second DEV, selector change, native retraining, or external Laya/Jev evaluation is authorized.
