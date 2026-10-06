# HIRA V1 S69 matched scientific receipt — Query-Gated Identity Interaction Representation

Status: **FROZEN / CASE B**

Scientific run: `37420104460`  
Artifact: `11392672918`  
Artifact digest: `sha256:c24c00e9b79d23b3e7e3fe4d17013270e0410d822b85c62230bb85e972a2f94d`  
Scientific head: `2c856e483a1b12d21dd249758919cbc46ea67d26`

Outcome:
`HIRA_V1_S69_QUERY_GATED_INTERACTION_DEV_COMPLETE`

A0:
- run `37417409926`
- artifact `11391117254`
- digest `sha256:40c2e13e6a40f9e0b2a51a4cdeb7a6819511da4dd29d9ce19f3d688ad0617383`.

A first workflow attempt `37419340008` failed before TRAIN at import time and exposed no DEV. The canonical import was fixed, exact-head CI `37419628125` passed Python 3.10/3.12, and run `37420104460` is the unique S69 scientific exposure.

## Controlled variable

Reference representation:
`normalize([identity, joint_context])` — exact S59.

Treatment representation:
`normalize([identity, normalize(joint_context + 16*(identity ⊙ joint_context))])`.

Both:
- representation trainable params **0**
- representation dim **512**
- exact S59 pairwise head **32,832 params**
- bit-identical A/u initialization
- same gold-vs-distractor objective
- same rows/order/optimizer
- same correction trajectory
- same S64 gate + S66 responsibility composition
- same gate context source (exact S59 reference representation).

Treatment parameter advantage: **0**.

## Selected checkpoints

Reference epoch: **12**  
Treatment epoch: **12**

## Pairwise evidence

Reference:
- gold-pair accuracy **0.5173611111**
- gold-pair margin **0.0160528732**
- aggregate canonical accuracy **0.2786458333**
- aggregate paraphrase accuracy **0.2682291667**
- aggregate canonical margin **-0.2181044115**
- aggregate paraphrase margin **-0.1525925482**
- aggregate paired both-correct **0.0000000000**
- aggregate agreement **0.6666666667**
- aggregate question-swap **0.0000000000**
- aggregate JS **0.0040241013**.

Treatment:
- gold-pair accuracy **0.5516493056**
- gold-pair margin **0.0407122214**
- aggregate canonical accuracy **0.2890625000**
- aggregate paraphrase accuracy **0.3177083333**
- aggregate canonical margin **-0.1976180427**
- aggregate paraphrase margin **-0.1600465896**
- aggregate paired both-correct **0.0052083333**
- aggregate agreement **0.6354166667**
- aggregate question-swap **0.1093750000**
- aggregate JS **0.0045963578**.

Treatment minus reference:
- gold-pair accuracy **+3.428819 pp**
- gold-pair margin **+0.0246593482**
- aggregate canonical accuracy **+1.041667 pp**
- aggregate paraphrase accuracy **+4.947917 pp**
- aggregate canonical margin **+0.0204863688**
- aggregate paraphrase margin **-0.0074540414**
- aggregate paired both-correct **+0.520833 pp**
- aggregate question-swap **+10.9375 pp**
- aggregate agreement **-3.125 pp**
- aggregate JS **+0.0005722565 worse**.

## Final decision

Reference:
- canonical accuracy **0.4557291667**
- paraphrase accuracy **0.2968750000**
- paired both-correct **0.1875**
- question-swap **0.65625**
- agreement **0.3020833333**
- JS **0.1199116266**.

Treatment:
- canonical accuracy **0.4583333333**
- paraphrase accuracy **0.2994791667**
- paired both-correct **0.1875**
- question-swap **0.6770833333**
- agreement **0.2994791667**
- JS **0.1205244033**.

Treatment minus reference:
- canonical **+0.260417 pp**
- paraphrase **+0.260417 pp**
- paired both-correct **0.00 pp**
- question-swap **+2.083333 pp**
- agreement **-0.260417 pp**
- JS **+0.0006127767 worse**.

## Frozen interpretation

**Case B.**

The representation intervention materially improves fresh pairwise evidence, especially gold-pair accuracy/margin and paraphrase aggregate accuracy, but the gain barely transfers to the final fused decision.

Therefore the S69 representation is worth freezing as the stronger pairwise representation, and the next bottleneck is **composition / pairwise evidence utilization**, not representation construction.

No S69 scale/formula/normalization/head/loss sweep, retry, second DEV, reliability-target reopening or external Laya/Jev evaluation is authorized.
