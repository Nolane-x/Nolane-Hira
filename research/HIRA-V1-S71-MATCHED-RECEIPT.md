# HIRA V1 S71 matched scientific receipt — Multi-Stat Pairwise Row Composer

Status: **FROZEN / CASE C**

Scientific run: `37459447097`  
Artifact: `11412505173`  
Artifact digest: `sha256:aa116237d5508131db18f88a23893d9107d05ae67f46ccffe638520fe8098a50`  
Scientific head: `4f1680ae1622e017a37c4e21166366aad8b29154`

Outcome:
`HIRA_V1_S71_MULTISTAT_ROW_COMPOSER_DEV_COMPLETE`

A0 authority:
- run `37457754519`
- artifact `11410541387`
- digest `sha256:52f38ac6d30177c43205b023114a4192333b3374c98c083aed30fa494aececb5`.

The first A0 verifier failure was numerical only. Row-stat reductions were hardened with float64 internal accumulation without changing formulas, architecture, parameter count, target or thresholds. The rerun passed before any fresh S71 TRAIN/DEV exposure.

## Controlled variable

Both arms:
- exact S69 query-gated interaction representation
- exact S59 pairwise head
- one shared pairwise training trajectory
- one shared correction trajectory
- exact S66 per-view responsibility target
- exact bounded scalar residual direction based on pairwise row mean
- exactly **80 trainable composer params**
- bit-identical initialization.

Reference row channels:
`[mean,0,0,0]`.

Treatment row channels:
`[mean,max,min,rms]`.

Treatment parameter advantage: **0**.

## Selected checkpoints

Reference epoch: **3**  
Treatment epoch: **3**

Raw pairwise evidence is exactly matched:
- gold-pair accuracy **0.5581597222 vs 0.5581597222**
- gold-pair margin **0.0103764294 vs 0.0103764294**.

## Composer policy

Reference:
- mean alpha **0.0928301895**
- min alpha **0.0896966010**
- max alpha **0.0969994515**
- std alpha **0.0014486869**.

Treatment:
- mean alpha **0.0922235601**
- min alpha **0.0893864557**
- max alpha **0.0964062661**
- std alpha **0.0013451465**.

Treatment minus reference:
- mean alpha **-0.0006066294**
- min alpha **-0.0003101453**
- max alpha **-0.0005931854**
- std alpha **-0.0001035404**.

The policy changes only microscopically.

## Final decision

Reference:
- canonical accuracy **0.4947916667**
- paraphrase accuracy **0.3854166667**
- paired both-correct **0.2552083333**
- question-swap **0.6354166667**
- agreement **0.5260416667**
- JS **0.0520106362**.

Treatment:
- canonical accuracy **0.4947916667**
- paraphrase accuracy **0.3854166667**
- paired both-correct **0.2552083333**
- question-swap **0.6354166667**
- agreement **0.5260416667**
- JS **0.0520003916**.

Treatment minus reference:
- canonical **0.00 pp**
- paraphrase **0.00 pp**
- paired both-correct **0.00 pp**
- question-swap **0.00 pp**
- agreement **0.00 pp**
- JS **-0.0000102446** better
- canonical margin **-0.00001820**
- paraphrase margin **+0.00008332**.

## Frozen interpretation

**Case C.**

Multi-stat row evidence does not materially change the scalar composer policy or final decision.

The bottleneck is now the **scalar residual direction itself**, not the amount of row evidence shown to the scalar gate.

No S71 row-stat/width/normalization/target/residual-direction sweep, retry, second DEV or external Laya/Jev evaluation is authorized.
