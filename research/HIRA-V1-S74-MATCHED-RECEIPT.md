# HIRA V1 S74 matched scientific receipt — Direct Set Arbitration Core

Status: **FROZEN / CASE D**

Scientific run: `37487578737`  
Artifact: `11423428905`  
Artifact digest: `sha256:cb2d8e2718d3aa426bf3ee44795d36c25e5960ca727fb8cc6280d12cfbefdb21`  
Scientific head: `9d4c9598915c7b81ab18f9e54eeb46aa19ba65bd`

Outcome:
`HIRA_V1_S74_DIRECT_SET_ARBITRATION_CORE_DEV_COMPLETE`

A0:
- run `37484291103`
- artifact `11422648744`
- digest `sha256:68620a7c563e5a22325b4415b53020720e37ba05e8f2eb675fef5d83d4089093`
- outcome `HIRA_V1_S74_A0_DIRECT_SET_ARBITRATION_CORE_READY`.

## Controlled variable

Both arms:
- exact same DSAC architecture
- exactly **257 trainable params**
- bit-identical initialization
- same fused/native/correction trajectory
- same S69/S59 pairwise-head trajectory
- same TRAIN rows/order
- same optimizer/LR/weight decay
- same frozen selector
- final logits are direct DSAC scores
- objective `0.5*(CE_c+CE_p)+0.10*JS`
- no teacher/pseudo-target/DEV target.

Reference:
- fused/native channels live
- eight relational channels exactly zero.

Treatment:
- same fused/native channels
- eight relational channels live.

Treatment parameter advantage: **0**.

Selected epoch: **15** both arms.

Raw pairwise evidence matched exactly:
- gold-pair accuracy **0.5694444444**
- gold-pair margin **0.0501904170**.

Treatment relational feature max abs:
**1.7313132286**.

## Reference DSAC selected DEV

- canonical accuracy **0.4296875000**
- paraphrase accuracy **0.3020833333**
- paired both-correct **0.1562500000**
- question-swap **0.7135416667**
- cross-view agreement **0.3984375000**
- cross-view JS **0.0981791988**
- canonical margin **-0.3146479772**
- paraphrase margin **-0.6378193051**
- canonical decision loss **1.2524763296**.

## Treatment DSAC selected DEV

- canonical accuracy **0.4375000000**
- paraphrase accuracy **0.3151041667**
- paired both-correct **0.1250000000**
- question-swap **0.6510416667**
- cross-view agreement **0.3723958333**
- cross-view JS **0.1087691685**
- canonical margin **-0.3287584993**
- paraphrase margin **-0.6298749028**
- canonical decision loss **1.2883038074**.

## Treatment minus reference

- canonical accuracy **+0.78125 pp**
- paraphrase accuracy **+1.302083 pp**
- paired both-correct **-3.125 pp**
- question-swap **-6.25 pp**
- agreement **-2.604167 pp**
- JS **+0.01058997 worse**
- canonical margin **-0.01411052 worse**
- paraphrase margin **+0.00794440**
- canonical decision loss **+0.03582748 worse**.

## Frozen interpretation

**Case D.**

Relational channels are live and alter the decision, but their integration materially degrades paired correctness and cross-view stability while increasing decision loss. Small single-view accuracy gains do not offset the larger structural regressions.

The S74 relational DSAC path is rejected.

No width/channel/loss/activation sweep, residual fallback, target change, retry, second DEV or external Laya/Jev evaluation is authorized.
