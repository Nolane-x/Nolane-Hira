# HIRA V1 S62 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #309  
PR: #310

## Parent S61

Merged main:
`e6c7053a8d24c75b5fcf1db75c8aa9f30cea35e2`

Fresh S61:
- run `37302829277`
- artifact `11343045582`
- digest `sha256:6e99ba1a5fab92cb6031c41eea54636721b78e690236dfb39788066803785f74`
- verdict **Case B**.

## Qualified S62-A0

Run: `37305254725`  
Artifact: `11342769947`  
Digest: `sha256:3db27b3a3a9fdc4c0b276c72c05649a6b77f59b868b2329914e94ccd7d483d0e`

Outcome:
`HIRA_V1_S62_A0_TRAIN_ONLY_RELIABILITY_SUPERVISED_ADAPTIVE_GATE_READY`

Qualified:
- correction params **114,688**
- pairwise params **32,832**
- reference gate **5 params**
- treatment gate **5 params**
- added treatment params **0**
- bit-identical gate initialization
- feature dimension **4**
- alpha probe **0.35**
- target tolerance **1e-8**
- beneficial probe => target 1
- correctness harm => target 0
- stability harm => target 0
- both harm => target 0
- mixed target includes both classes
- reference gold-CE gradients live
- treatment reliability-BCE gradients live
- upstream gradients zero
- no DEV target dependency
- K=3/7/255 PASS
- one encoder/state-once
- checkpoint replay exact.

Observed real-cache reliability-positive fraction at A0:
**0.25**.

## Fresh S62 authority

- seed **83001**
- TRAIN **768**
- DEV **192**
- **12 fresh S62 domains**
- exact S61 state/question/option overlap **0**
- K=4
- epochs **24**
- one DEV only.

## Matched scientific trajectory

Shared exactly:
- one immutable native/cache evidence surface
- one correction trajectory, **114,688 params**
- one explicit pairwise-head trajectory, **32,832 params**
- same TRAIN ordering
- same 4 gate features
- same 5-param gate architecture
- same gate initialization
- same alpha max **0.35**
- same initial alpha **0.10**
- same optimizer/LR/weight decay
- same frozen S17 selector.

Reference:
- independent 5-param gate
- objective = exact S61 canonical+paraphrase **gold CE**.

Treatment:
- independent 5-param gate
- objective = **TRAIN-only reliability BCE**.

Reliability target:
- fixed probe alpha **0.35**
- positive iff paired probe CE is non-worse AND paired JS strictly improves
- otherwise negative
- target detached
- no DEV target input.

At every TRAIN batch:
1. update shared correction from existing private objective;
2. update shared pairwise head from gold-pair objective;
3. recompute detached fused/pairwise surfaces;
4. update reference gate by gold CE;
5. update treatment gate by reliability BCE.

Gate gradients cannot enter correction, pairwise head, native runtime or cache.

## DEV comparison

Selection arms:
- reference = gold-CE adaptive gate
- treatment = reliability-BCE adaptive gate.

Diagnostic only:
- fused shadow baseline is reported but **never participates in selection**.

At every epoch freeze/report:
- correction SHA
- pairwise-head SHA
- reference gate SHA
- treatment gate SHA
- TRAIN reliability-positive fraction
- fused shadow metrics
- reference DEV metrics
- treatment DEV metrics.

## Pinned parent native authority

S51:
- run `37192490832`
- artifact `11299783210`
- runtime digest `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`.

## Stop rule

Once S62 TRAIN begins:
- no alpha-probe sweep
- no tolerance change
- no target-rule change
- no BCE weighting
- no feature addition/removal
- no hidden layer
- no alpha max/init sweep
- no optimizer/LR/weight-decay change
- no gate regularizer
- no gradient coupling
- no architecture/capacity change
- no native retraining
- no selector change
- no retry for scientific weakness
- no gate weakening
- no second S62 DEV
- no external Laya/Jev evaluation.

## Authorization rule

Marker:
`research/HIRA-V1-S62-ENABLE-TRAIN-DEV`

It MUST remain absent until the exact final pre-DEV staging head passes generic CI on Python 3.10 and 3.12.
