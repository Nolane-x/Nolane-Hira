# HIRA V1 S56 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #295  
PR: #296

## Qualified S56-A0

Run: `37215444292`  
Artifact: `11307997189`  
Digest: `sha256:48ee087d9a5fdd76790fbbdb625ebbd381ad44581a0df04bb047772baeff49cd`  
Authorization head: `abcf19a712c296ed5bd371f8ba5c5a119ac53e34`

Outcome:
`HIRA_V1_S56_A0_CROSS_VIEW_DECISION_CONSISTENCY_READY`

Qualified:
- private params **114,688 / arm**
- added trainable params **0**
- bit-identical initialization
- identical-logit JS 0
- disagreement JS live
- offset invariance exact
- positive-scale invariance exact
- ordering sign-flip sensitivity live
- inactive flat ordering exact
- reference auxiliary exact zero
- treatment auxiliary gradient live
- uniform-collapse diagnostics expose zero discrimination
- K=3/7/255 PASS
- full-K probability mass PASS
- native/cache isolation PASS.

## Fresh S56 authority

- seed **77001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh domains
- exact S55 state/question/option overlap **0**
- K=4
- private epochs **24**
- batch **16** semantics inherited through immutable cache.

## Matched court

Both arms:
- `JointStateQueryOptionPrivateCorrectionFork`
- correction params **114,688**
- added params **0**
- identity params **0**
- same S51 persisted native artifact
- same immutable TRAIN/DEV cache
- same initialization
- same optimizer/LR/weight decay/grad clip
- same base correctness/relation objective
- same checkpoint selector.

Reference:
- decision coefficient **0.0**
- ordering coefficient **0.0**

Treatment:
- standardized full-K decision JS coefficient **0.10**
- pairwise ordering coefficient **0.05**

Frozen:
- standardization epsilon **1e-6**
- ordering threshold **0.25**
- ordering margin floor **0.05**

## Pinned authority chain

S51 native:
- run `37192490832`
- artifact `11299783210`
- runtime/native digest `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`

S56 A0:
- run `37215444292`
- artifact `11307997189`
- digest `sha256:48ee087d9a5fdd76790fbbdb625ebbd381ad44581a0df04bb047772baeff49cd`

## Stop rule

Once private TRAIN begins:
- no coefficient sweep
- no ordering threshold/margin sweep
- no entropy auxiliary
- no architecture/capacity change
- no native retraining
- no selector change
- no retry for scientific weakness
- no gate weakening
- no second S56 DEV
- no external Laya/Jev evaluation.

## Authorization rule

Marker:
`research/HIRA-V1-S56-ENABLE-TRAIN-DEV`

The marker MUST remain absent until the exact final pre-DEV staging head passes generic CI on Python 3.10 and 3.12.
