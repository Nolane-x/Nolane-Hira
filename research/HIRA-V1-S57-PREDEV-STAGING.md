# HIRA V1 S57 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #297  
PR: #298

## Qualified S57-A0

Run: `37266891469`  
Artifact: `11326757222`  
Digest: `sha256:fcfbececa46f5bac0e1fe75ae51348d1b4c543b5b7dc3166410c148a35dd75ce`  
Authorization head: `8394339694c9dfece088711f13b75ea4b767c93c`

Outcome:
`HIRA_V1_S57_A0_DISCRETE_PAIRWISE_RANKING_CONSISTENCY_READY`

Qualified:
- anchor detach PASS
- sign-flip sensitivity live
- shared-offset invariance exact
- positive-scale invariance PASS
- wrong gold anchors filtered
- correct gold anchors retained
- non-gold pairs remain eligible
- flat/uniform anti-collapse PASS
- option permutation PASS
- K=3/7/255 PASS
- full-K probability mass PASS
- treatment auxiliary gradient live
- native/cache isolation PASS.

## Fresh S57 authority

- seed **78001**
- TRAIN **768**
- DEV **192**
- 12 ordinal-consistency domains
- exact S56 state/question/option overlap **0**
- K=4
- private epochs **24**

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
- ordinal coefficient **0.0**

Treatment:
- ordinal coefficient **0.05**

Frozen ordinal mechanics:
- standardization epsilon **1e-6**
- active threshold **0.25**
- preserved margin floor **0.05**
- detached anchor signs
- gold-order protection enabled.

## Pinned authority chain

S51 native:
- run `37192490832`
- artifact `11299783210`
- runtime/native digest `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`

S57 A0:
- run `37266891469`
- artifact `11326757222`
- digest `sha256:fcfbececa46f5bac0e1fe75ae51348d1b4c543b5b7dc3166410c148a35dd75ce`

## Stop rule

Once private TRAIN begins:
- no coefficient sweep
- no threshold/margin sweep
- no anchor-filter variant
- no teacher/consensus anchor
- no architecture/capacity change
- no native retraining
- no selector change
- no retry for scientific weakness
- no gate weakening
- no second S57 DEV
- no external Laya/Jev evaluation.

## Authorization rule

Marker:
`research/HIRA-V1-S57-ENABLE-TRAIN-DEV`

The marker MUST remain absent until the exact final pre-DEV staging head passes generic CI on Python 3.10 and 3.12.
