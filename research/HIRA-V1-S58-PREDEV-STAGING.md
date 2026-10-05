# HIRA V1 S58 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #299  
PR: #300

## Qualified S58-A0

Run: `37274963437`  
Artifact: `11330280320`  
Digest: `sha256:f4785071bafcf3674cecefbba98468c9d2dcb42a5c1212f0c446177bde5f5ebc`  
Authorization head: `fac398d0d42709c79fe0d82178b5e72fc6d994a7`

Outcome:
`HIRA_V1_S58_A0_CONSENSUS_TEACHER_PAIRWISE_RANKING_READY`

Qualified:
- frozen S57 reference teacher replay exact
- teacher trainable parameters **0**
- teacher logits require-grad **false**
- consensus target detached
- strong same-sign consensus active
- low-confidence teacher pairs inactive
- teacher sign-disagreement pairs inactive
- wrong gold teacher pairs filtered
- non-gold consensus remains eligible
- flat-student anti-collapse live
- reference auxiliary exact zero
- treatment auxiliary gradient live
- K=3/7/255 PASS
- one encoder/state-once preserved
- full-K probability mass PASS.

Observed A0 diagnostics:
- real-cache teacher active consensus fraction: **0.5885416865**
- real-cache wrong-gold filtered fraction: **0.0737704933**
- real-cache student violation fraction: **0.0929203555**
- teacher deterministic replay max abs error: **0**
- treatment auxiliary gradient L1: **1.6191167831**
- max probability-mass error: **2.384185791015625e-7**.

## Frozen teacher authority

S57 reference teacher:
- source run `37271509208`
- source artifact `11327849211`
- branch **reference**
- selected epoch **19**
- checkpoint `reference-private-candidate.pt`
- checkpoint SHA-256 `804dc0b31ca75a77400c4a658ae893058d39916ed287d9fe2d45643798b739aa`
- private/correction params **114,688**
- S57 ordinal coefficient **0.0**
- trainable parameters in S58 **0**
- no S58-based teacher selection.

## Fresh S58 authority

- seed **79001**
- TRAIN **768**
- DEV **192**
- **12 wholly fresh S58 domains**
- exact S57 state/question/option overlap **0**
- K=4
- private epochs **24**
- one DEV only.

## Matched court

Both arms:
- `JointStateQueryOptionPrivateCorrectionFork`
- correction params **114,688**
- added params **0**
- identity params **0**
- same S51 persisted native artifact
- same immutable TRAIN/DEV cache
- same bit-identical student initialization
- same optimizer/LR/weight decay/grad clip
- same base correctness/relation objective
- same frozen checkpoint selector
- same frozen teacher checkpoint
- same teacher targets precomputed exactly once before student training.

Reference:
- teacher-consensus coefficient **0.0**

Treatment:
- teacher-consensus coefficient **0.05**

Frozen teacher-consensus mechanics:
- teacher active threshold **0.25**
- student target margin **0.05**
- both teacher views must clear threshold
- both teacher views must agree in sign
- wrong gold teacher pairs are inactive
- non-gold consensus pairs remain eligible
- consensus sign detached
- no full-distribution JS auxiliary
- no self-generated anchor.

## Pinned authority chain

S51 native:
- run `37192490832`
- artifact `11299783210`
- runtime/native digest `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`

S57 teacher:
- run `37271509208`
- artifact `11327849211`
- checkpoint SHA `804dc0b31ca75a77400c4a658ae893058d39916ed287d9fe2d45643798b739aa`

S58 A0:
- run `37274963437`
- artifact `11330280320`
- digest `sha256:f4785071bafcf3674cecefbba98468c9d2dcb42a5c1212f0c446177bde5f5ebc`

## Stop rule

Once private TRAIN begins:
- no teacher swap
- no teacher-threshold sweep
- no target-margin/coefficient sweep
- no consensus-mask variant
- no architecture/capacity change
- no native retraining
- no selector change
- no retry for scientific weakness
- no gate weakening
- no second S58 DEV
- no external Laya/Jev evaluation.

## Authorization rule

Marker:
`research/HIRA-V1-S58-ENABLE-TRAIN-DEV`

The marker MUST remain absent until the exact final pre-DEV staging head passes generic CI on Python 3.10 and 3.12.
