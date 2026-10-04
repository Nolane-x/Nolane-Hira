# HIRA V1 S53 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #289  
PR: #290

## Qualified S53-A0

Run: `37205099128`  
Artifact: `11304188134`  
Digest: `sha256:08dbedcf29c7c71646c0c17a7e079de7df4d763ea0888db01282887867dbc841`  
Authorization head: `35ae7f9aa2aa9dc90e5f1a74d172527ffc14ac8c`

Outcome:
`HIRA_V1_S53_A0_TOKEN_QUERY_OPTION_LATE_INTERACTION_READY`

Qualified:
- parent native authority exact
- native trainable params 0
- shared cache inference/grad tensor counts 0
- reference/treatment correction 114,688 each
- treatment late-interaction params 0
- identity params 0
- correction initialization bit-identical
- K=3/7/255
- probability mass
- context norm
- attention mass
- query padding/mask invariance
- query-token permutation invariance
- option permutation equivariance
- all-masked query rejection
- per-option context diversity
- no pooled-query bypass
- informative-token context sensitivity
- deterministic activated-path test proves token perturbation reaches treatment logits once frozen zero-init correction B/W are nonzero.

## Fresh S53 authority

`src/nmd/v1_s53_authority.py`

Frozen:
- seed **74001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S53 domains
- K=4
- canonical/paraphrase paired views
- no exact overlap with S52 TRAIN/DEV
- no exact overlap with S51 TRAIN/DEV.

## Parent native authority

Reuse exact S51 artifact:
- run `37192490832`
- artifact `11299783210`
- runtime/native digest:
  `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA:
  `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`

Native retraining forbidden.

## Matched private court

Script:
`scripts/hira_v1_s53_train_dev.py`

Workflow:
`.github/workflows/hira-v1-s53-token-query-option-train-dev.yml`

Marker:
`research/HIRA-V1-S53-ENABLE-TRAIN-DEV`

Reference:
- `QueryFreeIdentityPrivateCorrectionFork`
- pooled global query context.

Treatment:
- `TokenLateInteractionQueryFreePrivateCorrectionFork`
- option-conditioned token-level query context.

Both:
- query-free state↔option identity
- correction params **114,688**
- private trainable **114,688**
- identity params **0**
- same correction initialization
- same shared cache bytes
- same TRAIN order
- same AdamW LR/weight decay
- same CE + relation-JS objective
- same grad clip
- same checkpoint selector
- private epochs **24**.

Controlled variable only:
- pooled global query vs option-conditioned token-level query context.

Treatment fixed:
- cosine token↔option score
- temperature **0.10**
- masked softmax aggregation
- no token projection
- no pooled-query bypass.

## Treatment DEV diagnostics

- mean token-attention entropy
- mean max token weight
- context cross-view same-option cosine
- context norm max error.

## Stop rule

After one S53 DEV begins:
- no temperature sweep
- no aggregation-family switch
- no trainable token projection
- no pooled-query bypass
- no native retraining
- no identity variant
- no correction capacity change
- no selector change
- no retry
- no gate weakening
- no second S53 DEV
- no external Laya/Jev evaluation.

## Authorization rule

`research/HIRA-V1-S53-ENABLE-TRAIN-DEV` MUST remain absent until the exact final staging head passes generic CI on Python 3.10 and Python 3.12.
