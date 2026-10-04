# HIRA V1 S52 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #287  
PR: #288

## Qualified S52-A0

Run: `37199183308`  
Artifact: `11302402275`  
Digest: `sha256:b99bb216580bbd7092126fc028c9581519ca23bfa72a0a8e174c7175daa76b5a`  
Authorization head: `f294124c7f4a3fe1ddc173dd9e9b5123237a68f9`

Outcome:
`HIRA_V1_S52_A0_QUERY_RELATION_CANONICALIZATION_READY`

Qualified mechanics:
- sealed parent native authority exact
- native trainable params 0
- cache inference/grad tensor counts 0
- correction params 114,688 each
- canonicalizer params 32,768 each
- total private trainable 147,456 each
- query-free identity params 0
- reference/treatment initialization bit-identical
- zero-init raw-query error <=1e-7
- K=3/7/255
- treatment auxiliary nonzero
- auxiliary canonicalizer gradient >0
- auxiliary correction gradient =0
- reference auxiliary =0.

## Fresh S52 authority

`src/nmd/v1_s52_authority.py`

Frozen:
- seed **73001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S52 domains
- K=4
- 2 relation questions per semantic case
- canonical/paraphrase paired views
- no exact overlap with S51 TRAIN/DEV
- no exact overlap with S52-A0.

## Parent native authority

Reuse exact S51 artifact:
- run `37192490832`
- artifact `11299783210`
- runtime/native digest:
  `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA:
  `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`

Native retraining is forbidden.

## Matched private court

Script:
`scripts/hira_v1_s52_train_dev.py`

Workflow:
`.github/workflows/hira-v1-s52-query-relation-train-dev.yml`

Marker:
`research/HIRA-V1-S52-ENABLE-TRAIN-DEV`

Per arm:
- correction **114,688**
- canonicalizer **32,768**
- total private trainable **147,456**
- identity params **0**
- same initialization
- same cache bytes
- same TRAIN order
- same AdamW LR/weight decay
- same grad clip
- same correctness CE + relation JS
- same checkpoint selector
- private epochs **24**

Controlled variable only:
- reference relation-code auxiliary coefficient **0.0**
- treatment coefficient **0.10**

Frozen treatment auxiliary:
- same-relation paraphrase alignment
- different-relation centroid hinge
- separation ceiling **0.25**

Checkpoint selection stores:
- correction adapter A/B
- correction bilinear W
- canonicalizer A/B.

## Query-code DEV diagnostics

Selected checkpoints report:
- same-relation A cosine
- same-relation B cosine
- mean same-relation cosine
- A-vs-B centroid cosine
- relation separation margin
- raw-vs-canonicalized query cosine
- canonicalizer residual norm.

## Stop rule

After the one S52 DEV begins:
- no auxiliary coefficient sweep
- no separation-margin sweep
- no hidden dimension change
- no raw-query bypass
- no native retraining
- no identity variant
- no correction/canonicalizer capacity change
- no selector change
- no retry
- no gate weakening
- no second S52 DEV
- no external Laya/Jev evaluation.

## Authorization rule

`research/HIRA-V1-S52-ENABLE-TRAIN-DEV` MUST remain absent until the exact final staging head passes generic CI on Python 3.10 and 3.12.
