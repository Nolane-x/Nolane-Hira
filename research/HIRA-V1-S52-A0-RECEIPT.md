# HIRA V1 S52 A0 receipt — Paired-View Query Relation Canonicalization

Status: **QUALIFIED**

Run: `37199183308`  
Artifact: `11302402275`  
Artifact digest: `sha256:b99bb216580bbd7092126fc028c9581519ca23bfa72a0a8e174c7175daa76b5a`  
Authorization head: `f294124c7f4a3fe1ddc173dd9e9b5123237a68f9`

Outcome:
`HIRA_V1_S52_A0_QUERY_RELATION_CANONICALIZATION_READY`

## Parent authority

- native runtime hash:
  `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA:
  `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`
- native trainable params: **0**

## Matched private surface

Per arm:
- correction params: **114,688**
- query canonicalizer params: **32,768**
- query-free identity params: **0**
- total private trainable: **147,456**

Initialization:
- reference/treatment bit-identical: **true**
- initial branch logit max abs error: **0**
- zero-init raw-query max abs error: **2.98e-8**

## Full-K mechanics

- K=3 PASS
- K=7 PASS
- K=255 PASS
- probability-mass max error **1.192e-7**
- second encoder pass **false**

## Shared cache

- digest `cca23237eba206a420a8137749c7addea85e9b458302fe117859be040d55994f`
- inference tensor count **0**
- requires-grad tensor count **0**

## Relation-code auxiliary court

Treatment:
- same-relation loss **0.006224**
- different-relation hinge **0.741297**
- weighted auxiliary **0.0747521**
- canonicalizer gradient L1 **0.00846146**
- correction gradient L1 **0**

Reference:
- auxiliary coefficient **0**
- auxiliary value **0**

Therefore the controlled treatment auxiliary reaches the query canonicalizer while remaining isolated from the correction A/B/W surface.

A0 is diagnostic only:
- model selection **false**
- production-ready claim **false**

Fresh S52 TRAIN/DEV remains separately gated.
