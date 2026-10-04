# HIRA V1 S52 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #287  
PR: #288

## Parent

S51 merged main:
`a7f9fb501c9f5e08ee59f1dd2e9d6c02be97cef9`

S51 valid matched court:
- run `37197889785`
- artifact `11301721375`
- digest `sha256:07cb321dd6f390bea5685f9c0786069ca77e4be757960c55bd3cbad8f21f7d43`
- interpretation **Case C**

Key S51 result:
- fused canonical **+2.08 pp**
- fused paraphrase **+2.08 pp**
- paired both-correct **+4.69 pp**
- fused agreement **-0.52 pp**
- relation agreement **-10.94 pp**

Query-free identity remained geometrically stable but did not solve cross-view instability.

## Frozen S52 variable

Both arms contain:
- query-free state↔option identity params **0**
- correction params **114,688**
- query relation canonicalizer params **32,768**
- total private trainable **147,456**
- bit-identical initialization.

Canonicalizer:
`q* = normalize(q + B(gelu(Aq)))`

- A **64×256**
- B **256×64**
- seed **73052**
- B zero-init
- no raw-query bypass.

Controlled variable:
- reference relation-code auxiliary coefficient **0.0**
- treatment coefficient **0.10**

Frozen treatment auxiliary:
- same-relation paraphrase cosine alignment
- A-vs-B relation centroid hinge
- separation ceiling **0.25**
- no sweep.

## Parent native authority

S52 reuses the sealed S51 native authority:
- run `37192490832`
- artifact `11299783210`
- runtime/native digest:
  `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA:
  `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`

No native retraining is part of S52-A0.

## Core staged

- `research/HIRA-V1-S52-CONTRACT.md`
- `research/HIRA-V1-S52-INTERPRETATION-PLAN.md`
- `src/nmd/v1_query_relation_canonicalization.py`
- `tests/test_v1_query_relation_canonicalization.py`

Core courts:
- 32,768 canonicalizer params
- 147,456 total private trainable
- zero-init identity behavior
- bit-identical initialization
- K=3/7/255
- full-K/probability mass
- no raw-query bypass
- reference auxiliary exactly zero
- treatment auxiliary nonzero
- auxiliary gradients reach canonicalizer
- auxiliary gradients into correction exactly zero
- distinct-relation non-collapse.

## A0 staged

- `scripts/hira_v1_s52_a0_query_relation_canonicalization.py`
- `tests/test_v1_s52_a0_harness.py`
- `.github/workflows/hira-v1-s52-a0-query-relation-canonicalization.yml`

A0 is mechanical/diagnostic only:
- no model selection
- no S52 TRAIN/DEV
- no native retraining
- no external Laya/Jev evaluation.

## Authorization rule

A0 marker:
`research/HIRA-V1-S52-ENABLE-A0`

The marker MUST remain absent until the exact final pre-A0 staging head passes generic CI on Python 3.10 and 3.12.
