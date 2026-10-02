# HIRA V1 S32 interpretation plan — frozen before S32-A0/DEV exposure

Status: **FROZEN**

Issue: #247
PR: #248

## Controlled question

S32 compares two training-only global relation objectives on the exact S17 runtime:

- control: S31 gold-only global contrastive;
- treatment: global contrastive over every query-option relation.

No inference, capacity, fusion, optimizer or selector change is permitted.

## Frozen operators

Control:
- relation CE coefficient **0.10**
- gold-only global contrastive coefficient **0.15**
- temperature **0.10**

Treatment:
- relation CE coefficient **0.10**
- all-query×option global contrastive coefficient **0.15**
- temperature **0.10**

Both add **0 learned parameters**.

## Frozen fresh authority

- seed **53001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S32 domains
- K=4
- two state views
- two question views per semantic query
- two option semantic views
- 24 epochs
- batch 16
- AdamW lr 2e-4
- weight decay 0.01
- grad clip 1.0
- identical rows and batch order across arms
- independent runtime/model/optimizer state

## Frozen selector

Use the S17 lexicographic selector independently in each arm:
1. paired
2. fused canonical
3. relation canonical
4. relation canonical margin
5. fused canonical margin
6. question-swap
7. fused agreement
8. signature cosine
9. signature discrimination
10. lower canonical decision loss
11. earlier epoch

## DEV_READY gates

Unchanged:
- fused canonical >= .85
- paired >= .75
- question-swap >= .80
- fused agreement >= .95
- fused JS <= .05
- fused canonical margin >= .15
- relation canonical >= .80
- relation margin >= .15
- signature cosine >= .90
- signature discrimination >= .15
- option/mass/full-K/state-once/capacity/freeze gates PASS

## Matched interpretation

Primary comparison is:

`all-option global - gold-only global`

Report raw deltas; do not create one scalar score.

### A — transport restored and semantics retained/improved
All-option remains viable for a later wholly fresh confirmation.

### B — transport restored but semantics materially regress
All-option over-constrains distractor relations. Close this topology.

### C — little/no transport gain
Gold-only coverage was not the main source of S31 transport collapse. Close global relation-contrastive topology and move to an architectural query-explicit relation representation.

### D — all-option DEV_READY
Freeze immediately. No second S32 DEV.

## Stop rule

After DEV exposure:
- no temperature/coefficient tuning
- no hard-negative mining change
- no local+global mixture
- no batch-size/rank/LR/seed/epoch retry
- no selector/gate weakening
- no second DEV

No Laya/Jev benchmark unless a later confirmed candidate reaches DEV_READY.
