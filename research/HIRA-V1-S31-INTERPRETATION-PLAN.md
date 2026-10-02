# HIRA V1 S31 interpretation plan — frozen before S31-A0/DEV exposure

Status: **FROZEN**

Issue: #245
PR: #246

## Controlled question

Compare on identical fresh rows:

- **Local control**: exact S17 local K=4 relation-signature canonicalization.
- **Global treatment**: same S17 system, but replace only local signature canonicalization with symmetric cross-case query-specific InfoNCE.

The treatment uses temperature **0.10** and the same outer relation regularization coefficient **0.15**.

## Frozen architecture

Both arms:
- final attention LoRA **16,384**
- shared projection **32,768**
- total trainable **49,152**
- rank 8 / alpha 8 / dropout 0
- original A13 frozen
- HIRACore frozen
- S13 relation expert
- S14 equal standardized fusion
- S15 relation-logit detach
- S17 relation-priority norm-balanced optimizer
- S17 primary block unchanged
- zero learned downstream params

## Global operator

For Q=2N semantic queries in a batch:
- choose each query's gold relation signature in canonical view;
- choose its gold relation signature in paraphrase view;
- normalize;
- diagonal is positive pairing;
- all off-diagonal semantic queries are negatives;
- symmetric CE canonical->paraphrase and paraphrase->canonical;
- temperature **0.10**.

No within-query wrong-option signatures are added to this global bank. Relation CE remains responsible for K=4 correct-vs-wrong discrimination.

## Frozen matched TRAIN/DEV

- seed **52001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S31 domains
- K=4
- 2 state views
- 2 question views per semantic query
- 2 option views
- epochs **24**
- batch **16**
- AdamW lr **2e-4**
- weight decay **0.01**
- grad clip **1.0**
- identical rows and batch order across arms
- independent runtimes/optimizers

## Selector

Each arm independently uses exact S17 lexicographic selector:
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

No cross-arm epoch mixing.

## Gates

Each arm independently uses the existing v1 semantic/mechanical gate set.

## Matched interpretation

Report:
`global treatment - local control`

No scalar winner score.

### A — coherent global improvement
If global treatment improves relation accuracy/margins and fused endpoints without material transport regression, advance only to a fresh confirmation stage.

### B — local control matches/exceeds
Close global cross-case contrastive as a useful v1 rescue direction.

### C — split evidence
Close S31 as unresolved. Do not tune temperature, coefficient or negative mining on exposed DEV.

### D — treatment independently reaches DEV_READY
Freeze immediately. S31 itself still does not authorize production/Laya/Jev claims without next-stage confirmation.

## Stop rule

After one matched DEV:
- no temperature change
- no coefficient change
- no negative-bank redesign
- no batch-size retry
- no seed/LR/epoch retry
- no extra loss stacking
- no second DEV
- no weaker gates
