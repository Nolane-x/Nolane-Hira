# HIRA V1 S32 contract — Global Query×Option Relation Matrix Canonicalization

Status: **OPEN / PREREGISTERED BEFORE S32-A0 EXPOSURE**

Issue: #247

Parent:
- S31 issue #245 / PR #246
- S31 merged main `05b29a42a6539a806649408a17ec8a6fa2d2e8ed`
- S31 outcome `HIRA_V1_S31_MATCHED_RELATION_DEV_COMPLETE`

## 1. Scientific question

S31 showed that global cross-case query-specific discrimination can improve fresh semantic relation accuracy and margins, especially under paraphrase.

But the S31 treatment globally paired only the gold relation signature for each semantic query. The other K-1 query-option relations received no global cross-view identity constraint.

Fresh S31 DEV exposed the result:
- local all-option signature cosine: **0.9415669243**
- global gold-only signature cosine: **0.6501545608**
- best global gold-only signature cosine across 24 epochs: **0.7729473710**

S32 tests one variable only:

> Does extending the same global cross-case contrastive topology from gold-only relations to the complete query×option relation matrix preserve S31's semantic-discrimination gains while restoring cross-view identity for every candidate relation?

## 2. Frozen architecture

Both arms use exact S17 architecture and optimization shell:
- final A13 attention LoRA: **16,384**
- shared bias-free 256→128 projection: **32,768**
- exact trainable surface: **49,152**
- original A13 frozen
- HIRACore frozen
- S13 relation expert
- S14 equal standardized full-K fusion, epsilon **1e-6**
- S15 fused-primary relation-logit detach
- S17 primary/relation loss partition
- S17 relation-priority norm-balanced gradient rule
- state-once
- full-K
- opaque option IDs
- no learned downstream scorer/router/gate/calibrator

Inference is identical across arms.

## 3. Arm A — gold-only global control

Exact S31 global treatment:
- relation CE coefficient **0.10**
- global gold-signature contrastive coefficient **0.15**
- temperature **0.10**
- for each semantic query q:
  - canonical gold signature <-> paraphrase gold signature is positive
  - gold signatures of all other semantic queries in the batch are negatives
- zero added learned parameters/state

## 4. Arm B — all query×option global treatment

Relation CE remains **0.10**.

Replace the gold-only global regularizer with:

For every semantic query q and logical option k:
- canonical signature (q,k) <-> paraphrase signature (q,k) is the positive pair;
- every other query-option relation (q',k') in the batch is a negative;
- flatten [Q,K,D] -> [Q*K,D];
- L2 normalize;
- symmetric canonical->paraphrase and paraphrase->canonical InfoNCE;
- temperature **0.10**;
- outer coefficient **0.15**;
- zero added learned parameters/state.

This is not local K=4 canonicalization:
- negatives span the complete batch;
- same-query wrong options are negatives;
- cross-query options are negatives;
- every logical option receives a cross-view positive identity target.

## 5. Required A0 discriminator

A0 is diagnostic only.

It must establish exact inference identity and exact 49,152 capacity in both arms.

It must also directly distinguish the treatment from S31 gold-only:

Construct controlled [Q,K,D] signatures and designated gold indices.

### Non-gold perturbation court

Perturb or mis-pair exactly one **non-gold** query-option relation while leaving all gold relations untouched.

Required:
- S31 gold-only loss changes by **exactly 0**
- S32 all-option loss increases materially
- preregistered materiality threshold: **>= 0.10**

### Symmetry/equivariance court

Required:
- matched query permutation preserves all-option loss
- matched option permutation preserves all-option loss
- canonical/paraphrase view swap preserves all-option loss
- errors <= **1e-7**

### Gradient court

Under all-option loss:
- canonical non-gold signature gradient L1 > 0
- paraphrase non-gold signature gradient L1 > 0
- full canonical gradient finite
- full paraphrase gradient finite

Real semantic A0:
- attention LoRA B gradients nonzero
- projection gradient nonzero
- primary/relation gradient blocks nonzero and finite
- S17 norm-balanced update finite

Mechanics:
- full-K
- state-once
- option-order flip 0
- probability mass <= 1e-6
- checkpoint roundtrip
- original A13/HIRACore frozen

A0 semantic accuracy cannot tune S32.

## 6. Fresh matched TRAIN/DEV

Only after qualified A0, frozen receipt, staged workflow, exact-head CI and explicit one-shot authorization.

Frozen:
- seed **53001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S32 domains
- K **4**
- two state views
- two question views per semantic query
- two semantic option views
- epochs **24**
- batch **16**
- AdamW lr **2e-4**
- weight decay **0.01**
- grad clip **1.0**
- identical rows and per-epoch batch order across arms
- independent runtime/model/optimizer state
- independent zero-init LoRA/projection copies from the same W28 T0

Freshness:
- no exact S0-S31 exposed rows
- no S32-A0 rows
- no M5 final/confirmatory rows
- no W29-W34 sealed rows

## 7. Selector

Each arm independently uses the frozen S17 lexicographic selector:
1. paired both-correct
2. fused canonical
3. relation canonical
4. relation canonical margin
5. fused canonical margin
6. question-swap
7. fused cross-view agreement
8. same-option signature cosine
9. signature discrimination margin
10. lower canonical decision loss
11. earlier epoch

No cross-arm epoch mixing.

## 8. DEV_READY gates

Each arm independently:
- fused canonical >= **0.85**
- paired >= **0.75**
- question-swap >= **0.80**
- fused agreement >= **0.95**
- fused JS <= **0.05**
- fused canonical margin >= **0.15**
- relation canonical >= **0.80**
- relation canonical margin >= **0.15**
- same-option signature cosine >= **0.90**
- signature discrimination >= **0.15**
- option flip <= **0.02**
- probability mass <= **1e-6**
- full-K
- state-once
- exact capacity
- original A13 frozen
- HIRACore frozen
- fusion params 0
- relation delta 0

## 9. Matched interpretation

Report exact selected DEV deltas:

`all-option global - gold-only global`

Do not collapse into one scalar winner score.

### A — transport restored and semantic gains retained

If all-option global materially restores signature cosine and keeps/improves the main relation/fused semantic endpoints without a new major regression, the all-option global family remains viable for a later fresh confirmation.

### B — transport restored, semantics regress

The complete relation-matrix objective over-constrains distractors. Close this global topology; do not tune temperature/coefficient from exposed DEV.

### C — little/no transport gain

The S31 collapse is not explained by gold-only supervision coverage. Close global relation-contrastive topology and move to a query-explicit relation representation.

### D — treatment DEV_READY

Freeze immediately. No second S32 DEV.

## 10. Stop rule

After matched S32 DEV exposure:
- no temperature change
- no coefficient change
- no negative mining change
- no local+global mixture
- no batch-size change
- no rank/LR/seed/epoch retry
- no loss stacking
- no gate weakening
- no second DEV run

No Laya/Jev benchmark unless a later confirmed candidate reaches DEV_READY.
