# HIRA V1 S33 contract — Query-Explicit Relation Coordinates

Status: **OPEN / PREREGISTERED BEFORE S33-A0 EXPOSURE**

Issue: #249

Parent:
- S32 issue #247 / PR #248
- S32 outcome `HIRA_V1_S32_MATCHED_GLOBAL_MATRIX_DEV_COMPLETE`
- merged main `7d3b48726d38400ae30bc6187de413d0e5e002ff`

## 1. Scientific question

S31/S32 show that adding global relation-discrimination losses can move semantic margins but does not restore stable all-option transport.

The remaining structural hypothesis is:

> the relation path uses the question indirectly to select role evidence, but discards explicit query coordinates before the final relation signature and score.

S33 tests whether retaining an explicit query-option coordinate inside the relation representation improves fresh semantics and transport.

## 2. Frozen shared runtime

Both matched arms use exact S17:
- final attention LoRA **16,384**
- shared bias-free projection **32,768**
- exact trainable surface **49,152**
- original A13 frozen
- HIRACore frozen
- S14 equal standardized full-K fusion
- S15 relation-logit detach
- S17 primary/relation loss partition
- S17 relation-priority norm-balanced gradients
- state-once
- full-K
- opaque option IDs
- no learned downstream router/head/gate/calibrator

Return to the local S13 relation training topology.
S31/S32 global relation-contrastive objectives are not used.

## 3. Control

Exact S13 `CrossViewRelationCanonicalizer`:
- role temperature **0.10**
- pair temperature **0.10**
- contrastive temperature **0.10**
- relation CE coefficient **0.10**
- local cross-view signature canonicalization coefficient **0.15**
- separation margin **0.20**

## 4. Treatment

**QueryExplicitRelationCanonicalizer**

Zero learned parameters/state.

First obtain the exact S13 base outputs:
- base relation logits
- base relation signature [B,K,128]

Then, from the same inherited projection:
1. project question tokens and L2 normalize;
2. masked-mean pool valid projected question tokens;
3. normalize to one query anchor [B,128];
4. recompute the exact S13 query-conditioned option role anchor [B,K,V,128];
5. compute per-option-view:
   `query_option_delta = normalize(option_anchor - query_anchor)`;
6. average valid query-option deltas across option views and normalize to [B,K,128];
7. treatment signature:
   `normalize(concat(base_signature, query_option_signature))`
   -> exact width **256**.

Explicit query-option compatibility:
- per view: `dot(query_anchor, option_anchor)`;
- average valid views;
- divide by the same contrastive temperature.

Treatment logits:
`0.5 * base_S13_logits + 0.5 * query_option_logits`.

Weights **0.50 / 0.50** are frozen before A0 and cannot be tuned.

## 5. What does not change

- no new trainable tensor
- no encoder widening
- no projection widening
- no global contrastive loss
- no new optimization rule
- no fusion change
- no selector change
- no gate change
- no external benchmark exposure

## 6. Required A0

Use fresh S33-A0 rows only.

A0 must prove:
- treatment parameter count **0**;
- runtime physical trainable surface remains **49,152**;
- original A13 trainable **0**;
- HIRACore trainable **0**;
- base S13 logits returned inside treatment are exact vs control;
- base S13 signature block returned inside treatment is exact vs control;
- query-anchor implementation matches the frozen masked-mean projected formula;
- treatment signature width **256**;
- option permutation equivariance;
- question intervention with fixed state/options changes:
  - query anchor
  - query-option signature block
  - explicit query-option compatibility/logits;
- controlled qA/qB geometry prefers the corresponding option;
- finite degenerate geometry;
- real semantic LoRA-B gradients nonzero;
- real projection gradient nonzero;
- local relation/signature losses finite;
- full-K/state-once/option-order/probability-mass/checkpoint mechanics PASS.

A0 semantic accuracy is diagnostic only.

## 7. Fresh matched TRAIN/DEV

Only after qualified A0, frozen receipt, exact-head CI, and a separate one-shot marker.

Frozen:
- seed **54001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S33 domains
- K=4
- two state views
- two question views per semantic query
- two semantic option views
- epochs **24**
- batch **16**
- AdamW lr **2e-4**
- weight decay **0.01**
- grad clip **1.0**
- identical rows and batch order across arms
- independent runtime/optimizer state

Freshness:
- exact S0-S32 exposed rows excluded
- S33-A0 rows excluded
- M5 final/confirmatory rows excluded
- W29-W34 sealed rows excluded

## 8. Independent selector and gates

Use exact S17 selector independently per arm.

Keep existing DEV_READY gates:
- fused canonical >= .85
- paired >= .75
- question-swap >= .80
- fused agreement >= .95
- fused JS <= .05
- fused canonical margin >= .15
- relation canonical >= .80
- relation margin >= .15
- same-option signature cosine >= .90
- signature discrimination >= .15
- full-K/state-once/capacity/freeze mechanics

## 9. Stop rule

After one fresh matched S33 DEV:
- no 0.50/0.50 tuning
- no query pooling change
- no signature block reweighting
- no temperature tuning
- no global-loss stacking
- no seed/LR/epoch/batch retry
- no gate/selector weakening
- no second DEV

Scientific FAIL or split evidence is valid.

No Laya/Jev benchmark unless a later confirmed candidate reaches DEV_READY.
