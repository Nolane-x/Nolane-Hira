# HIRA V1 S34 contract — Query-Conditioned Entropic Relation Transport

Status: **OPEN / PREREGISTERED BEFORE S34-A0 EXPOSURE**

Issue: #251

Parent:
- S33 issue #249 / PR #250
- S33 outcome `HIRA_V1_S33_MATCHED_QUERY_EXPLICIT_DEV_COMPLETE`
- merged main `c8d6f6a5c71385e6a5d9b7ee66c800d4e5b26213`

## 1. Scientific question

S33 proved that retaining one explicit query coordinate after the S13 anchor/additive pipeline is mechanically active but materially worsens fresh semantics and cross-view transport.

S10-S13/S33 repeatedly collapse relation structure before final scoring through:
- pooled evidence;
- fixed positional windows;
- independently selected best pairs;
- role/value anchors;
- additive/residual signatures.

S34 asks:

> Does preserving a full many-to-many state↔option correspondence plan, with query-conditioned token marginals, improve fresh relation semantics and transport?

## 2. Frozen shared runtime

Both matched arms use exact S17:
- final A13 attention LoRA **16,384**
- shared bias-free 256->128 projection **32,768**
- exact trainable surface **49,152**
- original A13 frozen
- HIRACore frozen
- S14 equal standardized full-K fusion
- S15 relation-logit detach
- S17 primary/relation partition
- S17 relation-priority norm-balanced gradients
- state-once
- full-K
- opaque option IDs
- no learned downstream router/head/gate/calibrator

Training topology returns to local relation CE + local cross-view signature canonicalization.
No S31/S32 global contrastive objective.

## 3. Control

Exact S13 `CrossViewRelationCanonicalizer`:
- role temperature **0.10**
- pair temperature **0.10**
- contrastive temperature **0.10**
- relation CE coefficient **0.10**
- local signature canonicalization coefficient **0.15**
- separation margin **0.20**

## 4. Treatment

**QueryConditionedEntropicRelationTransport**

Zero learned parameters/state.

For each semantic query and each option semantic view:

1. project state/question/option tokens with the inherited shared projection;
2. L2-normalize valid projected tokens;
3. state relevance:
   `r_s = max_q cosine(q, state_s)`;
4. option relevance:
   `r_o = max_q cosine(q, option_o)`;
5. state marginal:
   masked softmax(`r_s / 0.10`);
6. option marginal:
   masked softmax(`r_o / 0.10`);
7. state-option kernel:
   `K = exp(cosine(state_s, option_o) / 0.10)`;
8. run exactly **12 Sinkhorn iterations** against the two query-conditioned marginals;
9. option-view relation score:
   expected state-option cosine under the transport plan divided by **0.10**;
10. option-view relation signature:
    normalize the transport-weighted mean of normalized `option_token - state_token` pair deltas;
11. average active option views and renormalize signature.

Frozen hyperparameters:
- state relevance temperature **0.10**
- option relevance temperature **0.10**
- kernel temperature **0.10**
- relation-logit temperature **0.10**
- Sinkhorn iterations **12**
- numerical epsilon **1e-12**

Forbidden:
- S13 base-logit mixing
- fixed local positional windows
- learned transport head
- learned query/key transport maps
- global cross-case loss
- external memory/router

## 5. What does not change

- no new learned tensor
- no encoder widening
- no projection widening
- no optimizer rule change
- no primary scorer change
- no fusion change
- no selector change
- no DEV gate change
- no external benchmark exposure

## 6. Required S34-A0

Use fresh S34-A0 rows only.

A0 must prove:
- treatment parameter count **0**
- exact physical trainable surface **49,152**
- original A13 trainable **0**
- HIRACore trainable **0**
- state-token permutation equivariance
- option-token permutation equivariance
- logical-option permutation equivariance
- masked-padding invariance
- controlled qA/qB changes state marginals
- controlled qA/qB changes option marginals
- controlled qA/qB changes transport plan
- controlled qA/qB switches the intended option
- row/column Sinkhorn marginal residual <= **1e-4**
- finite one-token geometry
- finite duplicate-token geometry
- real semantic LoRA-B gradients nonzero
- real projection gradient nonzero
- local relation CE/signature losses finite
- full-K/state-once/probability/checkpoint mechanics PASS

A0 semantics are diagnostic only.

## 7. Fresh matched TRAIN/DEV

Only after qualified A0, frozen receipt, exact-head CI and a separate one-shot marker.

Frozen intended authority:
- seed **55001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S34 domains
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
- exact S0-S33 exposed rows excluded
- S34-A0 rows excluded
- M5 final/confirmatory rows excluded
- W29-W34 sealed rows excluded

## 8. Selector and gates

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

After one fresh matched S34 DEV:
- no temperature tuning
- no Sinkhorn-iteration tuning
- no kernel redesign
- no score/signature reweighting
- no seed/LR/epoch/batch retry
- no local+global loss stacking
- no selector/gate weakening
- no second DEV

Scientific FAIL is valid.

No Laya/Jev benchmark unless a later confirmed candidate reaches DEV_READY.
