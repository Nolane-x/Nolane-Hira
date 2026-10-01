# HIRA V1 S28 contract — Anchor-Level Cross-View Factor Transport

Status: **OPEN / PREREGISTERED BEFORE S28-A0 EXPOSURE**

Issue: #239

Parent:
- S27 issue #237 / PR #238
- S27 outcome `HIRA_V1_S27_BLOCKWISE_CANONICALIZATION_DEV_FAIL`
- S27 closure head `bdeb239f0fb83575679107a10a8d9787d606da1a`

## 1. Fixed hypothesis

S27 can improve final relation-signature alignment without producing strong correct-vs-wrong relation separation.

S28 tests one upstream hypothesis:

> Equivalent wording must extract the same query-conditioned state role anchor and state value/content anchor **before option scoring**. Enforcing transport at these latent anchors, with explicit cross-query anti-collapse negatives, will generalize better than aligning only the downstream option relation signature.

## 2. Inference identity

S28 production inference remains exactly S26/S27:
- S21 primary unchanged
- S26 factorized relation logits unchanged
- S26 factorized relation signatures unchanged
- S14 equal standardized full-K fusion unchanged
- no new inference path/state/parameter
- exact trainable surface **81,920**

The S28 anchor extractor exists only in the training/A0 court.

Any production inference difference is an A0 failure.

## 3. Training-only anchor extractor

Inputs:
- relation-private projection
- state token embeddings/mask
- question token embeddings/mask

Frozen role extraction:
1. project state/question with the S26 relation-private projection;
2. query-state cosine score via dot product;
3. max over active question tokens;
4. masked softmax with role temperature **0.10**;
5. normalized weighted state role anchor.

Frozen value/content extraction:
1. remove the role-anchor direction from projected state tokens;
2. normalize residual token vectors;
3. content weights = normalized `(1 - role_weight)` over active state tokens;
4. normalized weighted residual state value anchor.

These formulas must match the state-side latent construction used by S26.

Extractor learned parameters: **0**.

## 4. Anchor transport objective

For canonical and paraphrase views of corresponding semantic queries, for each factor independently:

- same-pair cosine `s_i = cos(canonical_i, paraphrase_i)`;
- hardest wrong cosine `w_i = max_{j != i} cos(canonical_i, paraphrase_j)`;
- alignment = mean(`1 - s_i`);
- separation = mean(`relu(0.20 - (s_i - w_i))`);
- factor total = alignment + separation.

Frozen combination:
- role factor weight **0.50**
- value factor weight **0.50**
- anchor separation margin **0.20**

Combined anchor transport loss:
- `0.50 * role_total + 0.50 * value_total`.

Frozen outer coefficient in the relation training block:
- **0.15**

S28 **replaces** S27's `0.15 * blockwise_output_canonicalization` term with `0.15 * anchor_transport`; it does not stack both canonicalization objectives.

This objective uses semantic-query pairing only.
No option IDs.
No option order.
No learned temperature or weighting.

A constant/collapsed anchor set has same≈wrong and therefore pays at least the separation margin.

## 5. Inherited capacity / optimizer shell

Keep:
- shared A13 LoRA **16,384**
- primary-private projection **32,768**
- relation-private projection **32,768**
- total trainable **81,920**
- original A13 frozen
- HIRACore frozen
- S25 private/shared gradient ownership
- neutral-bisector only on shared LoRA, epsilon **1e-12**
- S14 equal fusion
- relation detach from fused-primary objective
- AdamW / 24 epochs / batch 16 / lr 2e-4 / wd 0.01 / grad clip 1.0

S28 does not change inference score weights or temperatures.

## 6. Required A0

### Inference
- primary logits identity vs S27/S26 exactly
- relation logits identity exactly
- factorized signature identity exactly
- fused logits identity exactly
- exact 81,920 physical surface
- extractor/objective added params **0**

### Extractor
- output role/value anchors normalized
- reference S26-formula role anchor identity <= **1e-7**
- reference S26-formula value anchor identity <= **1e-7**
- finite degenerate geometry

### Objective
Synthetic orthogonal court:
- identical well-separated canonical/paraphrase anchors -> total/role/value loss near zero
- role-only mismatch activates role total, leaves value total near zero
- value-only mismatch activates value total, leaves role total near zero
- constant/collapsed role anchors incur positive role separation
- constant/collapsed value anchors incur positive value separation
- simultaneous permutation of semantic-query batch preserves loss
- option permutation leaves anchor loss exactly unchanged because options are absent

### Gradient ownership
Fresh S28-A0 text rows:
- relation-private gradient nonzero
- relation shared-LoRA gradient nonzero
- primary-private gradient nonzero from primary block
- primary -> relation-private leakage **0**
- relation -> primary-private leakage **0**
- full-K/state-once preserved

A0 semantic accuracy is diagnostic only.

## 7. Fresh TRAIN/DEV

Only after qualified A0 + frozen interpretation + exact-head CI + one-shot authorization.

Frozen intended authority:
- seed **49001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S28 domains
- K=4
- two state views
- two question views per semantic query
- two option views

No exact S0-S27 exposed row may enter S28 TRAIN/DEV.

## 8. Stop rule

After DEV exposure:
- no anchor-weight tuning
- no margin tuning
- no anchor-loss coefficient tuning
- no temperature change
- no seed/LR/epoch retry
- no inference/capacity/fusion change
- no selector/gate change
- no second DEV run

Scientific FAIL is valid.

No sealed confirmation, multilingual transfer, or Laya/Jev benchmark unless DEV_READY.
