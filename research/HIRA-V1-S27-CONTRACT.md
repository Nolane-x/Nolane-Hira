# HIRA V1 S27 contract — Blockwise Cross-View Factor Canonicalization

Status: **OPEN / PREREGISTERED BEFORE S27-A0 EXPOSURE**

Issue: #237

Base main:
`68490a0d196b5503280c55847496e65da354e9c6`

Parent:
- S26 outcome `HIRA_V1_S26_FACTORIZED_RELATION_DEV_FAIL`
- S26 selected checkpoint `ba99c7e4c40ccc85fd2577a1eec636138f1277556068945f296d5f09b0db88c5`

## 1. Fixed hypothesis

S26 inference produces a 256D factorized relation signature:
- role-relation block: 128D
- value/content-relation block: 128D

S26 canonicalizes the concatenated signature as one vector. Fresh authority shows canonical discrimination but poor transport across equivalent wording.

S27 tests one variable only:

> Canonicalizing the role block and value block independently prevents one factor from compensating for or drifting against the other across wording views.

## 2. Inference identity

S27 inference is **exactly S26**:
- S21 primary unchanged
- S26 factorized relation logits unchanged
- S26 factorized 256D signatures unchanged
- S14 equal standardized full-K fusion unchanged
- no new inference parameter/state/path
- exact trainable surface remains **81,920**

Any inference-logit/signature difference from S26 is an A0 failure.

## 3. Frozen blockwise loss

Input:
- canonical factorized signature [B,K,256]
- paraphrase factorized signature [B,K,256]

Split exactly:
- role block = [:128]
- value block = [128:]

Normalize each block independently.

For each block:
- same-option alignment = mean(1 - cosine(canonical_i, paraphrase_i))
- wrong-option separation = relu(0.20 - (same_i - strongest_wrong_i))

Block total:
- alignment + separation

S27 blockwise loss:
- **0.50 × role-block total + 0.50 × value-block total**

Frozen:
- block weights **0.50 / 0.50**
- separation margin **0.20**
- outer canonicalization coefficient in TRAIN loss **0.15**

No learned block weights.
No temperature.
No extra head.

## 4. Inherited optimization/capacity

Keep S26:
- shared A13 LoRA **16,384**
- primary-private projection **32,768**
- relation-private projection **32,768**
- exact total **81,920**
- original A13 frozen
- HIRACore frozen
- S25 private/shared gradient ownership
- neutral-bisector only on shared LoRA, epsilon **1e-12**
- relation detach from fused-primary objective
- role/value inference score weights **0.50 / 0.50**
- all inference temperatures unchanged
- 24 epochs / batch 16 / AdamW 2e-4 / wd 0.01 / grad clip 1.0
- other loss coefficients unchanged

## 5. Required A0

A0 must prove before fresh DEV:

### Identity
- S27 primary logits == S26 primary logits
- S27 relation logits == S26 relation logits
- S27 factorized signatures == S26 factorized signatures
- S27 fused logits == S26 fused logits
- exact physical surface **81,920**
- added learned params **0**

### Loss localization
Synthetic orthogonal signatures must prove:
- identical well-separated views -> total/role/value loss near zero;
- role-only mismatch increases role loss while value loss remains near zero;
- value-only mismatch increases value loss while role loss remains near zero;
- both-factor mismatch increases both;
- swapping role/value blocks swaps corresponding component losses;
- option permutation preserves total and component losses.

### Gradient ownership
On fresh S27-A0 text rows:
- relation-private gradient nonzero
- relation shared-LoRA gradient nonzero
- primary→relation-private leakage exactly zero
- relation→primary-private leakage exactly zero
- no new parameter storage
- full-K/state-once preserved

A0 semantic accuracy is diagnostic only.

## 6. Fresh TRAIN/DEV

Only after qualified A0 + frozen interpretation + exact-head CI + one-shot enable.

Use wholly fresh S27 authority:
- seed **48001**
- TRAIN **768**
- DEV **192**
- 12 fresh S27-only domains
- K=4
- two state views
- two question views
- two option views

No S0-S26 exposed row may be reused.

## 7. DEV gates

Keep the existing frontier gates:
- fused canonical >= **0.85**
- paired >= **0.75**
- question-swap >= **0.80**
- fused agreement >= **0.95**
- fused JS <= **0.05**
- fused canonical margin >= **0.15**
- relation canonical >= **0.80**
- relation canonical margin >= **0.15**
- whole-signature same-option cosine >= **0.90**
- whole-signature discrimination margin >= **0.15**

S27 additionally records role-block and value-block cross-view cosine/margins, but these diagnostics may not weaken the existing gates.

## 8. Stop rule

After DEV exposure:
- no block-weight tuning
- no separation-margin tuning
- no canonicalization-coefficient tuning
- no seed/LR/epoch retry
- no inference change
- no capacity change
- no selector change
- no gate weakening
- no second DEV run

Scientific FAIL is valid.

No sealed confirmation, multilingual transfer, or Laya/Jev benchmark unless DEV_READY.
