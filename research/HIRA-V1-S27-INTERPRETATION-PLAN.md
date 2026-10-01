# HIRA V1 S27 interpretation plan — frozen before S27-A0/DEV exposure

Status: **FROZEN**

Issue: #237  
PR: #238

## Fixed hypothesis

S27 tests exactly one claim:

> With S26 inference/capacity/gradient ownership frozen, canonicalizing the 128D role block and 128D value block independently improves cross-view semantic transport more reliably than canonicalizing the concatenated 256D factorized signature as one vector.

No inference rule changes.
No new learned params.
No new fusion.
No optimizer change.

## Frozen architecture

Inherited exactly from S26:
- shared A13 LoRA **16,384**
- primary-private projection **32,768**
- relation-private projection **32,768**
- exact trainable surface **81,920**
- original A13 frozen
- HIRACore frozen
- S21 primary
- S26 factorized role/value relation inference
- S14 equal standardized full-K fusion
- S25 shared/private gradient ownership
- neutral-bisector only on shared LoRA
- relation detach from fused-primary objective

S27 changes only canonicalization loss:
- split signature into role/value 128D blocks
- independent block normalization
- same-option alignment + wrong-option separation on each block
- block weights **0.50 / 0.50**
- separation margin **0.20**
- outer canonicalization coefficient **0.15**

## Frozen fresh TRAIN/DEV

- seed **48001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S27 domains
- K **4**
- two state views
- two question views per semantic query
- two option views
- epochs **24**
- batch size **16**
- AdamW lr **2e-4**
- weight decay **0.01**
- grad clip **1.0**

Other loss shell remains inherited:
- swap coefficient **0.25**
- swap margin **0.20**
- option alignment **0.05**
- option temperature **0.10**
- fused cross-view JS **0.25**
- relation CE **0.10**
- blockwise canonicalization outer coefficient **0.15**

## Frozen DEV selector

Lexicographic:
1. fused paired-both-correct rate
2. fused canonical accuracy
3. relation canonical accuracy
4. relation canonical signed margin
5. fused canonical signed margin
6. question-swap change
7. fused selected-choice agreement
8. whole-signature same-option cosine
9. whole-signature discrimination margin
10. lower canonical decision loss
11. earlier epoch tie-break

Blockwise role/value diagnostics are **not** allowed to alter epoch selection after DEV exposure.

## Existing DEV_READY gates

All remain required:
- fused canonical >= **0.85**
- paired both-correct >= **0.75**
- question-swap >= **0.80**
- fused cross-view agreement >= **0.95**
- fused mean JS <= **0.05**
- fused canonical margin >= **0.15**
- relation canonical >= **0.80**
- relation canonical margin >= **0.15**
- whole-signature same-option cosine >= **0.90**
- whole-signature discrimination margin >= **0.15**
- option-order/mass/full-K/state-once/capacity/ownership gates PASS

S27 additionally records:
- role-block same-option cosine
- value-block same-option cosine
- role-block discrimination margin
- value-block discrimination margin

These diagnostics cannot substitute for failed existing gates.

## Preregistered interpretation

### Outcome A — DEV_READY

Blockwise factor canonicalization is a viable solution to S26's cross-view transport failure. Freeze the candidate before any confirmation/multilingual/Laya-Jev work.

### Outcome B — cross-view metrics improve materially, but DEV_FAIL

The S26 diagnosis was partly correct: block drift contributed to failure, but independent block alignment alone is insufficient. Close S27 without tuning and use residual metrics for a new hypothesis.

### Outcome C — one block stabilizes while the other remains weak

The remaining bottleneck is localized to that factor family. Close S27 and preregister a factor-specific structural hypothesis; do not reweight 0.50/0.50 after DEV.

### Outcome D — canonical performance improves but paraphrase/cross-view remains weak

Blockwise loss is still being satisfied on TRAIN without semantic transport. Move away from canonicalization-only interventions.

### Outcome E — no material improvement / regression

Reject blockwise canonicalization as sufficient. Do not tune margin, coefficient, block weights, or capacity.

## Stop rule

Exactly one fresh S27 TRAIN/DEV authority.

After DEV exposure:
- no block weight change
- no separation-margin change
- no canonicalization coefficient change
- no inference modification
- no seed/LR/epoch/batch retry
- no capacity change
- no fusion change
- no selector change
- no gate weakening
- no second DEV run

Scientific FAIL is valid.

No sealed confirmation, multilingual transfer, or Laya/Jev benchmark unless DEV_READY.
