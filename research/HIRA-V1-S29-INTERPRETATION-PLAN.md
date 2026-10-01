# HIRA V1 S29 interpretation plan — frozen before S29-A0/DEV exposure

Status: **FROZEN**

Issue: #241
PR: #242

## Controlled question

S29 tests one architectural variable only:

> Does extending zero-init rank-8 LoRA from the final A13 attention sublayer to the complete final Transformer block, including both FFN dense layers, improve fresh semantic discrimination while preserving the strongest S17 inference/training shell?

No new downstream scorer, fusion, loss family, router, calibrator or task head is introduced.

## Frozen S17 shell

Keep exactly:
- S13 cross-view relation expert/canonicalizer
- S14 equal standardized full-K fusion, epsilon **1e-6**
- S15 primary-loss relation-logit detach
- S17 relation-priority norm-balanced shared-gradient rule
- S17 loss partition and coefficients
- one shared bias-free 256→128 projection
- state-once/full-K
- opaque option IDs
- original A13 frozen
- HIRACore frozen

## S29 trainable surface

Final A13 block only, rank **8**, alpha **8**, dropout **0**:
- attention Q/K/V/output LoRA: **16,384**
- FFN intermediate 256→1024 LoRA: **10,240**
- FFN output 1024→256 LoRA: **10,240**
- total A13 LoRA: **36,864**
- shared relation projection: **32,768**
- exact total: **69,632**

No earlier layer, bias, LayerNorm or embedding is trainable.

## Required A0 interpretation

A0 is diagnostic only.

It must establish:
- exact S17 initialization identity for token/pooled outputs
- exact primary/relation/fused logit identity
- exact selected-choice identity
- exact six-module LoRA layout and capacity
- checkpoint roundtrip
- nonzero gradient on at least one attention-LoRA B
- nonzero FFN-intermediate B gradient
- nonzero FFN-output B gradient
- nonzero shared projection gradient
- S17 primary/relation norm-balanced partition finite
- full-K/state-once/option-order/mass preserved

A0 semantic scores may not tune S29.

## Frozen fresh TRAIN/DEV

After qualified A0 only:
- seed **50001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S29 domains
- K **4**
- two state views
- two question views per semantic query
- two option semantic views
- epochs **24**
- batch size **16**
- AdamW lr **2e-4**
- weight decay **0.01**
- grad clip **1.0**

All S17 loss coefficients remain unchanged.

## Frozen selector

Use the S17 lexicographic selector unchanged:
1. fused paired-both-correct
2. fused canonical accuracy
3. relation canonical accuracy
4. relation canonical signed margin
5. fused canonical signed margin
6. question-swap change
7. fused cross-view selected-choice agreement
8. same-option signature cosine
9. signature discrimination margin
10. lower canonical decision loss
11. earlier epoch

## DEV_READY gates

Keep frontier gates unchanged:
- fused canonical >= **0.85**
- paired >= **0.75**
- question-swap >= **0.80**
- fused agreement >= **0.95**
- fused JS <= **0.05**
- fused canonical margin >= **0.15**
- relation canonical >= **0.80**
- relation canonical margin >= **0.15**
- same-option signature cosine >= **0.90**
- signature discrimination margin >= **0.15**
- option-order / mass / full-K / state-once / capacity / freeze gates PASS

## Preregistered interpretation

### A — DEV_READY
Final-block nonlinear adaptation is sufficient to advance. Freeze immediately before sealed confirmation or external matched evaluation.

### B — clear semantic jump but still DEV_FAIL
FFN adaptation unlocks useful representation capacity but one residual bottleneck remains. Close S29 without rank/layer tuning and diagnose from fresh residuals.

### C — primary improves, relation remains weak
Encoder semantic representation helps direct decision geometry more than relation binding. Next track must address relation construction, not widen LoRA further.

### D — relation improves, primary remains weak
FFN adaptation helps relational geometry but direct primary scoring remains limiting. Next track must address primary semantic readout without reopening exposed S29 settings.

### E — little/no gain
Reject final-block FFN LoRA as sufficient. Do not expand rank, more layers or alpha inside S29.

## Stop rule

Exactly one fresh S29 TRAIN/DEV authority after A0 qualification.

After DEV exposure:
- no rank/alpha/dropout change
- no additional layer adaptation
- no loss change
- no seed/LR/epoch/batch retry
- no fusion/scorer/optimizer change
- no selector/gate weakening
- no second DEV run

Scientific FAIL is valid.

No sealed confirmation, multilingual transfer, or Laya/Jev benchmark unless DEV_READY.
