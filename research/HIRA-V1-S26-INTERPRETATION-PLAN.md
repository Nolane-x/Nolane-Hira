# HIRA V1 S26 interpretation plan — frozen before fresh DEV exposure

Status: **FROZEN**

Issue: #235  
PR: #236

## Fixed hypothesis

S26 tests exactly one claim:

> Keeping S25 capacity/ownership fixed, an explicit factorized role-relation + value/content-relation signature improves fresh semantic discrimination more reliably than S13's additive relation signature.

S26 does not test more capacity, a new fusion rule, a learned gate, a different primary scorer, or a new optimizer.

## Frozen architecture

Inherited exactly:
- shared A13 LoRA **16,384**
- primary-private projection **32,768**
- relation-private projection **32,768**
- total trainable surface **81,920**
- original A13 frozen
- HIRACore frozen
- S21 primary unchanged
- S14 equal standardized full-K fusion **0.5 / 0.5**
- relation logits detached from fused-primary objective
- S25 gradient ownership
- neutral-bisector only on shared LoRA, epsilon **1e-12**

S26 relation representation:
- zero learned params
- query-conditioned role anchors
- option-independent state value extraction
- option-local value extraction
- role/value score weights **0.50 / 0.50**
- factorized signature = normalized concatenation of role delta and value delta
- real signature width **256**
- role temperature **0.10**
- contrastive temperature **0.10**

No post-A0 formula/weight/temperature change is allowed.

## Frozen TRAIN/DEV

- one fresh English authority
- seed **47001**
- TRAIN **768**
- DEV **192**
- **12 wholly fresh S26 domains**
- K **4**
- two state views
- two question views per semantic query
- two option views
- epochs **24**
- batch size **16**
- AdamW lr **2e-4**
- weight decay **0.01**
- grad clip **1.0**

Loss shell remains the S25/S23 family:
- swap coefficient **0.25**
- swap margin **0.20**
- option alignment **0.05**
- option temperature **0.10**
- fused cross-view JS **0.25**
- relation CE **0.10**
- signature canonicalization **0.15**
- signature separation margin **0.20**

## Frozen DEV selector

Lexicographic order:
1. fused paired-both-correct rate;
2. fused canonical accuracy;
3. relation canonical accuracy;
4. relation canonical signed margin;
5. fused canonical signed margin;
6. question-swap choice-change;
7. fused cross-view selected-choice agreement;
8. same-option signature cosine;
9. signature discrimination margin;
10. lower canonical decision loss;
11. earlier epoch tie-break.

No alternate epoch selection after DEV exposure.

## DEV_READY gates

All must pass:
- fused canonical accuracy >= **0.85**
- fused paired-both-correct >= **0.75**
- fused question-swap choice-change >= **0.80**
- fused cross-view selected-choice agreement >= **0.95**
- fused cross-view mean JS <= **0.05**
- fused canonical signed margin >= **0.15**
- relation canonical accuracy >= **0.80**
- relation canonical signed margin >= **0.15**
- same-option signature cosine >= **0.90**
- signature discrimination margin >= **0.15**
- fused option-order flip <= **0.02**
- fused probability-mass error <= **1e-6**
- full-K PASS
- state-once PASS
- exact cross-private leakage **0 / 0**
- exact **81,920** trainable surface
- S26 relation operator added params **0**
- original A13/HIRACore frozen

## Interpretation

### DEV_READY

Factorized role/value relation geometry is a viable solution to the residual binding/discrimination bottleneck. Freeze the selected checkpoint before any confirmation, multilingual, or Laya/Jev work.

### Relation/signature improvement but DEV_FAIL

Factorization addresses part of the relation collapse, but the fixed S21/S14 decision stack still cannot generalize reliably. Close S26 without tuning and use the residual failure pattern to preregister a new hypothesis.

### Alignment improves but discrimination remains weak

Cross-view canonicalization still dominates semantic separation. Reject factorization as sufficient and do not tune role/value weights from exposed DEV.

### No material improvement / regression

Reject the S26 representation rule and move away from this family rather than increasing width or coefficient strength.

## Stop rule

Exactly one fresh S26 TRAIN/DEV authority.

After DEV exposure:
- no seed/LR/epoch/batch retry;
- no role/value weight change;
- no temperature change;
- no value-extraction change;
- no signature composition change;
- no capacity change;
- no loss coefficient change;
- no fusion change;
- no selector change;
- no gate weakening;
- no second DEV run.

Scientific FAIL is valid.

No sealed confirmation, multilingual transfer, or Laya/Jev benchmark unless DEV_READY.
