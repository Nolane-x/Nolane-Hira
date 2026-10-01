# HIRA V1 S25 interpretation plan — frozen before fresh DEV exposure

Status: **FROZEN**

Issue: #233  
PR: #234

## Fixed hypothesis

S25 tests one claim:

> With the shared A13/LoRA encoder fixed, separating the primary and relation 256→128 projection surfaces improves semantic generalization by removing destructive private-representation coupling.

Only projection ownership changes relative to the S23-style objective family.

## Frozen architecture

- shared A13 LoRA: **16,384**
- primary-private projection: **32,768**
- relation-private projection: **32,768**
- total physical trainable surface: **81,920**
- original A13 frozen
- HIRACore frozen
- no learned downstream router/gate/calibrator/head
- S21 role-gated-content primary unchanged
- S13 relation canonicalizer unchanged
- frozen S14 equal standardized full-K fusion **0.5 / 0.5**
- relation logits detached from fused-primary objective

Gradient ownership:
- primary/private only from primary block
- relation/private only from relation block
- shared LoRA receives both blocks
- neutral-bisector applies only to shared LoRA
- epsilon **1e-12**

## Frozen training shell

- one fresh English TRAIN/DEV authority
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S25 domains
- K **4**
- two state views
- two question views per semantic query
- two option views
- epochs **24**
- batch size **16**
- AdamW lr **2e-4**
- weight decay **0.01**
- grad clip **1.0**
- seed **46001**

Losses remain S23-family:
- swap coefficient **0.25**
- swap margin **0.20**
- option alignment **0.05**
- option temperature **0.10**
- fused cross-view JS **0.25**
- relation CE **0.10**
- signature canonicalization **0.15**
- signature separation margin **0.20**
- role temperature **0.10**
- role/content weights **0.50 / 0.50**

## Frozen DEV selection

Lexicographic selector, in order:
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

No alternate epoch may be selected after DEV exposure.

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
- fused option-order flip rate <= **0.02**
- fused probability-mass error <= **1e-6**
- full-K PASS
- state-once PASS
- exact shared/private gradient ownership PASS
- exact **81,920** physical surface PASS
- original A13/HIRACore frozen

## Interpretation

### DEV_READY
Decoupled private projections are a viable solution to the representation-coupling bottleneck. Freeze the selected checkpoint before any confirmatory/multilingual/Laya-Jev work.

### Semantic improvement but DEV_FAIL
Private projection interference was a real contributor but not sufficient. Close S25 without post-DEV tuning and use the residual failure pattern to preregister a new representation hypothesis.

### No material improvement / regression
Reject decoupled private projections as the primary missing ingredient. Do not grow projection capacity or alter private/shared optimizers on exposed DEV.

## Stop rule

Exactly one fresh S25 TRAIN/DEV authority.

After DEV exposure:
- no seed retry;
- no LR/epoch/batch retry;
- no capacity change;
- no projection-sharing interpolation;
- no fusion change;
- no loss coefficient change;
- no alternate selector;
- no gate weakening;
- no second DEV run.

Scientific FAIL is valid.

No sealed confirmation, multilingual transfer, or Laya/Jev benchmark until DEV_READY.
