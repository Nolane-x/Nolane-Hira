# HIRA V1 S24 interpretation plan — frozen before TRAIN/DEV

Status: **FROZEN BEFORE FRESH DEV EXPOSURE**

Issue: #231  
PR: #232

## Fixed hypothesis

S24 tests exactly one fusion claim:

> With the S23 experts, training objective, 49,152-parameter surface and neutral-bisector optimizer fixed, deterministic reliability-weighted standardized full-K fusion can avoid suppressing the stronger expert better than S14 fixed 0.5/0.5 fusion.

Reliability is frozen as standardized top1-top2 gap plus `1e-6`, followed by symmetric normalization across primary and relation experts.

No learned gate, router, calibrator, head, coefficient, temperature, additional state, or parameter is allowed.

## Why S24 exists

S23 selected DEV:
- primary canonical **0.5651041667**
- primary paraphrase **0.5625**
- relation canonical **0.6875**
- relation paraphrase **0.6276041667**
- fixed-fusion canonical **0.6432291667**
- fixed-fusion paraphrase **0.609375**
- fixed-fusion paired **0.4166666667**
- question-swap **0.75**
- fused agreement **0.6770833333**
- fused JS **0.0387389847**
- fused canonical margin **+0.0783118053**

Relation outperformed the fixed 50/50 fused result on canonical accuracy. S24 therefore isolates fusion as the next bottleneck instead of reopening optimizer-priority variants.

## Frozen S24 configuration

- seed **42001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S24 domains
- K **4**
- two state views
- two question views
- two option views
- 24 epochs
- batch size **16**
- AdamW lr **2e-4**
- weight decay **0.01**
- grad clip **1.0**
- physical trainable params **49,152**
- A13 LoRA **16,384**
- shared projection **32,768**
- S21 role-gated content primary
- role temperature **0.10**
- role/content weights **0.50 / 0.50**
- S13 relation expert/canonicalizer
- neutral-bisector norm-balanced optimizer
- balance epsilon **1e-12**
- fused cross-view JS **0.25**
- relation CE **0.10**
- signature canonicalization **0.15**
- signature separation margin **0.20**
- reliability epsilon **1e-6**
- reliability = standardized top1-top2 gap + epsilon
- fusion weights = normalized primary/relation reliabilities
- relation logits detached from fused-primary objective
- no post-DEV retry/tuning

## Frozen DEV selection

Use exactly the preregistered S24/S23 lexicographic selector, led by:
1. fused paired-both-correct rate;
2. fused canonical accuracy;
3. relation canonical accuracy;
4. relation canonical margin;
5. fused canonical margin;
6. question-swap change;
7. cross-view agreement;
8. same-option signature cosine;
9. signature margin;
10. lower canonical decision loss;
11. earlier epoch tie-break.

No alternate epoch selection may be introduced after DEV exposure.

## DEV_READY gates

All existing S24 gates remain authoritative:
- fused canonical accuracy >= **0.85**
- fused paired-both-correct >= **0.75**
- fused question-swap change >= **0.80**
- fused cross-view choice agreement >= **0.95**
- fused cross-view mean JS <= **0.05**
- fused canonical margin >= **0.15**
- relation canonical accuracy >= **0.80**
- relation canonical margin >= **0.15**
- same-option signature cosine >= **0.90**
- signature margin >= **0.15**
- fused option-order flip <= **0.02**
- fused probability-mass error <= **1e-6**
- full-K, relation isolation, state-once and exact parameter-surface gates all pass.

## Preregistered interpretations

### Outcome A — DEV_READY

Interpretation:
- reliability-weighted fusion is a genuine solution to the fixed-fusion bottleneck under the frozen S23 substrate;
- freeze the candidate and only then open sealed confirmation/multilingual/Laya-Jev stages according to project protocol.

### Outcome B — clear fused improvement but not DEV_READY

Interpretation:
- fixed 50/50 fusion was a real bottleneck, but reliability weighting alone is insufficient;
- close S24 as a useful ablation;
- next research may investigate expert calibration/representation only in a new preregistered track, never by tuning S24 on exposed DEV.

### Outcome C — relation remains stronger than fused

Interpretation:
- the standardized top1-top2 reliability proxy does not reliably identify which expert should dominate;
- do not tune epsilon or weights after exposure;
- close this deterministic reliability rule and move to a separately preregistered fusion hypothesis.

### Outcome D — global regression/no material benefit

Interpretation:
- reject the S24 reliability-weighted fusion hypothesis;
- preserve S23/S17 evidence and move away from this fusion rule.

## Stop rule

One fresh S24 TRAIN/DEV authority is allowed.

After DEV exposure:
- no formula change;
- no epsilon change;
- no learned/fitted calibration;
- no optimizer change;
- no seed/LR/template retry;
- no role weight/temperature change;
- no gate weakening;
- no alternate selector;
- no hidden extra DEV run.

Scientific FAIL is a valid closure result.

No sealed confirmation, multilingual transfer, or Laya/Jev reopening unless S24 is DEV_READY.
