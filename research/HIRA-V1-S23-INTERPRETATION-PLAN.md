# HIRA V1 S23 interpretation plan — frozen before TRAIN/DEV

Status: **FROZEN BEFORE FRESH DEV EXPOSURE**

Issue: #229  
PR: #230

## Fixed hypothesis

S23 tests exactly one optimizer claim:

> With S21 role/content architecture fixed, symmetric norm equalization followed by the neutral bisector of primary/relation gradient directions generalizes better than either relation-priority or primary-priority conflict projection.

No model/inference/capacity change is allowed.

## References

S17 frontier:
- fused canonical **0.7213541667**
- paraphrase **0.5546875**
- paired **0.5104166667**
- question-swap **0.984375**
- fused margin **+0.3138313380**
- relation canonical **0.640625**
- relation margin **+0.2073315941**

S21 relation-priority + factorized primary:
- fused canonical **0.6145833333**
- paraphrase **0.5651041667**
- paired **0.3697916667**
- question-swap **0.8333333333**
- fused agreement **0.6588541667**
- relation canonical **0.6484375**
- relation margin **+0.2356135895**

S22 primary-priority:
- fused canonical **0.5729166667**
- paraphrase **0.546875**
- paired **0.28125**
- question-swap **0.5729166667**
- fused agreement **0.6796875**
- relation canonical **0.6015625**
- relation margin **+0.0338486681**

## Frozen S23 configuration

- seed **40001**
- 24 epochs
- batch size **16**
- AdamW lr **2e-4**
- weight decay **0.01**
- grad clip **1.0**
- exact trainable params **49,152**
- role temperature **0.10**
- role/content weights **0.50 / 0.50**
- fused JS coefficient **0.25**
- swap coefficient/margin **0.25 / 0.20**
- option alignment **0.05**, temperature **0.10**
- relation CE **0.10**
- signature canonicalization **0.15**
- signature separation margin **0.20**
- S14 equal-weight standardized fusion
- neutral bisector optimizer
- no asymmetric projection
- epsilon **1e-12**

No S18/S19/S20 intervention.

## Preregistered interpretations

### Outcome A — neutral bisector succeeds

Evidence:
- material improvement over S21/S22 and toward S17;
- canonical/paired recover without collapsing relation margin;
- or DEV_READY.

Interpretation:
- asymmetric conflict surgery was part of the optimizer bottleneck.

### Outcome B — preserves S21 relation quality but not canonical

Interpretation:
- removing projection protects relation geometry but does not solve factorized-primary canonical deficit;
- optimizer family is exhausted.

### Outcome C — improves primary but hurts relation/fusion

Interpretation:
- shared surface remains fundamentally competitive;
- neutral symmetry alone cannot reconcile the experts.

### Outcome D — global regression or no meaningful gain

Interpretation:
- optimizer-priority hypothesis family is rejected.

## Stop rule

If S23 does not materially improve the S17/S21 frontier:
- close optimizer-priority family;
- do not create another S24 priority/projection variant;
- return to representation/fusion research.

No post-DEV optimizer, seed, LR, role-weight, temperature or template tuning.

No sealed confirmation, multilingual transfer or Laya/Jev unless DEV_READY.
