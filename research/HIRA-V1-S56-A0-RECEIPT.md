# HIRA V1 S56 A0 receipt — Explicit Cross-View Decision Consistency

Status: **QUALIFIED**

Run: `37215444292`  
Artifact: `11307997189`  
Digest: `sha256:48ee087d9a5fdd76790fbbdb625ebbd381ad44581a0df04bb047772baeff49cd`  
Authorization head: `abcf19a712c296ed5bd371f8ba5c5a119ac53e34`

Outcome:
`HIRA_V1_S56_A0_CROSS_VIEW_DECISION_CONSISTENCY_READY`

## Matched surface

- reference private params **114,688**
- treatment private params **114,688**
- added trainable params **0**
- identity params **0**
- initialization bit-identical **true**
- native trainable params **0**
- second encoder pass **false**

## Decision-consistency mechanics

- identical decision JS **0**
- controlled disagreement JS **0.3275965**
- shared-offset invariance error **0**
- positive-scale invariance error **0**

## Ordering mechanics

- matching confident ordering loss **0**
- controlled sign-flip ordering loss **1.682993**
- sign-flip disagreement fraction **1.0**
- flat/inactive ordering loss **0**
- flat active-pair fraction **0**

## Gradient ownership

- reference auxiliary exact zero **true**
- treatment auxiliary value **0.00819233**
- treatment auxiliary gradient L1 **3.44179**
- real-cache active pair fraction **0.942708**
- cache inference tensor count **0**
- cache requires-grad tensor count **0**

## Anti-collapse

Uniform logits:
- top1-top2 probability gap **0**
- logit RMS **0**

Therefore a trivial uniform solution is mechanically visible and cannot be interpreted as successful stability.

## Full-K

- K=3 PASS
- K=7 PASS
- K=255 PASS
- max probability-mass error **1.19e-7**

A0 was not used for model selection. Fresh S56 TRAIN/DEV remains separately gated.
