# HIRA V1 S22 interpretation plan — frozen before TRAIN/DEV

Status: **FROZEN BEFORE FRESH DEV EXPOSURE**

Issue: #227  
PR: #228

## Fixed hypothesis

S22 tests exactly one claim:

> S21's high conflict rate made S17's relation-priority projection suppress useful role/content-primary updates; protecting the primary direction instead can recover canonical/paired quality while retaining the relation expert.

No model architecture, inference, loss coefficient, capacity, seed policy, gate or selection rule is changed for this hypothesis.

## Reference frontier

S21 selected:
- fused canonical: **0.6145833333**
- fused paraphrase: **0.5651041667**
- paired: **0.3697916667**
- question-swap: **0.8333333333**
- fused agreement: **0.6588541667**
- fused JS: **0.0395950909**
- fused canonical margin: **+0.1440208815**
- primary canonical/paraphrase: **0.5807291667 / 0.5182291667**
- relation canonical/paraphrase: **0.6484375 / 0.5963541667**
- relation canonical margin: **+0.2356135895**
- relation paraphrase margin: **+0.1948801869**
- mean gradient conflict: **0.5998263889**

S17 remains the broader best fused-canonical frontier at **0.7213541667**.

## Frozen S22 configuration

- seed: **38001**
- TRAIN: 768 fresh cases
- DEV: 192 fresh cases
- 12 fresh domains
- epochs: **24**
- batch: **16**
- AdamW lr: **2e-4**
- weight decay: **0.01**
- grad clip: **1.0**
- exact trainable params: **49,152**
- role temperature: **0.10**
- role/content weights: **0.50 / 0.50**
- fused cross-view JS: **0.25**
- swap coefficient/margin: **0.25 / 0.20**
- option alignment coefficient: **0.05**
- relation CE: **0.10**
- relation-signature canonicalization: **0.15**
- signature separation margin: **0.20**
- S14 equal-weight fusion
- primary-priority normalized conflict projection
- epsilon: **1e-12**

## Preregistered outcomes

### Outcome A — optimizer priority was the bottleneck

Evidence:
- primary/fused canonical and paired metrics materially improve versus S21;
- relation canonical/margin remain strong;
- question-swap remains >=0.80;
- no mechanical regression.

Interpretation:
- S21 architecture is useful and relation-priority conflict surgery was suppressing it.

Only DEV_READY opens sealed confirmation.

### Outcome B — primary improves while relation deteriorates

Evidence:
- primary canonical/margin improve;
- relation accuracy/margin fall materially;
- fused gains are limited or unstable.

Interpretation:
- shared surface contains a real expert-priority tradeoff.
- Next research should target expert decoupling/isolation while holding capacity small.

Do not tune priority after DEV.

### Outcome C — relation remains strong but primary/fused do not improve

Evidence:
- relation metrics are S21-like;
- primary/fused canonical/paired remain flat or worse.

Interpretation:
- relation-priority projection was not the main cause of S21's canonical deficit.
- Return to S17/S21 evidence and investigate representation/fusion, not optimizer priority.

### Outcome D — global regression

Evidence:
- primary, relation and fused metrics all regress;
- conflict trajectory remains high or training destabilizes.

Interpretation:
- primary-priority optimization is harmful under the shared surface.
- Reject S22 optimizer rule.

## Scientific discipline

Use only the gates and selection frozen in `HIRA-V1-S22-CONTRACT.md`.

Scientific FAIL is valid.

No post-DEV:
- relation/primary priority switching
- alternating priority
- role/content weight tuning
- role-temperature tuning
- seed/LR/template retry
- gate weakening
- S22 DEV reuse

No Laya/Jev, sealed confirmation or multilingual transfer unless DEV_READY.
