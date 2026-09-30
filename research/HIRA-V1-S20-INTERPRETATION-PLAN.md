# HIRA V1 S20 interpretation plan

Status: **FROZEN AFTER QUALIFIED A0, BEFORE TRAIN/DEV RESULT**

Qualified A0:
- run `36677124443`
- artifact `11080820457`
- digest `sha256:39769042b9620b5444f5a478a1ef12d63acb43c7abbd5332b8978b5177be4b06`
- outcome `HIRA_V1_S20_A0_IDENTITY_READY`

## Frozen hypothesis

S19 failed because raw-softmax JS was nearly blind to relative top-option changes on extremely flat triadic logits.

S20 keeps the S17 optimizer/objective frontier and changes only the consistency geometry:
- no S18 paired margin;
- no S19 raw-softmax JS;
- exact S14 center/RMS standardization on raw triadic full-K evidence;
- cross-view standardized-evidence MSE coefficient **0.25**;
- epsilon **1e-6**;
- relation block unchanged;
- inference unchanged.

A0 demonstrates that the new consistency target is not inert:
- raw triadic RMS is ~1.22e-4;
- raw softmax JS is ~7.4e-9;
- standardized mismatch loss is ~0.435;
- gradients to both views are large and finite.

## Outcome classes

### A. Triadic and fused cross-view consistency improve while S17 discrimination is preserved
Evidence:
- raw-triadic agreement rises materially;
- paired/fused agreement improve;
- fused canonical/paraphrase and relation metrics remain near or above S17;
- question-swap remains high.

Interpretation: scale-free relative evidence was the missing consistency geometry.

### B. Standardized consistency trains strongly but semantic discrimination degrades
Interpretation: forcing relative triadic evidence to align removes useful query/view-specific ranking signal. Close S20; do not coefficient-tune.

### C. Standardized consistency falls on TRAIN but fresh DEV view consistency does not improve
Interpretation: wording generalization, not local consistency signal visibility, remains the bottleneck.

### D. Triadic agreement improves but fused/paired quality stays weak
Interpretation: relation/triadic expert coordination or fused top-1 geometry becomes the next bottleneck.

### E. Relation quality remains strong but triadic semantic accuracy remains weak
Interpretation: triadic capacity/binding, not view consistency, is still limiting.

### F. DEV_READY
Only the frozen gate may open:
1. one-shot sealed English confirmation;
2. separately preregistered zero-training Vietnamese transfer.

## Diagnostic priority

1. paired both-correct
2. fused canonical/paraphrase accuracy
3. fused cross-view agreement
4. raw-triadic cross-view agreement
5. raw-triadic canonical/paraphrase accuracy
6. relation canonical/paraphrase accuracy and margins
7. question-swap and fused signed margin
8. signature cosine/margin
9. TRAIN standardized-consistency trajectory
10. raw triadic RMS / flat-rate diagnostics

## Forbidden after DEV exposure

Do not:
- tune standardized-consistency coefficient;
- tune standardization epsilon;
- move the consistency target again;
- re-add S18 paired margin;
- re-add S19 raw-softmax JS;
- change norm balancing;
- tune fusion or temperatures;
- change seed/LR/templates/gates;
- reuse S20 DEV in a next track;
- reopen Laya/Jev unless DEV_READY.
