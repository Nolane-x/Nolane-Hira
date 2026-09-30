# HIRA V1 S19 interpretation plan

Status: **FROZEN AFTER QUALIFIED A0, BEFORE TRAIN/DEV RESULT**

Qualified A0:
- run `36588963863`
- artifact `11042832648`
- digest `sha256:f54819e15bc642f3424272b05ba3dc66e2418b34314e536709d72171ce858276`
- outcome `HIRA_V1_S19_A0_IDENTITY_READY`

## Frozen hypothesis

S17's relation expert was substantially more view-stable than its triadic expert.

S19 returns to the S17 loss/optimizer and changes only the cross-view consistency target:
- no S18 paired margin
- raw-triadic symmetric JS replaces fused-output symmetric JS
- coefficient remains exactly 0.25
- relation block remains unchanged

## Outcome classes

### A. Triadic and fused view consistency improve while S17 discrimination is preserved
Evidence:
- raw-triadic cross-view agreement rises materially;
- fused paired/cross-view agreement improve;
- fused/relation accuracy and positive margins stay near or above S17;
- question-swap remains high.

Interpretation: branch-specific triadic instability was the dominant residual bottleneck.

### B. Triadic consistency improves but semantic discrimination degrades
Interpretation: forcing the triadic expert invariant removes useful wording-conditioned semantic signal. Close S19; no coefficient tuning.

### C. Triadic consistency does not improve
Interpretation: residual view instability lives upstream in representation/binding rather than in expert-output distribution alignment.

### D. Relation quality remains strong but fused/triadic paired quality stays weak
Interpretation: triadic semantic capacity/binding, not merely consistency, is the remaining bottleneck.

### E. TRAIN triadic JS improves but fresh DEV does not
Interpretation: triadic wording generalization remains the bottleneck. No template/seed retry.

### F. DEV_READY
Only the frozen gate may open sealed English confirmation and separately preregistered zero-training Vietnamese transfer.

## Diagnostic priority

1. paired both-correct
2. canonical/paraphrase fused accuracy
3. fused cross-view agreement
4. raw-triadic cross-view agreement
5. raw-triadic canonical/paraphrase accuracy
6. relation accuracy/margins
7. fused margins and question-swap
8. signature cosine/margin
9. TRAIN raw-triadic JS trajectory

## Forbidden after DEV exposure

Do not:
- tune JS coefficient or move its target again;
- re-add S18 paired margin;
- change norm balancing;
- tune fusion/epsilon/temperatures;
- change seed/LR/templates/gates;
- reuse S19 DEV in a next track;
- reopen Laya/Jev unless DEV_READY.
