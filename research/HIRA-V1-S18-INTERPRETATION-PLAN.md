# HIRA V1 S18 interpretation plan

Status: **FROZEN AFTER QUALIFIED A0, BEFORE TRAIN/DEV RESULT**

Qualified A0:
- run `36582445491`
- artifact `11040940678`
- digest `sha256:cb78b283fbcf1514e88aa8ad08f35c2bcad0a47a90302a8fd97e47640a3af452`
- outcome `HIRA_V1_S18_A0_IDENTITY_READY`

## Frozen hypothesis

S17 solved much of semantic discrimination but equivalent wording views remain jointly unreliable.

S18 keeps S17 inference and norm-balanced optimization unchanged and adds only the paired both-view margin term:

`0.25 * 0.5 * [relu(0.20-m_c) + relu(0.20-m_p)]`

The mechanism is intended to make both wording views independently carry a positive gold-vs-hardest-wrong margin.

## Outcome classes

### A. Paired correctness improves while S17 discrimination is preserved
Evidence:
- paired both-correct rises materially;
- paraphrase accuracy closes the canonical gap;
- cross-view agreement improves;
- fused/relation margins stay positive;
- question-swap remains high.

Interpretation: paired worst-view weakness was a real remaining bottleneck.

### B. Paired consistency improves but discrimination collapses
Interpretation: paired hinge is over-constraining the shared semantic surface under fixed coefficient 0.25. Close S18; do not coefficient-tune exposed DEV.

### C. Discrimination remains strong but paired consistency does not improve
Interpretation: logit-level paired margin is insufficient; remaining mismatch is representation/view alignment rather than output margin.

### D. TRAIN paired loss improves but fresh DEV does not
Interpretation: paired wording generalization remains the bottleneck. No template/seed retry.

### E. Relation quality improves but fused paired quality does not
Interpretation: fusion/triadic view instability dominates despite relation preservation.

### F. DEV_READY
Only the frozen gate may open:
1. one-shot sealed English confirmation
2. separately preregistered zero-training Vietnamese transfer

## Diagnostic priority

1. paired both-correct
2. canonical and paraphrase fused accuracy
3. fused cross-view selected-choice agreement
4. fused/relation signed margins
5. question-swap
6. relation accuracy
7. signature cosine/margin
8. paired TRAIN loss trajectory

## Forbidden after DEV exposure

Do not:
- tune paired coefficient or margin;
- change norm-balancing rule;
- retune fusion weights/epsilon;
- change seed/LR/templates/gates;
- reuse S18 DEV in the next track;
- reopen Laya/Jev unless DEV_READY is achieved.
