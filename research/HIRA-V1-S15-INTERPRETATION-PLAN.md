# HIRA V1 S15 — interpretation plan

Status: **FROZEN AFTER QUALIFIED A0, BEFORE TRAIN/DEV RESULT**

Qualified A0:
- run `36556776351`
- artifact `11027644171`
- digest `sha256:2f9025fc14c5bc6ca8c816834d7c5f0cf19c858cff61be93a8edc4c260145e5a`
- outcome `HIRA_V1_S15_A0_IDENTITY_READY`

## Frozen hypothesis

S14 achieved real zero-parameter fusion synergy but relation semantics remained unstable. S15 changes only the direct fused-primary gradient route, detaching relation logits from fused CE / swap / cross-view JS while retaining fully differentiable relation CE and relation-signature canonicalization.

Inference is unchanged from S14.

## Outcome classes

### A. Fusion gain preserved and relation semantics improve
Evidence:
- fused accuracy/paired correctness stay at or above S14-like levels;
- relation accuracy/margin and signature cosine/margin improve materially;
- cross-view decision agreement improves.

Interpretation: direct fused-primary gradient interference was a real bottleneck.

### B. Relation semantics improve but fusion/primary degrades
Interpretation: isolation protects relation geometry but triadic-side adaptation is no longer sufficient to exploit it.

Next track may test parameter-free inference-only use of a frozen relation expert, not extra capacity.

### C. Fusion remains strong but relation semantics do not improve
Interpretation: interference is dominated by the shared physical LoRA/projection surface, not the direct relation-logit gradient path.

### D. Both fusion and relation semantics degrade
Interpretation: S14 benefit depended on joint coupled gradients; detach is not the right control.

### E. TRAIN improves but fresh DEV does not
Interpretation: lexical/template generalization remains the primary bottleneck.

### F. DEV_READY
Only the frozen gate may open sealed English confirmation and separately preregistered zero-training Vietnamese transfer.

## Diagnostic priority

1. fused paired both-correct
2. fused canonical accuracy
3. canonical relation accuracy and signed margin
4. fused signed margin
5. question-swap sensitivity
6. fused cross-view agreement
7. signature cosine and signature margin
8. raw triadic diagnostics

## Forbidden after DEV exposure

Do not:
- change detach route;
- tune fusion weights or epsilon;
- retune relation/canonicalization coefficients;
- change temperature, seed, LR, templates or gates;
- reuse S15 DEV rows in the next track;
- reopen Laya/Jev unless DEV_READY is achieved.
