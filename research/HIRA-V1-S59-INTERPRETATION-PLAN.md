# HIRA V1 S59 interpretation plan

Status: **FROZEN BEFORE S59-A0**

Issue: #301

S59 is not allowed to claim success from lower JS alone.

Primary comparison is treatment minus equal-capacity reference at each arm's preregistered selected checkpoint.

## Primary correctness/discrimination

Track:
- fused canonical accuracy
- fused paraphrase accuracy
- paired both-correct
- question-swap choice-change
- canonical/paraphrase relation accuracy
- canonical/paraphrase gold margins.

## Primary stability

Track:
- fused selected-choice agreement
- fused cross-view JS
- relation selected-choice agreement
- relation cross-view JS.

## Interpretation

A requires a material stability improvement **without a material correctness/discrimination collapse**.

B is any clear stability gain accompanied by a meaningful correctness/discrimination regression analogous to S56-S58.

C is retained/improved correctness without material stability recovery.

D is joint regression.

E is reserved for the existing full DEV_READY gate set and requires a separate confirmation stage before any external Laya/Jev evaluation.

No post-DEV threshold selection or reinterpretation is permitted.
