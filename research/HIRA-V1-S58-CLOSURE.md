# HIRA V1 S58 closure — Consensus-Teacher Pairwise Ranking

Status: **SCIENTIFICALLY CLOSED — CASE B**

Issue: #299  
PR: #300

## Verdict

Fresh S58 court:
- run `37276076841`
- artifact `11330721014`
- digest `sha256:cea8203e98899feda6c1ca1558b9941a8900b987124dfb43958c90e5d0285e50`

Treatment vs reference:

Positive:
- fused agreement **+3.65 pp**
- fused JS **-0.010048**
- relation JS **-0.075205**

Negative:
- fused canonical **-6.25 pp**
- paired both-correct **-6.77 pp**
- question-swap **-14.06 pp**
- relation canonical **-11.46 pp**
- relation agreement **-1.30 pp**

Both arms remain DEV_READY false.

## Scientific conclusion

S58 validates that frozen cross-view teacher consensus is mechanically safer than self-generated anchoring, but it still does not solve Hira's joint correctness/stability target.

The treatment follows teacher-consensus targets more strongly and reduces score-distribution disagreement, yet loses substantial canonical correctness and discrimination. Therefore the bottleneck is no longer well described as only an anchor-selection problem.

A fixed teacher decision boundary is too restrictive. The next family must learn pairwise decisions directly from fresh supervised data, while preserving:
- full-K behavior;
- one encoder/state-once;
- persisted S51 native authority;
- immutable cache;
- explicit option-permutation equivariance;
- anti-symmetric pairwise mechanics;
- strict parameter/runtime accounting.

## Stop-rule compliance

No:
- teacher swap
- threshold/margin/coefficient sweep
- consensus-mask variant
- architecture/capacity change after DEV
- native retraining
- selector change
- retry
- gate weakening
- second S58 DEV
- external Laya/Jev evaluation.

S58 is closed.
