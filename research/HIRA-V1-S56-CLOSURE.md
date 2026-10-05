# HIRA V1 S56 closure — Explicit Cross-View Decision Consistency

Status: **SCIENTIFICALLY CLOSED — CASE B**

Issue: #295  
PR: #296

## Verdict

Fresh S56 court:
- run `37265241846`
- artifact `11326048054`
- digest `sha256:7284f9e9d4bd2e29ddb7269726b12147f636c62b2df0a51b697a5c1fee181003`

Treatment vs reference:

Stability:
- fused agreement **+6.51 pp**
- relation agreement **+16.93 pp**
- fused JS **-0.013664**
- relation JS **-0.037892**

Correctness:
- fused canonical **-13.54 pp**
- fused paraphrase **-9.90 pp**
- paired both-correct **-8.85 pp**
- canonical relation accuracy **-16.15 pp**
- paraphrase relation accuracy **-10.68 pp**

Both arms remain DEV_READY false.

## Scientific conclusion

S56 is the first strong evidence that direct cross-view pressure can materially move Hira's stability in the desired direction.

However, whole-distribution consistency over-smooths the decision surface and destroys correctness.

The next step must preserve only the **discrete ordinal structure that should be invariant** while allowing view-specific probability calibration/magnitude.

Therefore S57 should test pairwise ranking/ordinal consistency rather than JS matching of full distributions.

## Stop-rule compliance

No:
- coefficient sweep
- ordering threshold/margin sweep
- entropy auxiliary
- architecture/capacity change
- native retraining
- selector change
- retry
- gate weakening
- second S56 DEV
- external Laya/Jev evaluation.

S56 is closed.
