# HIRA V1 S57 closure — Discrete Pairwise Ranking Consistency

Status: **SCIENTIFICALLY CLOSED — CASE B**

Issue: #297  
PR: #298

## Verdict

Fresh S57 court:
- run `37271509208`
- artifact `11327849211`
- digest `sha256:868faf45b1994286e52cbac4c35adfbbb420e2ba8a6c561545067da013c05147`

Treatment vs reference:

Positive:
- fused agreement **+8.07 pp**
- fused JS **-0.001187**

Negative:
- fused canonical **-9.64 pp**
- fused paraphrase **-5.21 pp**
- paired both-correct **-5.21 pp**
- question-swap **-17.71 pp**
- relation canonical **-7.03 pp**
- relation paraphrase **-4.17 pp**
- relation agreement **-7.55 pp**

Both arms remain DEV_READY false.

## Scientific conclusion

S57 confirms the decision-level direction from S56: selective ordinal pressure can materially move fused selected-choice stability without matching the whole probability distribution.

However, self-anchored pairwise targets are unsafe. Even with detached anchors and gold-order filtering, the treatment can preserve or reinforce view-specific/wrong orderings, degrading correctness and relation consistency.

Therefore the next family should use **consensus/teacher anchoring**:
- derive pairwise targets only from orderings supported across both views and/or a frozen reference teacher;
- leave disputed pairs unconstrained;
- keep architecture/capacity unchanged.

## Stop-rule compliance

No:
- coefficient/threshold/margin sweep
- anchor-filter change
- teacher addition after seeing S57 within this stage
- architecture/capacity change
- native retraining
- selector change
- retry
- gate weakening
- second S57 DEV
- external Laya/Jev evaluation.

S57 is closed.
