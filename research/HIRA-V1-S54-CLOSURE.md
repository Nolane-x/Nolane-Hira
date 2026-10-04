# HIRA V1 S54 closure — Joint State–Query–Option Late Interaction

Status: **SCIENTIFICALLY CLOSED — CASE C**

Issue: #291  
PR: #292

## Verdict

S54 produced a valid matched court and is classified **Case C**.

Treatment minus reference:
- fused canonical **+1.04 pp**
- fused paraphrase **+3.91 pp**
- paired both-correct **+1.04 pp**
- fused agreement **-0.52 pp**
- fused JS **+0.000668** worse
- relation agreement **0.00 pp**
- relation JS **+0.001828** worse.

Treatment context itself is highly stable:
- cross-view same-option context cosine **0.981316**
- mean state support **0.958278**
- mean option support **0.956517**.

## Scientific conclusion

Adding direct state support to S53's query↔option late interaction improves useful correctness, especially paraphrase correctness, but still does not improve selected-choice stability.

This means:
- query-only representation was not enough (S51–S53);
- parameter-free triadic factorization is also not enough (S54);
- the remaining bottleneck likely requires a **learned joint relation transform** rather than fixed cosine/max support.

The next family should add a small matched learned triadic interaction with identical capacity in reference and treatment, changing only whether state evidence participates in the learned joint relation operator.

## Stop-rule compliance

No:
- temperature sweep
- aggregation sweep
- learned projection added after DEV
- native retraining
- identity/correction capacity change
- alternate selector
- retry
- gate weakening
- second S54 DEV
- external Laya/Jev evaluation.

S54 is closed.
