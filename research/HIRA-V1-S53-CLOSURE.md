# HIRA V1 S53 closure — Token-Level Query↔Option Late Interaction

Status: **SCIENTIFICALLY CLOSED — CASE C**

Issue: #289  
PR: #290

## Verdict

S53 produced a valid matched court and is classified **Case C**.

Treatment minus reference:
- fused canonical accuracy **0.00 pp**
- fused paraphrase accuracy **+0.52 pp**
- paired both-correct **0.00 pp**
- fused selected-choice agreement **-5.47 pp**
- fused JS **+0.006217** worse
- relation agreement **-10.68 pp**
- relation JS **-0.008454** better.

The treatment token context itself is highly stable:
- same-option cross-view context cosine **0.97809**

but that does not translate into stable fused/relation choices.

## Scientific conclusion

Changing from one pooled query vector to option-conditioned token-level query evidence is **not sufficient**.

S51 rejected query-free identity as the sole fix.
S52 rejected global learned query canonicalization.
S53 now shows that even stable token-local query context does not solve the downstream decision instability.

The remaining bottleneck is likely in the **factorization itself**: state→option identity and query→option context are computed separately and only fused late.

The next family should test a direct **joint state–query–option interaction** where relation evidence is formed conditionally from all three sources together.

## Stop-rule compliance

No:
- token temperature/aggregation sweep
- token projection
- pooled-query bypass
- native retraining
- identity change
- correction capacity change
- alternate selector
- retry
- gate weakening
- second S53 DEV
- external Laya/Jev evaluation.

S53 is closed.
