# HIRA V1 S52 closure — Paired-View Query Relation Canonicalization

Status: **SCIENTIFICALLY CLOSED — CASE D**

Issue: #287  
PR: #288

## Verdict

S52 produced a valid matched court and is classified **Case D**.

Treatment minus reference:
- fused canonical accuracy **-3.39 pp**
- fused paraphrase accuracy **-3.39 pp**
- paired both-correct **-6.25 pp**
- fused selected-choice agreement **-5.73 pp**
- fused JS **+0.013655** worse
- relation agreement **+3.65 pp**
- relation JS **+0.027290** worse.

The treatment successfully makes paired paraphrase query codes more similar:
- same-relation cosine **+0.1540**

but it also makes different A/B relation centroids more similar:
- cross-relation centroid cosine **+0.1368**

so the relation separation margin only improves **+0.0172** and remains negative.

## Scientific conclusion

The remaining instability is not solved by compressing each whole query into one canonical 256D relation vector and aligning paraphrases in that vector space.

This family over-couples relation representations: paraphrases align, but distinct relation semantics also move together.

S52 therefore rejects **global pooled query-relation canonicalization**.

The next family should preserve token-level/local relation evidence instead of forcing a single global query vector.

## Stop-rule compliance

No:
- auxiliary coefficient sweep
- separation ceiling sweep
- hidden-dimension sweep
- raw-query bypass
- native retraining
- identity variant
- correction/canonicalizer capacity change after DEV
- alternate selector
- retry
- gate weakening
- second S52 DEV
- external Laya/Jev evaluation.

S52 is closed.
