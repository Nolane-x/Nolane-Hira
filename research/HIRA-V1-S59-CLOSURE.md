# HIRA V1 S59 closure — Explicit Learned Pairwise Decision Head

Status: **SCIENTIFICALLY CLOSED — CASE C**

Issue: #302  
PR: #304

## Verdict

Fresh S59 court:
- run `37289522292`
- artifact `11335428630`
- digest `sha256:72a348c27f95c1f7ea2dc0dbcba8db7afac56fa5cd4d284d0593f7321d603e4c`

Treatment explicit pairwise decision vs reference fused decision:

Positive:
- selected-choice agreement **+32.55 pp**
- cross-view JS **-0.128684**
- paraphrase accuracy **+2.60 pp**
- paraphrase margin **+0.379362**

Negative:
- canonical accuracy **-15.36 pp**
- paired both-correct **-21.87 pp**
- question-swap **-66.15 pp**
- canonical margin **-0.084871**

Both arms remain DEV_READY false.

## Scientific conclusion

S59 is the strongest localization so far.

The explicit gold-supervised pairwise head is not collapsing mechanically:
- anti-symmetry exact
- no teacher
- no pseudo-target
- no self-anchor
- no raw logit bypass
- no pairwise gradient into correction/native
- stable cross-view choice improves dramatically.

Therefore the remaining failure is **composition**.

A pairwise-only final decision throws away useful absolute evidence from the existing fused shell. The next family must combine:
1. the existing fused decision as the correctness/discrimination anchor; and
2. the learned pairwise score as a calibrated residual rather than a replacement.

S59 is closed. No post-DEV tuning is authorized.
