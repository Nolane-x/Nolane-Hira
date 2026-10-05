# HIRA V1 S64 interpretation plan

Status: **FROZEN BEFORE A0**

Use treatment minus reference deltas.

## Materiality

Stability success requires BOTH:
- selected-choice agreement >= +0.010000
- cross-view mean JS <= -0.002000.

Correctness retained requires ALL:
- canonical accuracy >= -0.005000
- paraphrase accuracy >= -0.005000
- paired both-correct >= -0.005000
- question-swap discrimination >= -0.005000.

## Cases

**A** — stability success and correctness retained.  
Context information solves the S63 representation bottleneck. Freeze and move to a separate fresh confirmation unless E applies.

**B** — stability success false while correctness is not materially damaged.  
The fixed 4D context projection is too lossy or context alone is insufficient. A next family may preregister a learned low-rank contextual compressor or joint surface+context representation. No S64 projection sweep.

**C** — stability success true but correctness retained false.  
Context identifies useful stabilization, but inference needs a separately preregistered hard correctness-preserving veto.

**D** — stability weak and correctness materially regresses.  
Reject contextual substitution; only a separately preregistered joint surface+context family may continue.

**E** — full DEV_READY.  
Freeze immediately and open a separate fresh confirmation court before any external Laya/Jev evaluation.

## Stop rule

After one S64 DEV:
- no projection seed/dimension sweep
- no learned projection retrofit
- no context normalization change
- no feature/width/pooling change
- no initialization change
- no target/alpha-probe/tolerance change
- no BCE weighting
- no optimizer/LR/weight-decay change
- no regularizer
- no gradient coupling
- no native retraining
- no selector/threshold change
- no retry
- no second DEV
- no external Laya/Jev evaluation.

Scientific failure is valid.
