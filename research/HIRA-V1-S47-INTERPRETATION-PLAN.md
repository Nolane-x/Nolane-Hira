# HIRA V1 S47 interpretation plan — frozen before S47-A0/DEV exposure

Status: **FROZEN**

Issue: #277

## Controlled comparison

Same selected S45-mechanics treatment checkpoint.

Baseline:
- exact legacy S45 two-expert equal-mean standardized shell.

Treatment:
- S47 ordinal pairwise majority
- Copeland score
- corrected-private ordinal tie-break
- exact K-derived lexicographic multiplier `2K-1`.

Training, capacity, optimizer, correction objective, epoch selection, and encoder count do not change.

## Primary reporting

Same-checkpoint:
- fused canonical accuracy
- fused paraphrase accuracy
- paired both-correct
- question-swap change
- selected-choice agreement
- cross-view JS after softmax
- canonical/paraphrase gold margins
- option-order flip rate
- probability-mass error.

Unchanged expert evidence:
- raw primary canonical/paraphrase
- native relation canonical/paraphrase
- corrected relation canonical/paraphrase
- corrected relation agreement/JS
- native signature cosine/margin.

Ordinal diagnostics:
- fraction of option pairs with 3/3 agreement
- fraction with 2/3 majority
- fraction tied due expert-level exact ties
- fraction of queries where private tie-break changes the top set among Copeland-tied options.

## Frozen interpretation

**A** — fused correctness remains useful and cross-view stability materially recovers relative to the exact same-checkpoint legacy S45 shell:
score magnitude was the dominant downstream instability.

**B** — stability materially recovers but fused correctness collapses:
ordinal majority is too conservative/weak; close.

**C** — fused correctness remains but stability does not materially recover:
instability is in expert ordering itself rather than score magnitude; change family.

**D** — S47 satisfies the complete DEV_READY gate set:
freeze immediately and open a separate fresh confirmation before any external Laya/Jev evaluation.

No scalar winner score.

## Evidence hierarchy

1. same-checkpoint S47 vs legacy S45 shell;
2. ordinal invariance and zero-parameter A0 courts;
3. ownership/runtime invariants;
4. fresh absolute DEV metrics.

Do not directly compare absolute S47 fresh-domain values with S46 fresh-domain values as if they shared rows.

## Stop rule

No post-DEV:
- alternate rank aggregator
- Borda variant
- majority threshold
- tie-break expert change
- rank temperature
- learned gate
- scalar weighting
- correction objective change
- capacity/optimizer change
- second encoder
- retry
- gate weakening
- second DEV.

Scientific failure is valid.
