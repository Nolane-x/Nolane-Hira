# HIRA V1 S46 interpretation plan — frozen before S46-A0/DEV exposure

Status: **FROZEN**

Issue: #275

## Controlled comparison

Same selected runtime + private-correction checkpoint.

Baseline:
- exact legacy S45 two-expert shell:
  `0.5 * (standardized primary + standardized corrected relation)`

Treatment:
- S46 zero-parameter coordinate-wise median across:
  - standardized primary
  - standardized native relation
  - standardized corrected relation.

No training parameter, optimizer, correction objective, epoch-selection rule or encoder count changes.

## Primary comparison

Use same-checkpoint S46 minus S45 legacy shell on the single fresh S46 DEV.

Report:
- fused canonical accuracy
- fused paraphrase accuracy
- paired both-correct
- question-swap change
- fused selected-choice agreement
- fused JS
- canonical/paraphrase gold margins
- option-order flips
- probability mass.

Also report unchanged expert metrics:
- raw primary canonical/paraphrase
- native relation canonical/paraphrase
- corrected relation canonical/paraphrase
- corrected relation agreement/JS
- native signature cosine/margin.

## Frozen interpretation

**A** — private correctness remains useful and S46 fused cross-view stability materially recovers relative to the exact same-checkpoint S45 shell:
robust consensus solves the downstream instability.

**B** — stability materially recovers but fused correctness collapses:
median consensus is too conservative; close.

**C** — correctness remains but fused stability does not materially recover:
single-outlier robust consensus is insufficient; change decision family.

**D** — S46 shell satisfies the complete DEV_READY gate set:
freeze immediately and open a separate fresh confirmation before any external Laya/Jev evaluation.

No scalar winner score.

## Evidence hierarchy

Strongest:
1. same-checkpoint S46 vs S45 legacy shell;
2. native/correction ownership invariants;
3. fresh absolute DEV metrics.

Do not use S45 absolute DEV scores as directly representative of S46 fresh domains.

## Stop rule

No post-DEV:
- alternate robust aggregator
- median variant
- confidence threshold
- entropy weighting
- temperature
- learned gate
- correction objective change
- capacity change
- optimizer change
- second encoder
- retry
- gate weakening
- second DEV.

Scientific failure is valid.
