# HIRA V1 S48 interpretation plan — frozen before S48-A0/DEV exposure

Status: **FROZEN**

Issue: #279

## Controlled variable

Reference:
`PrivateCorrectionRepresentationFork`
with raw normalized mean query summary.

Treatment:
`QueryQuotientPrivateCorrectionFork`
with the same A/B/W capacity and initialization, but query input replaced everywhere by the option-difference quotient `q*`.

Nothing else may change.

## Primary DEV reporting

For reference and treatment:
- fused canonical/paraphrase accuracy
- paired both-correct
- question-swap change
- fused selected-choice agreement
- fused cross-view JS
- fused canonical/paraphrase gold margin
- corrected relation canonical/paraphrase accuracy
- corrected relation selected-choice agreement
- corrected relation cross-view JS
- corrected relation canonical/paraphrase margin
- option-order flip
- probability-mass error
- native trajectory fingerprints.

Treatment-only diagnostics:
- quotient norm mean
- quotient zero fraction
- raw-query vs quotient cosine
- quotient cross-view cosine
- corrected option-ranking cross-view agreement.

## Frozen interpretation

**A** — treatment preserves/improves useful correctness and materially recovers corrected/fused cross-view ordering:
query-surface nuisance was a major source of instability.

**B** — stability materially recovers but correctness collapses:
the quotient removes useful relation information; close.

**C** — correctness remains but stability does not materially recover:
query nuisance is not dominant; move deeper into state-option signature formation.

**D** — treatment satisfies the full DEV_READY gate set:
freeze immediately and open separate fresh confirmation before external Laya/Jev evaluation.

## Evidence hierarchy

1. matched reference vs treatment on identical S48 DEV rows;
2. exact native trajectory identity;
3. corrected relation ordering/agreement;
4. fused correctness/stability;
5. A0 quotient invariants.

Do not compare absolute S48 metrics directly against S47 as if they shared rows.

## Stop rule

After one S48 DEV:
- no quotient blend/sweep;
- no learned projection;
- no raw-query bypass;
- no normalization/epsilon sweep;
- no alternate checkpoint rule;
- no loss-weight/capacity/optimizer change;
- no second encoder;
- no retry;
- no gate weakening;
- no second DEV.

Scientific failure is valid.
