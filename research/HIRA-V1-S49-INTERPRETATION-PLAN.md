# HIRA V1 S49 interpretation plan — frozen before S49-A0/DEV exposure

Status: **FROZEN**

Issue: #281

## Controlled comparison

Reference:
question-conditioned native relation signature + raw-query private correction.

Treatment:
query-free state↔option identity signature + the exact same raw-query private correction.

Everything else remains matched.

## Primary DEV reporting

For each arm:
- fused canonical/paraphrase accuracy
- paired both-correct
- question-swap
- fused agreement / JS / margins
- corrected relation canonical/paraphrase accuracy
- corrected relation agreement / JS / margins
- option-order flip / probability mass
- native runtime trajectory fingerprints.

Treatment-only diagnostics:
- query-free identity cross-state-view cosine
- identity same-option vs strongest-wrong margin
- correction raw-query sensitivity with fixed identity.

## Frozen interpretation

**A** — correctness remains/improves and corrected/fused selected-choice stability materially recovers:
question contamination of private option identity was a major bottleneck.

**B** — selected-choice stability materially recovers but correctness collapses:
query-free identity is too coarse.

**C** — correctness remains but selected-choice stability does not materially recover:
private option identity was not the dominant instability.

**D** — both correctness and selected-choice stability materially regress:
question-conditioned identity contains necessary semantic information; close family and move deeper into state↔option semantic matching.

**E** — full DEV_READY:
freeze immediately and open a separate fresh confirmation before any external Laya/Jev evaluation.

No scalar winner score.

## Evidence hierarchy

1. matched reference vs treatment on identical S49 rows;
2. native trajectory identity;
3. corrected relation selected-choice behavior;
4. fused correctness/stability;
5. query-free identity A0 invariants.

## Stop rule

No post-DEV:
- identity operator variant
- pair-temperature/direct-relative sweep
- query leak
- raw/native identity blend
- learned projector
- correction/loss/capacity change
- alternate checkpoint rule
- retry
- gate weakening
- second DEV.

Scientific failure is valid.
