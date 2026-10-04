# HIRA V1 S50 interpretation plan — frozen before S50-A0/DEV exposure

Status: **FROZEN**

Issue: #283

## Controlled variable

Both branches consume one immutable shared-native evidence cache.

Reference private signature:
`question_conditioned_native_relation_signature`

Treatment private signature:
`query_free_state_option_identity`

Everything else in the private phase is matched.

## Evidence hierarchy

1. cache identity and immutable native evidence;
2. bit-identical correction initialization;
3. absence of native optimizer/live native graph in private phase;
4. corrected relation selected-choice behavior;
5. fused correctness/stability;
6. query-free identity A0 invariants.

If evidence items 1–3 fail, the court is invalid and cannot receive A/B/C/D/E.

## Primary DEV reporting

For each branch:
- canonical/paraphrase correction accuracy
- paired both-correct
- question-swap
- cross-view selected-choice agreement
- cross-view JS
- canonical/paraphrase margins
- option-order flip
- probability mass
- selected private epoch.

Treatment-only:
- identity cross-state-view same-option cosine
- identity same-vs-strongest-wrong margin
- fixed-identity raw-query sensitivity.

Shared:
- native cache digest
- row count
- native checkpoint identity
- replay-order digest.

## Frozen interpretation

**A** — correctness remains/improves and selected-choice stability materially recovers:
query-free identity is useful once native confounding is removed.

**B** — selected-choice stability materially recovers but correctness materially collapses:
query-free identity is too coarse.

**C** — correctness remains but selected-choice stability does not materially recover:
private option identity is not the dominant instability.

**D** — both correctness and selected-choice stability materially regress:
reject query-free identity and move deeper into state↔option semantic matching.

**E** — full DEV_READY:
freeze immediately and open a separate fresh confirmation before external Laya/Jev evaluation.

No scalar winner score.

## Stop rule

After one S50 DEV:
- no cache regeneration
- no native branch retraining
- no identity variant
- no pair-temperature/direct-relative sweep
- no query leak
- no blend/projector
- no loss/capacity change
- no alternate selector
- no retry
- no gate weakening
- no second DEV.
