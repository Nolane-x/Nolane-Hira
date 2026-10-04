# HIRA V1 S51 interpretation plan — frozen before S51-A0/authority exposure

Status: **FROZEN**

Issue: #285

## Controlled variable

Both private branches consume evidence generated from the exact same persisted native checkpoint artifact.

Reference:
`question_conditioned_native_relation_signature`

Treatment:
`query_free_state_option_identity`

Everything else is matched.

## Evidence hierarchy

1. Phase-A artifact integrity and authority binding;
2. Phase-B loaded runtime hash equals Phase-A runtime hash;
3. native parameters frozen / native optimizer absent;
4. reference/treatment cache byte identity;
5. bit-identical correction initialization;
6. corrected relation behavior;
7. fused correctness/stability;
8. treatment identity diagnostics.

Failure of items 1–5 invalidates the court and forbids A–E classification.

## Primary DEV reporting

Reference and treatment:
- selected private epoch;
- canonical/paraphrase relation accuracy;
- canonical/paraphrase fused accuracy;
- paired both-correct;
- question-swap;
- cross-view selected-choice agreement;
- cross-view JS;
- canonical/paraphrase gold margin;
- option-order flip;
- probability mass error;
- state-view encodes during private phase (=0).

Shared:
- Phase-A authority run/artifact name;
- artifact integrity digest;
- native checkpoint file SHA;
- native logical tensor digest;
- Phase-A runtime-state hash;
- Phase-B loaded runtime-state hash;
- TRAIN/DEV cache digests.

Treatment-only:
- query-free identity cross-state-view same-option cosine;
- identity same-vs-strongest-wrong margin;
- fixed-identity raw-query readout sensitivity.

## Frozen interpretation

**A** — useful correctness remains/improves and selected-choice stability materially improves:
query-free identity is useful.

**B** — stability materially improves but useful correctness materially collapses:
query-free identity is too coarse.

**C** — useful correctness remains but selected-choice stability does not materially improve:
query-free identity is not the dominant instability.

**D** — correctness and stability materially regress:
reject query-free identity and move deeper into state↔option semantic matching.

**E** — full DEV_READY:
freeze immediately and open separate fresh confirmation before external Laya/Jev evaluation.

No scalar winner score.

## Stop rule

After one Phase-B S51 DEV:
- no authority regeneration;
- no native retraining;
- no identity variant;
- no temperature/mixing sweep;
- no query leak/blend/projector;
- no loss/capacity change;
- no selector change;
- no retry;
- no gate weakening;
- no second DEV.
