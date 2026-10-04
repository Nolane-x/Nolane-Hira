# HIRA V1 S53 interpretation plan — frozen before S53-A0/DEV exposure

Status: **FROZEN**

Issue: #289

## Controlled variable

Reference:
`pooled_global_query_context`

Treatment:
`option_conditioned_token_level_query_context`

Everything else is matched.

## Evidence hierarchy

1. exact sealed S51 native authority;
2. native optimizer absent;
3. shared cache byte identity;
4. bit-identical correction initialization;
5. matched correction parameter count;
6. token-context A0 invariants;
7. relation selected-choice behavior;
8. fused correctness/stability.

Failure of items 1–5 invalidates A–E classification.

## Primary DEV reporting

Reference/treatment:
- selected private epoch
- canonical/paraphrase fused accuracy
- paired both-correct
- question-swap
- fused selected-choice agreement
- fused JS
- fused margins
- canonical/paraphrase relation accuracy
- relation selected-choice agreement
- relation JS
- relation margins
- option-order flip
- probability mass
- state-view encodes (=0).

Treatment diagnostics:
- mean token-attention entropy
- mean max token weight
- context norm error
- context cross-view cosine for same option
- informative-token sensitivity.

## Frozen interpretation

**A** — selected-choice stability materially improves while useful correctness remains/improves:
token-local late interaction is useful.

**B** — stability improves but correctness materially collapses:
late interaction is too selective/noisy.

**C** — correctness remains but stability does not materially improve:
global query pooling is not the dominant bottleneck.

**D** — correctness and stability both materially regress:
reject token-level query↔option late interaction.

**E** — full DEV_READY:
freeze and open a separate fresh confirmation before external Laya/Jev evaluation.

No scalar winner score.

## Stop rule

After one S53 DEV:
- no temperature/aggregation sweep
- no token projection
- no pooled-query bypass
- no native retraining
- no identity change
- no correction capacity change
- no selector change
- no retry
- no gate weakening
- no second DEV.
