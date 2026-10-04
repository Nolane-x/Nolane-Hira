# HIRA V1 S55 interpretation plan — frozen before S55-A0/DEV exposure

Status: **FROZEN**

Issue: #293

## Controlled variable

Reference:
`learned_joint_transform(q_k, zero_state_channel, option_identity)`

Treatment:
`learned_joint_transform(q_k, s54_joint_context, option_identity)`

All trainable capacity and optimization are matched.

## Evidence hierarchy

1. exact sealed S51 native authority;
2. native optimizer absent;
3. shared cache byte identity;
4. bit-identical correction initialization;
5. bit-identical learned-transform initialization;
6. equal correction/joint/total private parameter counts;
7. S55 A0 state-channel isolation/no-bypass invariants;
8. relation selected-choice behavior;
9. fused correctness/stability.

Failure of items 1–7 invalidates A–E classification.

## Primary DEV reporting

Reference/treatment:
- selected private epoch
- fused canonical/paraphrase accuracy
- paired both-correct
- question-swap
- fused selected-choice agreement
- fused JS
- fused margins
- relation canonical/paraphrase accuracy
- relation selected-choice agreement
- relation JS
- relation margins
- option-order flip
- probability mass
- state-view encodes (=0).

Learned-transform diagnostics:
- learned relation-code cross-view same-option cosine
- residual norm
- q↔relation-code cosine
- treatment explicit joint-channel norm
- treatment relation-code sensitivity to joint channel
- reference zero-state-channel invariant error.

## Frozen interpretation

**A** — selected-choice stability materially improves while useful correctness remains:
learned joint relation mapping is useful.

**B** — stability improves but correctness materially collapses:
learned transform overfits/over-smooths.

**C** — correctness remains but stability does not materially improve:
correction-factorization family is exhausted; move to explicit cross-view decision consistency at decision level.

**D** — correctness and stability both materially regress:
reject learned joint relation interaction.

**E** — full DEV_READY:
freeze immediately and open separate fresh confirmation before external Laya/Jev evaluation.

No scalar winner score.

## Stop rule

After one S55 DEV:
- no hidden-dimension/seed/channel-scale sweep
- no neutral-channel variant
- no bypass
- no native retraining
- no identity/correction capacity change
- no selector change
- no retry
- no gate weakening
- no second DEV.
