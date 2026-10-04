# HIRA V1 S56 interpretation plan — frozen before S56-A0/DEV exposure

Status: **FROZEN**

Issue: #295

## Controlled variable

Reference:
- `lambda_dec = 0.0`
- `lambda_order = 0.0`

Treatment:
- `lambda_dec = 0.10`
- `lambda_order = 0.05`

Architecture/capacity/initialization/data/optimizer/selector are matched.

## Evidence hierarchy

1. exact sealed S51 native authority;
2. native optimizer absent;
3. immutable shared cache byte identity;
4. bit-identical private initialization;
5. equal private parameter counts;
6. S56 A0 invariance/gradient/anti-collapse mechanics;
7. selected-choice stability;
8. correctness/discrimination.

If items 1–6 fail, A–E classification is forbidden.

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

Decision-collapse diagnostics:
- mean fused entropy
- mean top1-top2 probability gap
- mean standardized logit RMS.

Training diagnostics:
- mean decision-consistency auxiliary
- mean ordering-consistency auxiliary
- active ordering-pair fraction.

## Frozen interpretation

**A** — selected-choice stability materially improves while correctness/discrimination remains:
decision-level consistency is useful.

**B** — stability improves but correctness/margins/top1-top2 discrimination materially collapse:
consistency causes over-smoothing/collapse.

**C** — correctness remains but selected-choice stability does not materially improve:
move to a discrete pairwise-ranking/ordinal decision family.

**D** — correctness and stability both materially regress:
reject direct decision consistency.

**E** — full DEV_READY:
freeze immediately and open a separate confirmation before external Laya/Jev evaluation.

No scalar winner score.

## Stop rule

After one S56 DEV:
- no coefficient/threshold/margin sweep
- no entropy auxiliary
- no architecture/capacity change
- no native retraining
- no selector change
- no retry
- no gate weakening
- no second DEV.
