# HIRA V1 S57 interpretation plan — frozen before S57-A0/DEV exposure

Status: **FROZEN**

Issue: #297

## Controlled variable

Reference:
- pairwise ordinal coefficient **0.0**

Treatment:
- pairwise ordinal coefficient **0.05**

Architecture/capacity/init/data/optimizer/selector are identical.

Frozen mechanics:
- standardized logits epsilon **1e-6**
- active threshold **0.25**
- preserved margin floor **0.05**
- gold-correct directional anchor filtering.

## Evidence hierarchy

1. exact sealed S51 native authority;
2. native optimizer absent;
3. immutable shared cache byte identity;
4. bit-identical private initialization;
5. equal private parameter counts;
6. S57-A0 detach/filter/invariance/anti-collapse mechanics;
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

Ordinal training diagnostics:
- mean weighted ordinal auxiliary
- mean ordinal loss
- active directional-anchor fraction
- gold-filtered anchor fraction
- sign-disagreement fraction among retained anchors.

Decision discrimination:
- mean fused entropy
- mean top1-top2 probability gap
- mean raw logit RMS.

## Frozen interpretation

**A** — stability materially improves while correctness/discrimination remains:
ordinal consistency is the correct decision-level constraint.

**B** — stability improves but correctness materially collapses:
anchors are still unsafe; next family uses consensus/teacher anchoring.

**C** — correctness remains but stability does not materially improve:
ordinal auxiliary is too weak; move to explicit pairwise decision modeling.

**D** — correctness and stability both materially regress:
reject this objective.

**E** — full DEV_READY:
freeze immediately and open separate confirmation before external Laya/Jev evaluation.

No scalar winner score.

## Stop rule

After one S57 DEV:
- no coefficient/threshold/margin sweep
- no anchor-filter change
- no teacher/consensus addition
- no architecture/capacity change
- no native retraining
- no selector change
- no retry
- no gate weakening
- no second DEV.
