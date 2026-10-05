# HIRA V1 S58 interpretation plan — frozen before S58-A0/DEV exposure

Status: **FROZEN**

Issue: #299

## Controlled variable

Reference:
- teacher-consensus coefficient **0.0**

Treatment:
- teacher-consensus coefficient **0.05**

Architecture/capacity/initialization/data/optimizer/selector are identical.

Frozen teacher:
- S57 reference checkpoint from run `37271509208`
- artifact `11327849211`
- checkpoint SHA `804dc0b31ca75a77400c4a658ae893058d39916ed287d9fe2d45643798b739aa`
- selected epoch **19**
- no gradients / no S58-based selection.

Frozen consensus mechanics:
- teacher confidence threshold **0.25**
- student target margin **0.05**
- both teacher views must be confident and agree in sign
- wrong gold-order consensus filtered
- detached teacher sign.

## Evidence hierarchy

1. exact sealed S51 native authority;
2. exact S57 teacher checkpoint authority;
3. teacher frozen/no gradients;
4. immutable shared cache byte identity;
5. bit-identical student initialization;
6. equal student parameter counts;
7. S58-A0 consensus/filter/anti-collapse mechanics;
8. selected-choice stability;
9. correctness/discrimination.

If items 1–7 fail, A–E classification is forbidden.

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

Teacher-consensus diagnostics:
- eligible consensus pair fraction
- weak/inactive fraction
- teacher disagreement fraction
- wrong-gold filtered fraction
- non-gold eligible fraction
- student violation fraction
- weighted auxiliary.

Decision discrimination:
- mean fused entropy
- mean top1-top2 probability gap
- mean raw logit RMS.

## Frozen interpretation

**A** — stability improves while correctness/discrimination remains:
teacher consensus is the safe decision-level constraint.

**B** — stability improves but correctness materially collapses:
teacher still transmits unsafe orderings; move to explicit learned pairwise head.

**C** — correctness remains but stability does not materially improve:
consensus is too conservative; move to explicit learned pairwise head.

**D** — correctness and stability both materially regress:
reject teacher-consensus ranking.

**E** — full DEV_READY:
freeze immediately and open separate confirmation before external Laya/Jev evaluation.

No scalar winner score.

## Stop rule

After one S58 DEV:
- no teacher change
- no coefficient/threshold/margin sweep
- no consensus-mask variant
- no architecture/capacity change
- no native retraining
- no selector change
- no retry
- no gate weakening
- no second DEV.
