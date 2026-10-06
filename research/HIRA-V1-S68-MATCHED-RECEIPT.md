# HIRA V1 S68 matched scientific receipt — Context-Modulated Antisymmetric Pairwise Comparator

Status: **FROZEN / CASE C**

Scientific run: `37389861004`  
Artifact: `11381470793`  
Artifact digest: `sha256:0156bb57125f6b8c4f380649505803ed76cf49b8edf1c5bd4774fd8790386eaa`  
Scientific head: `d3412d4c3910cd8de331cdfbe601e12cf743019d`

Outcome:
`HIRA_V1_S68_CONTEXT_MODULATED_PAIRWISE_DEV_COMPLETE`

## Controlled variable

Reference and treatment both use:
- exact S59 detached 512D option representation;
- hidden width **64**;
- `A [64,512]`;
- `u [64]`;
- `G [64,8]`;
- total pairwise trainable params **33,344**;
- exact same parameter initialization;
- exact same gold-vs-distractor pairwise objective;
- same rows/order/optimizer;
- same correction trajectory;
- same S64 contextual reliability gate with S66 per-view responsibility BCE.

Treatment parameter advantage: **0**.

Only context source differs:
- reference modulation sees identity-half set context;
- treatment sees full identity + joint state/query/option set context.

## Selected checkpoints

Reference epoch: **15**  
Treatment epoch: **15**

## Fresh pairwise evidence

Reference:
- gold-pair accuracy **0.5911458333**
- mean gold-pair margin **0.0400770746**
- aggregate canonical accuracy **0.3619791667**
- aggregate paraphrase accuracy **0.3567708333**
- aggregate cross-view JS **0.00207579818**
- aggregate selected-choice agreement **0.6484375**.

Treatment:
- gold-pair accuracy **0.5911458333**
- mean gold-pair margin **0.0398169186**
- aggregate canonical accuracy **0.3619791667**
- aggregate paraphrase accuracy **0.3567708333**
- aggregate cross-view JS **0.00204449892**
- aggregate selected-choice agreement **0.6484375**.

Treatment minus reference:
- gold-pair accuracy **0.00 pp**
- gold-pair margin **-0.0002601561**
- aggregate canonical accuracy **0.00 pp**
- aggregate paraphrase accuracy **0.00 pp**
- aggregate JS **-0.0000312993**
- aggregate agreement **0.00 pp**.

## Final decision metrics

Reference:
- canonical **0.4791666667**
- paraphrase **0.34375**
- paired both-correct **0.203125**
- question-swap **0.6770833333**
- agreement **0.2552083333**
- JS **0.1436485316**.

Treatment:
- canonical **0.4791666667**
- paraphrase **0.34375**
- paired both-correct **0.203125**
- question-swap **0.6770833333**
- agreement **0.2552083333**
- JS **0.1436694302**.

Treatment minus reference:
- canonical **0.00 pp**
- paraphrase **0.00 pp**
- paired both-correct **0.00 pp**
- question-swap **0.00 pp**
- agreement **0.00 pp**
- JS **+0.0000208986 worse**.

## Mechanical validity

- reference G gradient live: yes
- treatment G gradient live: yes
- antisymmetry max error: **0**
- diagonal max error: **0**
- full-K: yes
- second DEV: **false**
- external Laya/Jev evaluation: **false**.

## Frozen interpretation

**Case C.**

Context modulation is active, but fresh pairwise evidence does not materially improve. Therefore the comparator can consume contextual modulation, yet the detached S59 representation still does not expose the semantic interaction required for robust pairwise discrimination.

Per the frozen interpretation plan, the next intervention moves upstream into **representation construction**, not another comparator/gate/target refinement.
