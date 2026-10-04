# HIRA V1 S53 matched scientific receipt — Token-Level Query↔Option Late Interaction

Status: **FROZEN / CASE C**

Scientific run: `37205794717`  
Artifact: `11304553484`  
Artifact digest: `sha256:3ab2f778eef687e3cf8c3be459ae75facb14f964b3a5345d5aa7ad1b8bb7c962`  
Scientific head: `041dde0028c97c88cb933278dcd4ddb05af8f607`

Outcome:
`HIRA_V1_S53_TOKEN_QUERY_OPTION_LATE_INTERACTION_DEV_COMPLETE`

## Matched authority

- parent S51 native run `37192490832`
- parent native artifact `11299783210`
- runtime/native digest `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- native trainable params **0**
- native optimizer absent
- native retraining **false**
- S53-A0 run `37205099128`
- S53-A0 artifact `11304188134`
- shared TRAIN cache digest `b4af43a9877b93a342556c8dbe165e953a6a6c29824950f4f95a74363b05e2c5`
- shared DEV cache digest `83227ed77eb6b15aba889b00064323ad5ff328f481aeeb6032b91b455e1c1c50`
- reference/treatment same cache bytes **true**
- private state-view encodes **0**

Matched private surface:
- reference private trainable **114,688**
- treatment private trainable **114,688**
- late-interaction trainable params **0**
- initialization bit-identical **true**
- token temperature **0.1**

## Selected checkpoints

Reference selected epoch: **16**  
Treatment selected epoch: **9**

## Reference — pooled global query context

- fused canonical accuracy **0.4322916666666667**
- fused paraphrase accuracy **0.4791666666666667**
- paired both-correct **0.13541666666666666**
- fused agreement **0.6640625**
- fused JS **0.02748964385439952**
- canonical relation accuracy **0.4817708333333333**
- paraphrase relation accuracy **0.4635416666666667**
- relation agreement **0.6901041666666666**
- relation JS **0.016512935748323798**

## Treatment — option-conditioned token-level query context

- fused canonical accuracy **0.4322916666666667**
- fused paraphrase accuracy **0.484375**
- paired both-correct **0.13541666666666666**
- fused agreement **0.609375**
- fused JS **0.03370636717105905**
- canonical relation accuracy **0.4270833333333333**
- paraphrase relation accuracy **0.5182291666666666**
- relation agreement **0.5833333333333334**
- relation JS **0.008059095174151784**

## Treatment minus reference

Correctness:
- fused canonical **0.00 pp**
- fused paraphrase **0.52 pp**
- paired both-correct **0.00 pp**
- canonical relation accuracy **-5.47 pp**
- paraphrase relation accuracy **5.47 pp**

Stability:
- fused selected-choice agreement **-5.47 pp**
- fused JS **+0.006217**
- relation selected-choice agreement **-10.68 pp**
- relation JS **-0.008454**

Margins:
- fused canonical **+0.041953**
- fused paraphrase **+0.084220**
- relation canonical **+0.029770**
- relation paraphrase **+0.028703**

## Token diagnostics

- context cross-view same-option cosine **0.9780888011058172**
- mean token-attention entropy **2.891828238964081**
- mean max token weight **0.07445693016052246**
- context norm max error **1.7881393432617188e-7**

## Frozen interpretation

**Case C — useful correctness remains, but selected-choice stability does not materially improve.**

Treatment preserves fused canonical accuracy exactly, improves paraphrase accuracy only **+0.52 pp**, and leaves paired both-correct unchanged.

At the same time:
- fused agreement falls **5.47 pp**;
- fused JS worsens by **0.006217**;
- relation agreement falls **10.68 pp**.

Token-level context is mechanically stable across paraphrases (same-option context cosine ~**0.9781**), but that stability does not propagate into stable option decisions.

Therefore the dominant instability is deeper than global query pooling versus token-local query evidence.

S53 closes the query-only representation family. The next experiment must make state, query and option evidence interact jointly rather than deriving a query context first and combining it afterward.

No S53 retry or aggregation/temperature sweep is authorized.
