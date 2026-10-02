# HIRA V1 S32 closure — Global Query×Option Relation Matrix Canonicalization

Status: **CLOSED — CASE C / NO TRANSPORT RESTORATION / GLOBAL RELATION-CONTRASTIVE TOPOLOGY CLOSED**

Issue: #247
PR: #248

## Authority

Canonical A0:
- run `36978637856`
- artifact `11214148806`
- digest `sha256:19cab638c1dbb73b382aae669f42b3ab254ddb993145810dc26b3e12a61942a0`
- authority head `47be09c51dcd5b5d1307d911171266ca229c59f9`
- outcome `HIRA_V1_S32_A0_GLOBAL_QUERY_OPTION_MATRIX_READY`

Fresh matched TRAIN/DEV:
- run `36979535007`
- artifact `11216646829`
- digest `sha256:67d659a1e47542ee19b9baebe12e68511cb25618e1b9da78d7e725b2071e5994`
- scientific head `d229275f0dcd444704ebd8a528daab5de4734f1e`
- outcome `HIRA_V1_S32_MATCHED_GLOBAL_MATRIX_DEV_COMPLETE`

No second DEV.
No post-DEV tuning.
No temperature/coefficient/negative-set changes.
No local+global mixture.
No sealed confirmation.
No multilingual probe.
No Laya/Jev evaluation.

## Matched setup

Both arms use exact S17 runtime/optimizer shell:
- final attention LoRA **16,384**
- shared projection **32,768**
- exact trainable surface **49,152**
- original A13 frozen
- HIRACore frozen
- S13 relation expert geometry
- S14 equal standardized fusion
- S15 relation-logit detach
- S17 relation-priority norm-balanced gradients
- relation CE coefficient **0.10**
- global regularizer coefficient **0.15**
- temperature **0.10**
- zero added learned params/state
- identical fresh TRAIN **768**
- identical fresh DEV **192**
- same 12 domains
- seed **53001**
- same per-epoch batch order
- 24 epochs, batch 16
- AdamW lr 2e-4, weight decay 0.01, clip 1.0

Control:
- S31 gold-only global contrastive.

Treatment:
- global contrastive over every query×option relation.

## A0 discriminator

S32 A0 proves treatment observes non-gold relation geometry that S31 cannot.

Controlled all-option matched loss:
- **0.0006808108**

Perturb exactly one non-gold relation:
- all-option loss **0.4214525521**
- increase **+0.4207717478**

Gold-only on the same perturbation:
- good **0.0001362469**
- perturbed **0.0001362469**
- exact change **0**

Non-gold gradients:
- gold-only canonical/paraphrase **0 / 0**
- all-option canonical **6.7918653488**
- all-option paraphrase **6.5863475800**

Symmetry:
- query+option permutation error **0**
- view-swap error **0**

Real semantic gradients:
- LoRA-B Q/K/V/out **0.2595716119 / 0.2727566063 / 2.6033835411 / 13.6473093033**
- projection L1 **757.2583007812**

Mechanics/capacity/checkpoint PASS.

Therefore the DEV result is not an inactive-treatment failure.

## Selected DEV — gold-only control

Selected epoch **18**
Checkpoint:
`a6d6e2edb473233d7df3ad3b9fb54a0f109aaec5721dbf0e5e7c5d1d7fa046b8`

Fused:
- canonical **0.5338541667**
- paraphrase **0.4557291667**
- paired **0.2604166667**
- question-swap **0.5208333333**
- agreement **0.5703125**
- JS **0.0349437164**
- canonical margin **-0.0016525799**
- paraphrase margin **-0.0791223080**

Primary:
- canonical **0.375**
- paraphrase **0.2916666667**
- agreement **0.7239583333**

Relation:
- canonical **0.40625**
- paraphrase **0.4322916667**
- canonical margin **-0.0329136128**
- paraphrase margin **-0.0223923276**
- agreement **0.59375**

Signatures:
- same-option cosine **0.5694199651**
- discrimination **0.2072677830**

Gate result:
- **14/22 PASS**
- **8/22 FAIL**
- DEV_READY false

Failed:
- relation canonical >= 0.80
- relation canonical margin >= 0.15
- fused canonical >= 0.85
- fused canonical margin >= 0.15
- fused agreement >= 0.95
- paired >= 0.75
- question-swap >= 0.80
- signature cosine >= 0.90

## Selected DEV — all-query×option treatment

Selected epoch **20**
Checkpoint:
`fad69ff8900dc5833c4181acd3dd39692a56e0e86ffe1229e5115b90c87de3cf`

Fused:
- canonical **0.5911458333**
- paraphrase **0.4322916667**
- paired **0.3385416667**
- question-swap **0.625**
- agreement **0.5208333333**
- JS **0.0373732445**
- canonical margin **+0.0824745664**
- paraphrase margin **-0.2076232824**

Primary:
- canonical **0.4322916667**
- paraphrase **0.2890625**
- agreement **0.5651041667**

Relation:
- canonical **0.3880208333**
- paraphrase **0.4166666667**
- canonical margin **-0.0333805978**
- paraphrase margin **-0.0262401228**
- agreement **0.515625**

Signatures:
- same-option cosine **0.5596104686**
- discrimination **0.2281091238**

Gate result:
- **14/22 PASS**
- **8/22 FAIL**
- DEV_READY false

## Exact selected delta — matrix minus gold-only

Treatment improves:
- fused canonical **+0.0572916667**
- paired **+0.078125**
- question-swap **+0.1041666667**
- fused canonical margin **+0.0841271463**
- raw primary canonical **+0.0572916667**
- signature discrimination **+0.0208413408**

Treatment regresses:
- fused paraphrase **-0.0234375**
- fused paraphrase margin **-0.1285009744**
- fused agreement **-0.0494791667**
- fused JS **+0.0024295281**
- relation canonical **-0.0182291667**
- relation paraphrase **-0.015625**
- relation agreement **-0.078125**
- same-option signature cosine **-0.0098094965**

Relation signed margins are essentially unchanged/slightly worse:
- canonical delta **-0.0004669850**
- paraphrase delta **-0.0038477952**

## Best observed DEV — gold-only

Across 24 epochs:
- fused canonical **0.5338541667**, epoch 18
- fused paraphrase **0.4765625**, epoch 22
- paired **0.2604166667**, epoch 18
- question-swap **0.5208333333**, epoch 18
- agreement **0.6015625**, epoch 20
- min JS **0.0186663722**, epoch 3
- canonical margin best **-0.0016525799**, epoch 18
- paraphrase margin best **-0.0791223080**, epoch 18
- relation canonical **0.4088541667**, epoch 16
- relation paraphrase **0.4401041667**, epoch 22
- relation canonical margin best **-0.0323069369**, epoch 20
- relation paraphrase margin best **-0.0187581306**, epoch 24
- relation agreement **0.5989583333**, epoch 24
- signature cosine **0.7555072655**, epoch 4
- signature discrimination **0.2130700195**, epoch 22

## Best observed DEV — all-option matrix

Across 24 epochs:
- fused canonical **0.5911458333**, epoch 20
- fused paraphrase **0.4609375**, epoch 24
- paired **0.3385416667**, epoch 20
- question-swap **0.625**, epoch 17/20
- agreement **0.6197916667**, epoch 4
- min JS **0.0204291795**, epoch 1
- canonical margin **+0.0824745664**, epoch 20
- paraphrase margin best **-0.1695766974**, epoch 24
- raw canonical **0.4557291667**, epoch 18
- raw paraphrase **0.3255208333**, epoch 8
- relation canonical **0.4036458333**, epoch 24
- relation paraphrase **0.4348958333**, epoch 15
- relation canonical margin best **-0.0313672746**, epoch 24
- relation paraphrase margin best **-0.0241754850**, epoch 24
- relation agreement **0.5598958333**, epoch 24
- signature cosine **0.7438737303**, epoch 2
- signature discrimination **0.2337449466**, epoch 22

Critical preregistered endpoint:
- selected signature cosine **0.5694 -> 0.5596**
- best signature cosine **0.7555 -> 0.7439**

All-option coverage does **not** restore transport.

## Optimization dynamics

Gold-only epoch 1 -> 24:
- total TRAIN loss **1.9622865220 -> 1.0797751298**
- relation block **0.3435288354 -> 0.1513234666**
- global regularizer **1.3757332191 -> 0.1606096124**
- LoRA-B norm **0.5287879705 -> 2.7274179459**
- mean gradient conflict **0.3125**

All-option epoch 1 -> 24:
- total TRAIN loss **2.1450297385 -> 1.2668152551**
- relation block **0.5258538667 -> 0.2762579483**
- all-option regularizer **2.5912927662 -> 0.9625349330**
- LoRA-B norm **0.5340690613 -> 3.2149510384**
- mean gradient conflict **0.3376736111**

Both arms optimize successfully.
This is not a harness, dead-gradient or capacity failure.

## Scientific interpretation

S32 falls under preregistered **Case C — little/no transport gain**.

The S31 hypothesis that transport collapsed merely because K-1 non-gold relations lacked global positives is falsified.

Giving every query-option relation a global cross-view positive:
- does not improve same-option transport;
- does not improve relation accuracy/margins coherently;
- improves canonical decision geometry and question sensitivity;
- but worsens paraphrase margin and relation agreement.

Therefore the missing ingredient is not broader global contrastive coverage.

The common structure that remains is architectural:
question tokens determine state/option role weights, but the final S13 relation signature is

`normalize(state_sig + option_sig + role_delta)`

and the relation logit is based on

`best_pair + state/option role compatibility`.

A query representation is not retained explicitly in the final relation coordinates or final relation score.

## Closed directions

Close the **global relation-contrastive topology** as the next v1 rescue direction.

Do not create S32b by:
- temperature/coefficient tuning
- hard-negative mining
- local+global mixture
- more batch negatives
- batch-size changes
- seed/LR/epoch retries
- selecting exposed intermediate epochs
- widening encoder layers
- weakening gates

## Next controlled direction

**S33 — Query-Explicit Relation Coordinates**

Return to exact S17 local training topology. Do not inherit S31/S32 global losses.

Matched fresh court:

Control:
- exact S17 / S13 relation canonicalizer and logits.

Treatment:
- parameter-free query-explicit relation canonicalizer;
- retain the S13 base relation path;
- compute normalized pooled projected question anchor;
- compute per-option-view query-option residual:
  `q_option_delta = normalize(option_anchor - question_anchor)`;
- final signature:
  `normalize(concat(base_relation_signature, q_option_delta))`;
- relation signature width **256 = 128 base + 128 query-option**;
- add explicit query-option compatibility to relation score with fixed equal block weighting:
  - base S13 view score = existing `0.5*(best_pair + role_compatibility)`
  - treatment view score = `0.5*base_view_score + 0.5*dot(question_anchor, option_anchor)`
- no learned parameters/state added;
- exact S17 LoRA/projection trainable surface **49,152** unchanged.

This is not S11/S12:
- those use question tokens to locate role/value evidence;
- S33 retains the query as an explicit final relation coordinate and final scoring term.

This is not S26:
- S26 factorizes role/value residuals;
- S33 introduces direct query-option coordinates.

A0 must prove:
- control capacity unchanged and treatment adds 0 params;
- query anchor exact masked projected pool;
- option permutation equivariance;
- direct query-option compatibility responds to controlled question changes;
- query-option signature block changes under question intervention even when state/options are held fixed;
- base S13 block remains exact;
- finite degenerate geometry;
- real LoRA/projection gradients nonzero;
- full-K/state-once/mass/checkpoint mechanics.

Only wholly fresh S33 rows may decide the result.

Production-ready remains false.
Laya/Jev parity remains unestablished.
