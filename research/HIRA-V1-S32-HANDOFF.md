# HIRA V1 S32 handoff — to S33 Query-Explicit Relation Coordinates

S32 is frozen as:

`HIRA_V1_S32_MATCHED_GLOBAL_MATRIX_DEV_COMPLETE`

## Canonical authority

A0:
- run `36978637856`
- artifact `11214148806`
- digest `sha256:19cab638c1dbb73b382aae669f42b3ab254ddb993145810dc26b3e12a61942a0`
- outcome `HIRA_V1_S32_A0_GLOBAL_QUERY_OPTION_MATRIX_READY`

Fresh matched TRAIN/DEV:
- run `36979535007`
- artifact `11216646829`
- digest `sha256:67d659a1e47542ee19b9baebe12e68511cb25618e1b9da78d7e725b2071e5994`
- scientific head `d229275f0dcd444704ebd8a528daab5de4734f1e`
- outcome `HIRA_V1_S32_MATCHED_GLOBAL_MATRIX_DEV_COMPLETE`

Gold-only selected:
- epoch **18**
- fused canonical/paraphrase **0.5338541667 / 0.4557291667**
- paired **0.2604166667**
- question-swap **0.5208333333**
- relation canonical/paraphrase **0.40625 / 0.4322916667**
- relation margins **-0.0329136128 / -0.0223923276**
- signature cosine/discrimination **0.5694199651 / 0.2072677830**
- gates **14/22 PASS**

All-option matrix selected:
- epoch **20**
- fused canonical/paraphrase **0.5911458333 / 0.4322916667**
- paired **0.3385416667**
- question-swap **0.625**
- relation canonical/paraphrase **0.3880208333 / 0.4166666667**
- relation margins **-0.0333805978 / -0.0262401228**
- signature cosine/discrimination **0.5596104686 / 0.2281091238**
- gates **14/22 PASS**

## Verdict

Preregged Case C:
**little/no transport gain**.

All-option coverage does not rescue the S31 transport failure:
- selected signature cosine delta **-0.0098094965**
- best-across-24 signature cosine delta is also negative.

It improves some canonical endpoints but worsens paraphrase margin, relation accuracy/agreement and fused cross-view agreement.

Close global relation-contrastive topology.
Do not tune S31/S32.

## S33 target

**Query-Explicit Relation Coordinates**

Base both matched arms on exact S17 local training shell.

Control:
- exact S13 canonicalizer/logit.

Treatment:
- no new learned params;
- pooled projected question anchor retained explicitly;
- per-option-view query-option delta:
  `normalize(option_anchor - question_anchor)`;
- signature:
  `normalize(concat(base_S13_signature, query_option_delta))`
  -> width **256**;
- scoring:
  - preserve exact S13 base view score;
  - fixed treatment score
    `0.5*base_view_score + 0.5*query_option_compatibility`;
- query-option compatibility:
  `dot(question_anchor, option_anchor)`.

Keep:
- final attention LoRA **16,384**
- shared projection **32,768**
- total trainable **49,152**
- original A13 frozen
- HIRACore frozen
- S14 fusion
- S15 detach
- S17 norm-balanced gradient rule
- local relation CE + local cross-view signature canonicalization
- no S31/S32 global contrastive objective

A0 must separate direct query-explicit behavior from the inherited indirect role-selection path and prove no added parameter/state.

Use one wholly fresh matched S33 authority.
No S32 DEV rows.
No post-DEV weighting/temperature/query-pooling tuning.
No second DEV.
No Laya/Jev reopening before DEV_READY.
