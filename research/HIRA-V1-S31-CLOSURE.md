# HIRA V1 S31 closure — Matched Local-vs-Global Relation Canonicalization

Status: **CLOSED — MATCHED DEV COMPLETE / GLOBAL GOLD-ONLY IMPROVES DISCRIMINATION BUT COLLAPSES ALL-OPTION TRANSPORT**

Issue: #245
PR: #246

## Authority

Canonical A0:
- run `36970587382`
- artifact `11211836648`
- artifact digest `sha256:6520619fbcb1fb74464770848022d8658ac234f909451ed2338e819be3c12862`
- authority head `422b81f96ae0cd0c1affa547521842b7ce0e650c`
- outcome `HIRA_V1_S31_A0_GLOBAL_RELATION_CONTRASTIVE_READY`

Earlier A0 run `36969870003` is a documented harness-only non-result:
- no scientific receipt
- no artifact
- no model-selection authority
- science settings unchanged

Fresh matched TRAIN/DEV:
- run `36973026624`
- artifact `11213780623`
- artifact digest `sha256:3c5d24679a96dda2bd49d2a09252f66e7eb57eb5931adbe2dcf1dcb254b3acb3`
- scientific head `7ab2e2b3696af119b00f9b8d5a20243fdc4853b3`
- outcome `HIRA_V1_S31_MATCHED_RELATION_DEV_COMPLETE`

No second DEV run.
No post-DEV tuning.
No temperature/coefficient/negative-bank retry.
No sealed confirmation.
No multilingual probe.
No Laya/Jev evaluation.

## Frozen matched setup

Both arms:
- exact S17 architecture
- final attention LoRA **16,384**
- shared projection **32,768**
- exact trainable surface **49,152**
- original A13 frozen
- HIRACore frozen
- S13 relation expert
- S14 equal standardized full-K fusion
- S15 relation-logit detach
- S17 primary/relation partition
- S17 relation-priority norm-balanced gradient rule
- same fresh TRAIN **768**
- same fresh DEV **192**
- same 12 domains
- seed **52001**
- same row order per epoch
- 24 epochs
- batch 16
- AdamW lr 2e-4
- weight decay 0.01
- grad clip 1.0
- independent runtime/optimizer state

Local control relation regularizer:
- `0.15 * local K=4 cross-view signature canonicalization`

Global treatment relation regularizer:
- `0.15 * cross-case gold-signature InfoNCE`
- temperature **0.10**
- canonical/paraphrase gold relation signature for each semantic query are positives
- all other semantic queries in the batch are negatives
- zero added learned params/state

Relation CE remains **0.10** in both arms.

## A0 mechanism result

A0 proved the global operator is active and exact-inference-preserving:

- local/global token identity max abs **0**
- pooled identity max abs **0**
- raw/relation/fused/signature identity max abs **0**
- exact logits **1.0**
- exact choices **1.0**
- physical surface **49,152 / 49,152**
- operator learned params **0**
- matched-positive controlled loss **0.0003177615**
- shuffled-positive controlled loss **2.0794413090**
- shuffled-minus-matched **+2.0791234970**
- matched query permutation error **0**
- canonical/paraphrase view-swap error **0**
- canonical signature gradient L1 **9.3638286591**
- paraphrase signature gradient L1 **8.6360034943**
- real semantic LoRA/projection gradients nonzero
- checkpoint/full-K/state-once/mechanics PASS

The DEV outcome is therefore not caused by an inactive or miswired treatment.

## Selected DEV — local control

Selected epoch: **20**
Checkpoint:
`fe99252f4ab48123ef95d39f092879b2688055168ab77ba3766fa906ff0f0fd1`

Fused:
- canonical **0.484375**
- paraphrase **0.3723958333**
- paired **0.2239583333**
- question-swap **0.5625**
- cross-view agreement **0.5598958333**
- JS **0.0470730269**
- canonical margin **-0.1444363470**
- paraphrase margin **-0.3916657949**

Primary:
- canonical **0.4270833333**
- paraphrase **0.3489583333**
- cross-view agreement **0.5182291667**

Relation:
- canonical **0.3671875**
- paraphrase **0.3255208333**
- canonical margin **-0.1254987555**
- paraphrase margin **-0.1419940343**
- cross-view agreement **0.6588541667**

Relation signatures:
- same-option cosine **0.9415669243**
- same-vs-strongest-wrong margin **0.1938003438**

Gate result:
- PASS **15/22**
- FAIL **7/22**
- DEV_READY false

Failed semantic gates:
- relation canonical >= 0.80
- relation canonical margin >= 0.15
- fused canonical >= 0.85
- fused canonical margin >= 0.15
- fused agreement >= 0.95
- paired >= 0.75
- question-swap >= 0.80

## Selected DEV — global gold-only treatment

Selected epoch: **24**
Checkpoint:
`576deb92ba06897f2c45182f10a0d4dfd84523391618c09b0bd5a754aa17bd2b`

Fused:
- canonical **0.5286458333**
- paraphrase **0.53125**
- paired **0.234375**
- question-swap **0.4479166667**
- cross-view agreement **0.5416666667**
- JS **0.0318575419**
- canonical margin **-0.0369015603**
- paraphrase margin **-0.0173759734**

Primary:
- canonical **0.40625**
- paraphrase **0.3984375**
- cross-view agreement **0.6692708333**

Relation:
- canonical **0.4166666667**
- paraphrase **0.4557291667**
- canonical margin **-0.0427276517**
- paraphrase margin **-0.0189892376**
- cross-view agreement **0.6302083333**

Relation signatures:
- same-option cosine **0.6501545608**
- same-vs-strongest-wrong margin **0.1934766614**

Gate result:
- PASS **14/22**
- FAIL **8/22**
- DEV_READY false

The extra failed gate versus local is:
- same-option signature cosine >= **0.90**

## Exact selected delta — global minus local

Global gold-only improves:
- fused canonical **+0.0442708333**
- fused paraphrase **+0.1588541667**
- paired **+0.0104166667**
- fused JS **-0.0152154850** (lower is better)
- fused canonical margin **+0.1075347867**
- fused paraphrase margin **+0.3742898215**
- relation canonical accuracy **+0.0494791667**
- relation paraphrase accuracy **+0.1302083333**
- relation canonical margin **+0.0827711038**
- relation paraphrase margin **+0.1230047966**
- raw primary paraphrase **+0.0494791667**

Global gold-only regresses:
- question-swap **-0.1145833333**
- fused agreement **-0.0182291667**
- raw primary canonical **-0.0208333333**
- relation agreement **-0.0286458333**
- same-option signature cosine **-0.2914123634**

Signature discrimination is essentially unchanged:
- delta **-0.0003236824**

This is split evidence.

## Best observed DEV — local

Across 24 epochs:
- fused canonical **0.5000**, epoch 13
- fused paraphrase **0.4401041667**, epoch 13
- paired **0.2239583333**, epoch 14/20
- question-swap **0.5625**, epoch 20
- agreement **0.6484375**, epoch 9
- minimum JS **0.0162682267**, epoch 8
- canonical margin best **-0.1177786738**, epoch 13
- paraphrase margin best **-0.1439059818**, epoch 1
- relation canonical **0.4140625**, epoch 4
- relation paraphrase **0.4427083333**, epoch 8
- relation canonical margin best **-0.0744152839**, epoch 2
- relation paraphrase margin best **-0.0404291724**, epoch 4
- relation agreement **0.8151041667**, epoch 12
- signature cosine **0.9515904039**, epoch 17
- signature discrimination **0.1938003438**, epoch 20

## Best observed DEV — global gold-only

Across 24 epochs:
- fused canonical **0.5286458333**, epoch 24
- fused paraphrase **0.53125**, epoch 24
- paired **0.234375**, epoch 24
- question-swap **0.484375**, epoch 15
- agreement **0.6015625**, epoch 12
- minimum JS **0.0206344179**, epoch 6
- canonical margin best **+0.0647712932**, epoch 13
- paraphrase margin best **-0.0173759734**, epoch 24
- relation canonical **0.4244791667**, epoch 12
- relation paraphrase **0.4635416667**, epoch 23
- relation canonical margin best **-0.0358179659**, epoch 10
- relation paraphrase margin best **-0.0181150883**, epoch 12
- relation agreement **0.6901041667**, epoch 11
- signature cosine best only **0.7729473710**, epoch 2
- signature discrimination **0.1970955127**, epoch 21

Critical fact:
- global gold-only never approaches the 0.90 same-option signature cosine gate;
- its best is only **0.7729473710**.

## Optimization dynamics

Local epoch 1 -> 24:
- total TRAIN loss **1.7336203903 -> 0.9450189409**
- relation block **0.1839396513 -> 0.1120342384**
- local canonicalization loss **0.3059594150 -> 0.0221805492**
- LoRA-B norm **0.4597776830 -> 3.4910919666**
- mean gradient conflict **0.2734375**

Global epoch 1 -> 24:
- total TRAIN loss **1.9476929680 -> 1.0503578683**
- relation block **0.3621240230 -> 0.1467500388**
- global relation regularizer **1.4978154985 -> 0.1500581348**
- LoRA-B norm **0.5109469891 -> 2.9169785976**
- mean gradient conflict **0.3246527778**

Both optimize successfully.
The global treatment is learnable on TRAIN.

## Scientific interpretation

S31 provides positive evidence for **cross-case query-specific discrimination pressure**:
- relation accuracies improve;
- relation signed margins move substantially toward zero;
- fused canonical and especially paraphrase accuracy improve;
- fused paraphrase margin improves by **+0.3743**;
- JS improves.

However, the gold-only topology creates a new structural failure:

> Only the gold query-option relation is explicitly paired across wording views. The other K-1 query-option relations receive no global cross-view identity constraint.

Fresh DEV exposes this directly:
- local all-option signature cosine **0.9416**
- global gold-only all-option signature cosine **0.6502**
- best global all-option signature cosine across all epochs only **0.7729**

At the same time signature same-vs-wrong discrimination remains about the same.

Therefore S31 does **not** reject global cross-case discrimination itself.
It rejects **gold-only global relation canonicalization as a sufficient solution**.

The remaining controlled question is whether global discrimination can be applied to the complete query×option relation matrix while preserving cross-view identity for every candidate relation.

## Track closure

Close:
**gold-only cross-case relation contrastive as the v1 solution**.

Do not create S31b by:
- tuning temperature
- tuning coefficient
- adding harder negatives
- changing batch size
- mixing local + global after exposed DEV
- selecting an exposed intermediate epoch
- seed/LR/epoch retry
- weakening the signature-cosine gate

## Next controlled direction

**S32 — Global Query×Option Relation Matrix Canonicalization**

Use exact S17 architecture and optimization shell.

Matched fresh two-arm court:

Control:
- S31 global gold-only relation contrastive
- temperature 0.10
- coefficient 0.15

Treatment:
- replace gold-only contrastive with **all-query-option global contrastive**
- for every semantic query q and every logical option k:
  - canonical relation signature (q,k) and paraphrase signature (q,k) are positives;
  - every other query-option pair in the batch is a negative;
- flatten [Q,K,D] -> [Q*K,D];
- L2 normalize;
- symmetric canonical->paraphrase and paraphrase->canonical InfoNCE;
- temperature **0.10**
- coefficient **0.15**
- zero learned params/state
- no inference change

This is not local K=4 canonicalization:
- negatives span all query-option relations across the batch;
- same-query wrong options remain negatives;
- cross-query options remain negatives;
- every option receives a cross-view positive identity constraint.

A0 must prove a discriminator that S31 cannot:
- perturbing/shuffling a **non-gold** option relation must leave S31 gold-only loss unchanged;
- the S32 all-option loss must increase materially;
- matched query+option permutation equivariance;
- view-swap symmetry;
- gradients to canonical/paraphrase non-gold signatures;
- zero added params;
- exact 49,152 surface;
- exact inference identity;
- full-K/state-once/checkpoint/mechanics.

Only wholly fresh S32 data may decide the result.

Production-ready remains false.
Laya/Jev parity remains unestablished.
