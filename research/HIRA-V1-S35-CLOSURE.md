# HIRA V1 S35 closure — Native A13 Relation Geometry Court

Status: **CLOSED — MATCHED DEV COMPLETE / PREREGISTERED CASE B / NATIVE TRANSPORT IMPROVES, CORRECTNESS REGRESSES**

Issue: #253
PR: #254

## Canonical authority

A0:
- run `37008578553`
- artifact `11227286171`
- digest `sha256:a32475c6b9278ea5feada4f4c002188089b1a2add17ae732f3a4b6ae1fa16a47`
- authority head `72dca7b0a464d6083421d6a8c297740070a956ba`
- outcome `HIRA_V1_S35_A0_NATIVE_RELATION_GEOMETRY_READY`

Fresh matched TRAIN/DEV:
- run `37014291731`
- artifact `11231711883`
- artifact digest `sha256:570486a4460b420f628db155edda626a8a05aa0cee708003b45bb055c27171c7`
- scientific head `40a64d519636411e7c18bb762d0d717cc07bce95`
- outcome `HIRA_V1_S35_MATCHED_NATIVE_DEV_COMPLETE`

No second DEV.
No post-DEV projected/native mixing.
No dimension/normalization/temperature retry.
No learned native metric was introduced inside S35.
No sealed confirmation.
No multilingual probe.
No Laya/Jev evaluation.

## Frozen matched setup

Both arms:
- exact S17 final-attention LoRA **16,384**
- shared primary projection **32,768**
- exact physical trainable surface **49,152**
- original A13 frozen
- HIRACore frozen
- S14 equal standardized full-K fusion
- S15 relation-logit detach
- S17 primary/relation partition
- S17 relation-priority norm-balanced gradients
- relation CE coefficient **0.10**
- local relation signature canonicalization coefficient **0.15**
- signature separation margin **0.20**
- TRAIN **768**
- DEV **192**
- 12 fresh S35 domains
- seed **56001**
- 24 epochs
- batch 16
- AdamW 2e-4 / wd .01 / clip 1.0
- identical rows and epoch order across arms

Control:
- exact S13 projected 128D relation geometry.

Treatment:
- exact S13-equivalent relation algorithm directly in adapted A13 native **256D** token geometry;
- shared 256->128 projection absent from the treatment relation path;
- native relation signature width **256**;
- zero added learned parameters/state.

## A0 mechanism result

S35-A0 cleanly isolated geometry:

Under an exact 256D identity projection:
- native vs S13 relation-logit max abs **0**
- native vs S13 signature max abs **0**

Primary control/treatment:
- primary-logit max abs **0**

After strong shared-projection perturbation:

Control relation:
- logit change **2.7219405174**
- signature change **0.2809848785**

Native treatment:
- logit change **0**
- signature change **0**

Primary:
- projection-sensitive logit change **0.0016246404**

Gradient ownership:
- native relation block -> shared projection L1 **0**
- native relation block -> A13 LoRA L1 **0.5091757309**
- primary block -> shared projection L1 **1024.1845703125**

Mechanics:
- state/question/option/logical-option permutation contracts PASS
- masked padding PASS
- finite degenerate geometry
- probability-mass error **1.1920929e-7**
- full-K PASS
- checkpoint roundtrip PASS

DEV outcome is therefore not a projection-bypass wiring failure.

## Selected DEV — projected control

Selected epoch: **22**
Checkpoint:
`5a24a1e3b020bffe3f23bdb8dcc3de18bd7a77db839c6b8134d864d843bedaad`

Fused:
- canonical **0.53125**
- paraphrase **0.5390625**
- paired both-correct **0.2291666667**
- question-swap **0.5416666667**
- cross-view agreement **0.5598958333**
- JS **0.0403928248**
- canonical margin **+0.0845258972**
- paraphrase margin **+0.0660107670**

Primary:
- canonical **0.4635416667**
- paraphrase **0.3802083333**
- cross-view agreement **0.625**

Relation:
- canonical **0.4921875**
- paraphrase **0.6354166667**
- canonical margin **-0.1438346890**
- paraphrase margin **+0.1756806920**
- cross-view agreement **0.5989583333**

Signatures:
- same-option cosine **0.8087541759**
- same-vs-strongest-other-option margin **0.0595523777**

Gate:
- PASS **13/22**
- FAIL **9/22**
- DEV_READY false

## Selected DEV — native treatment

Selected epoch: **23**
Checkpoint:
`9f43360d091916518d477d3223d28718e035f4171dc31a5b232f7f8fbfe6e339`

Fused:
- canonical **0.4479166667**
- paraphrase **0.4791666667**
- paired both-correct **0.1614583333**
- question-swap **0.4583333333**
- cross-view agreement **0.7265625**
- JS **0.0204525727**
- canonical margin **+0.0776920589**
- paraphrase margin **-0.1018054830**

Primary:
- canonical **0.4791666667**
- paraphrase **0.4791666667**
- cross-view agreement **0.6875**

Relation:
- canonical **0.3619791667**
- paraphrase **0.328125**
- canonical margin **-0.1168135876**
- paraphrase margin **-0.1040974433**
- cross-view agreement **0.703125**

Signatures:
- same-option cosine **0.9196729809**
- same-vs-strongest-other-option margin **0.1631556302**

Gate:
- PASS **15/22**
- FAIL **7/22**
- DEV_READY false

## Exact selected delta — native minus projected

Native improves representation/transport:
- fused JS **-0.0199402521** (lower is better)
- fused agreement **+0.1666666667**
- raw primary canonical **+0.015625**
- raw primary paraphrase **+0.0989583333**
- raw primary agreement **+0.0625**
- relation agreement **+0.1041666667**
- same-option signature cosine **+0.1109188050**
- signature same-vs-other margin **+0.1036032525**
- canonical relation signed margin **+0.0270211014**

Native regresses semantic correctness:
- fused canonical **-0.0833333333**
- fused paraphrase **-0.0598958333**
- paired **-0.0677083333**
- question-swap **-0.0833333333**
- fused paraphrase margin **-0.1678162500**
- relation canonical accuracy **-0.1302083333**
- relation paraphrase accuracy **-0.3072916667**
- relation paraphrase margin **-0.2797781353**

This is a clean **transport/correctness decoupling**.

## Best observed DEV — projected control

Across 24 epochs:
- fused canonical **0.53125**, epoch 22
- fused paraphrase **0.5989583333**, epoch 17
- paired **0.2291666667**, epoch 22
- question-swap **0.5416666667**, epoch 22
- fused agreement **0.7135416667**, epoch 3
- minimum JS **0.0223716075**, epoch 1
- fused canonical margin **+0.1218805977**, epoch 14
- fused paraphrase margin **+0.1307144100**, epoch 14
- relation canonical **0.53125**, epoch 13
- relation paraphrase **0.6614583333**, epoch 18
- relation canonical margin best **-0.0797083974**, epoch 5
- relation paraphrase margin **+0.1838351451**, epoch 21
- relation agreement **0.7734375**, epoch 4
- signature cosine **0.8718786389**, epoch 7
- signature margin **0.1261771168**, epoch 5

## Best observed DEV — native treatment

Across 24 epochs:
- fused canonical **0.4609375**, epoch 22
- fused paraphrase **0.5182291667**, epoch 13
- paired **0.1614583333**, epoch 23
- question-swap **0.4583333333**, epoch 23
- fused agreement **0.7421875**, epoch 19
- minimum JS **0.0204525727**, epoch 23
- fused canonical margin **+0.0776920589**, epoch 23
- fused paraphrase margin best **-0.0195363301**, epoch 16
- raw primary canonical **0.5078125**, epoch 19
- raw primary paraphrase **0.5026041667**, epoch 16
- relation canonical **0.40625**, epoch 18
- relation paraphrase **0.3541666667**, epoch 13
- relation canonical margin best **-0.0452200671**, epoch 11
- relation paraphrase margin best **-0.0470973849**, epoch 12
- relation agreement **0.7213541667**, epoch 7
- signature cosine **0.9661531498**, epoch 8
- signature margin **0.1673370640**, epoch 22

The important pattern is robust:
- native signatures can satisfy the transport/discrimination gates at isolated and selected epochs;
- native relation logits still do not identify the gold option reliably.

## Optimization dynamics

Projected control epoch 1 -> 24:
- total TRAIN loss **1.8036970645 -> 0.8247363244**
- relation block **0.1790869832 -> 0.0620070368**
- canonicalization **0.2704375579 -> 0.1168375066**
- LoRA-B norm **0.4246610403 -> 2.5941526890**
- mean gradient conflict **0.3359375**

Native treatment epoch 1 -> 24:
- total TRAIN loss **1.7573343044 -> 1.2655232300**
- relation block **0.2179938654 -> 0.1504059161**
- canonicalization **0.5326724959 -> 0.0891047951**
- LoRA-B norm **0.7355453372 -> 4.6115479469**
- mean gradient conflict **0.4973958333**

Native training learns wording-stable option identities, but its fixed S13 correctness readout does not learn the gold relation.

## Preregistered interpretation

S35 matches **Case B — native transport improves but semantic decision quality regresses**.

Therefore:

> Bypassing the shared 128D projection exposes a substantially more wording-stable and option-discriminative relation representation, but the fixed S13 relation-logit formula cannot convert that representation into gold-option correctness.

This is not evidence to:
- mix projected/native logits from exposed DEV;
- retune temperatures;
- reduce native width;
- whiten native space;
- reopen S35 with another seed;
- claim native geometry is already a better decision model.

It is evidence that **representation identity and correctness readout are separate bottlenecks**.

## Metric boundary

The signature metrics do not identify the gold option.

`mean_signature_same_vs_strongest_wrong_margin` compares:
- the same logical option across canonical/paraphrase wording;
against
- other logical options across wording.

It measures cross-view option-identity discrimination, not gold-vs-wrong correctness.

S36 must therefore test a correctness readout directly rather than treating signature invariance as correctness.

## S35 closure

Close:
- projected-vs-native geometry localization.

Do not create S35b by:
- projected/native mixing
- dimension tuning
- whitening
- temperature tuning
- learned metric on S35-exposed rows
- seed/LR/epoch/batch retry
- second DEV

## Next controlled direction

**S36 — Native Signature Linear Correctness Readout Court**

Use wholly fresh S36 data.

Shared base:
- exact S35 native 256D relation representation;
- exact S17 primary path and optimizer shell;
- same native S35 base logits/signatures;
- no projected relation path in either matched arm.

Control:
- exact S35 native relation logits.

Treatment:
- exact S35 native relation logits
  **plus a zero-initialized shared linear residual readout over each 256D native relation signature**;
- one shared weight vector `w in R^256`;
- residual for option k: `signature_k dot w`;
- no bias;
- no option-ID embedding;
- no domain-specific head;
- no K-specific parameter;
- arbitrary-K / option-permutation equivariant;
- treatment adds exactly **256 trainable parameters**;
- A0 zero-init treatment is exact inference identity to the native control.

Scientific question:

> Does the wording-stable native relation signature contain a transferable linear correctness direction that the fixed S13/native scoring formula fails to read out?

A0 must prove:
- zero-init control/treatment relation-logit identity exactly **0**
- signature identity exactly **0**
- primary/fused identity exactly **0**
- treatment added params exactly **256**
- total treatment trainable surface exactly **49,408**
- readout gradient nonzero on fresh semantic A0
- readout receives relation-block gradient and zero primary-block gradient
- option permutation equivariance
- no option-ID/domain leakage
- native representation remains projection-independent
- full-K/state-once/probability/checkpoint mechanics

Fresh TRAIN/DEV:
- matched control/treatment on identical wholly fresh rows/order;
- exact S17 selector/gates;
- control surface **49,152**
- treatment surface **49,408**
- no readout-width/scale/bias/nonlinearity tuning after exposure.

If treatment coherently improves relation accuracy/margins and fused endpoints without destroying native transport, the readout family remains viable for a fresh confirmation.

If not, close linear correctness readout and conclude native signature identity is not linearly aligned with correctness.

Production-ready remains false.
Laya/Jev parity remains unestablished.
