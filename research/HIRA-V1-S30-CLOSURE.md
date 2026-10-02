# HIRA V1 S30 closure — Matched Attention-vs-FFN Adaptation Surface Court

Status: **CLOSED — MATCHED DEV COMPLETE / SPLIT EVIDENCE / NO DEV_READY**

Issue: #243  
PR: #244

## Authority

Canonical A0:
- run `36940968046`
- artifact `11199314198`
- artifact digest `sha256:2ec456fc02ca498729358d21cc7a8205aa8807230514902ac73f2ddaaebad013`
- authority head `04100aad4158170bfcea8c0dd1b41a7a529f58c8`
- outcome `HIRA_V1_S30_A0_MATCHED_ADAPTATION_READY`

Fresh matched TRAIN/DEV:
- run `36941827140`
- artifact `11201303347`
- artifact digest `sha256:aa7779664411acdf3e44d3bbef8f80688f1a55c3d20240b99050b7757e6a150b`
- scientific head `41a62c63eff9244c7ccc6bddd57ddf11d2a64f3a`
- outcome `HIRA_V1_S30_MATCHED_ADAPTATION_DEV_COMPLETE`

No second DEV run.  
No post-DEV tuning.  
No sealed confirmation.  
No multilingual probe.  
No Laya/Jev evaluation.

## Matched setup

Both arms used:
- the same fresh TRAIN 768 rows
- the same fresh DEV 192 rows
- the same 12 domains
- the same seed **51001**
- the same epoch row order
- the same S17 semantic/inference shell
- the same optimizer, epochs, batch, selector and gates
- independent runtime/model/optimizer state
- independent projection copies initialized from the same W28 T0 projection

Arm A — attention-only:
- final Q/K/V/attention-output LoRA **16,384**
- shared projection **32,768**
- total **49,152**

Arm B — FFN-only:
- final intermediate/output FFN LoRA **20,480**
- shared projection **32,768**
- total **53,248**

Both:
- rank 8
- alpha 8
- dropout 0
- original A13 frozen
- HIRACore frozen
- zero learned downstream head/router/gate/calibrator

## A0 result

The arms are exact zero-init inference identities:
- token output max abs **0**
- pooled output max abs **0**
- raw/relation/fused/signature max abs **0**
- exact logit identity **1.0**
- exact selected-choice identity **1.0**

Both surfaces are live.

Attention-only LoRA-B gradient L1:
- Q **0.2424240708**
- K **0.3624919653**
- V **2.2540881634**
- attention output **16.6883277893**

FFN-only LoRA-B gradient L1:
- intermediate **3.6626973152**
- output **1.9171190262**

Projection gradient L1:
- attention arm **657.9580688477**
- FFN arm **659.6528320312**

Checkpoint roundtrip and mechanics PASS for both.

## Selected DEV — attention-only

Selected epoch: **15**  
Checkpoint: `04fad98500ebc802f6d9441e6cdbc0274a170f7d3eb23d2f12655772c274883a`

Fused:
- canonical **0.5078125**
- paraphrase **0.2578125**
- paired **0.2447916667**
- question-swap **0.5208333333**
- agreement **0.4322916667**
- JS **0.0511925406**
- canonical margin **+0.0164120756**
- paraphrase margin **-0.4002887743**

Primary:
- canonical **0.4244791667**
- paraphrase **0.3046875**
- agreement **0.4947916667**

Relation:
- canonical **0.4036458333**
- paraphrase **0.2734375**
- canonical margin **-0.2553525219**
- paraphrase margin **-1.0135451568**
- agreement **0.609375**

Signature:
- cosine **0.8607332458**
- discrimination **0.1111970699**

Gate result:
- PASS **12/22**
- FAIL **10/22**
- DEV_READY false

## Selected DEV — FFN-only

Selected epoch: **17**  
Checkpoint: `f7b6667c0c009776fe2910646b8d27e4ac26b64613e989c3a68ffc3b06304620`

Fused:
- canonical **0.546875**
- paraphrase **0.2994791667**
- paired **0.2552083333**
- question-swap **0.4739583333**
- agreement **0.46875**
- JS **0.0449273751**
- canonical margin **+0.0585019166**
- paraphrase margin **-0.4301508628**

Primary:
- canonical **0.4010416667**
- paraphrase **0.2916666667**
- agreement **0.59375**

Relation:
- canonical **0.4401041667**
- paraphrase **0.2630208333**
- canonical margin **-0.0848894926**
- paraphrase margin **-0.5236856043**
- agreement **0.5885416667**

Signature:
- cosine **0.8562618246**
- discrimination **0.1187967975**

Gate result:
- PASS **13/22**
- FAIL **9/22**
- DEV_READY false

## Exact selected DEV deltas — FFN-only minus attention-only

Improved under FFN-only:
- fused canonical **+0.0390625**
- fused paraphrase **+0.0416666667**
- paired **+0.0104166667**
- fused agreement **+0.0364583333**
- fused JS **-0.0062651655** (lower is better)
- fused canonical margin **+0.0420898410**
- relation canonical **+0.0364583333**
- relation canonical margin **+0.1704630293**
- relation paraphrase margin **+0.4898595524**
- signature discrimination **+0.0075997276**

Worse under FFN-only:
- question-swap **-0.046875**
- fused paraphrase margin **-0.0298620885**
- raw primary canonical **-0.0234375**
- raw primary paraphrase **-0.0130208333**
- relation paraphrase accuracy **-0.0104166667**
- relation agreement **-0.0208333333**
- signature cosine **-0.0044714212**

This is split evidence, not a coherent FFN-only advantage.

## Best observed DEV — attention-only

Across 24 epochs:
- fused canonical **0.5234375**, epoch 8
- fused paraphrase **0.421875**, epoch 8
- paired **0.2447916667**, epoch 15
- question-swap **0.5989583333**, epoch 21
- agreement **0.7083333333**, epoch 1
- minimum JS **0.0101515114**, epoch 3
- canonical margin **+0.0164120756**, epoch 15
- paraphrase margin best **-0.1403524528**, epoch 8
- relation canonical **0.4479166667**, epoch 8
- relation paraphrase **0.3984375**, epoch 3
- relation canonical margin best **-0.0786680778**, epoch 5
- relation paraphrase margin best **-0.0845773319**, epoch 5
- relation agreement **0.7578125**, epoch 3
- signature cosine **0.9219787369**, epoch 7
- signature discrimination **0.1486528025**, epoch 5

## Best observed DEV — FFN-only

Across 24 epochs:
- fused canonical **0.546875**, epoch 17
- fused paraphrase **0.4140625**, epoch 6
- paired **0.2552083333**, epoch 17
- question-swap **0.5208333333**, epoch 24
- agreement **0.765625**, epoch 1
- minimum JS **0.0134552434**, epoch 4
- canonical margin **+0.0585019166**, epoch 17
- paraphrase margin best **-0.1928893609**, epoch 6
- relation canonical **0.4661458333**, epoch 9
- relation paraphrase **0.40625**, epoch 5
- relation canonical margin best **-0.0449635175**, epoch 10
- relation paraphrase margin best **-0.0807047511**, epoch 5
- relation agreement **0.7708333333**, epoch 4
- signature cosine **0.8767408530**, epoch 6
- signature discrimination **0.1482071715**, epoch 8

Neither arm ever gets positive relation margins on both wording views at the selected frontier, and neither approaches the complete DEV_READY gate set.

## Optimization dynamics

Attention-only epoch 1 -> 24:
- total TRAIN loss **1.7350225598 -> 0.8151920289**
- decision loss **1.4836433381 -> 0.6520511967**
- relation binding loss **1.3872773896 -> 0.8239778926**
- canonicalization loss **0.2663004544 -> 0.0713762327**
- LoRA-B norm **0.4424999654 -> 2.8693511486**
- mean gradient conflict **0.3394097222**

FFN-only epoch 1 -> 24:
- total TRAIN loss **1.7314815596 -> 0.8337798454**
- decision loss **1.4797446430 -> 0.6604183018**
- relation binding loss **1.3875295843 -> 0.9092871485**
- canonicalization loss **0.2698484560 -> 0.0798198303**
- LoRA-B norm **0.4169140458 -> 2.2308022976**
- mean gradient conflict **0.3541666667**

Both optimize successfully. This is not a harness or dead-gradient failure.

## Scientific interpretation

S30 answers the exact question left open by S29.

FFN-only has some independent value:
- better selected fused canonical/paraphrase accuracy
- better canonical relation accuracy/margin
- lower selected fused JS

But it does not produce a coherent semantic advantage:
- raw primary is slightly worse
- question sensitivity is worse
- relation paraphrase accuracy/agreement are worse
- signature transport is slightly worse
- paraphrase fused margin is worse
- neither arm reaches DEV_READY

Therefore:

> The S29 failure cannot be explained purely by attention/FFN co-adaptation interference, and choosing one final-block sublayer family is not the missing solution.

The encoder-surface widening/localization family is closed as the next v1 rescue direction.

## Next residual

A persistent pattern across S17-S30 is that relation-signature training is predominantly **within semantic case**:
- relation CE chooses among K=4 local options;
- signature canonicalization aligns equivalent views and separates wrong options within that same query/case.

The model can lower these objectives while still learning relation geometry that does not remain globally query-specific on fresh cases.

S28 independently showed high cross-view anchor similarity without enough discriminative separation. S30 again shows local training improvement without robust fresh relation margins.

The next controlled hypothesis should test **global cross-case relation discrimination**, not more encoder capacity and not another output-level consistency heuristic.

## Closed directions

Do not create S30b by:
- picking FFN-only and tuning LR/rank/alpha
- picking attention-only and widening layers
- combining the arms again
- adding earlier encoder layers
- arm-specific schedules
- seed retries
- extra epochs
- weaker gates

Do not reopen:
- S18 paired output margin
- S19/S20 output consistency family
- S22/S23 optimizer-priority family
- S24 reliability fusion family
- S26-S28 factor/transport-only family

## S31 direction

**S31 — Matched Local-vs-Global Relation Canonicalization Court**

Return to the frozen S17 attention-only architecture and S17 optimizer/fusion shell.

Use two matched arms on identical wholly fresh S31 rows:

Control:
- exact S17 relation block
- local K=4 cross-view signature canonicalization

Treatment:
- same relation CE
- replace the local signature canonicalization term with a parameter-free **cross-case query-specific relation contrastive loss**
- canonical gold relation signature for semantic query i must match paraphrase gold signature i
- all other semantic queries in the batch are negatives
- symmetric canonical->paraphrase and paraphrase->canonical InfoNCE
- fixed temperature **0.10**
- keep outer coefficient **0.15**
- no extra learned parameters
- no inference change

This directly tests whether global batch-level query-specific discrimination generalizes better than the current local K=4 canonicalization.

Use matched fresh data and identical optimization conditions.
No coefficient/temperature tuning after exposure.

Production-ready remains false.  
Laya/Jev parity remains unestablished.
