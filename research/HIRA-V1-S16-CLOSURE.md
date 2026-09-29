# HIRA V1 S16 closure — Shared-Surface Gradient Surgery

Status: **CLOSED — DEV FAIL / PARTIAL SUPPORT FOR SHARED-GRADIENT INTERFERENCE**

Issue: #210  
PR: #211

## Authority

A0:
- run `36564603152`
- artifact `11031676959`
- digest `sha256:cefd83bc1decb2b6e7ebbbfe94ff8fab37d68724f9d3bced095373b4843774b1`
- outcome `HIRA_V1_S16_A0_IDENTITY_READY`

Fresh TRAIN/DEV:
- run `36565350686`
- artifact `11031938669`
- digest `sha256:518dbe1474abea876a314555b8b648357dce0744d798ff454fe7c8e58eb221ef`
- outcome `HIRA_V1_S16_SHARED_GRADIENT_SURGERY_DEV_FAIL`
- selected epoch **24**
- checkpoint SHA256 `2b46263a9a6b0f66c64b0a4d6f486b7303637de49c67531ef29a822962939bd3`

No post-DEV tuning.
No sealed confirmation.
No multilingual probe.
No Laya/Jev reopening.

## Frozen surface

- A13 final-attention LoRA: **16,384**
- shared 256->128 projection: **32,768**
- total trainable: **49,152**
- original A13 frozen
- HIRACore frozen
- learned optimizer/router params: **0**
- fusion/canonicalizer/downstream learned params: **0**
- inference numerically identical to S14/S15

Gradient surgery:
- relation-priority global projection
- epsilon **1e-12**
- no random PCGrad order
- no symmetric second projection

## Selected DEV — epoch 24

Fused:
- canonical accuracy: **0.5520833333**
- paraphrase accuracy: **0.4947916667**
- paired both-correct: **0.3020833333**
- question-swap choice-change: **0.6302083333**
- cross-view selected-choice agreement: **0.546875**
- mean JS: **0.0236762010**
- canonical signed margin: **0.0233745320**

Raw triadic:
- canonical accuracy: **0.5598958333**
- paraphrase accuracy: **0.5286458333**
- cross-view agreement: **0.7213541667**

Relation:
- canonical accuracy: **0.34375**
- paraphrase accuracy: **0.2942708333**
- canonical signed margin: **-0.0640710592**
- paraphrase signed margin: **-0.0724555030**
- cross-view agreement: **0.5807291667**

Relation signatures:
- same-option cosine: **0.8635843198**
- same-vs-strongest-wrong margin: **0.0359043240**

Mechanical:
- option-order flip: **0.0**
- max mass error: **1.1920928955e-07**
- full-K: PASS
- state-once: PASS
- relation delta: **0.0**

## TRAIN gradient evidence

Mean conflict rate across 24 epochs:

**0.3767361111**

Per-epoch conflict rate ranged from:
- minimum **0.2291666667**
- maximum **0.5416666667**

Therefore shared primary-vs-relation anti-alignment is real and frequent.

TRAIN losses epoch 1 -> 24:
- total loss: **1.7833232184 -> 1.1380550166**
- decision loss: **1.4879555702 -> 0.8992755562**
- relation-binding loss: **1.3942700997 -> 1.3644827331**
- canonicalization loss: **0.5404188130 -> 0.2045007925**
- primary block: **1.5628333762 -> 0.9709316231**
- relation block: **0.2204898372 -> 0.1671233960**

Gradient magnitude diagnostics expose a second bottleneck.

Epoch 1 mean:
- primary gradient norm: **9.2504089226**
- relation gradient norm: **0.3249968948**

Epoch 24 mean:
- primary gradient norm: **4.2758044302**
- relation gradient norm: **0.0354909398**

At epoch 24 the primary gradient is roughly **120x** larger in norm than the relation-preservation gradient.

The relation-priority projection removes anti-aligned primary components when conflict occurs, but it does not correct this large magnitude imbalance when gradients are aligned or only weakly conflicting.

## Comparison

S14:
- fused canonical: **0.6354166667**
- relation canonical: **0.5833333333**

S15:
- fused canonical: **0.5182291667**
- relation canonical: **0.2890625**

S16:
- fused canonical: **0.5520833333**
- relation canonical: **0.34375**

S16 recovers:
- **+3.39 points** fused canonical over S15
- **+5.47 points** relation canonical over S15

but remains below S14 by:
- **8.33 points** fused canonical
- **23.96 points** relation canonical

## Preregistered interpretation

S16 gives **partial support for Outcome A**:
- conflict rate is nontrivial;
- fusion improves over S15;
- relation accuracy improves over S15.

However the recovery is insufficient for DEV_READY and the relation surface remains weak.

The strongest new evidence is not just conflict frequency but severe shared-gradient magnitude imbalance.

## Scientific conclusion

S16 establishes that destructive anti-alignment on the shared surface is a real bottleneck, but first-order conflict projection alone is insufficient.

The next controlled question is:

> **Can HIRA preserve both objectives when their gradient directions are given comparable influence without changing model capacity or fitting DEV-selected loss coefficients?**

The next track should test a fixed parameter-free gradient-direction balancing rule before conflict handling.

It must not:
- tune S16/S17 coefficients from exposed DEV;
- reuse S16 DEV;
- add a learned optimizer/router;
- widen the 49,152-parameter surface;
- reopen Laya/Jev.

Production-ready remains false.
Laya/Jev parity remains unestablished.
