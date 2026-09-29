# HIRA V1 S14 closure — Canonical Relation Evidence Fusion

Status: **CLOSED — DEV FAIL / MAJOR FUSION GAIN**

Issue: #204  
PR: #205

## Authority

A0 scientific exposure:
- run `36527407197`
- outcome `HIRA_V1_S14_A0_IDENTITY_READY`

A0 packaged reproduction:
- run `36527874581`
- artifact `11015447511`
- digest `sha256:3fccd10b3bde77a4396cec491d8a01a36c129e7d917dedfc688ff03fb0832976`

TRAIN/DEV first complete scientific exposure:
- run `36528589774`
- 24/24 epochs completed
- emitted `HIRA_V1_S14_EVIDENCE_FUSION_DEV_FAIL`
- upload blocked only by a stale scientific_authority metadata assertion after TRAIN

Deterministic packaged reproduction:
- run `36530243773`
- artifact `11017300477`
- digest `sha256:3354d9c635e1cad0a8062c3d8f74344e8e4121fd92a3c68947adfdc93c569b05`
- exact intended authority label `V1_S14_FRESH_ENGLISH_CANONICAL_RELATION_EVIDENCE_FUSION`
- selected epoch **16**
- checkpoint SHA256 `c3e54f33a7bcb683a1d4c9720d037064fbe967d9542b6fcf72e4d456dde3bb17`

The reproduction changed only authority metadata packaging. Model code, data, seed, optimizer, losses, fusion rule, selection order and gates remained frozen. Checkpoint SHA and selected DEV reproduced.

No post-DEV scientific tuning.
No sealed confirmation.
No multilingual probe.
No Laya/Jev reopening.

## Frozen surface

- A13 final-attention LoRA: **16,384**
- shared projection: **32,768**
- total trainable: **49,152**
- original A13 trainable: **0**
- HIRACore trainable: **0**
- canonicalizer-added params: **0**
- fusion-added params: **0**
- learned downstream scorer: **0**

Fusion remained the fixed parameter-free S14 rule:
`0.5 * standardized(triadic) + 0.5 * standardized(relation)`, epsilon `1e-6`.

## Selected DEV — epoch 16

Fused primary:
- canonical accuracy **0.6354166667**
- paraphrase accuracy **0.5390625**
- paired both-correct **0.375**
- question-swap choice-change **0.6979166667**
- canonical signed margin **0.1034617118**
- paraphrase signed margin **0.0882455171**
- cross-view selected-choice agreement **0.5572916667**
- cross-view mean JS **0.0479289759**

Raw experts:
- triadic canonical accuracy **0.5260416667**
- triadic paraphrase accuracy **0.5286458333**
- triadic cross-view agreement **0.5833333333**
- relation canonical accuracy **0.5833333333**
- relation paraphrase accuracy **0.5**
- relation canonical signed margin **0.0116562198**
- relation paraphrase signed margin **0.0013052672**
- relation cross-view agreement **0.5182291667**

Relation signatures:
- same-option cosine **0.7023841192**
- same-vs-strongest-wrong margin **0.0355867463**

Mechanical invariants:
- fused option-order flip **0.0**
- max probability-mass error **1.7881393433e-07**
- full-K PASS
- state-once PASS
- relation delta **0.0**

## Training dynamics

- total loss: **1.700388 -> 0.735353**
- fused decision loss: **1.417476 -> 0.504966**
- relation-binding loss: **1.417162 -> 1.245507**
- canonicalization loss: **0.478217 -> 0.255395**

Fresh DEV peaks:
- fused canonical accuracy **0.6354166667** at epoch 16
- paired both-correct **0.375** at selected epoch
- question-swap **0.703125** at epoch 11
- fused canonical margin **0.1362212275** at epoch 4
- relation canonical accuracy **0.5911458333** at epoch 18
- relation canonical margin **0.0166401391** at epoch 22
- signature cosine **0.7946747045** at epoch 3
- signature margin **0.0533725254** at epoch 5
- fused cross-view agreement **0.6875** at epoch 1

## Gate result

Mechanical/capacity gates passed. Semantic DEV_READY gates did not.

`HIRA_V1_S14_EVIDENCE_FUSION_DEV_READY` is therefore not authorized.

## Scientific conclusion

S14 is the strongest HIRA v1 primary semantic result so far.

At selected DEV:

`fused 63.54% > relation 58.33% > triadic 52.60%`

This is real zero-parameter fusion synergy and a large jump from S13 primary canonical accuracy 29.43%.

However the gain is not accompanied by sufficiently stable relation geometry:
- S13 selected signature cosine **0.765345** -> S14 **0.702384**
- S13 selected signature margin **0.065690** -> S14 **0.035587**
- fused cross-view choice agreement remains only **0.557292**

Thus fusion is validated as a productive primary architecture, but direct joint optimization still leaves semantic relation stability unresolved.

The next controlled hypothesis is **S15 Gradient-Isolated Evidence Fusion**:
- inference fusion remains numerically identical to S14;
- fused-primary CE/margin/JS direct gradients flow through the triadic path only;
- relation CE/canonicalization remain fully differentiable through the relation path;
- total trainable capacity stays exactly 49,152;
- no learned gate/head is added.

If relation geometry still degrades, the remaining interference is attributable to the shared physical parameter surface rather than direct relation-path fusion gradients.

Production-ready remains false.
Laya/Jev parity remains unestablished.
