# HIRA V1 S15 closure — Gradient-Isolated Evidence Fusion

Status: **CLOSED — DEV FAIL / SHARED-SURFACE GRADIENT CONFLICT EXPOSED**

Issue: #207  
PR: #208

## Authority

A0:
- run `36556776351`
- artifact `11027644171`
- digest `sha256:2f9025fc14c5bc6ca8c816834d7c5f0cf19c858cff61be93a8edc4c260145e5a`
- outcome `HIRA_V1_S15_A0_IDENTITY_READY`

Fresh TRAIN/DEV:
- run `36557765610`
- artifact `11029253333`
- digest `sha256:c15025d02b0b2e4afc3b2f93fdb104c1a8037420a84bb8530d49ee3df2b003c4`
- outcome `HIRA_V1_S15_GRADIENT_ISOLATED_FUSION_DEV_FAIL`
- selected epoch **12**
- checkpoint SHA256 `b14b644d739cff5559d3580e15f33e53b09f3e6ccbb6a3e6b6ee49d2ae1cc64e`

No post-DEV tuning.
No sealed confirmation.
No multilingual probe.
No Laya/Jev reopening.

## A0 gradient-route result

S15 inference remained exactly S14-equivalent:
- S15 vs S14 forward max abs: **0.0**

Direct fused-primary CE:
- triadic-logit gradient L1: **1.1707811356**
- relation-logit direct gradient L1: **0.0**

Dedicated relation CE:
- relation-logit gradient L1: **1.6726777554**

Therefore direct relation-logit isolation was implemented correctly.

## Selected DEV — epoch 12

Fused primary:
- canonical accuracy: **0.5182291667**
- paraphrase accuracy: **0.4088541667**
- paired both-correct: **0.2604166667**
- question-swap change: **0.671875**
- cross-view selected-choice agreement: **0.625**
- cross-view mean JS: **0.0271875365**
- canonical signed margin: **0.0106863134**

Raw triadic:
- canonical accuracy: **0.5755208333**
- paraphrase accuracy: **0.5**
- cross-view agreement: **0.6770833333**

Relation:
- canonical accuracy: **0.2890625**
- paraphrase accuracy: **0.28125**
- canonical signed margin: **-0.1557084620**
- paraphrase signed margin: **-0.1368227030**
- relation cross-view agreement: **0.6041666667**

Relation signatures:
- same-option cosine: **0.8722190807**
- same-vs-strongest-wrong margin: **0.0247949465**

Mechanical invariants:
- full-K: PASS
- option-order flip: **0.0**
- relation delta: **0.0**
- state-once: PASS
- max probability-mass error: **1.7881393433e-07**
- exact trainable surface: **49,152**
- fusion-added params: **0**

## TRAIN dynamics

Epoch 1 -> 24:
- total loss: **1.76195063 -> 1.18061710**
- relation-binding loss: **1.39966559 -> 1.37893081**
- canonicalization loss: **0.47704326 -> 0.21948369**

Fresh DEV:
- fused canonical accuracy peaked at **0.51822917** at epoch 12
- paired both-correct peaked at **0.26041667** at epoch 12
- relation canonical accuracy peaked only at **0.30729167** at epoch 21
- relation signed margin remained negative throughout; best **-0.13015850** at epoch 24
- same-option signature cosine peaked **0.87679177** at epoch 22
- signature discrimination margin peaked only **0.02803654** at epoch 4

## Comparison to S14

S14 selected DEV:
- fused canonical accuracy: **0.63541667**
- relation canonical accuracy: **0.58333333**
- raw triadic canonical accuracy: **0.52604167**
- paired both-correct: **0.375**
- fused margin: **0.10346171**

S15 selected DEV:
- fused canonical accuracy: **0.51822917**
- relation canonical accuracy: **0.2890625**
- raw triadic canonical accuracy: **0.57552083**
- paired both-correct: **0.26041667**
- fused margin: **0.01068631**

S15 improved the raw triadic path relative to S14 but substantially degraded the relation path and therefore lost fusion synergy.

## Scientific conclusion

S15 rejects the hypothesis that removing only the direct fused-primary gradient into relation logits is sufficient to preserve relation semantics.

The key structural result is:

> **The remaining interference occurs through the shared physical LoRA/projection surface.**

Although relation logits were detached from the fused-primary objective, triadic-primary gradients still update the same 49,152 physical parameters used by the relation expert. Dedicated relation CE/canonicalization also update that same surface.

The two objectives can therefore conflict at the shared parameter-gradient level even when relation logits are autograd-isolated.

## Next controlled hypothesis

S16 should keep:
- same inference as S14/S15
- same 49,152 physical trainable params
- zero learned optimizer/router params
- same fresh-evidence discipline

But explicitly measure and resolve conflict between:
1. primary/triadic gradient on the shared surface
2. relation/canonicalization gradient on the shared surface

A parameter-free projected-gradient rule is the strongest next test:
- compute primary and relation gradient vectors separately;
- when their dot product is negative, remove the conflicting component before applying the shared update;
- when non-conflicting, preserve both;
- do not fit a learned gate or add parameters.

This tests whether S14's fusion synergy can coexist with protected relation semantics when interference is handled at the true shared-parameter location.

Production-ready remains false.
Laya/Jev parity remains unestablished.
