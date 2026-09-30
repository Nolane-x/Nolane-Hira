# HIRA V1 S20 closure — Standardized Triadic Evidence Consistency

Status: **CLOSED — DEV FAIL / STANDARDIZED CONSISTENCY IS ACTIVE BUT DOES NOT GENERALIZE**

Issue: #223  
PR: #224

## Authority

A0:
- run `36677124443`
- artifact `11080820457`
- artifact digest `sha256:39769042b9620b5444f5a478a1ef12d63acb43c7abbd5332b8978b5177be4b06`
- A0 head `705539dddb8316f4bcfd5cf6a08d7885aba29d1a`
- outcome `HIRA_V1_S20_A0_IDENTITY_READY`

Fresh TRAIN/DEV:
- run `36677868762`
- artifact `11080892962`
- artifact digest `sha256:7f217c8bd86c4844328fd31b27ab6a97f1daa04e4186b5cd340112090320c220`
- scientific head `cb2c521a5d4d140991f9292f3da79819838cda51`
- outcome `HIRA_V1_S20_STANDARDIZED_TRIADIC_CONSISTENCY_DEV_FAIL`
- selected epoch **17**
- checkpoint SHA256 `dc87723defa224f012b513fcb679fe44794e0eb3a7d28cb0a10deb9f5772047d`

No post-DEV tuning.
No sealed confirmation.
No multilingual probe.
No Laya/Jev reopening.

## Frozen intervention

S20 returned to the S17 scientific frontier and changed only the cross-view consistency target.

Kept:
- S17 norm-balanced shared-gradient optimization
- S14 equal-weight full-K inference fusion
- relation CE/signature canonicalization
- option alignment
- exact A13 LoRA **16,384**
- shared projection **32,768**
- total trainable physical surface **49,152**
- 0 learned consistency/head/router parameters

Removed:
- S18 paired output-margin term
- S19 raw-softmax triadic JS

Added:
- exact S14 center/RMS standardization of raw triadic evidence;
- cross-view standardized evidence MSE coefficient **0.25**;
- standardization epsilon **1e-6**.

## A0 result

A0 proved that the S20 intervention is not inert.

Key invariants:
- inherited logit identity: **1.0**
- inherited choice identity: **1.0**
- inference max abs vs S14 family: **0.0**
- physical parameter surface: **49,152**
- runtime trainable parameters during A0: **0**
- standardized mismatch loss: **0.4353144765**
- canonical gradient L1: **0.8270463347**
- paraphrase gradient L1: **0.7869513631**
- positive-affine invariance max abs: **2.38e-7**
- common-scale invariance max abs: **8.94e-8**
- flat evidence -> exact zero neutral finite vector
- option permutation / view-swap invariants: PASS

This directly fixes the S19 measurement failure where raw-softmax JS was ~1e-8.

## Selected DEV — epoch 17

Fused:
- canonical accuracy: **0.5963541667**
- paraphrase accuracy: **0.4036458333**
- paired both-correct: **0.3385416667**
- question-swap choice-change: **0.734375**
- cross-view selected-choice agreement: **0.4583333333**
- cross-view mean JS: **0.0639626536**
- canonical signed margin: **+0.1902903815**
- paraphrase signed margin: **-0.2076632828**

Raw triadic:
- canonical accuracy: **0.5208333333**
- paraphrase accuracy: **0.3932291667**
- cross-view agreement: **0.5364583333**

Relation:
- canonical accuracy: **0.4427083333**
- paraphrase accuracy: **0.3515625**
- canonical signed margin: **-0.0772121996**
- paraphrase signed margin: **-0.4223929842**
- cross-view agreement: **0.5052083333**

Relation signatures:
- same-option cosine: **0.7979244739**
- same-vs-strongest-wrong margin: **0.0933141845**

Mechanical:
- full-K: PASS
- state-once: PASS
- relation delta: **0**
- option-order flip: **0**
- max probability-mass error: **1.7881393433e-07**
- exact trainable surface: **49,152**

## Gate result

PASS:
- fused canonical signed margin >= 0.15
- option-order flip <= 0.02
- probability-mass error <= 1e-6
- full-K/state-once/relation-delta-zero
- exact capacity/frozen invariants

FAIL:
- fused canonical >= 0.85
- paired >= 0.75
- question-swap >= 0.80
- fused cross-view agreement >= 0.95
- fused mean JS <= 0.05
- relation canonical >= 0.80
- relation signed margin >= 0.15
- same-option signature cosine >= 0.90
- signature discrimination margin >= 0.15

DEV_READY is not authorized.

## TRAIN dynamics

Epoch 1 -> 24:
- total loss: **1.7672188605 -> 0.8409315683**
- decision loss: **1.5031419595 -> 0.6814378686**
- relation-binding loss: **1.3884874831 -> 0.5131287264**
- canonicalization loss: **0.2866653974 -> 0.1342762240**
- primary block: **1.5853702923 -> 0.7694772619**
- relation block: **0.1818485629 -> 0.0714543072**

Crucially, standardized consistency is active:
- epoch 1: **0.0632902884**
- maximum: **0.1474452103** at epoch 13
- epoch 24: **0.0928988051**

Therefore S20 is not a failed-measurement repeat of S19.

Norm-balancing:
- mean conflict rate: **0.2873263889**
- rule: `equal_direction_relation_priority_projection`
- reference scale: arithmetic mean raw norm

## Fresh DEV extrema

- fused canonical best: **0.5963541667** at epoch 17
- fused paraphrase best: **0.5208333333** at epoch 9
- paired both-correct best: **0.3385416667** at epoch 17
- fused cross-view agreement best: **0.7239583333** at epoch 9
- question-swap best: **0.765625** at epoch 16
- raw-triadic agreement best: **0.8567708333** at epoch 1
- relation canonical best: **0.4921875** at epoch 15
- relation canonical signed margin best: **-0.0643456827** at epoch 15
- same-option signature cosine best: **0.9262867371** at epoch 1
- signature discrimination margin best: **0.1651247138** at epoch 7
- fused canonical signed margin best: **0.1919441468** at epoch 16
- fused JS minimum: **0.0141438896** at epoch 6

No epoch satisfies the semantic DEV_READY gate.

## Comparison to S17 frontier

S17 selected:
- fused canonical **0.7213541667**
- fused paraphrase **0.5546875**
- paired **0.5104166667**
- question-swap **0.984375**
- fused agreement **0.5286458333**
- fused canonical margin **+0.3138313380**
- raw triadic canonical/paraphrase **0.5911458333 / 0.4427083333**
- relation canonical/paraphrase **0.640625 / 0.6380208333**
- relation canonical margin **+0.2073315941**

S20 selected:
- fused canonical **0.5963541667**
- fused paraphrase **0.4036458333**
- paired **0.3385416667**
- question-swap **0.734375**
- fused agreement **0.4583333333**
- fused canonical margin **+0.1902903815**
- raw triadic canonical/paraphrase **0.5208333333 / 0.3932291667**
- relation canonical/paraphrase **0.4427083333 / 0.3515625**
- relation canonical margin **-0.0772121996**

S20 is a clear regression versus S17.

## Scientific conclusion

S20 rejects the hypothesis that insufficient cross-view consistency signal is the primary remaining bottleneck.

S19 showed:
- raw-softmax JS was effectively inert.

S20 fixed that measurement problem:
- standardized evidence consistency produced large, finite gradients;
- the term remained materially active throughout TRAIN.

Yet fresh semantic DEV regressed.

Therefore:

> **The remaining S17 failure is not explained by a lack of pressure for two wording views to produce similar triadic evidence. Forcing stronger view invariance can actively damage the semantic factorization that supported S17's relation/fused gains.**

The next experiment should not add another output/evidence consistency regularizer.

## Next controlled direction

S21 should return to the S17 frontier and target **semantic factorization / role-content binding generalization** directly.

The strongest next hypothesis must:
- preserve S17 norm-balanced optimization;
- preserve S17 inference;
- keep exact 49,152 trainable physical params unless a separately preregistered capacity hypothesis is justified;
- add no learned downstream head/router by default;
- explicitly train the representation to distinguish **field/role identity** from **field value/content**, rather than merely making two surface wordings look alike;
- use wholly fresh authority;
- preregister the exact role/content objective before A0/DEV.

Production-ready remains false.
Laya/Jev parity remains unestablished.
