# HIRA V1 S23 closure — Neutral Bisector Norm-Balanced Optimization

Status: **CLOSED — DEV FAIL / OPTIMIZER-PRIORITY FAMILY EXHAUSTED**

Issue: #229  
PR: #230

## Authority

A0:
- run `36721891033`
- artifact `11101960494`
- digest `sha256:bb4fed5838c5c2ad0ad10e3c20bcd89d47f1e38317cabecf427b2c0321f9798a`
- head `67bb65ce3416a6985f624768f9196c007a214a05`
- outcome `HIRA_V1_S23_A0_NEUTRAL_BISECTOR_READY`

Fresh TRAIN/DEV:
- run `36730706512`
- artifact `11106195105`
- digest `sha256:61a6b8ce293b6f43c6bcc6bcf8da2cf57950092c05eeb37009417d4a94a078af`
- scientific head `2ec1ce302ba2476c6afa4a08a6c4aecefa24e79d`
- outcome `HIRA_V1_S23_NEUTRAL_BISECTOR_DEV_FAIL`
- selected epoch **15**
- checkpoint SHA256 `f1ff5234b9e055843fa5ef2653a90abedfab21db07ca4c87e741440e52558d04`

Pre-authority A0 abort:
- run `36720789774`
- diagnostic `NameError: math` before receipt/artifact;
- not scientific evidence.

No post-DEV tuning.
No sealed confirmation.
No multilingual probe.
No Laya/Jev reopening.

## Frozen intervention

S23 retained the full S21 role/content architecture and all S17/S21 losses.

Only optimizer conflict handling changed:
- normalize primary and relation gradients independently;
- apply **no asymmetric projection**;
- neutral direction = normalize(`u_p + u_r`);
- scale = arithmetic mean raw gradient norm;
- epsilon **1e-12**.

Physical trainable surface remained exactly **49,152**.

## A0 result

Neutral-bisector court PASS:
- S21 logit identity **1.0**
- S21 choice identity **1.0**
- conflict pre-dot **-0.7999999523**
- conflict post-dot **-0.7999999523**
- projection coefficient **0.0**
- analytical direction error **0.0**
- exchange symmetry error **0.0**
- no-conflict vs S17 error **0.0**
- joint-scale error **0.0**
- near-opposite finite
- exact zero/single-expert cases
- role/content synthetic court preserved
- exact 49,152 physical surface.

## Selected DEV — epoch 15

Fused:
- canonical accuracy: **0.6432291667**
- paraphrase accuracy: **0.609375**
- paired both-correct: **0.4166666667**
- question-swap choice-change: **0.75**
- cross-view selected-choice agreement: **0.6770833333**
- cross-view mean JS: **0.0387389847**
- canonical signed margin: **+0.0783118053**
- paraphrase signed margin: **+0.0223363129**

Primary role/content expert:
- canonical accuracy: **0.5651041667**
- paraphrase accuracy: **0.5625**
- cross-view agreement: **0.6666666667**

Relation expert:
- canonical accuracy: **0.6875**
- paraphrase accuracy: **0.6276041667**
- canonical signed margin: **+0.1510315314**
- paraphrase signed margin: **+0.1357324384**
- cross-view agreement: **0.625**

Signatures:
- same-option cosine: **0.8025678347**
- same-vs-strongest-wrong margin: **0.0245468643**

Expert agreement:
- canonical top-1 agreement: **0.8125**
- paraphrase top-1 agreement: **0.84375**

Mechanical:
- full-K PASS
- state-once PASS
- relation delta **0**
- option-order flip **0**
- fused mass error **1.7881393433e-07**
- exact trainable surface **49,152**

## Gate result

PASS:
- fused mean JS <= 0.05
- relation canonical margin >= 0.15
- option-order / mass / full-K / state-once / capacity gates

FAIL:
- fused canonical >= 0.85
- paired >= 0.75
- question-swap >= 0.80
- fused agreement >= 0.95
- fused canonical margin >= 0.15
- relation canonical >= 0.80
- signature cosine >= 0.90
- signature discrimination margin >= 0.15

DEV_READY is not authorized.

## TRAIN dynamics

Epoch 1 -> 24:
- total loss: **2.1227133920 -> 0.7522931832**
- decision loss: **1.8617932424 -> 0.6030279038**
- relation binding: **1.3852356374 -> 0.5393575492**
- canonicalization: **0.3067044203 -> 0.1822889702**
- fused consistency JS: **0.0349881616 -> 0.0026079047**
- primary block: **1.9381841545 -> 0.6710140804**
- relation block: **0.1845292334 -> 0.0812791021**

Selected epoch 15:
- primary grad norm **1.4524126177**
- relation grad norm **0.6008722242**
- normalized pre/post dot **-0.3225354925**
- conflict rate **0.7291666667**
- projection coefficient **0.0**

## Conflict evidence

Conflict rate by epoch:
- epoch 1: **0.50**
- minimum: **0.1875** at epochs 6/7
- selected epoch 15: **0.7291666667**
- epoch 20: **0.8958333333**
- epoch 22: **0.8958333333**
- epoch 24: **0.9375**
- mean: **0.6545138889**

At epoch 24:
- normalized pre/post dot **-0.4557608059**
- no projection is applied.

Thus high conflict is intrinsic to the current shared surface, not an artifact of which expert receives conflict priority.

## Fresh DEV extrema

- fused canonical best: **0.6432291667** at epoch 15
- fused paraphrase best: **0.6354166667** at epoch 11
- paired best: **0.4166666667** at epoch 15
- question-swap best: **0.75** at epoch 15
- fused agreement best: **0.6953125** at epoch 7
- fused JS minimum: **0.0347947692** at epoch 7
- fused canonical margin best: **+0.0783118053** at epoch 15
- primary canonical best: **0.5703125** at epoch 11
- primary paraphrase best: **0.5963541667** at epoch 8
- primary agreement best: **0.7239583333** at epoch 7
- relation canonical best: **0.6875** at epoch 15
- relation paraphrase best: **0.6510416667** at epoch 11
- relation canonical margin best: **+0.1692667343** at epoch 21
- relation paraphrase margin best: **+0.2059508326** at epoch 11
- signature cosine best: **0.9222907623** at epoch 6
- signature discrimination margin best: **0.1015788831** at epoch 5

## Comparison

S17:
- fused canonical **0.7213541667**
- paraphrase **0.5546875**
- paired **0.5104166667**
- question-swap **0.984375**
- fused margin **+0.3138313380**
- relation canonical **0.640625**
- relation margin **+0.2073315941**

S21 relation-priority:
- fused canonical **0.6145833333**
- paraphrase **0.5651041667**
- paired **0.3697916667**
- relation canonical **0.6484375**
- relation margin **+0.2356135895**

S22 primary-priority:
- fused canonical **0.5729166667**
- paraphrase **0.546875**
- paired **0.28125**
- relation canonical **0.6015625**
- relation margin **+0.0338486681**

S23 neutral:
- fused canonical **0.6432291667**
- paraphrase **0.609375**
- paired **0.4166666667**
- relation canonical **0.6875**
- relation margin **+0.1510315314**

S23 is the strongest factorized-primary optimizer variant overall, but it still does not recover S17 canonical/paired performance.

## Scientific conclusion

The optimizer-priority family is exhausted.

Evidence:
1. relation-priority (S21), primary-priority (S22), and neutral bisector (S23) all fail DEV_READY;
2. S23 removes asymmetric projection entirely yet conflict remains extremely high;
3. changing conflict priority changes the trade-off but does not solve the shared representation bottleneck.

The next track must not be another gradient-priority/projection variant.

A second clear signal emerges from S23:

> At selected DEV, relation accuracy (**68.75%**) is higher than equal-weight fused accuracy (**64.32%**), while primary is **56.51%**.

The fixed S14 50/50 fusion is now capable of suppressing the stronger expert on the factorized-primary line.

## Next controlled direction

S24 should enter **fusion research**, not optimizer research.

Proposed hypothesis:
**parameter-free reliability-weighted full-K fusion**.

Keep S23 model/optimizer/capacity fixed.

For each expert:
1. center/RMS-standardize full-K evidence as S14;
2. compute a permutation-equivariant reliability from its standardized top-1 vs top-2 gap;
3. normalize the two nonnegative reliabilities to weights summing to 1;
4. fuse the two standardized vectors with those deterministic weights.

Required properties:
- 0 learned params/state;
- expert-swap symmetry;
- option-permutation equivariance;
- equal reliabilities reduce exactly to S14 equal fusion;
- flat/zero evidence is neutral and finite;
- full-K retained;
- no external calibration/fitting;
- S23 inference differs only at fusion.

Use wholly fresh S24 authority.

Production-ready remains false.
Laya/Jev parity remains unestablished.
