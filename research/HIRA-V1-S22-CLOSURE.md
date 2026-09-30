# HIRA V1 S22 closure — Primary-Priority Norm-Balanced Optimization

Status: **CLOSED — DEV FAIL / PRIMARY PRIORITY DOES NOT RECOVER S21 CANONICAL PERFORMANCE AND DAMAGES RELATION SEPARATION**

Issue: #227  
PR: #228

## Authority

A0:
- run `36709243180`
- artifact `11093551516`
- digest `sha256:217b60410f9dc3e63d0c5d82dc75af6dadccdb2dc181ae7fbe2885f0eb6ac638`
- head `592ae0a70c1085c79b8f8d85c6f0b2be632da2dd`
- outcome `HIRA_V1_S22_A0_PRIMARY_PRIORITY_READY`

Canonical fresh TRAIN/DEV:
- run `36713837067`
- artifact `11095183683`
- digest `sha256:3329c0641159d9e96b3bff32d21b6413085b5c37fa1632b2ec0fa56ed779e14e`
- scientific head `081c01d732bafd73f7dfe9bda281e11d490276bf`
- outcome `HIRA_V1_S22_PRIMARY_PRIORITY_DEV_FAIL`
- selected epoch **24**
- checkpoint SHA256 `82361edc7f5bb82a364248aee471559d41c95ad9b34f93092edef7b8ccc36af0`

Pre-exposure harness aborts preserved separately:
- run `36710305829`: stale A0 outcome label before S22 case generation;
- run `36713186950`: stale A0 identity field names before S22 case generation.

Neither abort produced TRAIN/DEV result/artifact or DEV exposure.

No post-DEV tuning.
No sealed confirmation.
No multilingual probe.
No Laya/Jev reopening.

## Frozen controlled change

S22 kept S21 model/inference/data geometry/losses/capacity and changed only normalized gradient conflict priority.

On conflict:
- protect normalized primary direction;
- project normalized relation direction away from primary;
- combine and renormalize;
- scale by arithmetic mean raw gradient norm.

Frozen:
- epsilon **1e-12**
- exact physical trainable surface **49,152**
- role-gated content primary
- role temperature **0.10**
- role/content weights **0.50 / 0.50**
- S13 relation expert
- S14 equal-weight standardized fusion
- S17/S21 loss coefficients
- 0 new learned params/state

## A0 proof

Optimizer-only identity:
- S21 logit identity rate: **1.0**
- S21 choice identity rate: **1.0**

Controlled conflict:
- normalized pre-dot: **-0.7999999523**
- normalized post-dot: **+4.47e-08**
- analytical expected-direction max abs: **5.96e-08**
- relation direction change L1: **1.073312521**
- protected primary direction identity max abs: **0.0**
- no-conflict S22 vs S17 update max abs: **0.0**
- positive joint-scale equivariance max abs: **0.0**

Physical surface / S21 mechanism:
- exact **49,152**
- factorization added params **0**
- role/content synthetic court PASS
- inference/mechanics/full-K/state-once PASS

## Selected DEV — epoch 24

Fused:
- canonical accuracy: **0.5729166667**
- paraphrase accuracy: **0.546875**
- paired both-correct: **0.28125**
- question-swap choice-change: **0.5729166667**
- cross-view selected-choice agreement: **0.6796875**
- cross-view mean JS: **0.0388114705**
- canonical signed margin: **-0.0605145351**
- paraphrase signed margin: **+0.0601067841**

Primary role-gated content expert:
- canonical accuracy: **0.5052083333**
- paraphrase accuracy: **0.5442708333**
- cross-view agreement: **0.6953125**

Relation expert:
- canonical accuracy: **0.6015625**
- paraphrase accuracy: **0.5520833333**
- canonical signed margin: **+0.0338486681**
- paraphrase signed margin: **+0.0481495907**
- cross-view agreement: **0.65625**

Signatures:
- same-option cosine: **0.8385203779**
- same-vs-strongest-wrong margin: **0.0534842378**

Expert interaction:
- canonical expert top-1 agreement: **0.8098958333**
- paraphrase expert top-1 agreement: **0.8828125**

Mechanical:
- full-K PASS
- state-once PASS
- relation delta **0**
- option-order flip **0**
- max fused probability-mass error **1.1920928955e-07**
- exact trainable surface **49,152**

## Gate result

PASS:
- fused mean JS <= 0.05
- mechanics/capacity/frozen gates

FAIL:
- fused canonical >= 0.85
- paired >= 0.75
- question-swap >= 0.80
- fused agreement >= 0.95
- fused canonical margin >= 0.15
- relation canonical >= 0.80
- relation margin >= 0.15
- signature cosine >= 0.90
- signature margin >= 0.15

DEV_READY is not authorized.

## TRAIN dynamics

Epoch 1 -> 24:
- total loss: **2.0079146673 -> 0.7142906189**
- decision loss: **1.7437711631 -> 0.5540641819**
- fused CE: **1.5311444402 -> 0.5539039013**
- swap loss: **0.8505069638 -> 0.0006411286**
- option alignment: **1.3372881462 -> 1.3207484956**
- relation-binding loss: **1.3828186144 -> 0.6934820972**
- canonicalization loss: **0.3261756124 -> 0.1575701398**
- fused consistency JS: **0.0402835375 -> 0.0048211352**
- primary block: **1.8207064619 -> 0.6213068912**
- relation block: **0.1872082092 -> 0.0929837332**

Training learns strongly, but fresh semantic DEV does not reach S21/S17 quality.

## Gradient-conflict diagnostics

Conflict rate:
- epoch 1: **0.6458333333**
- minimum: **0.0833333333** at epochs 11/14
- selected epoch 24: **0.7083333333**
- mean: **0.5**

Selected epoch 24:
- primary norm: **2.1886563649**
- relation norm: **0.5512624482**
- normalized pre-dot: **-0.2038679257**
- normalized post-dot: **+0.1864412293**
- projection coefficient: **-0.3903091506**

Primary-priority does reduce average conflict activation versus S21 (~0.600 -> 0.500) but does not improve the semantic result.

## Fresh DEV extrema

- fused canonical best: **0.59375** at epoch 21
- fused paraphrase best: **0.5572916667** at epoch 19
- paired best: **0.28125** at epoch 24
- question-swap best: **0.6354166667** at epoch 21
- fused agreement best: **0.7734375** at epoch 1
- fused JS minimum: **0.0167599258** at epoch 1
- fused canonical margin best: **+0.0927376735** at epoch 18
- primary canonical best: **0.5364583333** at epoch 21
- primary paraphrase best: **0.5598958333** at epoch 16
- primary agreement best: **0.7552083333** at epoch 1
- relation canonical best: **0.6197916667** at epoch 18
- relation paraphrase best: **0.5677083333** at epoch 21
- relation canonical margin best: **+0.0922444065** at epoch 18
- relation paraphrase margin best: **+0.0570655006** at epoch 21
- signature cosine best: **0.9211881757** at epoch 11
- signature margin best: **0.1733834445** at epoch 10

No epoch approaches the full DEV_READY requirement.

## Comparison

### S17 frontier

- fused canonical **0.7213541667**
- paraphrase **0.5546875**
- paired **0.5104166667**
- question-swap **0.984375**
- fused margin **+0.3138313380**
- relation canonical **0.640625**
- relation margin **+0.2073315941**

### S21 relation-priority + factorized primary

- fused canonical **0.6145833333**
- paraphrase **0.5651041667**
- paired **0.3697916667**
- question-swap **0.8333333333**
- fused agreement **0.6588541667**
- primary canonical/paraphrase **0.5807291667 / 0.5182291667**
- relation canonical **0.6484375**
- relation margin **+0.2356135895**
- mean conflict **0.5998263889**

### S22 primary-priority + same factorized primary

- fused canonical **0.5729166667**
- paraphrase **0.546875**
- paired **0.28125**
- question-swap **0.5729166667**
- fused agreement **0.6796875**
- primary canonical/paraphrase **0.5052083333 / 0.5442708333**
- relation canonical **0.6015625**
- relation margin **+0.0338486681**
- mean conflict **0.5**

Primary-priority is a regression versus S21 on canonical/fused/relation separation and does not recover S17.

## Scientific conclusion

S22 rejects the hypothesis that S21's canonical deficit is primarily caused by relation-priority conflict projection suppressing the factorized primary expert.

Protecting primary instead:
- does not recover canonical or paired accuracy;
- severely weakens relation signed separation;
- reduces question-swap sensitivity;
- still leaves substantial gradient conflict.

Thus neither asymmetric priority extreme is sufficient evidence for a better optimizer.

One clean optimizer ablation remains before leaving this family: **no asymmetric conflict projection at all** after norm equalization.

## Next controlled hypothesis

S23 — **Neutral Bisector Norm-Balanced Optimization**

Keep the entire S21 role/content architecture and objective.

For nonzero gradients:
- `u_p = g_p / ||g_p||`
- `u_r = g_r / ||g_r||`
- do **not** project either expert, regardless of sign of dot product;
- `d = normalize(u_p + u_r)`
- `s = 0.5*(||g_p|| + ||g_r||)`
- `g = s*d`

Zero-gradient behavior stays inherited.

This is symmetric under exchanging primary/relation and tests whether asymmetric conflict surgery itself causes the S21/S22 trade-off.

If S23 does not materially improve the S17/S21 frontier, close the optimizer-priority family and return to representation/fusion research.

Production-ready remains false.
Laya/Jev parity remains unestablished.
