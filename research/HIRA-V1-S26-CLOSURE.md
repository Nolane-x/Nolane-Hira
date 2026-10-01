# HIRA V1 S26 closure — Factorized Role-Value Relation Signatures

Status: **CLOSED — DEV FAIL / FACTORIZATION IMPROVES CANONICAL MARGIN BUT DOES NOT TRANSPORT ACROSS WORDING VIEWS**

Issue: #235  
PR: #236

## Authority

Canonical A0:
- run `36849791728`
- artifact `11155305741`
- artifact digest `sha256:1cfe67edb48649a3cf7931403f1b3abd50b47d0030f05c0242a18b11f52564be`
- authority head `0df6f21b3e5b3a065e19a83c975e3ed3d4da7411`
- outcome `HIRA_V1_S26_A0_FACTORIZED_RELATION_READY`

Fresh TRAIN/DEV:
- run `36851831847`
- artifact `11155554912`
- artifact digest `sha256:a64bdbcbb0e217fc454f2a15653b5374bc0110ed2d05452e27f7a9b1158bd349`
- scientific head `a2480be6577e0da29583e648af6193eae38f3af0`
- outcome `HIRA_V1_S26_FACTORIZED_RELATION_DEV_FAIL`
- selected epoch **17**
- checkpoint SHA256 `ba99c7e4c40ccc85fd2577a1eec636138f1277556068945f296d5f09b0db88c5`

Pre-DEV staged head:
- `c54af655536fbf6c897bb59d932a6c4c88cc9d8a`
- generic CI run `36851367529`: Python 3.10 PASS / Python 3.12 PASS / preflight PASS

Authorization-head generic CI:
- run `36851837104`: PASS

No second DEV run.
No post-DEV tuning.
No sealed confirmation.
No multilingual probe.
No Laya/Jev reopening.

## Frozen intervention

S26 changed the **relation representation/scoring rule only**.

Inherited S25 surface:
- shared A13 LoRA **16,384**
- primary-private projection **32,768**
- relation-private projection **32,768**
- exact trainable surface **81,920**
- original A13 frozen
- HIRACore frozen
- S21 primary unchanged
- S14 equal standardized full-K fusion unchanged
- S25 gradient ownership unchanged

S26 factorized relation:
- learned params **0**
- factorized signature width **256**
- explicit role-relation block
- explicit value/content-relation block
- fixed role/value scoring weights **0.50 / 0.50**
- option-independent state value extraction
- option-local value extraction
- signature = normalized concatenation of role delta + value delta

## A0 result

The factorized mechanism is real and mechanically valid:
- relation intervention max abs **5.0234084129**
- correct vs same-role/wrong-value margin **+5.0**
- correct vs wrong-role/same-value margin **+9.9998397827**
- correct vs wrong-role/wrong-value margin **+10.0**
- exact factorized signature width **256**
- operator params **0**
- primary → relation-private gradient **0**
- relation → primary-private gradient **0**
- both private gradients nonzero
- both shared-LoRA gradients nonzero
- option permutation exact for primary/relation/signature
- mapped fused option-order flip **0**
- full-K/state-once PASS

Thus S26 DEV failure is not an inactive intervention or broken court.

## Selected DEV — epoch 17

Fused:
- canonical accuracy **0.609375**
- paraphrase accuracy **0.3307291667**
- paired both-correct **0.3072916667**
- question-swap choice-change **0.6145833333**
- cross-view selected-choice agreement **0.2734375**
- cross-view mean JS **0.1250924325**
- canonical signed margin **+0.1865183748**
- paraphrase signed margin **-0.5469577868**

Primary:
- canonical accuracy **0.4765625**
- paraphrase accuracy **0.4166666667**
- cross-view agreement **0.2005208333**

Relation:
- canonical accuracy **0.5416666667**
- paraphrase accuracy **0.2135416667**
- canonical signed margin **+0.0418145005**
- paraphrase signed margin **-0.7501472371**
- cross-view agreement **0.2057291667**

Factorized signatures:
- same-option cosine **0.5707240601**
- same-vs-strongest-wrong margin **0.0249184022**

Mechanical:
- full-K PASS
- state-once PASS
- relation delta **0**
- option-order flip **0**
- probability-mass error **1.1920929e-7**
- exact physical surface **81,920**
- factorized relation operator params **0**

## Gate result

PASS **19 / 28**:
- fused canonical signed margin >= **0.15**
- full-K
- option-order
- probability mass
- zero fusion params
- zero S26 relation-operator params
- exact 256D factorized signature
- exact 81,920 total surface
- exact LoRA / primary-private / relation-private surfaces
- private projection storage distinct
- cross-private leakage **0 / 0**
- relation isolation
- state-once train/dev
- original A13 frozen
- HIRACore frozen

FAIL **9 / 28**:
- fused canonical >= **0.85**
- paired >= **0.75**
- question-swap >= **0.80**
- fused cross-view agreement >= **0.95**
- fused cross-view JS <= **0.05**
- relation canonical >= **0.80**
- relation canonical margin >= **0.15**
- same-option signature cosine >= **0.90**
- signature discrimination margin >= **0.15**

DEV_READY is not authorized.

## Fresh DEV extrema

Across 24 epochs:
- fused canonical best **0.6119791667** at epoch **21**
- fused paraphrase best **0.5911458333** at epoch **9**
- paired best **0.3072916667** at epoch **17**
- question-swap best **0.6145833333** at epoch **17**
- fused agreement best **0.375** at epoch **11**
- fused JS minimum **0.0842635958** at epoch **1**
- fused canonical margin best **+0.1865183748** at epoch **17**
- fused paraphrase margin best **-0.0488879737** at epoch **9**
- relation canonical best **0.5651041667** at epoch **24**
- relation paraphrase best **0.4505208333** at epoch **8**
- relation canonical margin best **+0.0795259426** at epoch **24**
- relation paraphrase margin best **-0.0218745681** at epoch **9**
- relation cross-view agreement best **0.3776041667** at epoch **1**
- same-option signature cosine best **0.6428080897** at epoch **9**
- signature discrimination margin best **0.0398570701** at epoch **24**

No epoch approaches the complete DEV_READY gate set.

## Optimization dynamics

Training is healthy:

Epoch 1 → 24:
- total loss **1.7042866473 → 0.7012357600**
- decision loss **1.4389918372 → 0.5545007090**
- relation binding loss **1.2521387438 → 0.5365756272**
- canonicalization loss **0.4114237173 → 0.1630000764**
- fused consistency JS **0.0414977422 → 0.0027224146**

Gradient ownership remains healthy:
- minimum primary-private gradient L1 **46.5480728149**
- minimum relation-private gradient L1 **9.9770936966**
- minimum primary shared-LoRA gradient L1 **2.0867624283**
- minimum relation shared-LoRA gradient L1 **0.9196392298**
- cross-private leakage **0 / 0**
- mean shared-LoRA conflict rate **0.3949652778**

The model learns the frozen TRAIN objective strongly. This is not an optimizer crash or dead-surface failure.

## Scientific interpretation

S26 rejects the claim that role/value factorization by itself solves fresh semantic binding.

It does establish an important positive result:
- the selected fused canonical signed margin is **+0.1865**, enough to pass its frozen margin gate;
- canonical relation margin also becomes positive;
- the factorized relation geometry is trainable and discriminative on one wording regime.

The dominant failure is now **cross-view transport**.

Evidence:
- selected relation canonical **54.17%** vs paraphrase **21.35%**;
- selected relation margin **+0.0418** vs paraphrase **-0.7501**;
- relation cross-view agreement only **20.57%**;
- same-option signature cosine only **0.5707**;
- fused cross-view agreement only **27.34%**;
- fused JS **0.1251**;
- the best paraphrase relation margin across every epoch remains negative (**-0.0219**).

There is also a strong temporal trade-off:
- paraphrase metrics peak around epochs 8–9;
- canonical relation/fused metrics continue improving toward epochs 17–24;
- continued training therefore sharpens one view while the equivalent wording view drifts away.

This is evidence that the role/value factor blocks themselves are **not being canonicalized independently across wording views**. The current whole-signature canonicalization objective can reduce TRAIN loss without maintaining stable role-block and value-block semantics on fresh paraphrases.

Cross-track raw percentages use different fresh authorities and must not be treated as paired same-row deltas. The valid conclusion is within S26: canonical discrimination emerges while cross-view semantic transport remains decisively below gate.

## Track closure

Close:
**factorized role/value relation representation as a sufficient solution**.

Do not create S26b by:
- changing role/value weights;
- changing role/pair/contrastive temperatures;
- increasing projection width;
- changing fusion;
- retrying seed/LR/epochs;
- weakening cross-view gates.

## Next controlled direction

S27 should keep S26 inference **exactly unchanged** and change only the canonicalization objective.

Proposed track:
**S27 — Blockwise Cross-View Factor Canonicalization**

Controlled hypothesis:
- S26's 256D signature contains two semantically distinct 128D blocks;
- aligning the concatenated vector as one unit permits role/value factors to compensate or drift across wording views;
- align and separate the role block and value block independently.

Frozen candidate:
- exact S26 81,920 trainable surface;
- exact S26 factorized inference/logits;
- exact S14 fusion;
- exact S25 gradient ownership;
- no new learned params/state.

S27 canonicalization loss:
- split the factorized signature into role block + value block;
- apply the same cross-view alignment/separation rule to each block independently;
- combine block losses with fixed **0.50 / 0.50** weighting;
- keep the overall canonicalization coefficient **0.15** unchanged;
- separation margin **0.20** unchanged.

A0 must prove:
- inference identity vs S26 exactly;
- blockwise loss zero/near-zero on identical views;
- blockwise loss activates when role changes with value fixed;
- blockwise loss activates when value changes with role fixed;
- option permutation;
- no new params;
- gradients reach relation-private + shared LoRA without cross-private leakage.

Production-ready remains false.
Laya/Jev parity remains unestablished.
