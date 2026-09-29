# HIRA V1 S18 closure — Paired Both-View Margin Consistency

Status: **CLOSED — DEV FAIL / TRAIN PAIRED HINGE LEARNS, FRESH VIEW GENERALIZATION COLLAPSES**

Issue: #217  
PR: #218

## Authority

A0:
- run `36582445491`
- artifact `11040940678`
- digest `sha256:cb78b283fbcf1514e88aa8ad08f35c2bcad0a47a90302a8fd97e47640a3af452`
- outcome `HIRA_V1_S18_A0_IDENTITY_READY`

Fresh TRAIN/DEV:
- run `36583384787`
- artifact `11042370479`
- digest `sha256:b97df1baf7b5a23ed04cab55f6f815ce757a32004da2df039c29bdbd0f57bc8c`
- outcome `HIRA_V1_S18_PAIRED_VIEW_MARGIN_DEV_FAIL`
- selected epoch **10**
- checkpoint SHA256 `bf4e378c3fd14ce6be2b0bb65b176aa095c58fdeecd2fad27d593063899368dc`

No post-DEV tuning.
No sealed confirmation.
No multilingual probe.
No Laya/Jev reopening.

## Frozen surface / intervention

S18 retained:
- S17 norm-balanced shared-gradient optimization
- S14/S17 equal-weight inference fusion
- A13 LoRA **16,384**
- shared projection **32,768**
- total trainable **49,152**
- original A13 frozen
- HIRACore frozen
- 0 learned paired/consensus/router parameters

Added only:

`L_pair = 0.5 * [relu(0.20 - m_c) + relu(0.20 - m_p)]`

with fixed coefficient **0.25** in the primary block.

## Selected DEV — epoch 10

Fused:
- canonical accuracy: **0.703125**
- paraphrase accuracy: **0.4114583333**
- paired both-correct: **0.5208333333**
- question-swap choice-change: **0.78125**
- cross-view selected-choice agreement: **0.4739583333**
- cross-view mean JS: **0.0413906077**
- canonical signed margin: **0.2828457902**
- paraphrase signed margin: **-0.1637737309**

Raw triadic:
- canonical accuracy: **0.5078125**
- paraphrase accuracy: **0.34375**
- cross-view agreement: **0.4791666667**

Relation:
- canonical accuracy: **0.421875**
- paraphrase accuracy: **0.390625**
- canonical signed margin: **-0.0914296707**
- paraphrase signed margin: **-0.1773813193**
- cross-view agreement: **0.6354166667**

Relation signatures:
- same-option cosine: **0.9314097762**
- same-vs-strongest-wrong margin: **0.1460411654**

Mechanical:
- option-order flip: **0.0**
- max probability-mass error: **1.7881393433e-07**
- full-K/state-once/relation-delta-zero: PASS

## TRAIN dynamics

Epoch 1 -> 24:
- total loss: **1.8444628442 -> 0.8242938829**
- paired margin loss: **0.4779098456 -> 0.0725741908**
- relation-binding loss: **1.3923915749 -> 0.7257500415**
- canonicalization loss: **0.2825087278 -> 0.1016922978**
- primary block: **1.6628473749 -> 0.7364650269**
- relation block: **0.1816154715 -> 0.0878288516**

The paired hinge is therefore highly learnable on TRAIN.

## Fresh DEV dynamics

Key extrema:
- paired both-correct best: **0.5208333333** at epoch 10
- fused canonical best: **0.703125** at epoch 10
- fused paraphrase best: **0.515625** at epoch 7
- fused cross-view agreement best: **0.6432291667** at epoch 5
- question-swap best: **0.9635416667** at epoch 21
- canonical relation accuracy best: **0.6276041667** at epochs 23/24
- canonical relation margin best: **+0.1708478009** at epoch 24
- same-option signature cosine best: **0.9527875185** at epoch 5
- signature margin best: **0.1793983920** at epoch 7

But later optimization becomes strongly view-asymmetric.

Epoch 24:
- fused canonical: **0.6354166667**
- fused paraphrase: **0.296875**
- canonical relation: **0.6276041667**
- paraphrase relation: **0.2760416667**
- canonical relation margin: **+0.1708478009**
- paraphrase relation margin: **-1.2794760962**
- fused cross-view agreement: **0.3333333333**

## Comparison to S17

S17 selected:
- fused canonical: **0.7213541667**
- fused paraphrase: **0.5546875**
- paired both-correct: **0.5104166667**
- fused cross-view agreement: **0.5286458333**
- relation canonical: **0.640625**
- relation paraphrase: **0.6380208333**
- relation canonical margin: **+0.2073315941**

S18 selected:
- fused canonical: **0.703125**
- fused paraphrase: **0.4114583333**
- paired both-correct: **0.5208333333**
- fused cross-view agreement: **0.4739583333**
- relation canonical: **0.421875**
- relation paraphrase: **0.390625**
- relation canonical margin: **-0.0914296707**

The paired gain is only **+1.04 percentage points**, while paraphrase and relation quality regress sharply.

## Preregistered interpretation

S18 matches primarily **Outcome D**:
- TRAIN paired loss improves strongly;
- fresh DEV paired quality does not materially improve;
- fresh paraphrase quality degrades.

It also rejects the useful form of Outcome A: paired correctness does not improve materially while S17 discrimination is preserved.

## Scientific conclusion

S18 rejects the hypothesis that an output-level paired gold-margin hinge is sufficient to solve S17's cross-view inconsistency.

The key evidence from S17 remains more informative:
- relation canonical/paraphrase were already nearly symmetric (**64.06% / 63.80%**);
- raw triadic canonical/paraphrase were much more asymmetric (**59.11% / 44.27%**);
- fused agreement followed the weak triadic branch.

Therefore the residual cross-view instability after S17 is most plausibly concentrated in the **triadic expert**, while the S18 fused hinge perturbs the whole shared surface and damages relation generalization.

## Next controlled hypothesis

S19 should return to the **S17 norm-balanced objective** and directly regularize triadic cross-view consistency instead of keeping the failed S18 paired hinge.

A strong controlled test is **expert-separated cross-view consistency**:
- remove S18 paired-margin term;
- retain S17 relation-signature canonicalization;
- move the primary cross-view JS target from fused logits to raw triadic logits, with the same fixed coefficient **0.25**;
- inference remains unchanged;
- total trainable surface remains **49,152**;
- add 0 learned parameters.

This directly targets the branch that was asymmetric in S17 without forcing the already-strong relation expert through a fused-output hinge.

Production-ready remains false.
Laya/Jev parity remains unestablished.
