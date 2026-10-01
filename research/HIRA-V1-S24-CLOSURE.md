# HIRA V1 S24 closure — Reliability-Weighted Full-K Expert Fusion

Status: **CLOSED — DEV FAIL / TOP2-GAP RELIABILITY RULE REJECTED**

Issue: #231  
PR: #232

## Authority

Canonical A0:
- run `36793018851`
- artifact `11133405764`
- digest `sha256:4eed0771f2e2037a9cccbee547ffc7831140e53781256ff2b4dd9ce1c67810b3`
- exact head `970768c70d3b558a03258863b4c6f3ae09dfe02d`
- outcome `HIRA_V1_S24_A0_RELIABILITY_FUSION_READY`

Fresh TRAIN/DEV:
- run `36819995046`
- artifact `11143228027`
- artifact digest `sha256:e6437365d3b1f94bffa390f676d6516a8839cb843f4e7fca54f926ac89cf1775`
- scientific head `2a33d54026de879bdb717e9e095cc876238edcb6`
- outcome `HIRA_V1_S24_RELIABILITY_FUSION_DEV_FAIL`
- selected epoch **22**
- checkpoint SHA256 `b4fc574c2b1058172642bd10b41c4e18b8404952a1e08fa55d12b8c7a872eb08`

Pre-DEV staged head:
- `129902a930b8698d1c2706e28ac1cf7f57fb6838`
- generic CI run `36819677570`: Python 3.10 PASS / Python 3.12 PASS / preflight PASS

Authorization-head generic CI:
- run `36819998701`: PASS

No post-DEV tuning.
No second DEV run.
No sealed confirmation.
No multilingual probe.
No Laya/Jev reopening.

## Frozen intervention

S24 changed fusion only.

Frozen from S23:
- S21 role-gated content primary
- S13 relation expert/canonicalizer
- role temperature **0.10**
- role/content weights **0.50 / 0.50**
- neutral-bisector norm-balanced optimizer
- AdamW lr **2e-4**
- weight decay **0.01**
- 24 epochs
- batch size **16**
- exact physical trainable surface **49,152**
- A13 LoRA **16,384**
- shared projection **32,768**

S24 fusion:
- center/RMS-standardize both full-K expert logits
- reliability = standardized top1-top2 gap + **1e-6**
- normalize the two reliabilities to weights summing to one
- weighted full-K sum
- relation detached from fused-primary objective
- learned fusion params/state **0**

## A0 result

The operator itself is mechanically valid:
- S23 primary logits/choices identity **1.0 / 1.0**
- S23 relation logits/choices identity **1.0 / 1.0**
- expert-swap error **0**
- option-permutation error **0**
- equal reliability -> exact S14 **0.5 / 0.5**
- equal-gap vs S14 max abs **0**
- flat/flat neutral exact
- flat/nonflat relation weight **0.9999992847**
- positive affine invariance error **5.9604645e-8**
- weight-sum error **5.9604645e-8**
- fusion parameters **0**
- fused mass error **1.1920929e-7**
- full-K/state-once/isolation preserved
- `fusion_vs_s14_forward_max_abs = 0.7367611527`

Thus S24 is a real fusion intervention, not a refactor or broken court.

## Selected DEV — epoch 22

Fused:
- canonical accuracy: **0.390625**
- paraphrase accuracy: **0.3151041667**
- paired both-correct: **0.0572916667**
- question-swap choice-change: **0.2239583333**
- cross-view selected-choice agreement: **0.4713541667**
- cross-view mean JS: **0.0962312045**
- canonical signed margin: **-0.4352511764**
- paraphrase signed margin: **-0.7936199997**

Primary role/content expert:
- canonical accuracy: **0.3619791667**
- paraphrase accuracy: **0.2890625**
- cross-view agreement: **0.4557291667**

Relation expert:
- canonical accuracy: **0.4010416667**
- paraphrase accuracy: **0.296875**
- canonical signed margin: **-0.1124879544**
- paraphrase signed margin: **-0.2615829383**
- cross-view agreement: **0.4322916667**

Signatures:
- same-option cosine: **0.8084561278**
- same-vs-strongest-wrong margin: **0.0858259757**

Expert agreement:
- canonical top-1 agreement: **0.7213541667**
- paraphrase top-1 agreement: **0.8046875**

Mechanical:
- full-K PASS
- state-once PASS
- relation delta **0**
- option-order flip **0**
- fused mass error **2.3841858e-7**
- exact trainable surface **49,152**

## Gate result

PASS **12 / 22**:
- full-K
- option-order
- probability mass
- zero fusion params
- HIRACore frozen
- original A13 frozen
- relation isolation
- state-once train/dev
- exact LoRA **16,384**
- exact projection **32,768**
- exact total **49,152**

FAIL **10 / 22**:
- fused canonical >= **0.85**
- paired >= **0.75**
- question-swap >= **0.80**
- fused agreement >= **0.95**
- fused JS <= **0.05**
- fused canonical margin >= **0.15**
- relation canonical >= **0.80**
- relation canonical margin >= **0.15**
- signature cosine >= **0.90**
- signature discrimination margin >= **0.15**

DEV_READY is not authorized.

## Fresh DEV extrema

Across all 24 epochs:
- fused canonical best: **0.4296875** at epoch **18**
- fused paraphrase best: **0.3828125** at epoch **8**
- paired best: **0.0572916667** at epoch **22**
- question-swap best: **0.2552083333** at epoch **24**
- fused agreement best: **0.6875** at epoch **7**
- fused canonical margin best: **-0.3642630999** at epoch **18**
- primary canonical best: **0.4010416667** at epoch **3**
- primary paraphrase best: **0.3776041667** at epoch **2**
- relation canonical best: **0.4375** at epoch **20**
- relation paraphrase best: **0.3697916667** at epoch **3**
- relation canonical margin best: **-0.0571382120** at epoch **1**
- relation paraphrase margin best: **-0.0927978531** at epoch **1**
- signature cosine best: **0.8174792528** at epoch **20**
- signature discrimination margin best: **0.1069177647** at epoch **14**

No epoch approaches DEV_READY.

## TRAIN dynamics

Epoch 1 -> 24:
- total loss: **2.1084498093 -> 0.7987292434**
- decision loss: **1.8482493758 -> 0.6216041545**
- relation binding loss: **1.3898478970 -> 0.9206026209**
- canonicalization loss: **0.3024477111 -> 0.0943797231**
- fused consistency JS: **0.0315884668 -> 0.0128074290**

The training objective decreases strongly while fresh DEV remains weak. This is not an optimization-run crash.

Neutral-bisector conflict:
- mean across epochs: **0.2951388889**
- minimum: **0.1458333333** at epoch 24
- maximum: **0.4375** at epoch 9
- projection coefficient remains **0**

Compared with S23's much higher conflict regime, S24 authority does not show conflict itself explaining the failure.

## Scientific interpretation

The preregistered S24 hypothesis is rejected.

The deterministic standardized top1-top2 gap is not a sufficient reliability signal for expert arbitration.

More importantly, on fresh S24 authority both experts themselves are weak:
- primary selected canonical **36.20%**
- relation selected canonical **40.10%**
- fused selected canonical **39.06%**

Therefore the S24 failure cannot be repaired merely by changing the mixture weight after the fact. A different post-DEV epsilon, confidence transform, learned coefficient, seed, selector, or retry is forbidden and scientifically unjustified.

S23 and S24 use separate fresh authorities, so their raw percentages are contextual rather than paired same-row comparisons. The valid conclusion is the within-S24 one: the preregistered fusion rule fails every semantic DEV_READY family of gates while mechanics remain intact.

## Track closure

Close:
**standardized top1-top2-gap reliability-weighted fusion**.

Do not create S24b/S24c by:
- changing epsilon;
- exponentiating gaps;
- adding an arbitrary temperature;
- clipping weights;
- hard winner-take-all on S24 DEV;
- retrying seed/LR/templates;
- weakening gates.

## Next controlled direction

S25 should leave post-hoc confidence weighting and test **representation interference directly**.

Proposed track:
**S25 — Decoupled Expert Projection Surfaces**

Controlled hypothesis:
- keep the A13 encoder/LoRA shared;
- restore fixed S14 equal standardized fusion so fusion is no longer the variable;
- give primary and relation experts separate private projection surfaces instead of forcing both through one shared projection;
- keep relation isolation;
- use neutral-bisector balancing only on truly shared parameters;
- update private expert surfaces from their own objectives without cross-expert projection conflict.

Target trainable surface:
- shared A13 LoRA: **16,384**
- primary projection: **32,768**
- relation projection: **32,768**
- total: **81,920**
- no learned downstream router/gate/calibrator.

This remains tiny relative to the resident model while directly testing whether shared representation competition is suppressing expert quality.

S25 must use wholly fresh authority and freeze exact parameter ownership/gradient routing before A0.

Production-ready remains false.
Laya/Jev parity remains unestablished.
