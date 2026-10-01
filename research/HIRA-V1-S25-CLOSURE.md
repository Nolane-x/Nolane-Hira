# HIRA V1 S25 closure — Decoupled Expert Projection Surfaces

Status: **CLOSED — DEV FAIL / PRIVATE PROJECTION DECOUPLING INSUFFICIENT**

Issue: #233  
PR: #234

## Authority

Canonical A0:
- run `36841998584`
- artifact `11151613971`
- digest `sha256:75390a682778eafafbf9e16dd3271ee004770c06a8bd49fb9dd0d3e12c6a19ee`
- authority head `6f6a99ec5a230981a53b6a311446202c22b5965c`
- outcome `HIRA_V1_S25_A0_DECOUPLED_PROJECTIONS_READY`

Fresh TRAIN/DEV:
- run `36843918808`
- artifact `11152983824`
- artifact digest `sha256:40ba0d13aa30dd33dbc842384515643517cf4b59450932187e3300fa81107063`
- scientific head `41ba0bc6dfa2a21610dacc2f0da34aebd4f235a1`
- outcome `HIRA_V1_S25_DECOUPLED_PROJECTIONS_DEV_FAIL`
- selected epoch **17**
- checkpoint SHA256 `3d4405ae33b1651070416f0b31df4c27aa056acd5c12ea951721a50af0ece164`

Exact-head generic CI:
- run `36843922971`
- PASS

No post-DEV tuning.
No second DEV run.
No sealed confirmation.
No multilingual probe.
No Laya/Jev reopening.

## Frozen intervention

S25 changed representation ownership only.

Trainable surface:
- shared A13 LoRA **16,384**
- primary-private 256→128 projection **32,768**
- relation-private 256→128 projection **32,768**
- exact total **81,920**
- original A13 frozen
- HIRACore frozen
- no learned router/gate/calibrator/head

Inference:
- S21 role-gated-content primary
- S13 canonical-relation expert
- frozen S14 equal standardized full-K fusion
- relation logits detached from fused-primary objective

Gradient ownership:
- primary-private receives only primary block
- relation-private receives only relation block
- shared LoRA receives both blocks
- neutral-bisector applies only to shared LoRA

## A0 result

The architecture and ownership hypothesis is implemented correctly:
- both private projections start bit-identically from W28 T0
- private storage is distinct
- primary → relation-private gradient **0**
- relation → primary-private gradient **0**
- primary-private gradient nonzero
- relation-private gradient nonzero
- primary shared-LoRA gradient nonzero
- relation shared-LoRA gradient nonzero
- exact **81,920** physical surface
- S14 mapped option-order flip **0**
- full-K/state-once preserved
- checkpoint ownership roundtrip and frozen replay PASS

The earlier A0 aborts are preserved as non-results. Canonical A0 is run `36841998584`.

## Selected DEV — epoch 17

Fused:
- canonical accuracy **0.5104166667**
- paraphrase accuracy **0.3541666667**
- paired both-correct **0.1979166667**
- question-swap choice-change **0.375**
- cross-view selected-choice agreement **0.3880208333**
- cross-view mean JS **0.0955321817**
- canonical signed margin **-0.1282932429**
- paraphrase signed margin **-0.4498306389**

Primary:
- canonical accuracy **0.5130208333**
- paraphrase accuracy **0.4296875**
- cross-view agreement **0.3515625**

Relation:
- canonical accuracy **0.4296875**
- paraphrase accuracy **0.2942708333**
- canonical signed margin **-0.0703994036**
- paraphrase signed margin **-0.2418746576**
- cross-view agreement **0.5286458333**

Relation signatures:
- same-option cosine **0.7333363046**
- same-vs-strongest-wrong margin **0.0827438538**

Mechanics:
- option-order flip **0**
- fused probability-mass error **1.7881393e-7**
- relation delta **0**
- full-K PASS
- state-once PASS

## Gate result

PASS **16 / 26**:
- all capacity/freeze gates
- both cross-private leakage gates
- private projection storage gate
- relation isolation
- full-K
- state-once train/dev
- option-order
- probability mass
- zero learned fusion params

FAIL **10 / 26**:
- fused canonical >= **0.85**
- paired >= **0.75**
- question-swap >= **0.80**
- fused cross-view agreement >= **0.95**
- fused cross-view JS <= **0.05**
- fused canonical margin >= **0.15**
- relation canonical >= **0.80**
- relation canonical margin >= **0.15**
- same-option signature cosine >= **0.90**
- signature discrimination margin >= **0.15**

DEV_READY is not authorized.

## Best observed DEV values across 24 epochs

- fused canonical: **0.5104166667**, epoch 17
- fused paraphrase: **0.4244791667**, epoch 5
- paired: **0.1979166667**, epoch 17
- question-swap: **0.4895833333**, epoch 4
- fused agreement: **0.515625**, epoch 5
- lowest fused JS: **0.0571338832**, epoch 4
- best fused canonical margin: **-0.0897147230**, epoch 4
- relation canonical: **0.515625**, epoch 20
- best relation canonical margin: **-0.0100153784**, epoch 20
- same-option signature cosine: **0.9018671364**, epoch 4
- signature discrimination margin: **0.0832758718**, epoch 20
- primary canonical: **0.5442708333**, epoch 4
- primary paraphrase: **0.4895833333**, epoch 5

No epoch approaches the complete DEV_READY gate set.

## Optimization dynamics

Training is healthy rather than crashed.

Epoch 1 → 24:
- total loss **1.9571752523 → 0.7236845456**
- decision loss **1.6938289752 → 0.5531840697**
- relation binding loss **1.4048766692 → 0.8877493603**
- canonicalization loss **0.2804123291 → 0.0886717183**
- fused consistency JS **0.0530796330 → 0.0046049297**

Shared-LoRA mean conflict rate:
- **0.4348958333**

Across all training steps:
- minimum primary-private gradient L1 **53.7289695740**
- minimum relation-private gradient L1 **1.6365672350**
- minimum primary shared-LoRA gradient L1 **4.5629181862**
- minimum relation shared-LoRA gradient L1 **0.1469916999**
- cross-private leakage remains exactly **0 / 0**

So the failure is not caused by dead private surfaces or a broken optimizer route.

## Scientific interpretation

S25 rejects the strong claim that **shared 256→128 projection interference is the primary missing semantic ingredient**.

Decoupling private projection geometry is mechanically valid and allows both experts to specialize, but fresh DEV still shows:
- weak wrong-option discrimination;
- negative relation and fused gold margins;
- very weak canonical↔paraphrase decision stability;
- signature alignment can become high while signature discrimination remains low.

The strongest residual signal is therefore not simple capacity sharing. At epoch 4, same-option signature cosine already reaches **0.9019**, yet its discrimination margin is only **0.0064** and fused margin remains negative. At epoch 20, relation canonical reaches **51.56%** and its margin is still slightly negative.

This indicates **relation-signature collapse / insufficient role-value discriminability**: semantically corresponding views can align without placing the correct role-value relation far enough away from plausible wrong relations.

Cross-track percentages use different fresh authorities and must not be treated as paired same-row improvements. The valid S25 conclusion is entirely within its own frozen authority.

## Track closure

Close:
**private primary/relation projection decoupling as a sufficient solution**.

Do not create S25b by:
- increasing projection width;
- interpolating shared/private projections;
- changing private optimizer;
- retrying seed/LR/epochs;
- increasing relation coefficient;
- weakening gates.

## Next controlled direction

S26 should target **discriminative relation binding directly**, not more projection ownership.

Proposed track:
**S26 — Factorized Role-Value Relation Signatures**

Controlled hypothesis:
- keep S25 decoupled projection ownership;
- keep S14 equal fusion and the entire training shell fixed;
- replace only the S13 additive relation signature/logit construction;
- explicitly factor queried-role compatibility from value/content compatibility;
- require same-role/wrong-value and wrong-role/same-value negatives to be separated in A0;
- use no learned router/gate/calibrator and no extra parameter surface beyond S25 unless separately justified before A0.

The central S26 court must prove that relation signatures distinguish:
1. correct role + correct value;
2. correct role + wrong value;
3. wrong role + correct-looking value;
4. wrong role + wrong value;

while preserving wording-view alignment, option permutation, full-K, state-once and S25 gradient ownership.

Production-ready remains false.
Laya/Jev parity remains unestablished.
