# HIRA V1 S25 handoff — to S26 Factorized Role-Value Relation Signatures

S25 is frozen as:

`HIRA_V1_S25_DECOUPLED_PROJECTIONS_DEV_FAIL`

Canonical authority:
- A0 run `36841998584`
- A0 artifact `11151613971`
- A0 digest `sha256:75390a682778eafafbf9e16dd3271ee004770c06a8bd49fb9dd0d3e12c6a19ee`
- TRAIN/DEV run `36843918808`
- artifact `11152983824`
- artifact digest `sha256:40ba0d13aa30dd33dbc842384515643517cf4b59450932187e3300fa81107063`
- selected epoch **17**
- checkpoint `3d4405ae33b1651070416f0b31df4c27aa056acd5c12ea951721a50af0ece164`

Selected DEV:
- fused canonical **0.5104166667**
- fused paraphrase **0.3541666667**
- paired **0.1979166667**
- question-swap **0.375**
- fused agreement **0.3880208333**
- fused JS **0.0955321817**
- fused canonical margin **-0.1282932429**
- primary canonical/paraphrase **0.5130208333 / 0.4296875**
- relation canonical/paraphrase **0.4296875 / 0.2942708333**
- relation canonical margin **-0.0703994036**
- relation paraphrase margin **-0.2418746576**
- signature cosine **0.7333363046**
- signature discrimination margin **0.0827438538**

Best residual probes:
- same-option signature cosine reaches **0.9018671364** at epoch 4
- signature discrimination margin never exceeds **0.0832758718**
- relation canonical margin never becomes positive beyond **-0.0100153784**
- fused canonical margin remains negative at every epoch

Mechanics:
- exact surface **81,920**
- full-K/state-once PASS
- S14 mapped option-order flip **0**
- cross-private gradient leakage **0 / 0**
- private gradients never vanish
- shared LoRA receives both objectives

## Stop rule applied

Do not retry S25 with more width, different private optimizer, different seed/LR/epochs, or a modified fusion.

## S26 target

**Factorized Role-Value Relation Signatures**

Why:
S25 shows that view alignment can become high without correct-vs-wrong relation separation. The next experiment must target binding geometry itself.

Freeze before A0:
- inherit S25 shared A13 + LoRA;
- inherit S25 primary-private projection;
- inherit S25 relation-private projection;
- preserve exact S25 **81,920** trainable surface unless a separately preregistered parameter addition is essential;
- S21 primary unchanged;
- S14 equal standardized full-K fusion unchanged;
- relation detach unchanged;
- S25 gradient ownership unchanged;
- replace only S13 relation signature/logit construction.

Required S26 relation factorization:
- query-derived role anchor;
- state and option role compatibility scored explicitly;
- value/content evidence formed after removing role direction;
- relation score must expose separate role and value components;
- final relation evidence combines them symmetrically without option-specific parameters;
- signature must retain role identity and value residual as distinguishable components rather than one additive vector.

Required A0 synthetic quadrants:
- correct role + correct value;
- correct role + wrong value;
- wrong role + same/correct-looking value;
- wrong role + wrong value.

A0 must require positive margins from correct relation to both hard-negative classes.

Also prove:
- canonical/paraphrase semantic equivalence stability;
- option permutation;
- full-K;
- state-once;
- zero new learned downstream state if parameter-free;
- exact S25 capacity and gradient ownership;
- checkpoint roundtrip/replay;
- no hidden dependence on option IDs/order.

Use wholly fresh S26 authority.
No S25 DEV rows may enter S26 TRAIN/DEV.
No Laya/Jev until DEV_READY.
