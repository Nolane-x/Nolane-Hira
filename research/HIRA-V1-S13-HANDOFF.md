# HIRA V1 S13 handoff — to S14 Canonical Relation Evidence Fusion

S13 is frozen as:

`HIRA_V1_S13_CANONICAL_RELATION_DEV_FAIL`

## Canonical evidence

A0:
- run `36524047158`
- artifact `11013757416`
- digest `sha256:d2aeeec672dd7787aab3f3fe68799e83d303a99b74b4ebf8926ed75e44fbf8ba`

TRAIN/DEV:
- run `36524706099`
- artifact `11014078172`
- digest `sha256:0abe4623eef73dddda3aa49a2725731388dc02bb2ab2305262595f4c501e037a`
- selected epoch **24**
- checkpoint SHA256 `1ede44bc65a275941c8313c8ee78d82c8759eaf3985eb62080e13d48aa9f7360`

Selected DEV:
- canonical primary accuracy **0.2942708333**
- paraphrase primary accuracy **0.2786458333**
- paired both-correct **0.046875**
- question-swap choice-change **0.3489583333**
- primary cross-view agreement **0.3359375**
- canonical relation-binding accuracy **0.4921875**
- paraphrase relation-binding accuracy **0.5338541667**
- canonical relation margin **-0.1725222593**
- paraphrase relation margin **-0.0383211312**
- same-option signature cosine **0.7653450121**
- same-vs-strongest-wrong signature margin **0.0656896873**
- option-order flip **0.0**
- full-K/state-once/relation-delta-zero PASS

TRAIN dynamics:
- binding loss **1.39382457 -> 0.57022986**
- canonicalization loss **0.31337504 -> 0.12005968**

Fresh DEV transient peaks:
- signature cosine **0.85817918** at epoch 4
- signature margin **0.17837184** at epoch 12
- primary cross-view agreement **0.82291667** at epoch 9
- canonical relation-binding accuracy **0.4921875** at epoch 24

## Key result

S13 proves that the 49,152-parameter surface can learn substantially better relation evidence without a learned downstream head.

But the unchanged primary triadic scorer does not convert that evidence into correct decisions.

The clearest gap at selected DEV is:

`relation evidence ~49-53% accuracy > primary decision ~28-29% accuracy`

This is now the main architectural bottleneck.

## Do not do

Do not:
- retune S13 canonicalization coefficient;
- retune role/pair/contrastive temperatures;
- retune signature margin;
- select a different S13 epoch after seeing DEV;
- reuse S13 DEV rows for S14 fitting or selection;
- add a learned fusion head;
- increase model capacity by default;
- reopen M5/Laya/Jev.

## S14 target

S14 should test **Canonical Relation Evidence Fusion** while keeping the exact **49,152 trainable parameters**.

Controlled hypothesis:

1. retain the existing parameter-free triadic primary logits as one expert;
2. retain the S13 parameter-free canonical relation logits as a second expert;
3. normalize each expert per full-K decision in a permutation-equivariant, parameter-free way;
4. combine the two experts with a fixed symmetric rule — no fitted fusion coefficient and no learned gate;
5. use the fused full-K logits as the actual primary decision surface;
6. keep raw triadic and raw relation logits as diagnostics;
7. keep state-once, full-K, relation-refinement-off and option permutation invariance;
8. use wholly fresh S14 A0/TRAIN/DEV evidence;
9. preregister the fusion rule, losses, selection order and gates before exposure.

A strong default to investigate is an equal-weight standardized product-of-experts / additive log-evidence rule, because it has no tunable mixture coefficient and directly tests whether agreement between independent semantic surfaces can improve decision separation.

S14 must answer:

**If relation evidence is already stronger than the legacy primary path, can a zero-parameter symmetric fusion rule turn that evidence into substantially better full-K decisions without adding capacity?**
