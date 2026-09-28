# HIRA V1 S10-A0 — grounding identity/localization receipt

Status: **CLOSED — IDENTITY / LOCALIZATION ONLY**

Run: `36429235962`  
Head: `bb1ce0cc96cbe3a95f29e0736ef71022df2fea67`  
Artifact: `10973275125`  
Digest: `sha256:b3a632682cccd5c78a91d2d799e74d2d328a115113a4871a3e9b43d18218b5f3`

Outcome:

`HIRA_V1_S10_A0_IDENTITY_READY`

## Identity / mechanics

- 16 fresh English semantic cases
- 32 state wording views
- 64 decision rows
- K=4
- two semantic views per option
- A13 token output identity: **true**
- A13 pooled output identity: **true**
- exact final triadic logit identity: **1.0**
- exact selected-choice identity: **1.0**
- LoRA physical params: **16,384**
- projection physical params: **32,768**
- candidate physical params: **49,152**
- grounding added params: **0**
- runtime trainable params during A0: **0**
- original A13 trainable params: **0**
- state encodes: **32/32**
- option-order flip: **0.0**
- max probability mass error: `1.1920928955078125e-07`
- full-K: PASS
- relation refinement: OFF

## Fresh decision baseline

- all-view decision accuracy: **0.28125**
- canonical paired both-correct: **0.0**
- cross-view selected-choice agreement: **0.90625**
- cross-view mean JS: **6.138495933782906e-09**

## Fresh grounding baseline

Canonical:
- grounding accuracy: **0.34375**
- mean gold-vs-max-wrong grounding margin: **-0.19469201564788818**
- normalized state-attention entropy: **0.809822641313076**
- mean max state-attention weight: **0.22189235920086503**

Paraphrase:
- grounding accuracy: **0.34375**
- mean gold-vs-max-wrong grounding margin: **-0.249116912484169**
- normalized state-attention entropy: **0.810539148747921**
- mean max state-attention weight: **0.18987579457461834**

Cross-view grounding selected-choice agreement:
- **0.40625**

Frozen grounding temperatures:
- state attention: **0.10**
- grounding contrastive: **0.10**

## Interpretation boundary

A0 proves:
- S10 introduces no new learned capacity;
- zero initialization preserves the exact inherited S8/S9 decision model;
- the parameter-free grounding operator is active and measurable;
- fresh baseline grounding is weak, diffuse and has negative signed gold margin.

A0 does **not** select a model or authorize any conclusion about the S10 hypothesis.

All S10-A0 rows are now permanently exposed and forbidden for TRAIN/DEV fitting, model selection, architecture ranking or hyperparameter tuning.
