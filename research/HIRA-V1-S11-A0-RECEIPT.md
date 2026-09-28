# HIRA V1 S11-A0 — role-value mechanics receipt

Status: **CLOSED — MECHANICS / BASELINE ONLY**

Run: `36436701667`  
Head: `bc0d7ea57d6f34c963191b4c22475523b2a9c2fc`  
Artifact: `10975624156`  
Digest: `sha256:f92d5d105d58e9dbe1f4ebe44cc385cee7f604a252d044d8caa72c55ed7b0cf0`

Outcome:

`HIRA_V1_S11_A0_ROLE_VALUE_READY`

## Mechanics

- 16 fresh English semantic cases
- 32 state wording views
- 64 decisions
- K=4
- two semantic views per option
- A13 token output identity: **true**
- A13 pooled output identity: **true**
- LoRA physical params: **16,384**
- projection physical params: **32,768**
- candidate physical params: **49,152**
- binding-added params: **0**
- runtime trainable params during A0: **0**
- original A13 trainable params: **0**
- state encodes: **32/32**
- full-K: PASS
- option-order flip: **0.0**
- max probability mass error: **1.1920928955078125e-07**
- relation refinement: OFF

## Fresh factorized-binding baseline

Canonical:
- binding accuracy: **0.375**
- paired both-correct: **0.125**
- question-swap choice-change: **0.3125**
- signed binding gold margin: **-0.2620204985**
- role normalized entropy: **0.9745326471**
- state normalized entropy: **0.9570791423**
- expert selected-choice agreement: **0.21875**

Paraphrase:
- binding accuracy: **0.46875**
- signed binding gold margin: **-0.1452610493**
- role normalized entropy: **0.9833785724**
- state normalized entropy: **0.9781847447**
- expert selected-choice agreement: **0.375**

Cross-view:
- selected-choice agreement: **0.6875**
- mean JS: **0.0048230030**

State-presence diagnostic:
- paired-gold top-2 containment: **0.0625**

Frozen temperatures:
- role expert: **0.10**
- state expert: **0.10**

## Interpretation boundary

A0 proves:
- S11 changes no learned capacity;
- A13 semantic outputs remain exactly inherited before training;
- the factorized role/state experts and product-of-experts fusion execute correctly;
- option permutation, full-K and state-once mechanics remain intact;
- the baseline experts are diffuse and not already solving the task.

The semantic baseline values are diagnostic only. They are **not model-selection evidence** and may not be used to alter S11 temperatures, fusion, loss coefficients, optimizer, gates or architecture.

All S11-A0 rows are permanently exposed and forbidden for TRAIN/DEV fitting or selection.
