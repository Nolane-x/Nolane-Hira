# HIRA V1 S12-A0 — relation-binding identity/localization receipt

Status: **CLOSED — IDENTITY / LOCALIZATION ONLY**

Run: `36497191159`  
Head: `01d6d6d09e29bbee7e5dcb5a8dcf2b13e7dfbc66`  
Artifact: `11004450349`  
Artifact digest: `sha256:62340525b36377e08d5ec481d852dd9422264a43579a0347c7d126943180bbf5`

Outcome:

`HIRA_V1_S12_A0_IDENTITY_READY`

## Identity / mechanics

- 16 wholly fresh English semantic cases
- 32 state wording views
- 64 decisions
- K=4
- two semantic views per option
- A13 token output identity: **true**
- A13 pooled output identity: **true**
- exact inherited primary logits: **1.0**
- exact inherited selected choices: **1.0**
- LoRA physical params: **16,384**
- projection physical params: **32,768**
- candidate physical params: **49,152**
- relation-binding added params: **0**
- runtime trainable params during A0: **0**
- original A13 trainable params: **0**
- state encodes: **32/32**
- option-order flip: **0.0**
- max probability mass error: **1.1920928955078125e-07**
- full-K: PASS
- relation refinement: OFF

## Fresh primary-decision baseline

- all-view accuracy: **0.3125**
- canonical paired both-correct: **0.0**
- cross-view selected-choice agreement: **0.71875**
- cross-view mean JS: **9.061670525056797e-09**

## Fresh relation-binding baseline

Canonical:
- relation-binding accuracy: **0.46875**
- signed gold-vs-max-wrong margin: **-0.3835980594**
- state-role normalized entropy: **0.8865347318**
- state-role max weight: **0.1507578027**
- option-role normalized entropy: **0.8776449505**
- option-role max weight: **0.2356426856**
- mean best explicit pair score: **0.7325994652**

Paraphrase:
- relation-binding accuracy: **0.34375**
- signed gold-vs-max-wrong margin: **-0.3163611293**
- state-role normalized entropy: **0.8549689166**
- state-role max weight: **0.1517372511**
- option-role normalized entropy: **0.9135185573**
- option-role max weight: **0.1962880674**
- mean best explicit pair score: **0.7568734270**

Cross-view relation-binding selected-choice agreement:
- **0.71875**

Frozen temperatures:
- role: **0.10**
- contrastive: **0.10**

## Interpretation boundary

A0 proves:
- S12 introduces no new learned capacity;
- the exact inherited primary decision model is preserved;
- relation binding is independent of state token order by contract;
- explicit role-relative state/option token-pair evidence is active and measurable;
- the fresh zero-training baseline is above random in canonical relation-binding accuracy, but signed semantic separation remains negative.

A0 does **not** select a model and cannot authorize a success claim.

All S12-A0 rows are permanently exposed and forbidden for TRAIN/DEV fitting, model selection, hyperparameter choice, or architecture ranking.
