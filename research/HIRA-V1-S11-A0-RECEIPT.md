# HIRA V1 S11-A0 — role-preserving binding identity/localization receipt

Status: **CLOSED — IDENTITY / LOCALIZATION ONLY**

Run: `36440258349`  
Head: `9e1e9a0e661e01473cbb9fbcdb84698c5879031f`  
Artifact: `10978171127`  
Artifact digest: `sha256:95115c357f9b9174bf4f0c2b0beb3f5d5829829e432e00116165e5dcee0f080b`

Outcome:

`HIRA_V1_S11_A0_IDENTITY_READY`

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
- total candidate params: **49,152**
- binding-added params: **0**
- runtime trainable params during A0: **0**
- original A13 trainable params: **0**
- state encodes: **32/32**
- option-order flip: **0.0**
- max probability mass error: **1.1920928955078125e-07**
- full-K: PASS
- relation refinement: OFF

## Fresh primary-decision baseline

- all-view accuracy: **0.25**
- canonical paired both-correct: **0.0**
- cross-view selected-choice agreement: **0.4375**
- cross-view mean JS: **6.033971544638916e-09**

## Fresh role-binding baseline

Canonical:
- binding accuracy: **0.28125**
- signed gold-vs-max-wrong margin: **-0.3389761447906494**
- role normalized entropy: **0.8970667012035847**
- role max weight: **0.12862010311800987**
- value normalized entropy: **0.9805640578269958**
- value max weight: **0.06178424239624292**

Paraphrase:
- binding accuracy: **0.34375**
- signed gold-vs-max-wrong margin: **-0.34898996353149414**
- role normalized entropy: **0.8949157111346722**
- role max weight: **0.12019944959320128**
- value normalized entropy: **0.961082074791193**
- value max weight: **0.07305040257051587**

Cross-view binding selected-choice agreement:
- **0.59375**

Frozen operator:
- role temperature: **0.10**
- contrastive temperature: **0.10**
- value window: **4**

## Interpretation boundary

A0 proves:
- S11 introduces no new learned capacity;
- the exact inherited primary decision path is unchanged at zero initialization;
- role/value locality binding is active and measurable;
- the fresh untrained role and value distributions are diffuse;
- fresh signed binding margins are negative.

These diagnostics are localization only. They do **not** authorize changing the frozen temperatures, value window, optimizer, losses or DEV gates.

All S11-A0 rows are now permanently exposed and forbidden for S11 TRAIN/DEV fitting or model selection.
