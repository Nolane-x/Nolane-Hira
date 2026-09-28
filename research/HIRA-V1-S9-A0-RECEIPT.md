# HIRA V1 S9-A0 — identity and margin receipt

Status: **CLOSED — IDENTITY / LOCALIZATION ONLY**

Run: `36421792484`  
Artifact: `10969906709`  
Digest: `sha256:e0ff0ad9907bf452fe0771822c085962b3fdda05ca95b3cf8855f7054d5dbb29`

Outcome:

`HIRA_V1_S9_A0_IDENTITY_READY`

Observed:
- 16 fresh English semantic cases;
- 64 decisions across canonical/paraphrase views;
- K=4;
- two semantic views per option;
- A13 token identity: **true**;
- A13 pooled identity: **true**;
- final logit identity: **1.0**;
- selected-choice identity: **1.0**;
- candidate physical capacity: **49,152**;
- LoRA physical params: **16,384**;
- projection physical params: **32,768**;
- runtime trainable during A0: **0**;
- original A13 trainable: **0**;
- option-order flip: **0.0**;
- state encodes: **32/32 wording views**;
- full-K: PASS;
- probability max mass error: `1.1920928955078125e-07`.

Fresh separation diagnostics:
- accuracy all views: **0.25**;
- canonical paired both-correct: **0.0625**;
- cross-view selected-choice agreement: **0.46875**;
- cross-view mean JS: **1.1452669923528447e-08**;
- canonical mean gold-vs-max-wrong margin: **-8.804556273389608e-05**;
- paraphrase mean gold-vs-max-wrong margin: **-0.00012065676128258929**;
- canonical margin >= 0.20 satisfaction: **0.0**;
- paraphrase margin >= 0.20 satisfaction: **0.0**;
- canonical mean top1-top2 margin: **7.738269050605595e-05**;
- paraphrase mean top1-top2 margin: **0.00015975060523487628**.

Interpretation:
- S9 zero initialization is an exact functional identity;
- the baseline decision surface is essentially unseparated at the S9 margin scale;
- A0 is permanently exposed and forbidden for S9 fitting or model selection.
