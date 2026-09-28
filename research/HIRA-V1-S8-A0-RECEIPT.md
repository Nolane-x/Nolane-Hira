# HIRA V1 S8-A0 — paraphrase-invariance identity receipt

Status: **CLOSED — IDENTITY / LOCALIZATION ONLY**

Run: `36416813621`  
Head: `8bb3568117176bcde588160f7a82302d6b1bcdc7`  
Artifact: `10968037685`  
Digest: `sha256:cd7b84cf7de183a6bd263ab8923764d810ff97f44a56b930190893e213b2804a`

Outcome:

`HIRA_V1_S8_A0_IDENTITY_READY`

Observed:
- 16 fresh English semantic cases;
- 32 state wording views;
- 64 decisions;
- K=4;
- two semantic views per option;
- A13 token output identity: **true**;
- A13 pooled output identity: **true**;
- exact final logit identity: **1.0**;
- exact selected-choice identity: **1.0**;
- accuracy over all wording views: **0.328125**;
- canonical paired both-correct: **0.0625**;
- cross-view selected-choice agreement: **0.6875**;
- cross-view mean JS: **9.746536022703367e-09**;
- option-order flip: **0.0**;
- LoRA capacity: **16,384**;
- projection capacity: **32,768**;
- total candidate capacity: **49,152**;
- runtime trainable parameters during A0: **0**;
- original A13 trainable parameters: **0**;
- state encode calls: **32/32**;
- full-K: PASS;
- relation refinement: OFF;
- max probability mass error: `1.1920928955078125e-07`.

Interpretation:
- S8 adds no model capacity relative to S7;
- zero-init A13 LoRA + exact W28 T0 is an exact functional identity;
- A0 is exposed and permanently forbidden for fitting/model selection;
- the cross-view JS value is descriptive only because the frozen baseline is nearly uniform; it is not evidence that S8 has learned paraphrase invariance.
