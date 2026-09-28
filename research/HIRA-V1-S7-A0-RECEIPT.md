# HIRA V1 S7-A0 — joint identity receipt

Status: **CLOSED — IDENTITY / LOCALIZATION ONLY**

Run: `36411385031`  
Artifact: `10964497583`  
Digest: `sha256:38c9a23606053c4131143817a555677bdbe0e09dac645cbd725572a1d6cd4bdc`

Outcome:

`HIRA_V1_S7_A0_IDENTITY_READY`

Observed:
- 16 fresh English base states / 32 paired queries;
- K=4;
- two semantic views per option;
- A13 token output identity: **true**;
- A13 pooled output identity: **true**;
- exact final triadic logit identity: **1.0**;
- exact selected-choice identity: **1.0**;
- accuracy: **0.25**;
- paired both-correct: **0.0**;
- option-order flip: **0.0**;
- LoRA physical params: **16,384**;
- projection physical params: **32,768**;
- candidate physical params: **49,152**;
- runtime trainable during A0: **0**;
- original A13 trainable: **0**;
- state encode calls: 16/16;
- full-K: PASS;
- relation refinement: OFF;
- probability max mass error: `1.1920928955078125e-07`.

Interpretation:
- zero-initialized A13 LoRA plus exact W28 T0 projection is an exact functional identity;
- any S7 TRAIN/DEV movement is attributable to joint A13-W28 optimization;
- A0 is permanently exposed and forbidden for S7 fitting or model selection.
