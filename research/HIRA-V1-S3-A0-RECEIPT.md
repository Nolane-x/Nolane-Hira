# HIRA V1 S3-A0 — parameter-free triadic receipt

Status: **CLOSED — LOCALIZATION ONLY**

Run: `36393304057`  
Artifact:
- name: `hira-v1-s3-a0-parameter-free-triadic`
- ID: `10956484697`
- digest: `sha256:614c1b48421392e53906075c34670b879fc2033141f7e824c31cc848bf41e0e6`

Outcome:

`HIRA_V1_S3_A0_PARAMETER_FREE_TRIADIC_READY`

## Contract

- exact frozen M4 A13/W28 provenance;
- W34 excluded from S3 semantic scoring;
- 16 fresh English base states / 32 paired queries;
- K=4;
- direct coordinate triadic evidence:
  `sum_d state_d * question_d * option_d / sqrt(128)`;
- added parameters: **0**;
- trainable parameters: **0**;
- state-once: **16/16**;
- full-K: PASS;
- relation refinement: OFF;
- max probability mass error: `1.7881393432617188e-07`;
- not used for model selection.

## Observed S3-A0

- accuracy: **0.3125**
- paired both-correct: **0.0**
- question changes logits: **1.0**
- question changes final choice: **0.25**

Fresh frozen-v0 control on the exact same memory/schema:
- accuracy: **0.21875**
- paired both-correct: **0.0**
- question changes final choice: **0.0**

## Interpretation

Direct state × question × option geometry is materially different from the frozen v0/W34 path:
- every paired question changes the triadic logits;
- 4/16 paired states change selected option;
- accuracy is 9.375 percentage points above the frozen-v0 control on this localization suite.

This is only a localization result. It does **not** qualify S3 quality and may not be used for candidate selection, threshold tuning or architecture ranking after exposure.

The A0 rows are now permanently exposed and forbidden for learned S3 fitting/selection.

A0 authorizes only the preregistered learned Triadic CP TRAIN/DEV experiment.
