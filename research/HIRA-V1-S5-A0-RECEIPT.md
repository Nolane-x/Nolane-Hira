# HIRA V1 S5-A0 — W28 identity receipt

Status: **CLOSED — IDENTITY / LOCALIZATION ONLY**

Run: `36400104371`  
Artifact: `10960266188`  
Digest: `sha256:0ba8a60dd46a8cae47ed7098c558cce9fa7e8b4e137231116bb458943a386607`

Outcome:

`HIRA_V1_S5_A0_IDENTITY_READY`

Observed:
- 16 fresh English base states / 32 paired queries;
- K=4;
- two semantic views per option;
- projection parameter count: 32,768;
- projection trainable parameters: 0;
- exact logit identity vs parameter-free W28 triadic: **1.0**;
- exact selected-choice identity: **1.0**;
- accuracy: **0.3125**;
- paired both-correct: **0.0625**;
- option-order flip: **0.0**;
- frozen view InfoNCE: **0.9803502075374126**;
- state encode calls: 16/16;
- full-K: PASS;
- relation refinement: OFF;
- probability max mass error: `1.1920928955078125e-07`.

Interpretation:
- S5 starts exactly from W28 semantic geometry;
- the two-view representation alignment objective is finite before training;
- any S5 TRAIN/DEV movement is attributable to projection relearning.

A0 is permanently exposed and forbidden for S5 fitting or model selection.
