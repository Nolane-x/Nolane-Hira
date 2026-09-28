# HIRA V1 S4-A0 — identity receipt

Status: **CLOSED — LOCALIZATION / IDENTITY ONLY**

Run: `36397966988`  
Artifact: `10959630634`  
Digest: `sha256:1943d802440ec759e4d32c97c61fdb93714ce8fb0a1897df25bd3bcdbc071c59`

Outcome:

`HIRA_V1_S4_A0_IDENTITY_READY`

Observed:
- 16 fresh English base states / 32 paired queries;
- K=4;
- adapter parameter count: 16,384;
- adapter trainable parameters: 0;
- exact logit identity vs parameter-free triadic: **1.0**;
- exact selected-choice identity: **1.0**;
- accuracy: **0.28125**;
- paired both-correct: **0.0**;
- option-order flip: **0.0**;
- state encode calls: 16/16;
- full-K: PASS;
- relation refinement: OFF;
- probability max mass error: `1.1920928955078125e-07`.

Interpretation:
- the S4 shared residual adapter has an exact identity starting boundary;
- S4 changes no baseline semantic behavior before optimization;
- any later TRAIN/DEV movement is attributable to learned adapter weights, not a changed scorer or initialization offset.

A0 is permanently exposed and forbidden for S4 fitting or model selection.
