# HIRA V1 S1-A0 — zero-parameter query-keyed evidence receipt

Status: **CLOSED — LOCALIZATION ONLY**

Run: `36387872121`  
Artifact: `10955282264`  
Digest: `sha256:98428470ca03e5de015517c1355b4d16d2dc1aa6bf8688e6ae2f05476cf5da3a`

Outcome:

`HIRA_V1_S1_A0_PARAMETER_FREE_EVIDENCE_READY`

## Contract

- exact frozen M4 A13/W28/W34 base;
- 16 fresh English base states;
- 32 paired queries;
- K=4;
- top-2 state evidence selected by frozen W28 question↔state cosine;
- added parameters: **0**;
- trainable parameters: **0**;
- state-once: 16/16;
- full-K: PASS;
- relation refinement: OFF;
- probability max mass error: `1.1920928955078125e-07`;
- not used for model selection.

## Observed A0

- accuracy: **0.21875**
- paired both-correct: **0.0**
- question changes selected evidence: **0.625**
- question changes final choice: **0.0**

Fresh frozen-v0 control on the same rows:

- accuracy: **0.28125**
- paired both-correct: **0.0**
- question changes choice: **0.0**

## Interpretation

A question-conditioned evidence selector can change which state tokens reach W34 without adding parameters, so the S1 architectural axis is mechanically real.

However, hard top-2 frozen cosine extraction is not a semantic rescue:
- evidence changes in 10/16 base cases;
- those evidence changes do not change the final selected option;
- accuracy is below the fresh frozen-v0 control.

Therefore A0 authorizes only the preregistered learned QKEE TRAIN/DEV experiment. It does not support any quality, parity, multilingual or production claim.

A0 rows are now exposed and forbidden for S1-A fitting/selection.
