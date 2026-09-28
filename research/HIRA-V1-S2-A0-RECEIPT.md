# HIRA V1 S2-A0 — direct question-as-evidence receipt

Status: **CLOSED — LOCALIZATION ONLY**

Run: `36390086785`  
Artifact: `10955893093`  
Digest: `sha256:e17ccf529e503450686895c983d77df945b21e90570a8945da8c48e95c57c471`

Outcome:

`HIRA_V1_S2_A0_QUESTION_AS_EVIDENCE_READY`

## Contract

- exact frozen M4 A13/W28/W34 base;
- 16 fresh English base states;
- 32 paired queries;
- K=4;
- all state content tokens preserved;
- question content tokens appended to the frozen W34 evidence sequence;
- added parameters: **0**;
- trainable parameters: **0**;
- state-once: 16/16;
- full-K: PASS;
- relation refinement: OFF;
- probability max mass error: `1.1920928955078125e-07`;
- not used for model selection.

## Observed S2-A0

- accuracy: **0.25**
- paired both-correct: **0.0**
- question changes logits: **1.0**
- question changes final choice: **0.0625**

Fresh frozen-v0 control on the same rows:

- accuracy: **0.25**
- paired both-correct: **0.0**
- question changes final choice: **0.0**

## Interpretation

Appending question tokens to the W34 evidence sequence eliminates strict logit blindness: every paired question changes the logits.

However, it barely changes discrete decisions:
- only 1/16 paired states changes selected option;
- accuracy remains exactly equal to frozen v0 control;
- paired both-correct remains zero.

Therefore simply exposing question tokens as more state evidence is not sufficient.

A0 authorizes only the preregistered learned S2 QTRF TRAIN/DEV experiment. It does not qualify semantic quality, multilingual transfer, parity, reliability/OOD or production readiness.

A0 rows are permanently exposed and forbidden for S2 fitting/selection.
