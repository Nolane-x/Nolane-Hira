# HIRA V1 S33 interpretation plan — frozen before S33-A0/DEV exposure

Status: **FROZEN**

Issue: #249

## Controlled comparison

Control:
- exact S17/S13 local relation representation.

Treatment:
- exact S13 base relation block plus explicit query-option coordinate;
- fixed 0.50 base-score / 0.50 query-option-score weighting;
- signature width 256;
- zero learned params.

## Fresh authority

- seed **54001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh domains
- K=4
- two state views
- two question views per semantic query
- two option views
- 24 epochs
- batch 16
- AdamW 2e-4
- weight decay .01
- grad clip 1.0

## Frozen interpretation

### A — coherent treatment gain
If treatment improves relation margins/accuracy, fused endpoints and question sensitivity while retaining or improving cross-view signature transport, the query-explicit architecture remains viable for a fresh confirmation track.

### B — transport gain but semantic regression
Explicit query coordinates stabilize representation but distort option discrimination. Close fixed 0.50/0.50 query-explicit scoring; do not tune weights on exposed DEV.

### C — semantics gain but transport remains weak
Direct query information helps discrimination but does not solve representation stability. Close S33 and localize whether query anchor wording drift is the remaining issue.

### D — little/no coherent gain
Reject this query-explicit coordinate construction and move away from the S13 additive relation family.

### E — treatment DEV_READY
Freeze immediately; no second S33 DEV. A separate confirmation protocol is required before external matched evaluation.

## Stop rule

No post-DEV:
- score-weight tuning
- query-pool tuning
- block-weight tuning
- temperature tuning
- seed/LR/epoch/batch retry
- loss stacking
- selector/gate weakening
- second DEV

No external Laya/Jev benchmark from S33 alone.
