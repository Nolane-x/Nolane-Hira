# HIRA V1 S34 interpretation plan — frozen before S34-A0/DEV exposure

Status: **FROZEN**

Issue: #251

## Controlled comparison

Control:
- exact S17/S13 local relation operator.

Treatment:
- zero-parameter query-conditioned entropic state↔option transport;
- no S13 base-logit mixing;
- no fixed positional window;
- no global contrastive loss;
- exact S17 trainable surface.

## Fresh authority

- seed **55001**
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

### A — coherent transport gain
Treatment improves relation accuracy/margins, fused endpoints and question sensitivity while retaining or improving signature transport.
Then S34 remains viable for a fresh confirmation track.

### B — transport improves but semantic accuracy/margins regress
Many-to-many transport stabilizes correspondence but does not identify the correct semantic relation.
Close S34; do not tune temperatures or iterations.

### C — semantics improve but transport remains weak
The transport plan helps discrimination but wording-equivalent plans remain unstable.
Close S34 and localize transport-plan invariance, not scorer capacity.

### D — little/no coherent gain
Reject the entropic transport construction and move away from the current shared-projection relation geometry.

### E — treatment DEV_READY
Freeze immediately.
No second S34 DEV.
A separate confirmation protocol is required before external matched evaluation.

## Stop rule

No post-DEV:
- transport temperature tuning
- marginal temperature tuning
- Sinkhorn iteration tuning
- epsilon tuning
- score/signature reweighting
- seed/LR/epoch/batch retry
- loss stacking
- selector/gate weakening
- second DEV

No external Laya/Jev benchmark from S34 alone.
