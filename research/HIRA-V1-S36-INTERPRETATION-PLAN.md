# HIRA V1 S36 interpretation plan — frozen before S36-A0/DEV exposure

Status: **FROZEN**

Issue: #255

## Controlled comparison

Control:
- exact S35 native 256D relation representation and native logits.

Treatment:
- exact same S35 native representation/logits;
- one zero-init shared 256D linear residual correctness vector;
- `treatment = native_logits + signature @ w`;
- no bias/nonlinearity/learned scale;
- added params **256**.

## Frozen capacities

Control:
- LoRA **16,384**
- primary projection **32,768**
- total **49,152**

Treatment:
- LoRA **16,384**
- primary projection **32,768**
- readout **256**
- total **49,408**

## Frozen training authority

- seed **57001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S36 domains
- K=4
- two state views
- two question wording views per semantic query
- two option views
- 24 epochs
- batch 16
- AdamW lr 2e-4
- weight decay .01
- grad clip 1.0
- identical rows/order across arms

Treatment readout uses the same optimizer, LR and weight decay.
No separate schedule.

## Primary matched metrics

Report treatment-control deltas for:
- fused canonical/paraphrase
- paired
- question-swap
- fused agreement/JS
- fused canonical/paraphrase margins
- raw primary canonical/paraphrase
- relation canonical/paraphrase
- relation margins/agreement
- signature cosine/discrimination.

Do not collapse to a scalar winner score.

## Frozen interpretation

A — coherent correctness gain with preserved native transport:
readout family remains viable for fresh confirmation.

B — relation correctness gain with transport/fusion regression:
split evidence; close S36.

C — little/no correctness gain:
one shared linear correctness direction is not sufficient; close S36.

D — treatment DEV_READY:
freeze immediately; no second DEV.

## Stop rule

No post-DEV:
- readout scale/bias/nonlinearity changes
- MLP/rank/multi-vector retry
- readout-only LR
- projected/native mixing
- native representation tuning
- seed/LR/epoch/batch retry
- selector/gate weakening
- second DEV

No external Laya/Jev benchmark from S36 alone.
