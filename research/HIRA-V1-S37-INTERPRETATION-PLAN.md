# HIRA V1 S37 interpretation plan — frozen before S37-A0/DEV exposure

Status: **FROZEN**

Issue: #257

## Controlled comparison

Control:
- exact S35 native 256D relation representation and native logits.

Treatment:
- exact same native representation/logits;
- masked-mean + L2-normalized 256D native question summary;
- elementwise query gating of each native option signature;
- one zero-init shared 256D residual readout vector;
- no bias/nonlinearity/learned scale/rank expansion;
- added params **256**.

## Frozen capacities

Control:
- LoRA **16,384**
- primary projection **32,768**
- total **49,152**

Treatment:
- LoRA **16,384**
- primary projection **32,768**
- query-gated readout **256**
- total **49,408**

## Frozen query summary

- source: adapted A13 question tokens
- reducer: masked arithmetic mean
- final normalization: L2
- epsilon: **1e-12**
- no learned query projection
- no attention over question tokens
- no positional dependence

## Frozen training authority

- seed **58001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S37 domains
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
- signature cosine/discrimination

Do not collapse to a scalar winner score.

## Frozen interpretation

A — coherent query-conditioned correctness gain:
relation correctness/margins and both fused wording endpoints improve while native transport remains strong.

B — correctness gain with transport/fusion regression:
split evidence; close S37.

C — little/no direct correctness gain:
one diagonal query-signature interaction is insufficient; close S37.

D — treatment DEV_READY:
freeze immediately; no second DEV.

## Stop rule

No post-DEV:
- query-summary normalization changes
- readout scale/bias/nonlinearity changes
- rank/multi-vector retry
- readout-only LR
- projected/native mixing
- seed/LR/epoch/batch retry
- selector/gate weakening
- second DEV

No external Laya/Jev benchmark from S37 alone.
