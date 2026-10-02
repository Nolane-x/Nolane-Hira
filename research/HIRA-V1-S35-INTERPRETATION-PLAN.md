# HIRA V1 S35 interpretation plan — frozen before S35-A0/DEV exposure

Status: **FROZEN**

Issue: #253

## Controlled comparison

Control:
- exact S13 projected 128D relation geometry.

Treatment:
- exact S13-equivalent relation algorithm directly in adapted A13 256D token space;
- shared projection absent from the treatment relation path;
- primary S17 path unchanged and still projected;
- zero added learned parameters/state.

## Frozen training authority

- seed **56001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S35 domains
- K=4
- two state views
- two question wording views per semantic query
- two option views
- 24 epochs
- batch 16
- AdamW 2e-4
- weight decay .01
- grad clip 1.0
- identical rows/order across arms

## Frozen treatment

- native relation dimension **256**
- role temperature **0.10**
- pair temperature **0.10**
- contrastive temperature **0.10**
- no learned relation tensor
- no projection/native mixing
- no whitening
- no global loss
- no transport

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

A — coherent native-space gain:
native relation geometry is a viable family for fresh confirmation.

B — transport-only gain:
close S35; relation correctness remains unresolved.

C — semantic gain with transport regression:
close S35 and localize native-space invariance, without mixing geometries.

D — little/no coherent gain:
reject projection bypass and move beyond current A13 relation geometry localization.

E — DEV_READY:
freeze candidate immediately; no second DEV.

## Stop rule

No post-DEV:
- projected/native mixing
- dimension tuning
- whitening/metric learning
- normalization/temperature tuning
- seed/LR/epoch/batch retry
- loss stacking
- selector/gate weakening
- second DEV

No external Laya/Jev benchmark from S35 alone.
