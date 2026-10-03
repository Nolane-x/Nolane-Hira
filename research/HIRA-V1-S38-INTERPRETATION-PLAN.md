# HIRA V1 S38 interpretation plan — frozen before S38-A0/DEV exposure

Status: **FROZEN**

Issue: #259

## Controlled comparison

Control:
- exact S35 native 256D relation representation and native logits.

Treatment:
- exact same native representation/logits;
- exact S37 masked-mean + L2-normalized 256D native question summary;
- one zero-init shared full bilinear matrix `W in R^(256x256)`;
- residual `s_k^T W q`;
- no bias/nonlinearity/learned scale/factorization;
- added params **65,536**.

## Frozen capacities

Control:
- LoRA **16,384**
- primary projection **32,768**
- total **49,152**

Treatment:
- LoRA **16,384**
- primary projection **32,768**
- bilinear readout **65,536**
- total **114,688**

## Frozen query summary

- source: adapted A13 question tokens
- reducer: masked arithmetic mean
- final normalization: L2
- epsilon: **1e-12**
- no learned query projection
- no attention reducer
- no positional dependence

## Frozen bilinear map

- shape **256 x 256**
- exact zero initialization
- residual scale **1.0**
- no bias
- no factorization
- no rank constraint
- no separate regularizer beyond shared AdamW weight decay
- no W-only learning rate

## Frozen training authority

- seed **59001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S38 domains
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
- independent runtime/optimizer state

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

A — coherent cross-coordinate correctness gain:
relation correctness/margins and both fused wording endpoints improve while native transport remains viable.

B — correctness gain with transport/fusion regression:
split evidence; close S38.

C — little/no direct correctness gain:
full bilinear query-signature readout is insufficient; close the family.

D — treatment DEV_READY:
freeze immediately; no second DEV.

## Stop rule

No post-DEV:
- W regularization changes
- factorization/rank retry
- residual scale/bias/nonlinearity changes
- W-only LR
- projected/native mixing
- seed/LR/epoch/batch retry
- selector/gate weakening
- second DEV

No external Laya/Jev benchmark from S38 alone.
