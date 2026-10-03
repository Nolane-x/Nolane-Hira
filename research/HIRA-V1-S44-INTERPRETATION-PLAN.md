# HIRA V1 S44 interpretation plan — frozen before S44-A0/DEV exposure

Status: **FROZEN**

Issue: #271

## Controlled comparison

Reference:
- exact native S35/S17 runtime.

Treatment native path:
- exact same native runtime/objective/init/order as reference;
- must remain trajectory-identical.

Treatment correction path:
- detached native signature + detached query;
- private GELU adapter A/B;
- full bilinear W;
- exact S39 correction CE objective at coefficient 0.10;
- no gradient into native runtime.

## Frozen capacity

Native runtime:
- 49,152

Private adapter:
- A 32,768
- B 16,384
- total 49,152

W:
- 65,536

Correction-only:
- 114,688

Treatment total:
- **163,840**

No hidden-width/activation/capacity alternatives are in scope.

## Frozen objective

`CE_corr = 0.5 * (CE(canonical) + CE(paraphrase))`

`L_corr = 0.10 * CE_corr`

Exactly the S39 correction objective.

This makes S44 a controlled test of representation capacity rather than objective tuning.

## Frozen optimizer

Native arms:
- exact existing S35/S17 AdamW behavior.

Correction branch:
- AdamW lr 2e-4;
- weight decay .01;
- default existing S39 betas/eps;
- grad clip 1.0 over A/B/W;
- no separate W schedule.

## Frozen fresh authority

- seed **65001**
- TRAIN 768
- DEV 192
- 12 wholly fresh S44 domains
- K=4
- 24 epochs
- batch 16
- identical native rows/order.

## Required reporting

Native trajectory:
- epoch-by-epoch runtime-state fingerprint identity;
- max LoRA difference;
- max primary-projection difference;
- max native pre-correction logit difference;
- max native signature difference;
- encoder state-view count equality.

Decision metrics:
- fused canonical/paraphrase;
- paired;
- question-swap;
- fused agreement/JS;
- fused margins;
- raw primary canonical/paraphrase;
- native relation canonical/paraphrase;
- corrected relation canonical/paraphrase;
- relation agreement;
- native same-option cosine;
- native discrimination margin.

Private branch:
- A norm;
- B norm;
- W norm;
- private-vs-native signature displacement;
- correction residual magnitude.

## Frozen interpretation

A — native transport exact + private correction restores fused correctness:
private representation ownership resolves the S38/S43 conflict.

B — native transport exact but correctness remains weak:
post-encoder private representation is insufficient; close.

C — correctness rises but native trajectory identity breaks:
mechanism invalid; close.

D — treatment DEV_READY:
freeze immediately; no second S44 DEV; separate fresh confirmation before external Laya/Jev evaluation.

DEV_READY gates remain authoritative.
No scalar winner score.

## Stop rule

No post-DEV:
- width/activation/init sweep;
- bias/gate/layernorm;
- residual scale;
- objective/coefficient change;
- native-gradient leakage;
- second encoder;
- mixing coefficient;
- W-only scheduler;
- seed/LR/epoch/batch retry;
- gate weakening;
- second DEV.

No external Laya/Jev benchmark from S44 alone.
