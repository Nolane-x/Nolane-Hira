# HIRA V1 S39 interpretation plan — frozen before S39-A0/DEV exposure

Status: **FROZEN**

Issue: #261

## Controlled comparison

Control:
- exact S35 native relation training/evaluation on the S17 shell.

Treatment evaluation:
- exact same native representation
- exact same S38 full bilinear readout capacity
- `residual = signature^T W query`
- W shape **256x256**
- W params **65,536**
- total treatment trainable surface **114,688**

Treatment training:
- runtime primary/native-relation losses remain exactly control losses
- W correction CE consumes detached native logits/signatures/query features
- correction coefficient **0.10**
- runtime and W gradients are clipped separately at **1.0**
- same AdamW LR/WD
- no W-only schedule

## Frozen trajectory invariant

Treatment runtime is not allowed to co-adapt to W.

With identical seed/data/order:
- LoRA state must match control
- primary projection state must match control
- native pre-W relation logits/signatures must match control

Any selected DEV difference must therefore be attributable to W at evaluation, not a changed native runtime.

## Frozen training authority

- seed **60001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S39 domains
- K=4
- two state views
- two question wording views
- two option views
- 24 epochs
- batch 16
- AdamW lr 2e-4
- weight decay .01
- runtime clip 1.0
- W clip 1.0
- identical rows/order across arms

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

Additionally report:
- LoRA state max abs control-treatment
- projection state max abs control-treatment
- native pre-W logit max abs control-treatment
- native signature max abs control-treatment
- W norm

Do not collapse to a scalar winner score.

## Frozen interpretation

A — correctness retained + transport recovers:
direct correctness remains meaningfully above control and S38-like stability regressions substantially recover.

B — correctness retained but transport remains degraded:
gradient isolation insufficient; close S39.

C — transport recovers but correctness gain largely disappears:
S38 gain required co-adaptation; close isolated family.

D — treatment DEV_READY:
freeze immediately; no second DEV.

## Stop rule

No post-DEV:
- partial detach
- gradient mixing
- W-only LR/scheduler
- rank/factorization
- scale/bias/nonlinearity
- projected/native mixing
- seed/LR/epoch/batch retry
- gate weakening
- second DEV

No external Laya/Jev benchmark from S39 alone.
