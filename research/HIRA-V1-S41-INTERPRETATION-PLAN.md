# HIRA V1 S41 interpretation plan — frozen before S41-A0/DEV exposure

Status: **FROZEN**

Issue: #265

## Controlled comparison

Reference:
- exact native S35/S17 runtime
- frozen explicit AdamW semantics

Treatment:
- exact S38 full bilinear joint co-adaptation
- W 256x256 / 65,536 params
- runtime+W total 114,688
- actual AdamW runtime parameter delta constrained by matched-reference native-signature anchor
- W candidate AdamW delta unchanged

## Frozen anchor

`A_sig = mean(1 - cosine(s_treatment, stopgrad(s_reference)))`

No coefficient.
No learned anchor params.

## Frozen actual-step projection

For anchor gradient `a` and actual candidate AdamW runtime delta `delta_r`:

- if `a·delta_r <= 0`: identity
- if `a·delta_r > 0`: `delta_r' = delta_r - ((a·delta_r)/(||a||^2 + 1e-12))a`

W candidate delta remains exact AdamW delta.

AdamW moment/step state is advanced from the frozen clipped raw gradient and persisted unchanged by the projection.

## Frozen optimizer

- AdamW
- lr 2e-4
- betas (0.9, 0.999)
- eps 1e-8
- weight decay .01
- foreach false
- fused false
- grad clip 1.0 before moment update

## Frozen fresh authority

- seed 62001
- TRAIN 768
- DEV 192
- 12 wholly fresh S41 domains
- K=4
- two state views
- two question wording views
- two option views
- 24 epochs
- batch 16
- identical rows/order

## Primary matched metrics

Report treatment-reference deltas for:
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
- actual-step conflict rate
- mean pre-projection `a·delta`
- mean post-projection `a·delta'`
- mean anchor
- mean reference/treatment signature cosine
- W norm
- AdamW candidate-parity diagnostics from A0
- row/order identity

Do not collapse to a scalar winner score.

## Frozen interpretation

A — correctness retained + transport protected:
material S38-like correctness gain with recovery from S38 transport collapse.

B — correctness retained but transport still collapses:
optimizer-step anchor insufficient; close.

C — transport protected but correctness collapses toward S39:
conflicting actual movement was required for correctness; close.

D — treatment DEV_READY:
freeze immediately; no second DEV.

## Stop rule

No post-DEV:
- optimizer-state reinterpretation
- anchor target/weight/slack change
- partial detach
- gradient mixing
- W-only LR/scheduler
- rank/factorization
- scale/bias/nonlinearity
- projected/native mixing
- seed/LR/epoch/batch retry
- gate weakening
- second DEV

No external Laya/Jev benchmark from S41 alone.
