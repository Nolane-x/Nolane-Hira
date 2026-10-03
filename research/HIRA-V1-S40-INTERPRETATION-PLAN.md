# HIRA V1 S40 interpretation plan — frozen before S40-A0/DEV exposure

Status: **FROZEN**

Issue: #263

## Controlled comparison

Reference/control:
- exact native S35/S17 runtime.

Treatment:
- exact S38 full bilinear co-adaptation;
- W **256x256 / 65,536 params**;
- runtime+W total **114,688**;
- treatment runtime gradient constrained by matched-reference native-signature projection;
- W gradient unchanged by the anchor projection.

## Frozen anchor

`A_sig = mean(1 - cosine(s_treatment, stopgrad(s_reference)))`

No coefficient.
No learned anchor parameters.

The reference runtime is trained independently on exact native S35/S17 objectives using identical rows/order.

## Frozen projection

For treatment runtime gradient `g` and anchor gradient `a`:

- if `a·g >= 0`: identity;
- if `a·g < 0`: `g' = g - ((a·g)/(||a||^2 + 1e-12)) a`.

W is excluded from `a` and is never changed by this projection.

## Frozen training authority

- seed **61001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S40 domains
- K=4
- two state views
- two question wording views
- two option views
- 24 epochs
- batch 16
- AdamW lr 2e-4
- weight decay .01
- grad clip 1.0
- identical rows/order across reference and treatment

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
- anchor conflict rate
- mean pre-projection `a·g`
- mean post-projection `a·g'`
- mean anchor value
- mean treatment/reference native-signature cosine
- W norm
- reference/treatment row-order identity

Do not collapse to a scalar winner score.

## Frozen interpretation

A — correctness retained + transport protected:
material S38-like correctness gain with recovery from S38 transport collapse.

B — correctness retained but transport still collapses:
anchor projection insufficient; close.

C — transport protected but correctness collapses toward S39:
conflicting co-adaptation was required for correctness; close.

D — treatment DEV_READY:
freeze immediately; no second DEV.

## Stop rule

No post-DEV:
- anchor target change
- anchor coefficient/slack
- projection-margin tuning
- partial detach
- gradient mixing
- W-only LR/scheduler
- rank/factorization
- scale/bias/nonlinearity
- projected/native mixing
- seed/LR/epoch/batch retry
- gate weakening
- second DEV

No external Laya/Jev benchmark from S40 alone.
