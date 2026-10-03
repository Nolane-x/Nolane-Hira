# HIRA V1 S43 interpretation plan — frozen before S43-A0/DEV exposure

Status: **FROZEN**

Issue: #269

## Controlled comparison

Reference:
- exact native S35/S17 runtime;
- exact S41/S42 frozen AdamW semantics.

Treatment:
- exact S38 full bilinear joint co-adaptation;
- W 256x256 / 65,536;
- treatment runtime+W total 114,688;
- actual AdamW runtime movement constrained by S43 relative-gap anchor;
- W candidate movement unchanged.

## Frozen anchor

`G[b,i,j] = cosine(S_c[b,i], S_p[b,j])`

`R[b,i,j] = G[b,i,i] - G[b,i,j]`, for `j != i`.

`A_gap = mean((R_treatment - stopgrad(R_reference))^2)`

No strongest-wrong max.
No coefficient.
No gap weighting.
No numeric target margin.
No gold labels.
No learned anchor params.

## Frozen actual-step projection

For gap-anchor gradient `a` and candidate AdamW runtime movement `delta_r`:
- if `a dot delta_r <= 0`: identity;
- otherwise remove exactly the conflicting component with eps 1e-12.

W candidate movement remains unchanged.
AdamW state remains the transition from the original clipped correctness gradient.

## Frozen optimizer

- AdamW
- lr 2e-4
- betas (0.9, 0.999)
- eps 1e-8
- weight decay .01
- global treatment grad clip 1.0
- foreach false
- fused false
- no W-only optimizer/schedule

## Frozen fresh authority

- seed **64001**
- TRAIN 768
- DEV 192
- 12 wholly fresh S43 domains
- K=4
- two state views
- two question wording views
- two option views
- 24 epochs
- batch 16
- identical semantic rows/order.

## Primary matched metrics

Report treatment-reference deltas for:
- fused canonical/paraphrase;
- paired;
- question-swap;
- fused agreement/JS;
- fused canonical/paraphrase margins;
- raw primary canonical/paraphrase;
- relation canonical/paraphrase;
- relation margins/agreement;
- same-option signature cosine;
- signature same-vs-strongest-wrong discrimination margin.

Additionally:
- mean relative-gap anchor;
- conflict rate;
- mean pre/post-projection dot;
- mean applied gap-anchor dot;
- applied rounding bound;
- max runtime/W rounding ratios;
- selected W norm.

No scalar winner score.

## Relative-gap success criterion

S43 can be interpreted as protecting relative-gap geometry only if treatment avoids a material S41/S42-like degradation in:
- signature same-vs-strongest-wrong discrimination margin;

while not reintroducing collapse in:
- fused agreement;
- relation agreement;
- fused JS.

Existing DEV_READY gates remain authoritative. No new numeric gate is introduced.

## Frozen interpretation

A — correctness retained + relative-gap geometry protected:
material positive correctness while recovering from S41/S42 signature-discrimination failure without agreement/JS re-collapse.

B — correctness retained but gap geometry still collapses:
relative-gap anchoring is insufficient; close.

C — gap geometry protected but correctness collapses:
the constrained relative-gap movement was required for correctness; close.

D — treatment DEV_READY:
freeze immediately; no second S43 DEV; separate fresh confirmation before any external matched evaluation.

## Stop rule

No post-DEV:
- strongest-wrong variant;
- hinge/numeric margin target;
- gap weighting;
- S41/S42 anchor mixing;
- coefficient/slack;
- alternate norm;
- optimizer reinterpretation;
- partial detach;
- gradient mixing;
- W-only LR/scheduler;
- rank/factorization;
- residual scale/bias/nonlinearity;
- seed/LR/epoch/batch retry;
- gate weakening;
- second DEV.

No external Laya/Jev benchmark from S43 alone.
