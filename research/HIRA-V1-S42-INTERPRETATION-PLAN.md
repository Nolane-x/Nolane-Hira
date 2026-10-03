# HIRA V1 S42 interpretation plan — frozen before S42-A0/DEV exposure

Status: **FROZEN**

Issue: #267

## Controlled comparison

Reference:
- exact native S35/S17 runtime
- exact S41 frozen AdamW semantics

Treatment:
- exact S38/S41 full bilinear joint co-adaptation
- W 256x256 / 65,536
- runtime+W total 114,688
- actual AdamW runtime movement constrained by the S42 full cross-view KxK relational signature anchor
- W candidate AdamW movement unchanged

## Frozen anchor

For normalized canonical/paraphrase option signatures:

`G = S_c @ S_p^T`

Anchor:

`A_rel = mean((G_treatment - stopgrad(G_reference))^2)`

No coefficient.
No diagonal/off-diagonal weighting.
No margin.
No gold labels.
No learned anchor params.

## Frozen actual-step projection

For relational-anchor gradient `a` and actual candidate AdamW runtime movement `delta_r`:

- if `a·delta_r <= 0`: identity
- if `a·delta_r > 0`: remove exactly the conflicting component using epsilon 1e-12

W candidate delta remains unchanged.

AdamW moment/step state remains the state produced by the frozen clipped raw correctness gradient.

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

- seed **63001**
- TRAIN 768
- DEV 192
- 12 wholly fresh S42 domains
- K=4
- two state views
- two question wording views
- two option views
- 24 epochs
- batch 16
- identical semantic rows/order between arms

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
- same-option signature cosine
- signature same-vs-strongest-wrong discrimination margin

Additionally report:
- mean relational anchor
- actual-step conflict rate
- mean pre-projection dot
- mean projected dot
- mean applied relational-anchor dot
- applied rounding bound
- max runtime/W rounding ratios
- selected W norm
- row/order identity

No scalar winner score.

## Relative-geometry success criterion

The central transport distinction versus S41 is the signature discrimination failure.

S42 can be interpreted as protecting relative geometry only if the treatment avoids a material S41-like degradation in:
- signature same-vs-strongest-wrong margin

while not reintroducing S38-like collapse in:
- fused cross-view agreement
- relation cross-view agreement
- fused JS.

The existing frozen DEV_READY gates remain authoritative; this paragraph does not create a new gate or numeric threshold.

## Frozen interpretation

A — correctness retained + relative geometry protected:
material S41-like correctness gain with recovery from the S41 signature-discrimination failure.

B — correctness retained but relative geometry still collapses:
cross-view KxK relational anchoring is insufficient.

C — relative geometry protected but correctness collapses:
the constrained relational movement was required for S41 correctness.

D — treatment DEV_READY:
freeze immediately; no second S42 DEV.

## Stop rule

No post-DEV:
- per-signature + relational anchor mixing
- diagonal/off-diagonal weighting
- anchor coefficient/slack
- margin addition
- alternate matrix norm
- optimizer reinterpretation
- partial detach
- gradient mixing
- W-only LR/scheduler
- rank/factorization
- residual scale/bias/nonlinearity
- seed/LR/epoch/batch retry
- gate weakening
- second DEV

No external Laya/Jev benchmark from S42 alone.
