# HIRA V1 S18 handoff — to S19 Expert-Separated Triadic View Consistency

S18 is frozen as:

`HIRA_V1_S18_PAIRED_VIEW_MARGIN_DEV_FAIL`

Canonical authority:
- run `36583384787`
- artifact `11042370479`
- digest `sha256:b97df1baf7b5a23ed04cab55f6f815ce757a32004da2df039c29bdbd0f57bc8c`
- selected epoch **10**
- checkpoint `bf4e378c3fd14ce6be2b0bb65b176aa095c58fdeecd2fad27d593063899368dc`

Selected DEV:
- fused canonical **0.703125**
- fused paraphrase **0.4114583333**
- paired **0.5208333333**
- question-swap **0.78125**
- fused agreement **0.4739583333**
- fused canonical margin **0.2828457902**
- fused paraphrase margin **-0.1637737309**
- raw triadic canonical **0.5078125**
- raw triadic paraphrase **0.34375**
- raw triadic agreement **0.4791666667**
- relation canonical **0.421875**
- relation paraphrase **0.390625**
- signature cosine **0.9314097762**
- signature margin **0.1460411654**

TRAIN paired-margin loss:
- **0.4779098456 -> 0.0725741908**

## Key diagnosis

Output-level paired hinge learns on TRAIN but fails to generalize to the fresh paraphrase view and damages S17 relation quality.

S17 had:
- relation canonical **0.640625**
- relation paraphrase **0.6380208333**
- raw triadic canonical **0.5911458333**
- raw triadic paraphrase **0.4427083333**

The relation expert was already much more view-stable than the triadic expert.

## S19 target

**Expert-Separated Triadic View Consistency**

Start from the S17 scientific objective, not the failed S18 addition.

Keep:
- S17 norm-balanced shared-gradient rule
- exact 49,152 trainable physical params
- original A13/HIRACore frozen
- S14 equal-weight inference fusion
- relation CE and relation-signature canonicalization unchanged
- option alignment unchanged
- no paired-margin term
- 0 learned heads/routers

Controlled change:
- replace the S17 primary fused-output symmetric JS term with symmetric JS on **raw triadic canonical/paraphrase logits**;
- keep the same coefficient **0.25**;
- do not add an additional JS coefficient.

Hypothesis:
- relation canonicalization already provides branch-specific relation invariance;
- triadic-specific JS should address the weak expert directly;
- avoiding fused JS prevents the stable relation expert from masking triadic instability during consistency training.

A0 must prove:
- inference numerically identical to S17;
- raw-triadic JS is permutation equivariant;
- zero for identical triadic distributions;
- nonzero finite gradient for mismatched views;
- exact 49,152 surface;
- 0 added parameters/state.

Use wholly fresh S19 TRAIN/DEV.
Do not reuse S18 DEV.
Do not tune the 0.25 coefficient.
Do not reopen Laya/Jev before DEV_READY.
