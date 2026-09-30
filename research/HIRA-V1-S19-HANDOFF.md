# HIRA V1 S19 handoff — to S20 Standardized Triadic Evidence Consistency

S19 is frozen as:

`HIRA_V1_S19_TRIADIC_VIEW_CONSISTENCY_DEV_FAIL`

Canonical authority:
- run `36644174421`
- artifact `11068688127`
- digest `sha256:2f2f70f4d83704f5701238171faa0e23187bfe18293d8a96ffb793cc904762be`
- selected epoch **20**
- checkpoint `af302063a4188fdaa0c61d29810b2ae15dd77059767b002d83d61d4adaf930af`

Selected DEV:
- fused canonical **0.6302083333**
- fused paraphrase **0.453125**
- paired **0.359375**
- fused agreement **0.6067708333**
- question-swap **0.71875**
- raw triadic canonical **0.4609375**
- raw triadic paraphrase **0.3567708333**
- raw triadic agreement **0.5494791667**
- relation canonical **0.4791666667**
- relation paraphrase **0.4166666667**
- signature cosine **0.9174721142**
- signature margin **0.1374289120**

Critical finding:
- S19 TRAIN raw-triadic JS remains only **~4.6e-9 to ~1.15e-8** across all 24 epochs.
- Yet triadic top-1 agreement ranges **0.4661 to 0.8854** on DEV.

Therefore raw-softmax distribution similarity is not a useful proxy for relative triadic evidence/ranking stability.

## S20 target

**Standardized Triadic Evidence Consistency**

Return to S17 as the scientific base.

Keep:
- S17 norm-balanced shared-gradient optimizer
- exact 49,152 physical trainable params
- original A13/HIRACore frozen
- S14 equal-weight full-K inference fusion
- relation CE/signature canonicalization unchanged
- option alignment unchanged
- no S18 paired-margin term
- no S19 raw-logit JS
- 0 learned heads/routers

Controlled change:

For raw triadic logits `x`:
1. center over K;
2. divide non-flat centered vector by its RMS evidence magnitude;
3. epsilon **1e-6**;
4. flat vector -> zero neutral evidence.

Then:
`L_std = mean((z_c - z_p)^2)`

Use coefficient **0.25** in the primary block in place of S17 fused JS.

Required A0 invariants:
- zero learned params/state
- exactly zero for identical standardized evidence
- common positive affine transform `a*x+b` (a>0) leaves standardized evidence/loss unchanged within tolerance
- option permutation equivariant
- canonical/paraphrase swap symmetric
- mismatched relative evidence gives nonzero finite gradient
- flat expert handled without NaN
- inference numerically unchanged
- exact 49,152 physical surface

Wholly fresh S20 authority.
Do not reuse S19 DEV.
Do not coefficient/epsilon tune after DEV.
Do not reopen Laya/Jev before DEV_READY.
