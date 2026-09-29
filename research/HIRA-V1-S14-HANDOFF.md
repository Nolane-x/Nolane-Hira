# HIRA V1 S14 handoff — to S15 Gradient-Isolated Evidence Fusion

S14 is frozen as:

`HIRA_V1_S14_EVIDENCE_FUSION_DEV_FAIL`

## Canonical evidence

A0:
- run `36527874581`
- artifact `11015447511`
- digest `sha256:3fccd10b3bde77a4396cec491d8a01a36c129e7d917dedfc688ff03fb0832976`

TRAIN/DEV scientific exposure:
- run `36528589774`
- 24/24 epochs completed; post-TRAIN upload was blocked only by metadata verification

Packaged reproduction:
- run `36530243773`
- artifact `11017300477`
- digest `sha256:3354d9c635e1cad0a8062c3d8f74344e8e4121fd92a3c68947adfdc93c569b05`
- selected epoch **16**
- checkpoint SHA256 `c3e54f33a7bcb683a1d4c9720d037064fbe967d9542b6fcf72e4d456dde3bb17`

Selected DEV:
- fused canonical accuracy **0.6354166667**
- fused paraphrase accuracy **0.5390625**
- fused paired both-correct **0.375**
- fused question-swap change **0.6979166667**
- fused canonical margin **0.1034617118**
- fused cross-view agreement **0.5572916667**
- raw triadic canonical accuracy **0.5260416667**
- relation canonical accuracy **0.5833333333**
- relation canonical margin **0.0116562198**
- same-option signature cosine **0.7023841192**
- signature margin **0.0355867463**
- option-order flip **0.0**
- full-K/state-once/relation-delta-zero PASS

## Key result

S14 produces genuine fusion synergy:

`fused 63.54% > relation 58.33% > triadic 52.60%`

It is the strongest v1 primary semantic result so far.

But relation signatures regress relative to S13 while the fused decision improves. The current fused-primary objective backpropagates through both experts, while relation also receives dedicated relation CE + canonicalization gradients.

## Do not do

Do not:
- tune S14 0.5/0.5 weights or epsilon;
- choose another S14 epoch after DEV;
- tune S14 objective coefficients or temperatures;
- reuse S14 DEV rows;
- add a learned fusion gate;
- increase capacity by default;
- reopen M5/Laya/Jev.

## S15 target

**Gradient-Isolated Evidence Fusion**

Keep S14 numerical inference unchanged:

`fused = 0.5 * standardized(triadic) + 0.5 * standardized(relation)`

Change only primary-loss autograd:

`protected_fused = S14_fusion(triadic, stop_gradient(relation))`

Therefore:
- fused CE / swap margin / fused cross-view JS send direct gradients through triadic path only;
- relation CE remains non-detached;
- relation-signature canonicalization remains non-detached;
- option alignment remains unchanged;
- physical trainable capacity remains 49,152;
- added learned params remain 0.

This is gradient-route isolation, not parameter isolation; both experts still share the same LoRA/projection weights.

S15 must answer:

**Can HIRA preserve S14 fusion gain while protecting relation semantics from direct fused-primary interference?**

If not, the next bottleneck is the shared parameter surface itself.
