# R8-W29 closure — HIRA v0 semantic-core integration

Status: **CLOSED — W29_RUNTIME_INTEGRATION_FAIL**

Branch: `feat/r8-w29-hira-v0-semantic-core`  
Base main: `953c3324a43bd0f14f285dc13a6720ef9d996f87`  
Authoritative head: `4ba5b11f3c830adb9446d59fca8fd58c9f3cdcf2`  
PR: #148  
Issue: #147

## Authority

- workflow run: `36247969053`
- authoritative audit artifact: `10908298353`
- artifact digest: `sha256:ba5e881f7f8434d7f4785f5c9ec7ad4509f3dd4586c9e14e90e5cfab88baab1e`
- frozen T0 artifact copied by W29: `10908396270`
- all authority jobs completed successfully; the scientific verdict is negative, not an infrastructure failure.

## Frozen scientific outcome

`W29_RUNTIME_INTEGRATION_FAIL`

Reference adequacy passed on all fresh domains EW/EX/EY/EZ.

Runtime integrity passed on all fresh domains EW/EX/EY/EZ and globally:
- exact W28 T0 provenance;
- projection parameter count 32,768;
- projection trainable parameter count 0;
- relation refinement disabled;
- relation delta max abs = 0;
- full-K rate = 1.0;
- option-order invariance = 1.0;
- state-once rate = 1.0;
- schema cache reuse passed;
- probability mass max error = 1.1920928955078125e-07;
- state encode calls = 384 for 384 cases.

HIRA semantic quality did **not** pass any of the four domains.

Pooled HIRA:
- F0 top-1: 0.9244791666666666
- F1 top-1: 0.7578125
- F2 top-1: 0.7864583333333334
- factor-vector top-1: 0.5416666666666666
- composed severity top-1: 0.5416666666666666
- invalid factor-vector rate: 0.06770833333333333
- cross-primitive agreement: 1.0 for F0/F1/F2 across choice/score/noul.

Reference pooled:
- F0 BA: 1.0
- F1 BA: 1.0
- U BA: 1.0
- C BA: 1.0
- composed F2 BA: 1.0

The direct-F2 diagnostic was weaker, as preregistered, but the composed reference authority passed exactly.

## Interpretation boundary

W29 proves that the production runtime port is mechanically faithful and typed/state-once integration is correct.

W29 does **not** prove that the rescued W28 T0 projection transfers strongly enough to compact, newly worded production schema views.

The negative result therefore localizes the next work to semantic transfer / schema representation rather than:
- typed primitive wrappers;
- state caching;
- option ordering;
- probability normalization;
- relation refinement;
- candidate truncation;
- external reference adequacy.

No post-exposure W29 model, schema wording, gate, or runtime-path modification is authorized.

EW/EX/EY/EZ are permanently exposed and forbidden for future tuning or selection.

## Next wave

W30 must use wholly fresh training/dev/confirm material and attack semantic-transfer robustness without consulting W29 rows for tuning.

No `HIRA v0 semantic core` readiness claim is authorized by W29.
