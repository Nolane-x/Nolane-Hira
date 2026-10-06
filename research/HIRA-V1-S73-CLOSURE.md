# HIRA V1 S73 closure

Status: **CLOSED / CASE C**

Fresh authority:
- run `37480212536`
- artifact `11420533826`
- digest `sha256:0f08d05bea219333495529c6ae2c54d65fd9a276dd5f783adff0b0ac3925982f`
- scientific head `0d011eb4ed60032dc0447894b67dd152996bb858`.

S73 tested whether a tiny TRAIN-only safety predictor could selectively apply the S72 vector residual.

It did not.

The predictor's fresh DEV probabilities were centered around **0.3449**, and at the preregistered 0.5 threshold treatment accepted **0%** of cases.

Because treatment therefore reduced to the fused baseline everywhere, the small metric shifts are not evidence for learned selective safety.

Frozen verdict: **Case C**.

Scientific conclusion:
the S59→S73 bounded residual family is exhausted. Pairwise evidence is informative, but attaching it to the existing fused core through scalar/vector residuals and safety gates is not reliably converting evidence into decisions.

The next stage must redesign the **decision core**, not add another residual, gate, threshold, target, or feature tweak.
