# HIRA V1 S56 pre-A0 mechanical repair

Status: **PRE-EXPOSURE / SCIENCE UNCHANGED**

Failed core CI:
`37214624878`

Failure:
the identical-logit JS test observed `-2.02685e-8` from floating-point roundoff.

Mathematical invariant:
Jensen-Shannon divergence is non-negative and equals zero for identical distributions.

Repair:
`js_explicit.clamp_min(0.0)`

Scope:
- numerical underflow only;
- no coefficient change;
- no ordering threshold/margin change;
- no architecture/capacity change;
- no A0 exposure;
- no TRAIN/DEV exposure.

The existing identical-logit regression test remains the gate.
