# HIRA V1 S61 closure

Status: **CLOSED / CASE B**

S61 asked whether a tiny 5-parameter confidence-adaptive gate could preserve S60 correctness while recovering the strong S59 pairwise stability signal.

Fresh authority:
- run `37302829277`
- artifact `11343045582`
- digest `sha256:6e99ba1a5fab92cb6031c41eea54636721b78e690236dfb39788066803785f74`
- scientific head `d731fe8c33e97279681116327bc0500cd340bb83`.

Answer: **no, not with gold-CE-only gate calibration.**

The gate learned non-constant alpha but did not improve selected-choice agreement and worsened cross-view JS. Correctness remained broadly safe, so bounded adaptive composition itself is not rejected.

Frozen verdict: **Case B**.

Authorized handoff:
- retain bounded hybrid composition;
- retain detached confidence features as candidate inputs;
- replace pure gold-CE gate supervision with a preregistered TRAIN-only reliability/usefulness target;
- no S61 retry.

S61 is scientifically exhausted.
