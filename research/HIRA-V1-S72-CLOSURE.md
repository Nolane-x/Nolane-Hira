# HIRA V1 S72 closure

Status: **CLOSED / CASE B**

Fresh authority:
- run `37470998139`
- artifact `11416962628`
- digest `sha256:9a653a681e0cb3182819634fe24eb345ce82a48f2d078d15e510d1d3ebfb0834`
- scientific head `197d2df295e6b44a270b771ae38b8b159f969611`.

S72 changed only residual direction geometry.

The intervention was real:
- direction max difference **0.895**
- direction cosine **0.964**
- alpha policy **exactly matched**
- raw pairwise evidence **exactly matched**.

But final result was mixed:
- canonical **-0.52 pp**
- paraphrase **-0.52 pp**
- both-correct **-2.08 pp**
- agreement **+1.04 pp**
- question-swap **+0.52 pp**
- JS slightly worse.

Frozen verdict: **Case B**.

Scientific conclusion:
the residual can move decisions meaningfully, but it is unsafe to apply uniformly. S73 must learn a hard TRAIN-only counterfactual safety veto for the already-frozen S72 candidate residual. Do not sweep residual geometry again.
