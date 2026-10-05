# HIRA V1 S60 closure

Status: **CLOSED / CASE B**

Fresh scientific court:
- run `37297784471`
- artifact `11340397535`
- digest `sha256:c55205e9b74aac6196346a0ce9b705900aaec7fe4c6b30380af8ee0093d1b18b`
- scientific head `7352d5bd5d52233f0d6b815682102ce226a98209`.

S60 preserved the S59 pairwise signal only as a bounded residual with one TRAIN-calibrated global alpha.

Result:
- canonical **+0.52 pp**
- paraphrase **+0.78 pp**
- paired both-correct **-1.56 pp**
- question-swap **0.00 pp**
- agreement **+0.26 pp**
- JS **+0.000412** worse
- selected alpha **0.1050317**.

Interpretation: the residual composition is safe for correctness, but a single global alpha cannot know when pairwise evidence should matter. S60 is not DEV_READY and is closed without retry.

Next: S61 confidence-adaptive bounded hybrid gate.

Forbidden after closure:
- S60 alpha sweep
- S60 retry / second DEV
- post-hoc gate addition inside S60
- selector change
- native retraining
- external Laya/Jev evaluation.
