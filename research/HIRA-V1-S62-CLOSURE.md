# HIRA V1 S62 closure

Status: **CLOSED / CASE B**

S62 asked whether explicit TRAIN-only counterfactual reliability supervision could make the existing four-feature S61 gate recover the strong S59 stability signal without giving up correctness.

Fresh authority:
- run `37307658289`
- artifact `11344855774`
- digest `sha256:7812c348d707ea8675fc9370af573a87d9e5b1a0b10e55a1f335bf890ddb3305`
- scientific head `f86e6b6296737d6e3274c93498217054eb7bc3a6`.

Answer: **no, not with the four S61 scalar features.**

Reliability BCE made the gate substantially more conservative:
- reference mean alpha **0.105502400547266**
- treatment mean alpha **0.08251461200416088**.

But treatment did not improve stability over the gold-CE reference:
- agreement delta **-0.26 pp**
- JS delta **0.00024088782568770783**
- canonical accuracy delta **0.00 pp**
- paraphrase accuracy delta **-0.26 pp**.

Frozen verdict: **Case B**.

Authorized handoff:
- retain bounded pairwise residual and correctness-veto reliability target;
- retain detached/upstream-isolated gate training;
- move to a richer TRAIN-only reliability representation/head;
- do not retry or retune S62.

S62 is scientifically exhausted.
