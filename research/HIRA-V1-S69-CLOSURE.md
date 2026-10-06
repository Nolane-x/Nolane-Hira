# HIRA V1 S69 closure

Status: **CLOSED / CASE B**

Fresh authority:
- run `37420104460`
- artifact `11392672918`
- digest `sha256:c24c00e9b79d23b3e7e3fe4d17013270e0410d822b85c62230bb85e972a2f94d`
- scientific head `2c856e483a1b12d21dd249758919cbc46ea67d26`.

S69 tested a zero-parameter query-gated identity interaction before pair subtraction.

It worked at the pairwise-evidence level:
- gold-pair accuracy **+3.43 pp**
- gold-pair margin **+0.02466**
- pairwise aggregate paraphrase accuracy **+4.95 pp**
- pairwise aggregate canonical accuracy **+1.04 pp**
- pairwise aggregate question-swap **+10.94 pp**.

But downstream final transfer remained weak:
- canonical **+0.26 pp**
- paraphrase **+0.26 pp**
- paired both-correct **0**
- agreement **-0.26 pp**
- JS slightly worse.

Frozen verdict: **Case B**.

Scientific conclusion:
S69 resolves a real representation bottleneck. Freeze the treatment representation. The remaining bottleneck is how the pairwise matrix is aggregated/composed into the final decision.

Do not reopen S62-S69 target/representation sweeps.
