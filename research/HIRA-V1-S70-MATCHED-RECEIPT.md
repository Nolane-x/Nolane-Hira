# HIRA V1 S70 matched scientific receipt — Fused-Anchored Weighted Pairwise Aggregation

Status: **FROZEN / CASE C**

Scientific run: `37423476142`  
Artifact: `11394610000`  
Artifact digest: `sha256:8af711f789efb9717eaed0eff882fade7bab37b546f86a35502db82242ad221f`  
Scientific head: `b41620e3c5cb29ea40c5428b8200e28016a81b76`

Outcome:
`HIRA_V1_S70_FUSED_ANCHORED_PAIRWISE_DEV_COMPLETE`

A0:
- run `37422000871`
- artifact `11393387610`
- digest `sha256:6e021d9fe8807eb6ff097bfe16b9bb464a8ee87e949d064883ec43d6cfe5a9ff`.

## Controlled variable

Both arms:
- exact S69 query-gated interaction representation
- representation params **0**
- one shared S59 pairwise head **32,832 params**
- one shared pairwise training trajectory
- one shared correction trajectory
- exact S64 contextual gate **60 params per arm**
- exact S66 responsibility objective
- identical optimizer/selector/context path.

Reference aggregation:
- uniform row mean.

Treatment aggregation:
- fused-softmax opponent weighting, temperature **1.0**.

Aggregation params:
- **0 vs 0**.

## Pairwise evidence isolation

Selected epoch: **7** both arms.

Raw pairwise evidence is exactly matched:
- gold-pair accuracy **0.5525173611 vs 0.5525173611**
- gold-pair margin **0.0173085642 vs 0.0173085642**.

Therefore any outcome difference is attributable to aggregation/gate policy only.

## Aggregate-level treatment minus reference

- canonical accuracy **+0.260417 pp**
- paraphrase accuracy **+1.822917 pp**
- paired both-correct **+0.520833 pp**
- canonical margin **+0.007743**
- paraphrase margin **+0.004174**
- question-swap **-1.5625 pp**
- agreement **-1.5625 pp**
- JS **-0.00009477 better**.

This is a small mixed shift, not a material aggregate breakthrough.

## Final decision

Reference:
- canonical accuracy **0.3880208333**
- paraphrase accuracy **0.3776041667**
- paired both-correct **0.1354166667**
- question-swap **0.5000000000**
- agreement **0.3828125000**
- JS **0.1135650751**.

Treatment:
- canonical accuracy **0.3906250000**
- paraphrase accuracy **0.3776041667**
- paired both-correct **0.1406250000**
- question-swap **0.5000000000**
- agreement **0.3828125000**
- JS **0.1132933110**.

Treatment minus reference:
- canonical **+0.260417 pp**
- paraphrase **0.00 pp**
- paired both-correct **+0.520833 pp**
- question-swap **0.00 pp**
- agreement **0.00 pp**
- JS **-0.000271764 better**
- canonical margin **+0.000794**
- paraphrase margin **+0.000111**.

## Frozen interpretation

**Case C.**

Fused-anchored opponent weighting causes only microscopic/mixed changes. Uniform row averaging is therefore not the main bottleneck.

Per the preregistered plan, do not sweep temperature/top-k/weighting. Move directly to a richer correctness-preserving composition operator that can consume more than one scalar pairwise aggregate per option.

No second S70 DEV or external Laya/Jev evaluation is authorized.
