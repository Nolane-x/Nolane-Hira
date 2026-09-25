# R8-W13 prior-art note — semantic consistency as reliability

Status: **FROZEN PRE-DATA MOTIVATION.**

This note is motivation only. It does not alter W13 gates after empirical exposure.

## 1. Query-variation robustness

Hagen, Scells & Potthast, Findings of EMNLP 2024:
https://aclanthology.org/2024.findings-emnlp.248/

Relevant lesson:
retrieval rankings can be unstable under meaning-preserving query variations, including paraphrases. Therefore ranking invariance across paraphrases is a legitimate robustness/reliability property rather than a cosmetic text property.

W13 uses this only to motivate measuring frozen HIRA semantic-rank consistency. It does not import their training method or thresholds.

## 2. Paraphrastic consistency

Raj et al./TACL-style paraphrastic consistency work and related 2024 consistency studies motivate measuring whether a model preserves decisions under alternate formulations of the same semantics.

W13's D0/D1/D2 views are fixed before exposure. Agreement is label-free:
- STRICT: all three top1 candidate IDs agree;
- MAJORITY: two agree;
- SPLIT: all differ.

No generated paraphrase after exposure is permitted.

## 3. Confidence-aware / adaptive reranking

AcuRank:
https://arxiv.org/abs/2505.18512

CAR:
https://arxiv.org/abs/2605.04495

Relevant lesson:
reranking/refinement should not automatically be assumed beneficial for every query; uncertainty/reliability may justify selective intervention.

W13 does not reuse their models. It tests whether HIRA's own semantic-view consistency identifies when its production competitive path should or should not override candidate-independent semantic evidence.

## 4. Dense semantic understanding

SURE (ACL 2026):
https://aclanthology.org/2026.acl-long.2127/

Relevant lesson:
dense retrievers can struggle with fine-grained semantic equivalence and lexical paraphrases even when aggregate retrieval metrics look good.

This supports W13's decision to inspect candidate-level semantic equivalence stability explicitly instead of inferring reliability from one scalar margin.

## 5. Frozen HIRA-specific invariant

W10-W12 repeatedly showed:
- candidate-independent semantic geometry is often stronger than the complete production path;
- production transforms can destroy correct semantic rankings;
- the W12 top1-top2/MAD margin does not stably identify when this damage will occur.

W13 therefore asks one narrower question:

**Does independent semantic agreement across three meaning-equivalent schema descriptions provide a more stable reliability signal than a single-view score margin?**

## 6. Scientific boundary

Prior art does not authorize a new production gate.

W13 remains diagnostic:
- no training;
- no fitting;
- no weight search;
- no paraphrase search after exposure;
- no threshold search;
- no MiniLM supervision.

Only a preregistered >=3/4-domain stable classification may authorize the next mechanism lane.
