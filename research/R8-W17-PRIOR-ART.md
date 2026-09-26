# R8-W17 prior-art boundary — field-isolated state representation

Frozen before any W17 empirical exposure.

W17 does not claim novelty for:
- structured retrieval;
- multi-field retrieval;
- field-aware retrieval;
- query-conditioned field weighting;
- batching multiple structured fields through one encoder.

The HIRA-specific question is narrower:

> after W16 proved that unrelated typed fields contaminate the frozen A13/W9 contextual token geometry, can preserving field identity before self-attention restore typed semantic competence?

## 1. Multi-Field Adaptive Retrieval

Millicent Li, Tongfei Chen, Benjamin Van Durme, Patrick Xia.
"Multi-Field Adaptive Retrieval" (2024).
https://arxiv.org/abs/2410.20056

MFAR decomposes structured documents into fields and indexes field evidence independently before adaptive combination.

Relevance to W17:
- supports treating field structure as meaningful rather than flattening everything into one text.

Not imported:
- adaptive field weighting;
- retriever training;
- document-index design.

## 2. Field Aware Agent Skill Retrieval

Paimon Goulart, Liang Wu, Kelly Wan, Evangelos E. Papalexakis, Liangjie Hong.
"Field Aware Agent Skill Retrieval" (2026).
https://arxiv.org/abs/2608.02880

The work studies structured skill objects and reports that preserving fields such as name, description and body can improve retrieval compared with flat concatenation.

Relevance:
- directly motivates testing representation structure itself as a causal variable.

Not imported:
- learned field MLP;
- sparse+dense fusion;
- benchmark-specific training.

## 3. Multi-Field Tool Retrieval

Yichen Tang, Weihang Su, Yiqun Liu, Qingyao Ai.
"Multi-Field Tool Retrieval" (2026).
https://arxiv.org/abs/2602.05366

The work models distinct aspects of tool utility rather than treating raw tool documentation as one undifferentiated text.

Relevance:
- structured semantic aspects can require different matching treatment.

Not imported:
- its model architecture;
- tool-specific objectives;
- public benchmark labels.

## 4. THYME

Da Li, Keping Bi, Jiafeng Guo, Xueqi Cheng.
"Tailoring Table Retrieval from a Field-aware Hybrid Matching Perspective."
EMNLP 2025.
https://aclanthology.org/2025.emnlp-main.1409/

THYME reports that different table fields have different matching preferences.

Relevance:
- motivates retaining field identity through representation/matching.

Not imported:
- hybrid sparse/dense design;
- table-specific matching modules.

## 5. Context interference

Boyang Xue et al.
"Mitigating Context Interference for Reliable and Efficient Search Agents."
ACL 2026.
https://aclanthology.org/2026.acl-long/

The broader finding is that irrelevant context can reduce downstream reliability.

W17 does not import the paper's search-agent refinement method. The HIRA causal evidence comes from W16's matched R0/R1/R2 experiment.

## 6. Frozen W17 boundary

No external paper selects:
- intent/severity/confidence fields;
- primitive-to-field mapping;
- full/triplicate/isolated controls;
- three-sequence batch size;
- fresh domains;
- seeds;
- competence thresholds;
- causal gain thresholds;
- verdict precedence.

The field schema follows W15/W16's own typed gold construction.

No post-exposure literature search may alter the W17 protocol.
