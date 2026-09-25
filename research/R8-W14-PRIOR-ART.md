# R8-W14 prior-art note — continuous multiview semantic reliability

Status: **PRE-DIAGNOSTIC. Frozen before BV/BW/BX/BY empirical exposure.**

This note exists to prevent overclaiming novelty and to sharpen the scientific question behind W14.

## 1. What prior work already establishes

### Meaning-preserving paraphrases as a consistency probe

Elazar et al., *Measuring and Improving Consistency in Pretrained Language Models* (TACL 2021):
https://aclanthology.org/2021.tacl-1.60/

The work treats invariance under meaning-preserving alternations as a desirable model property and shows large variation in factual consistency across paraphrases.

HIRA implication:
- paraphrase agreement itself is not novel;
- W14 may only claim to test whether a specific candidate-ranking consistency representation predicts reliability in HIRA.

### Semantic consistency as reference-less performance evidence

Rabinovich et al., *Predicting Question-Answering Performance of Large Language Models through Semantic Consistency* (GEM 2023):
https://aclanthology.org/2023.gem-1.12/

They use semantic consistency across equivalent formulations as part of a framework for predicting QA performance without a reference answer.

HIRA implication:
- consistency-as-confidence has clear precedent;
- W14's contribution, if any, is restricted to the HIRA architecture and frozen rank-based diagnostic.

### Paraphrastic variability as an error source

Srikanth, Carpuat & Rudinger, *How Often Are Errors in Natural Language Reasoning Due to Paraphrastic Variability?* (TACL 2024):
https://aclanthology.org/2024.tacl-1.63/

They explicitly quantify paraphrastic consistency and separate reasoning performance from variability induced by wording.

HIRA implication:
- W14 should distinguish semantic competence from sensitivity to schema wording;
- agreement is evidence about robustness, not proof of correctness.

### Confidence-weighted self-consistency

Taubenfeld et al., *Confidence Improves Self-Consistency in LLMs* (Findings ACL 2025):
https://aclanthology.org/2025.findings-acl.1030/

Confidence-informed aggregation can reduce the number of sampled reasoning paths needed by weighting them by confidence.

HIRA implication:
- combining independent views and confidence is prior art;
- W14 deliberately avoids learned weights and uses an equal-weight frozen ensemble.

### Similarity/consistency for uncertainty quantification

Bhattacharjya et al., *SIMBA UQ: Similarity-Based Aggregation for Uncertainty Quantification in Large Language Models* (Findings EMNLP 2025):
https://aclanthology.org/2025.findings-emnlp.859/

The work studies consistency/similarity among multiple outputs as a proxy for uncertainty.

HIRA implication:
- multiview agreement as uncertainty evidence is not novel;
- W14 tests whether agreement among semantic *candidate rankings* transfers across domains.

### Ranking stability under paraphrases

Schlegel et al., *PRSM: A Measure to Evaluate CLIP's Robustness Against Paraphrases* (2025):
https://arxiv.org/abs/2511.11141

PRSM explicitly treats ranking stability under paraphrased text as a robustness object in a multimodal retrieval setting.

HIRA implication:
- rank stability under paraphrases has direct conceptual precedent;
- W14 must not claim novelty for Spearman ranking stability or paraphrase robustness metrics.

### Recent controlled paraphrase-robustness evaluation

Alhetelah & Ahmad, *Measuring LLMs' Sensitivity to Paraphrased Opinion Prompts* (WASSA 2026):
https://aclanthology.org/2026.wassa-1.5/

This work evaluates stability across multiple human-validated paraphrases under deterministic inference and reports complementary stability metrics.

HIRA implication:
- using multiple deterministic paraphrases and multiple agreement metrics is established methodology.

## 2. What W14 specifically tests

W14 is not proposing a new general uncertainty method.

It tests one HIRA-specific claim:

> When the exact frozen W9 semantic projection produces similar candidate rankings across three independently worded definitions of the same schema, does a simple label-free combination of vote strength, full-rank stability and top-3 overlap provide a stable reliability ordering across wholly fresh semantic domains?

The three components are deliberately elementary:
- V: top1 vote strength;
- S: full-rank Spearman stability;
- O: top-3 set overlap.

The frozen score is:
`R = (V + S + O) / 3`.

No weight is learned.
No feature is selected from W13.
No label participates in R.

## 3. Why W14 follows W13 rather than replacing its result

W13 showed:
- STRICT agreement was extremely accurate when present;
- but exact 3/3 K16 coverage fell below the preregistered 25% support floor on BR/BS/BT;
- the frozen W13 verdict therefore remained unresolved.

W14 does not lower that threshold.

Instead it asks whether consistency can be represented continuously over all 64 cases/domain/K.

This is a new fresh diagnostic, not a reanalysis or rescue of W13.

## 4. Scientific boundary

W14 may conclude only:
- whether continuous consistency provides stable reliability localization under its frozen gates;
- whether multiview anchor dominance or low-reliability production value is stable;
- whether the signal is unresolved.

W14 may not claim:
- a novel uncertainty-quantification method;
- a production-safe gate;
- a trained confidence model;
- superiority to the cited methods;
- broad semantic robustness;
- that the arithmetic V/S/O combination is optimal.

If W14 is successful, a later phase must still validate a bounded mechanism on fresh TRAIN/DEV/dual-CONFIRM data before any production claim.
