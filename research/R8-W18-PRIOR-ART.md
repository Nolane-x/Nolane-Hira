# R8-W18 prior-art boundary — latent-field extraction vs deterministic typed composition

Frozen before any W18 empirical exposure.

W18 does not claim novelty for neuro-symbolic reasoning, structured latent variables, probabilistic composition, semantic parsing, or deterministic rule execution.

## 1. Neural extraction separated from symbolic reasoning

Sharma, Sharma & Bedi, SemEval 2026:
"Lakksh at SemEval-2026 Task 11(1 & 2): Neuro-Symbolic Decomposition to Mitigate Content Bias in Syllogistic Reasoning"
https://aclanthology.org/2026.semeval-1.17/

Their architecture deliberately separates neural structure extraction from a deterministic symbolic validity checker. W18 uses this only as motivation for separating semantic field extraction from typed composition.

## 2. Translation/abstraction can be narrower than reasoning

Advani et al., SemEval 2026:
"lakshadvani at SemEval-2026 Task 11: A Neuro-Symbolic Approach to Content-Independent Syllogistic Reasoning"
https://aclanthology.org/2026.semeval-1.80/

The system restricts the language model to translation/abstraction and delegates deduction to a deterministic checker. W18 analogously asks whether HIRA's semantic geometry should identify atomic latent field values while deterministic typed equations handle known composition.

## 3. Extraction quality remains a separate bottleneck

Gupta, Goyal & Bedi, SemEval 2026:
"0704mis at SemEval-2026 Task 11: Single-Call Joint Abstraction for Robust Neuro-Symbolic Retrieval"
https://aclanthology.org/2026.semeval-1.244/

The reported ablations emphasize that a sophisticated downstream symbolic method cannot compensate when abstraction/extraction is wrong. This motivates W18's explicit atomic extraction gates before interpreting a composition result.

## 4. Architectural separation, not a broad claim

The cited work is about formal syllogistic reasoning, not HIRA, typed customer-service schemas, or semantic routing.

W18 therefore does not import:
- their models;
- datasets;
- prompts;
- formal languages;
- training losses;
- thresholds.

The HIRA-specific question comes from W17:
- field isolation caused very large diagnosis gains;
- direct response/review/urgency semantic matching stayed weak;
- those typed labels are deterministic functions of severity/confidence in the authority generator.

W18 freezes:
- atomic intent/severity/confidence extraction;
- deterministic hard composition;
- frozen probabilistic soft composition;
- oracle equation sanity;
- independent reference adequacy.

No post-exposure prior-art reading may alter:
- latent definitions;
- composition equations;
- gates;
- fresh domains/seeds;
- classification precedence.
