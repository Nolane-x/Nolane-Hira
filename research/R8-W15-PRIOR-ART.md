# R8-W15 prior-art boundary — anchor-preserving bounded residual redesign

Status: frozen before any W15 BZ/CA/CB/CC/CD/CE/CF empirical exposure.

## Purpose

W15 does not claim novelty for residual ranking, score fusion, late interaction, anchor-based reranking, or bounded logit corrections.

The HIRA-specific question is narrower:

> after W14 stably showed 4/4 fresh-domain multiview semantic-anchor dominance, can the anchor be made the primary production signal while competitive and relation reasoning are restricted to a tiny bounded residual?

## Relevant motivation

### ResRank

Ke et al., 2026, “ResRank: Unifying Retrieval and Listwise Reranking via End-to-End Joint Training with Residual Passage Compression.”

https://arxiv.org/abs/2604.22180

Relevant only as motivation that residual connections can be useful when retrieval and reranking representations must coexist.

W15 is not a reproduction of ResRank:
- no LLM reranker;
- no passage compression;
- no joint encoder training;
- no autoregressive ranking.

### Anchor-based reranking stress test

Ghosh & Chatterjee, 2026, “When Do Anchor-Based Pointwise LLM Rerankers Help? Retriever Quality, Statistical Scope, and Anchor Design.”

https://arxiv.org/abs/2608.10528

Relevant observation:
extra reranking/fusion need not help when the first-stage dense representation is already strong.

This motivates HIRA's W15 safety constraint:
a strong semantic anchor must not be freely overwritten by a downstream refinement stage.

W15 does not import that paper's anchor construction or pointwise LLM scoring.

### Multi-stage score fusion

Retrieval/reranking systems often combine first-stage and later-stage scores instead of treating later ranking as automatically superior.

Examples include late-fusion / reciprocal-rank-fusion retrieval systems and weighted first-stage/reranker score combinations.

This motivates comparing:
- current production only;
- semantic anchor only;
- anchor + residual;
- explicitly bounded anchor + residual.

## HIRA-specific architectural boundary

The W15 mechanism is intentionally tiny:
- frozen A13;
- frozen W9 projection;
- frozen CompetitiveCoarseScorer;
- frozen HIRACore;
- six primitive-specific scalar parameters only.

The residual cap is structural, not a learned router:
- competitive source <= .25 robust anchor spread per candidate;
- relation source <= .25 robust anchor spread per candidate;
- combined <= .50 robust anchor spread per candidate.

The equal-parameter unbounded control is mandatory.

## What W15 may establish

Only if untouched CE and CF satisfy frozen gates:
- anchor-primary production is reproducible;
- tiny typed/relation residuals add value without destroying semantics;
- explicit bounding may or may not be causally necessary.

## What W15 cannot establish

W15 cannot establish:
- broad external superiority;
- K32/K64 robustness;
- that three paraphrases are the final product interface;
- that A13 is optimal;
- that residual reranking is novel;
- that MiniLM or any external reference is part of HIRA.

Positive W15 must go to fresh W16 high-cardinality replication before any public/external authority.
