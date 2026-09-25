# R8-W12 prior-art reassessment — anchor-preserving residual competition

Status: **PRE-DIAGNOSTIC LITERATURE/ARCHITECTURE NOTE. No W12 empirical result is used here.**

This note records the reasoning that led from W11's unresolved micro-stage localization to the W12 higher-level invariant.

## 1. Problem statement

W10/W11 repeatedly showed that a candidate-independent semantic representation can rank fresh natural definitions materially better than the full production HIRA path.

But W11 did not find one universal harmful stage:
- question context was harmful in BL;
- common-mode subtraction was harmful in BJ;
- salient-min coverage was harmful in BK;
- BM was unresolved;
- candidate-relative IDF and HIRACore relation were pooled net-positive.

Therefore a global deletion of any one production transform is not scientifically supported.

The architectural question is instead:

**Should production competition be allowed to replace a strong semantic ranking unconditionally, or should it behave as a bounded correction conditioned on semantic uncertainty?**

## 2. Late interaction as semantic anchor

ColBERTv2:
- Santhanam et al., 2021/2022;
- https://arxiv.org/abs/2112.01488

Relevant lesson:
multi-vector token-level late interaction is a strong semantic retrieval primitive and can generalize outside its training domain.

W12 does not copy ColBERTv2 architecture. HIRA already has a late-interaction path. The relevance here is that a token-level semantic score can legitimately serve as an independent ranking anchor rather than only an internal feature later overwritten by list-dependent transforms.

## 3. Token transforms can help and hurt

TRIAL:
- Kang, Kim, Han, EMNLP 2025;
- https://aclanthology.org/2025.emnlp-main.854/

Relevant lesson:
naive token-level aggregation can be improved by modeling token importance and relations, so W12 must not conclude that IDF/salience/relations are inherently wrong.

This is consistent with HIRA's own history:
- W6f found crude salience removal much worse;
- W11 found candidate-relative IDF pooled beneficial;
- W11 found HIRACore relation pooled slightly beneficial.

The problem is not whether competitive corrections ever help. The problem is whether they should be trusted equally on every case.

## 4. Reranking is not monotonic

BRIGHT:
- Su et al., 2024;
- https://arxiv.org/abs/2407.12883

Relevant lesson:
a reranker trained under a different relevance distribution can reduce retrieval quality instead of monotonically improving it.

This supports treating a production reranker/competitive scorer as a correction that can fail under transfer, not as an automatically superior replacement for a semantic base ranking.

## 5. Uncertainty-aware refinement

AcuRank:
- Yoon et al., 2025;
- https://arxiv.org/abs/2505.18512

Relevant lesson:
reranking/computation can be allocated adaptively from ranking uncertainty rather than using one fixed refinement policy for every query.

W12 does not adopt AcuRank's Bayesian TrueSkill machinery.
The relevant principle is narrower:
**ranking confidence can be a legitimate control signal for whether further ranking intervention is warranted.**

## 6. Anchor + bounded corrective evidence

FaLCon:
- Pham et al., 2026;
- https://arxiv.org/abs/2608.09474

Relevant lesson:
a strong global semantic retrieval score can be retained as an anchor while finer semantic facets act as bounded corrective evidence, with uncertainty-aware consensus on ambiguous cases.

FaLCon is a different multimodal Sim2Real task and is not evidence that the same mechanism will work in HIRA.
It is cited only as contemporary architectural precedent for anchor-preserving refinement.

## 7. W12 invariant

The combined prior-art/HIRA evidence motivates exactly one diagnostic invariant:

> **If the candidate-independent semantic anchor is reliable on a case, production competition should not be allowed to overturn it without stronger evidence. If the anchor is uncertain, production residuals may be useful.**

This invariant is intentionally above W11's micro-stages.

It does not assert:
- question context is globally harmful;
- IDF is globally harmful;
- common-mode subtraction is globally harmful;
- salient-min is globally harmful;
- relation reranking is globally harmful.

## 8. Why W12 uses margin quartiles instead of a tuned threshold

A numeric confidence threshold learned on BJ/BK/BL/BM would reuse exposed evidence.

W12 therefore:
- computes a label-free normalized top1-top2 semantic-anchor margin;
- forms HIGH/LOW quartiles independently within each fresh domain and K;
- fixes exactly 25%/25% before exposure;
- uses deterministic base_id tie-breaking.

This is a diagnostic stratification, not a production threshold.

## 9. Why W12 includes an inverse guard control

If replacing any arbitrary 25% of production decisions with anchor decisions improves accuracy, the result does not prove uncertainty conditioning.

Therefore W12 compares:
- `G_high_anchor`: preserve anchor only in HIGH-confidence cases;
- `G_low_anchor_control`: preserve anchor only in LOW-confidence cases.

A future gating mechanism is authorized only if the preregistered high-confidence protection pattern is stable across >=3/4 wholly fresh domains.

## 10. Research boundary

W12 performs no optimization.

A stable signal would authorize a separate W13 mechanism authority with new TRAIN/DEV/dual-CONFIRM data.

A mixed/unresolved result blocks training and requires another architecture reassessment.

This note must not be amended using BN/BO/BP/BQ results except to append an explicit post-authority closure section after the frozen W12 outcome exists.
