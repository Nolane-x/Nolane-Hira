# HIRA V1 S14 A0 receipt — Canonical Relation Evidence Fusion

Status: **QUALIFIED**

Issue: #204

Original scientific exposure:
- run: `36527407197`
- head: `e049e31df92c2e929aab415f049de973ba691cb8`
- scientific A0 step: **PASS**
- logged outcome: `HIRA_V1_S14_A0_IDENTITY_READY`
- artifact packaging did not occur because receipt verification still contained two stale S13-only field assertions

Exact reproduction / artifact packaging:
- run: `36527874581`
- artifact: `11015447511`
- artifact digest: `sha256:3fccd10b3bde77a4396cec491d8a01a36c129e7d917dedfc688ff03fb0832976`
- outcome: `HIRA_V1_S14_A0_IDENTITY_READY`

The reproduction changed no scientific code, A0 row, model surface, fusion rule, epsilon, objective or gate.
All discrete decisions and qualification invariants reproduced.

## 1. Identity and capacity

- A13 token identity: PASS
- A13 pooled identity: PASS
- exact inherited raw triadic logits: **1.0**
- exact inherited raw triadic choices: **1.0**
- physical candidate surface: **49,152**
- runtime trainable during A0: **0**
- original A13 trainable: **0**
- canonicalizer-added params: **0**
- fusion-added params: **0**

## 2. Mechanical fusion invariants

- fusion epsilon: **1e-6**
- expert-swap max abs: **0.0**
- fused option-order flip: **0.0**
- raw option-order flip: **0.0**
- fused max probability-mass error: **1.1920928955078125e-07**
- raw max probability-mass error: **1.1920928955078125e-07**
- full-K: PASS
- relation refinement: OFF
- state encode calls: **32 / 32 expected**

## 3. Fresh A0 decision baselines

Canonical:
- raw triadic accuracy: **0.25**
- canonical relation accuracy: **0.40625**
- fused accuracy: **0.375**
- fused paired both-correct: **0.0625**

Paraphrase:
- raw triadic accuracy: **0.40625**
- relation accuracy: **0.40625**
- fused accuracy: **0.40625**

Cross-view:
- raw triadic selected-choice agreement: **0.5625**
- relation selected-choice agreement: **0.8125**
- fused selected-choice agreement: **0.5625**
- fused mean JS: **0.0251854807**

Signed margins:
- raw triadic canonical margin: **-0.0000945138**
- relation canonical margin: **-0.1821298748**
- fused canonical margin: **-0.3138211966**

Relation signatures:
- same-option cross-view cosine: **0.7710087299**
- same-vs-strongest-wrong signature margin: **0.0076525202**

## 4. Expert scale diagnostic

Before normalization:
- mean triadic RMS evidence: **0.0001341402**
- mean relation RMS evidence: **0.4713211776**

Expert top-1 agreement:
- **0.359375**

This scale disparity is diagnostic only.

S14 deliberately standardizes both experts independently, so a near-flat triadic surface receives equal normalized voting strength. A0 therefore exposes a credible failure mode: the symmetric rule may promote weak triadic noise.

This observation is **not** used to change the frozen fusion rule.
S14 TRAIN/DEV must test whether fused supervision can make the triadic expert acquire useful discriminative structure and become complementary to the relation expert.

A0 is not used for model selection.
