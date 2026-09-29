# HIRA V1 S19 A0 receipt — Expert-Separated Triadic View Consistency

Status: **QUALIFIED**

Issue: #221  
A0 run: `36588963863`  
Artifact: `11042832648`  
Artifact digest: `sha256:f54819e15bc642f3424272b05ba3dc66e2418b34314e536709d72171ce858276`  
Head exposed by A0: `1870aefae1b7486e27912e8e8062743838acc414`

Outcome:

`HIRA_V1_S19_A0_IDENTITY_READY`

## Identity / capacity

- A13 token identity: PASS
- A13 pooled identity: PASS
- exact inherited raw-triadic logits: **1.0**
- exact inherited raw-triadic choices: **1.0**
- physical candidate surface: **49,152**
- runtime trainable during A0: **0**
- original A13 trainable: **0**
- full-K/state-once: PASS
- fused option-order flip: **0.0**
- fused mass error: **1.1920928955e-07**

## Raw-triadic JS operator proof

Fixed coefficient: **0.25**

- identical-view JS: **6.3116183391e-09**
- mismatched-view JS: **0.1234456822**
- view-swap max abs: **0.0**
- option-permutation max abs: **0.0**
- canonical-view gradient L1: **0.1521519572**
- paraphrase-view gradient L1: **0.1794761866**
- gradients nonzero on both views: PASS
- added learned parameters/state: **0**

## Fresh semantic baseline

Raw triadic:
- canonical accuracy **0.21875**
- paraphrase accuracy **0.25**
- cross-view agreement **0.5**
- cross-view mean JS **7.3407484535e-09**

Relation:
- canonical accuracy **0.34375**
- paraphrase accuracy **0.28125**

Fused:
- canonical accuracy **0.375**
- paraphrase accuracy **0.25**
- cross-view agreement **0.625**

Signatures:
- same-option cosine **0.7017681599**
- same-vs-strongest-wrong margin **-0.0004072639**

## Real shared-surface gradient probe

- primary norm **22.0530300140**
- relation norm **0.3898453414**
- raw norm ratio primary/relation **56.5686636971**
- normalized cosine **0.0349002853**
- post-projection dot **0.0349002853**
- combined norm **11.2214365005**
- surface params **49,152**

A0 is diagnostic only and was not used for model selection.
