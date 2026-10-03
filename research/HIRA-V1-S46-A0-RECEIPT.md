# HIRA V1 S46 A0 receipt — Robust Three-Expert Evidence Consensus

Status: **QUALIFIED**

Run: `37134236894`  
Artifact: `11278770039`  
Artifact digest: `sha256:076a4aa432d3ace98cac0e1836ceaa547bc76e6b377af01812e6f32d2d7a54a3`  
Authorization head: `84c8287ad9ce6099a2355d2c2946b86df85e467f`

Outcome:
`HIRA_V1_S46_A0_ROBUST_THREE_EXPERT_CONSENSUS_READY`

## Frozen surface

- native trainable: **49,152**
- private adapter: **49,152**
- bilinear W: **65,536**
- correction-only: **114,688**
- treatment total: **163,840**
- fusion trainable parameters: **0**
- second encoder pass: **false**
- training mechanics: exact S45
- correction objective: `0.10 CE + 0.25 cross-view JS`

## Robust-fusion mechanics

- K=3: PASS
- K=7: PASS
- K=255: PASS
- logical-option permutation max abs error: **5.96e-8**
- independent shift/positive-scale max abs error: **1.073e-6**
- all-three-identical error: **0**
- one-extreme-outlier error: **2.205e-6**
- median lower/upper envelope violations: **0 / 0**
- flat expert finite: **true**
- max softmax probability-mass error: **1.192e-7**
- actual shell legacy-vs-robust max abs: **1.6826072**
- actual shell probability-mass error: **1.192e-7**
- actual shell one encoder batch: **true**

## Ownership

- matched native gradient max abs: **0**
- matched native one-step parameter max abs: **0**
- matched native one-step output max abs: **0**
- correction -> native runtime gradient L1: **0**
- native objective -> private correction gradient L1: **0**
- JS-only native-runtime gradient L1: **0**
- JS-only A gradient L1: **0.0003848664**
- JS-only B gradient L1: **0.0083554629**
- JS-only W gradient L1: **0.3616145253**

Warm-start remains:
**W -> B -> A**

## Interpretation

S46-A0 mechanically qualifies the new family.

The median shell:
- is genuinely different from the legacy S45 shell;
- adds no trainable parameters;
- supports K up to 255 in the A0 court;
- preserves option equivariance and per-expert shift/scale invariance;
- contains one arbitrarily extreme outlier when the other two experts agree;
- preserves exact native/private ownership and one-pass runtime mechanics.

A0 is diagnostic only and was not used for model selection.

Fresh S46 TRAIN/DEV remains separately gated.
