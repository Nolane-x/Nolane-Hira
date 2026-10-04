# HIRA V1 S46 closure — Robust Three-Expert Evidence Consensus

Status: **SCIENTIFICALLY CLOSED**

Issue: #275  
PR: #276

## Qualified A0

- run `37134236894`
- artifact `11278770039`
- digest `sha256:076a4aa432d3ace98cac0e1836ceaa547bc76e6b377af01812e6f32d2d7a54a3`
- outcome `HIRA_V1_S46_A0_ROBUST_THREE_EXPERT_CONSENSUS_READY`

A0 established:
- zero fusion parameters
- K=3/7/255
- option permutation equivariance
- independent shift/positive-scale invariance
- exact three-expert identity
- one-outlier containment
- finite flat-expert behavior
- one encoder pass/state-once
- exact native/private ownership

## Fresh scientific court

Run: `37163095749`  
Artifact: `11289416980`  
Digest: `sha256:2d3c6ef18a3082fcc4515f5977ad3997c625bd07a239c3ba4a4aa9ccf0735789`

Outcome:
`HIRA_V1_S46_ROBUST_THREE_EXPERT_CONSENSUS_DEV_COMPLETE`

Same-checkpoint S46 minus legacy S45:
- canonical accuracy **+3.39 pp**
- paraphrase accuracy **-0.78 pp**
- paired **+1.04 pp**
- selected-choice agreement **-8.85 pp**
- cross-view JS **+0.02876** worse
- paraphrase margin **-0.05703**

Robust DEV_READY: **false**.

## Frozen interpretation

**Case C.**

Magnitude-level median consensus is not the missing decision mechanism.

S46 shows that an expert may be numerically stable yet weak; injecting it into score-level robust fusion can worsen final decision stability even when canonical accuracy rises.

The next family must remove dependence on cross-expert score magnitudes rather than trying another magnitude aggregator.

## Stop-rule compliance

No:
- mean/median/trimmed-mean sweep
- confidence threshold
- entropy weighting
- temperature
- learned gate
- capacity change
- optimizer change
- native-gradient leakage
- second encoder
- seed/LR/epoch/batch retry
- gate weakening
- second S46 DEV
- external Laya/Jev evaluation.

Frozen evidence:
- `research/HIRA-V1-S46-MATCHED-RECEIPT.json`
- `research/HIRA-V1-S46-MATCHED-RECEIPT.md`

Next family:
**S47 — Ordinal Pairwise Consensus with Private Tie-Break**.
