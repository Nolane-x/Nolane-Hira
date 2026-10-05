# HIRA V1 S67 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #321  
PR: #322

## Parent S66

Merged main:
`e898314b51995db7e6fa18b06256a59551d0797e`

Fresh S66:
- run `37331468679`
- artifact `11354228532`
- digest `sha256:f3c6322eec8cce02a698807037a4c05e48c32b2addc5709a116a4384338379b1`
- verdict **Case B**.

## Qualified S67-A0

Run: `37335108984`  
Artifact: `11356196983`  
Digest: `sha256:14a77818c8e2158bdb3aff1b2cc864ef71fe47c1a451ce32d1d10cd234ffb793`

Outcome:
`HIRA_V1_S67_A0_SAFE_ORACLE_ALPHA_READY`

Qualified:
- reference gate **60 params**
- treatment gate **60 params**
- added treatment params **0**
- bit-identical initialization/context path
- oracle alpha lattice **[0, 0.0875, 0.175, 0.2625, 0.35]**
- oracle target levels **[0, 0.25, 0.5, 0.75, 1]**
- smaller-alpha tie break exact
- view-swap target error **0**
- treatment target distinct from S66 binary
- target levels all observed on deterministic probe bank
- output gradients live
- staged phi gradients live
- upstream gradients zero
- K=3/7/255 PASS
- residual bound violation 0
- probability mass error **1.19e-7**
- checkpoint replay exact
- one encoder/state-once
- no fresh S67 TRAIN/DEV exposure.

A0 observed:
- probe-bank canonical mean target **0.2615356**
- probe-bank paraphrase mean target **0.2532959**
- real-cache treatment target disagreement **0.28125**.

## Fresh S67 authority

- seed **88001**
- TRAIN **768**
- DEV **192**
- **12 fresh S67 domains**
- exact S66 state/question/option overlap **0**
- K=4
- epochs **24**
- one DEV only.

## Matched scientific trajectory

Shared exactly:
- immutable S51 native/cache evidence
- one correction trajectory, **114,688 params**
- one S59 pairwise-head trajectory, **32,832 params**
- exact S64 contextual gate architecture
- 60 gate params per arm
- bit-identical gate initialization
- same context projection/path
- alpha max **0.35**
- initial alpha **0.10**
- correctness tolerance **1e-8**
- same optimizer/LR/weight decay
- same TRAIN ordering
- same frozen S17 selector
- single-view inference.

Reference:
- exact S66 binary per-view responsibility BCE.

Treatment:
- S67 safe oracle-alpha continuous BCE.

Frozen treatment lattice:
`[0, 0.0875, 0.175, 0.2625, 0.35]`.

No target smoothing, lattice search at inference, learned target model or parameter advantage.

At each epoch record:
- reference binary-positive fraction
- canonical/paraphrase oracle mean target
- oracle target disagreement
- oracle nonbinary fraction
- full five-level oracle histogram
- fused shadow DEV diagnostics
- reference/treatment DEV metrics
- correction/head/gate SHAs.

Fused shadow is diagnostic only and never participates in selection.

## Stop rule

Once S67 TRAIN begins:
- no lattice change
- no tie-break change
- no correctness-tolerance change
- no target smoothing/interpolation/mixing
- no BCE weighting
- no context/projection/width/pooling change
- no optimizer/LR/weight-decay change
- no regularizer
- no gradient coupling
- no native retraining
- no selector change
- no retry for scientific weakness
- no second S67 DEV
- no external Laya/Jev evaluation.

## Authorization rule

Marker:
`research/HIRA-V1-S67-ENABLE-TRAIN-DEV`

The marker MUST remain absent until this exact final pre-DEV staging head passes generic CI on Python 3.10 and 3.12.
