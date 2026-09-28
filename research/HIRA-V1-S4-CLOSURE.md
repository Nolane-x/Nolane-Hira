# HIRA V1 S4 closure — Compact Semantic Representation Adaptation

Status: **CLOSED — HIRA_V1_S4_ADAPTER_DEV_FAIL**

Issue: #179  
PR: #180  
Branch: `feat/hira-v1-s4-semantic-adaptation`  
Base main: `040bbcc0f182b517011ebb848e3baebd233465c2`

## 1. Purpose

S4 followed `HIRA_V1_S3_CLOSED_DEV_FAIL`.

S3 removed W34 and still showed strong TRAIN/DEV divergence, shifting the primary bottleneck hypothesis toward the frozen A13/W28 representation substrate.

S4 therefore learned only a compact shared token adapter before frozen W28 while keeping the downstream triadic scorer parameter-free.

No S0-S3/M5/sealed row was used for S4 fitting or selection.

## 2. S4-A0 — exact identity authority

Authority:
- run: `36397966988`
- artifact: `10959630634`
- digest: `sha256:1943d802440ec759e4d32c97c61fdb93714ce8fb0a1897df25bd3bcdbc071c59`
- outcome: `HIRA_V1_S4_A0_IDENTITY_READY`

Fresh English localization:
- 16 base states / 32 queries;
- K=4;
- adapter parameter count: 16,384;
- adapter trainable params: 0;
- exact logit identity vs parameter-free triadic: **1.0**;
- exact selected-choice identity: **1.0**;
- accuracy: **0.28125**;
- paired both-correct: **0.0**;
- option-order flip: **0.0**;
- state-once/full-K/relation-delta-zero: PASS.

This proves S4 starts from an exact parameter-free triadic boundary.

## 3. S4-A — learned Shared Residual Semantic Adapter

Architecture:
- exact frozen A13 encoder;
- exact frozen W28 T0 projection;
- frozen HIRACore;
- W34 excluded from semantic scoring path;
- shared adapter applied identically to state/question/option token embeddings:
  - down 256->32 = 8,192 params;
  - GELU;
  - up 32->256 = 8,192 params;
  - residual connection;
- downstream parameter-free coordinate triadic scoring;
- no learned decision head.

Trainable parameters: **16,384 exact**.

## 4. Fresh TRAIN / DEV authority

Authority:
- run: `36398414087`
- artifact: `10959711099`
- digest: `sha256:112aecf9aa1f9c6d5925fd7beddab5167cce7d3eca6d94d15873909ee02309b4`
- outcome: `HIRA_V1_S4_ADAPTER_DEV_FAIL`

Frozen setup:
- seed: 9401;
- AdamW;
- 24 epochs;
- lr: 4e-4;
- weight decay: 0.01;
- grad clip: 1.0;
- CE + 0.25 paired swap-margin;
- margin: 0.20;
- 384 fresh English TRAIN states / 768 queries;
- 96 fresh English DEV states / 192 queries;
- six fresh domains:
  - solar_array
  - pharmacy_batch
  - satellite_task
  - quarry_sample
  - music_catalog
  - emergency_drill
- K=4.

No exact prior-track/S4-A0/M5/W29-W34 sealed row was used.

## 5. Selected DEV result

Selected epoch:

**15**

Selected checkpoint SHA256:

`956bfb4987ce6a5ad7c983be75fafbc16eb178cf3a87ad010743bf24c2327ed8`

Selected DEV:
- accuracy: **0.2760416667**
- paired both-correct: **0.0**
- question-swap choice-change: **0.09375**
- option-order flip: **0.0104166667**
- mean loss: **1.4353258523**
- max probability mass error: **1.1920928955078125e-07**
- full-K: PASS
- relation delta: 0
- TRAIN state encode: 384/384
- DEV state encode: 96/96

Mechanical isolation:
- adapter trainable params: 16,384 exact
- W28 trainable: 0
- encoder trainable: 0
- HIRACore trainable: 0
- learned downstream scorer params: 0

## 6. Frozen DEV gate

Failed quality gates:
- accuracy >= 0.85 — **FAIL**
- paired both-correct >= 0.75 — **FAIL**
- question-swap choice-change >= 0.80 — **FAIL**

Passed mechanical gates:
- option-order flip <= 0.02 — PASS
- probability mass error <= 1e-6 — PASS
- full-K — PASS
- relation delta = 0 — PASS
- state-once — PASS
- exact parameter isolation — PASS

No threshold or hyperparameter was weakened after DEV exposure.

## 7. Learning dynamics

TRAIN CE decreased:
- epoch 1: ~1.3821
- epoch 2: ~1.3550
- epoch 10: ~1.3174
- epoch 24: ~1.3141

The adapter learned something on TRAIN, but the improvement was modest and did not transfer.

DEV accuracy stayed near chance:
- epoch 1: 0.2604
- selected epoch 15: 0.2760
- several epochs: 0.2188-0.2708

Paired both-correct stayed **0.0 at every epoch**.

Question-swap choice-change remained unstable and low, peaking around 0.2708 while the selected epoch was 0.09375.

This is not a simple overfitting profile like S3. It indicates that the 32-rank residual adapter itself has insufficient leverage to reorganize the frozen A13/W28 relation geometry into a transferable semantic space.

## 8. Scientific interpretation

S4 removes another plausible rescue:

> a small shared pre-W28 residual adapter is not sufficient.

Across S0-S4:
- S0 fixed question blindness partially but did not generalize;
- S1 evidence compression failed;
- S2 made choices question-sensitive but not correct;
- S3 removed W34 but overfit TRAIN and failed DEV;
- S4 learned only representation adaptation but still stayed near chance.

The remaining representation hypothesis must now act more directly on the frozen relation map or semantic frontend.

The next defensible track should avoid another tiny outer adapter and test **direct semantic projection relearning and/or limited frontend adaptation** with strict compactness.

## 9. Evidence boundary

Because DEV failed:
- no post-DEV S4 tuning is authorized;
- no bottleneck-width retry;
- no learning-rate retry;
- no epoch-count retry;
- no activation change;
- no threshold weakening;
- no sealed confirm;
- no Vietnamese probe;
- no M5/Laya/Jev reopening.

S4 A0 and DEV rows are permanently exposed and forbidden for future fitting, selection, architecture ranking or hyperparameter tuning.

## 10. Next research direction — S5

Recommended hypothesis:

# Semantic Projection Relearning

Before unfreezing the A13 backbone itself, directly relearn the compact A13(256) -> relation(128) projection that all v1 tracks inherited from W28.

Candidate:
- A13 remains frozen;
- replace frozen W28 with a trainable shared 256->128 bias-free projection;
- parameter count: 32,768;
- use the same projection for state/question/option;
- downstream parameter-free triadic scoring;
- add a representation alignment objective in addition to paired decision CE;
- no learned task head;
- fresh TRAIN/DEV only.

This directly tests whether W28 relation geometry, rather than A13 itself, is the dominant remaining bottleneck.

Only if this fails should a later track consider limited A13 layer/LoRA adaptation.

## 11. Closure decision

S4-A0: **CLOSED — identity/localization only**  
S4-A TRAIN/DEV: **CLOSED — DEV FAIL**  
S4 sealed confirm: **NOT OPENED**  
S4 multilingual probe: **NOT OPENED**  
Production-ready: **false**  
Laya/Jev parity: **not claimed**
