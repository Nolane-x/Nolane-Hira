# HIRA V1 S1 closure — Query-Keyed Evidence Extraction

Status: **CLOSED — HIRA_V1_S1_QKEE_DEV_FAIL**

Issue: #173  
PR: #174  
Branch: `feat/hira-v1-s1-query-keyed-evidence`  
Base main: `5cfa41b297b389ebb37422028c7aab7bafcef1d8`

## 1. Purpose

S1 followed S0's `HIRA_V1_S0_CLOSED_DEV_FAIL`.

S0 proved that frozen Hira v0 semantic scoring is question-blind and that a low-rank multiplicative query gate is insufficient.

S1 tested a distinct hypothesis:

> use the question to decide which already-encoded state evidence is presented to frozen W34 option scoring.

No S0/M5/sealed rows were used for S1 fitting or selection.

## 2. S1-A0 — zero-parameter evidence extraction

Authority run:

`36387872121`

Artifact:
- name: `hira-v1-s1-a0-parameter-free-evidence`
- ID: `10955282264`
- digest: `sha256:98428470ca03e5de015517c1355b4d16d2dc1aa6bf8688e6ae2f05476cf5da3a`

Outcome:

`HIRA_V1_S1_A0_PARAMETER_FREE_EVIDENCE_READY`

Contract:
- exact M4 W28/W34 base;
- 16 fresh English base states / 32 queries;
- K=4;
- top-2 state content tokens selected by frozen W28 question↔state cosine;
- added parameters: 0;
- trainable parameters: 0;
- state-once: PASS;
- full-K: PASS;
- relation delta: 0.

Observed:
- A0 accuracy: **0.21875**
- paired both-correct: **0.0**
- question changes selected evidence: **0.625**
- question changes final choice: **0.0**

Fresh v0 control:
- accuracy: **0.28125**
- paired both-correct: **0.0**
- question changes choice: **0.0**

Interpretation:
- query-keyed evidence selection is mechanically real;
- hard frozen top-2 cosine routing is not a quality rescue;
- evidence changes alone are insufficient to move W34 decisions.

A0 remained localization-only and was never used for model selection.

## 3. S1-A learned QKEE candidate

Architecture:
- exact frozen A13;
- exact frozen W28 T0;
- exact frozen W34 residual / interaction / composition core;
- frozen HIRACore;
- query heads: 128→32 = 4,096 params;
- state keys: 128→32 = 4,096 params;
- two attention heads × rank 16;
- two question-specific weighted state evidence slots;
- W34 consumes those two slots.

Trainable extractor parameters:

**8,192**

Complete W34 + S1 candidate surface:

**16,384**

Runtime invariants:
- state encoder once per base state;
- dynamic question schema;
- full-K;
- option permutation equivariance;
- relation refinement OFF;
- adaptive budget OFF.

## 4. Fresh TRAIN / DEV

Authority run:

`36388286637`

Artifact:
- name: `hira-v1-s1-qkee-train-dev`
- ID: `10955218190`
- digest: `sha256:51b7a808531922f1e3c73c4d01d931562b7f0dc2032e39af7d9d68d5a575468d`

Outcome:

`HIRA_V1_S1_QKEE_DEV_FAIL`

TRAIN:
- 256 English base states;
- 512 paired queries;
- four fresh domains:
  - shipment_manifest
  - lab_specimen
  - network_node
  - festival_access
- K=4.

DEV:
- 64 fresh English base states;
- 128 paired queries;
- fresh templates and lexical banks;
- no exact TRAIN state/question overlap;
- no exact S0 row use;
- no A0 row use;
- no M5 final row use;
- no W29-W34 sealed row use.

Frozen optimizer:
- seed 6101;
- AdamW;
- 12 epochs;
- lr 5e-4;
- weight decay 0.01;
- grad clip 1.0;
- CE + 0.20 paired swap-margin;
- margin 0.15.

## 5. Selected DEV result

Selected epoch:

**7**

Checkpoint SHA256:

`50232f45f0e7b6123c84f9b5378473f8b3028978044e6e0163f93e5cfdc1a4c7`

Selected DEV:
- accuracy: **0.265625**
- paired both-correct: **0.0**
- question-swap choice-change: **0.0**
- option-order flip: **0.0**
- mean loss: **1.4179871696978807**
- max probability mass error: **1.1920928955078125e-07**
- full-K: PASS
- relation delta: 0
- state encode calls: 64/64

Training mechanics:
- TRAIN state encode calls: 256/256
- trainable parameters: 8,192 exact
- W34 base trainable: 0
- encoder trainable: 0
- HIRACore trainable: 0

## 6. DEV gate

Frozen gate required:
- accuracy >= 0.85;
- paired both-correct >= 0.75;
- question-swap choice-change >= 0.80;
- option-order flip <= 0.02;
- probability mass error <= 1e-6;
- full-K;
- relation delta = 0;
- state-once;
- exact 8,192 trainable params.

Passed mechanical gates:
- option-order flip
- probability integrity
- full-K
- relation delta
- state-once
- parameter isolation

Failed quality gates:
- accuracy
- paired both-correct
- question-swap choice-change

The failure is therefore **scientific / architectural**, not infrastructure-related.

## 7. Learning trajectory

Across 12 epochs:
- DEV accuracy stayed around 0.2578125–0.265625;
- paired both-correct stayed 0.0;
- question-swap choice-change was 0.0 for most epochs and only 0.015625 at epochs 5 and 8;
- TRAIN loss decreased only modestly.

The extractor did not learn a meaningful query-dependent decision routing mechanism.

## 8. Evidence boundary

No post-DEV tuning is authorized inside S1.

Therefore:
- no learning-rate retry;
- no attention-temperature tweak;
- no head-count tweak;
- no normalization change;
- no threshold weakening;
- no sealed confirm;
- no Vietnamese probe;
- no M5/Laya/Jev reopening.

S1 DEV rows are now permanently exposed and forbidden for future fitting, selection, architecture ranking or hyperparameter tuning.

## 9. Scientific interpretation

S1 establishes two useful negative results:

1. hard top-2 frozen cosine evidence routing can change selected state evidence but does not change decisions;
2. a compact two-head soft evidence compressor with 8,192 learned parameters also fails to make decisions meaningfully question-dependent on fresh DEV.

A likely architectural weakness is that S1 compresses a full state into only two weighted-average evidence slots. In addition, normalized rank-16 query/key vectors followed by sqrt(16) scaling constrain attention-logit range and can make routing nearly uniform.

These observations may motivate a future architecture, but S1 itself must remain frozen.

## 10. Next research direction — S2

S2 must use wholly fresh evidence and a distinct mechanism.

Recommended hypothesis:

# Query-Token Cross-Attention Residual Fusion

Instead of selecting or averaging state tokens, preserve **all state tokens** and let question tokens modify each state token's relation-space representation before W34 option scoring.

Candidate concept:
- frozen W28 relation vectors;
- learned low-rank state-query cross-attention;
- query context fused residually into each state relation token;
- no state compression to two slots;
- no dataset/domain-specific head;
- state encoder still exactly once.

This attacks the S1 failure directly: preserve sparse evidence identity while allowing the question to alter semantic geometry.

S2 must preregister new TRAIN/DEV and must not tune against S1 DEV.

## 11. Closure decision

S1-A0: **CLOSED — localization evidence only**  
S1-A TRAIN/DEV: **CLOSED — DEV FAIL**  
S1 sealed confirm: **NOT OPENED**  
S1 multilingual probe: **NOT OPENED**  
Production-ready: **false**  
Laya/Jev parity: **not claimed**
