# R8-W6c handoff — typed reliability calibration

Status: **PRE-AUTHORITY IMPLEMENTATION ACTIVE. No W6c empirical verdict is valid yet.**

Issue: #81
PR: #82

## Why W6c exists

W6b closed as `PRODUCTION_COMPETITIVE_PARTIAL`.

Untouched W6b CONFIRM for the selected scorer-only production path:
- overall 81.50%;
- choice 80.31%;
- score 88.75%;
- diagnosis K64 60%;
- noul 69.375%;
- hard Brier 0.3283;
- soft-target ECE 0.16373;
- score MAE 0.2367;
- state encodes/case 1.0.

Only two absolute gates failed:
- noul 69.375% < 70%;
- soft-target ECE 0.16373 > 0.15.

Do not reopen semantic binding search.

## Phase A production integration

`TypedReliabilityCalibrator` is optional and owned by `NolaneHira`, not `HIRACore`.

Modes:
- `primitive-temperature`: 3 trainable parameters;
- `primitive-temperature-noul-bias`: 4 trainable parameters.

Identity initialization:
- all log temperatures = 0 => T=1;
- noul true bias = 0.

The noul bias follows semantic option value=1 rather than option position.

Default runtime has no calibrator. HIRACore remains 422,159 parameters.

## Frozen base production checkpoint

Use only the W6b DEV-selected artifact:
- W6b authority run `35959578866`;
- selected artifact `10792488690`;
- artifact name `r8-w6b-selected`;
- artifact digest `sha256:3f0fe10a60701a0a654e5c71170432d55a96d1345b42fa2e53a38e9cb1778711`;
- selected candidate `competitive-w5i-scorer-only`;
- HIRA SHA `925f74094ac4ae583c015ea2a0be32ec692d885ec64dcf9cf3d94b64be0ccf42`;
- scorer SHA `50abb2e8136599bcaf5c41d61036e3c335a7589c6244536cc0292dcea15b1ef0`.

The W6b CONFIRM artifact is forbidden as an input.

HIRA and scorer remain frozen for all W6c candidates.

## Fresh W6c authority

Exact joint balancing is over:

`diagnosis K × severity × confidence = 4 × 4 × 3 = 48 strata`.

Counts:
- TRAIN 384 / 1,920 decisions = 8 cases per joint stratum;
- DEV 192 / 960 decisions = 4 cases per joint stratum;
- CONFIRM 192 / 960 decisions = 4 cases per joint stratum.

The earlier proposed DEV count 160 is invalid for exact joint balancing and was superseded before any W6c data generation.

Seeds:
- TRAIN 171137;
- DEV 172239;
- CONFIRM 173341;
- calibration 1009.

CONFIRM remains sealed until DEV selection freeze.

## Frozen cache boundary

W6c calibration does not train on encoder/HIRA/scorer tensors.

For each fresh state:
1. encode source state once;
2. compile each typed schema;
3. execute the exact frozen W6b competitive production path;
4. cache only final raw production logits + gold soft target + typed metadata;
5. discard model graph.

Calibration training consumes only this frozen logit cache.

This makes HIRA/scorer gradient leakage structurally impossible.

## Exactly three candidates

1. `frozen-production-control`
   - no calibrator;
   - 0 trainable params.

2. `primitive-temperature`
   - 3 trainable params.

3. `primitive-temperature-noul-bias`
   - 4 trainable params.

Optimization for trainable calibrators:
- Adam;
- lr 0.01;
- weight decay 0;
- exactly 8 epochs;
- deterministic shuffle seed 1009 + epoch;
- teacher KL 1.0 + soft Brier 1.0;
- no hard CE / hard Brier / ordinal MAE objective;
- no scheduler/warmup/clipping/AMP.

## DEV selection

Within each trainable candidate:
1. lower soft-target ECE;
2. higher noul accuracy;
3. lower soft Brier;
4. lower hard Brier;
5. higher overall accuracy;
6. lower score MAE;
7. earlier epoch.

Cross-candidate exact tie prefers the simpler 3-parameter temperature-only model.

Preserve fresh frozen-production control.

## Frozen CONFIRM gates

Absolute:
- overall >= 0.65;
- choice >= 0.65;
- noul >= 0.70;
- score >= 0.60;
- diagnosis K64 >= 0.55;
- hard Brier <= 0.50;
- soft-target ECE <= 0.15;
- score MAE <= 0.55;
- probability error <= 1e-6;
- state-once = 1.0.

Mechanism vs fresh control:
- soft-target ECE improvement >= 0.04;
- noul gain >= 0.01 OR control already passes noul >= 0.70;
- overall no worse by > 0.01;
- K64 diagnosis no worse;
- hard Brier no worse by > 0.02;
- score MAE no worse by > 0.05.

Frozen PARTIAL:
- ECE improvement >= 0.02 OR noul gain >= 0.01;
- overall delta >= -0.02.

Verdicts:
- `RELIABILITY_CALIBRATION_RESCUE`;
- `RELIABILITY_CONTROL_ALREADY_RESCUES`;
- `RELIABILITY_CALIBRATION_PARTIAL`;
- `RELIABILITY_CALIBRATION_FAIL`.

## Integrity boundary

Forbidden:
- W6b CONFIRM rows;
- typed-decisions final/test;
- prior W5 CONFIRM rows;
- public campaign cells;
- retraining HIRA or competitive scorer;
- changing thresholds after W6c CONFIRM generation.

The first empirical authority is valid only after the W6c unit/contract suite passes on the same exact head.
