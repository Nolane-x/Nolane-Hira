# HIRA V1 S72 pre-DEV staging receipt

Status: **STAGED / FRESH TRAIN-DEV NOT AUTHORIZED**

Issue: #331  
PR: #332

Parent S71:
- merged main `ef2e8a4b71ba9a46dd9caea07698df742abb6461`
- scientific run `37459447097`
- artifact `11412505173`
- digest `sha256:aa116237d5508131db18f88a23893d9107d05ae67f46ccffe638520fe8098a50`
- verdict **Case C**.

S72 A0:
- run `37463175304`
- artifact `11413616350`
- digest `sha256:a4d8052a0e76160310d4a99e86ef71f07810a96e77681cb34d747b85c98ef52a`
- outcome `HIRA_V1_S72_A0_OPPONENT_PROFILE_VECTOR_RESIDUAL_READY`.

## Frozen fresh authority

- seed **93001**
- TRAIN **768**
- DEV **192**
- **12** wholly fresh S72 domains
- exact S71 state/question/option overlap **0**
- K=4
- two semantic option views
- 24 epochs
- one DEV only.

Authority:
`src/nmd/v1_s72_authority.py`

Authority tests:
`tests/test_v1_s72_authority.py`

## Frozen controlled variable

Both arms use:
- exact S69 query-gated identity interaction representation;
- exact S59 pairwise head, **32,832 params**;
- one shared pairwise head trajectory;
- one shared correction trajectory;
- exact S71 **80-param mean-only composer**;
- bit-identical composer initialization;
- exact S66 TRAIN-only per-view responsibility supervision;
- no teacher/pseudo-target;
- no DEV-derived target;
- one encoder/state-once.

Reference direction:
`tanh(center(uniform_pairwise_row_mean)/rms(center(uniform_pairwise_row_mean)))`.

Treatment direction:
- fixed fused opponent prior, temperature **1.0**;
- self removed and opponent prior renormalized;
- signed weighted evidence divided by weighted absolute evidence;
- centered/RMS-normalized;
- tanh bounded.

Direction trainable params:
**0 vs 0**.

Treatment parameter advantage:
**0**.

## Isolation

At selected DEV:
- same pairwise head state digest;
- same correction state digest;
- same selected epoch;
- same raw pairwise gold-pair accuracy/margin;
- same composer family/capacity/objective.

Only residual direction may differ.

## Matched trainer

`scripts/hira_v1_s72_train_dev.py`

Guards:
- `tests/test_v1_s72_matched_harness.py`
- `tests/test_v1_s72_train_dev_workflow.py`

Workflow:
`.github/workflows/hira-v1-s72-opponent-profile-vector-residual-train-dev.yml`

## Authorization rule

Marker:
`research/HIRA-V1-S72-ENABLE-TRAIN-DEV`

The marker MUST remain absent until this exact pre-DEV staging head passes generic CI on Python 3.10 and 3.12.

After authorization:
- exactly one fresh S72 TRAIN/DEV court;
- no retry after scientific exposure;
- no temperature/profile/epsilon/normalization sweep;
- no composer/target/representation/head change;
- no second DEV;
- no external Laya/Jev evaluation.

Scientific weakness is valid.
