# HIRA V1 S70 pre-DEV staging receipt

Status: **STAGED / FRESH TRAIN-DEV NOT AUTHORIZED**

Issue: #327  
PR: #328

Parent S69:
- merged main `1ada877bcd9381486395599519363e87ef3dd831`
- scientific run `37420104460`
- artifact `11392672918`
- digest `sha256:c24c00e9b79d23b3e7e3fe4d17013270e0410d822b85c62230bb85e972a2f94d`
- verdict **Case B**.

S70 A0:
- run `37422000871`
- artifact `11393387610`
- digest `sha256:6e021d9fe8807eb6ff097bfe16b9bb464a8ee87e949d064883ec43d6cfe5a9ff`
- outcome `HIRA_V1_S70_A0_FUSED_ANCHORED_PAIRWISE_AGGREGATION_READY`.

## Frozen fresh authority

- seed **91001**
- TRAIN **768**
- DEV **192**
- **12** wholly fresh S70 domains
- K=4
- two semantic option views
- exact S69 state/question/option overlap **0**
- 24 epochs
- one DEV only.

Authority:
`src/nmd/v1_s70_authority.py`

Authority tests:
`tests/test_v1_s70_authority.py`

## Frozen controlled variable

Both arms use:
- exact S69 query-gated identity interaction representation;
- representation trainable params **0**;
- one shared S59 explicit pairwise head, **32,832 params**;
- one shared pairwise training trajectory;
- one shared correction trajectory;
- one shared upstream selection epoch chosen only from fused-shadow metrics;
- exact S64 contextual reliability gate, **60 params per arm**;
- exact S66 per-view responsibility objective;
- exact S59 reference representation as gate context.

Reference aggregation:
`uniform row mean`.

Treatment aggregation:
`fused softmax opponent weighted`, temperature **1.0**.

Aggregation trainable params:
**0 vs 0**.

Treatment parameter advantage:
**0**.

## Scientific isolation

Raw pairwise evidence must be exactly matched at selected DEV:
- same head state digest;
- same correction state digest;
- same selected epoch;
- same pairwise gold-pair accuracy;
- same pairwise gold-pair margin.

Only aggregation and the gate policy trained from that aggregation may differ.

## Matched trainer

`scripts/hira_v1_s70_train_dev.py`

Guards:
- `tests/test_v1_s70_matched_harness.py`
- `tests/test_v1_s70_train_dev_workflow.py`

Workflow:
`.github/workflows/hira-v1-s70-fused-anchored-pairwise-train-dev.yml`

## Authorization rule

Marker:
`research/HIRA-V1-S70-ENABLE-TRAIN-DEV`

The marker MUST remain absent until this exact final pre-DEV staging head passes generic CI on both Python 3.10 and 3.12.

After authorization:
- exactly one fresh S70 TRAIN/DEV court;
- no retry after scientific exposure;
- no temperature/top-k/weighting sweep;
- no representation/head change;
- no reliability-target reopening;
- no external Laya/Jev evaluation.

Scientific failure is valid.
