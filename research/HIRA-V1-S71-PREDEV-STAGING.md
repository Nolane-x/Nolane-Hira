# HIRA V1 S71 pre-DEV staging receipt

Status: **STAGED / FRESH TRAIN-DEV NOT AUTHORIZED**

Issue: #329  
PR: #330

Parent S70:
- merged main `751161746887b795149fcd924735ee2d043b43b3`
- scientific run `37423476142`
- artifact `11394610000`
- digest `sha256:8af711f789efb9717eaed0eff882fade7bab37b546f86a35502db82242ad221f`
- verdict **Case C**.

S71 A0:
- run `37457754519`
- artifact `11410541387`
- digest `sha256:52f38ac6d30177c43205b023114a4192333b3374c98c083aed30fa494aececb5`
- outcome `HIRA_V1_S71_A0_MULTISTAT_ROW_COMPOSER_READY`
- row permutation max abs error **0.0**
- 80 vs 80 composer params.

## Frozen fresh authority

- seed **92001**
- TRAIN **768**
- DEV **192**
- **12** wholly fresh S71 domains
- K=4
- two semantic option views
- exact S70 state/question/option overlap **0**
- 24 epochs
- one DEV only.

Authority:
`src/nmd/v1_s71_authority.py`

Authority tests:
`tests/test_v1_s71_authority.py`

## Frozen controlled variable

Both arms use:
- exact S69 query-gated identity interaction representation;
- representation params **0**;
- one shared S59 pairwise head, **32,832 params**;
- one shared pairwise training trajectory;
- one shared correction trajectory;
- one shared upstream selection epoch chosen only from fused-shadow metrics;
- exact S66 TRAIN-only per-view responsibility objective;
- exact S59 reference representation as composer context;
- uniform row-mean residual direction;
- alpha max **0.35**.

Reference composer:
- **80 params**
- row channels: normalized mean + zero + zero + zero.

Treatment composer:
- **80 params**
- row channels: normalized mean/max/min/RMS.

Parameter initialization:
- bit-identical.

Treatment parameter advantage:
- **0**.

## Scientific isolation

Selected DEV must have exactly matched:
- pairwise head state digest;
- correction state digest;
- selected epoch;
- pairwise gold-pair accuracy;
- pairwise gold-pair margin;
- pairwise aggregate metrics.

Only composer row-stat exposure and the resulting 80-param composer policy may differ.

## Matched trainer

`scripts/hira_v1_s71_train_dev.py`

Guards:
- `tests/test_v1_s71_matched_harness.py`
- `tests/test_v1_s71_train_dev_workflow.py`

Workflow:
`.github/workflows/hira-v1-s71-multistat-row-composer-train-dev.yml`

## Authorization rule

Marker:
`research/HIRA-V1-S71-ENABLE-TRAIN-DEV`

The marker MUST remain absent until this exact final pre-DEV staging head passes generic CI on both Python 3.10 and 3.12.

After authorization:
- exactly one fresh S71 TRAIN/DEV court;
- no retry after scientific exposure;
- no row-stat subset/width/normalization sweep;
- no representation/head/target/residual-direction change;
- no external Laya/Jev evaluation.

Scientific failure is valid.
