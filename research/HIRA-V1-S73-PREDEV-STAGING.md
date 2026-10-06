# HIRA V1 S73 pre-DEV staging receipt

Status: **STAGED / FRESH TRAIN-DEV NOT AUTHORIZED**

Issue: #333  
PR: #334

Parent S72:
- merged main `16eb7a583376cd824421d5e7ae9df948979b3953`
- scientific run `37470998139`
- artifact `11416962628`
- digest `sha256:9a653a681e0cb3182819634fe24eb345ce82a48f2d078d15e510d1d3ebfb0834`
- verdict **Case B**.

S73 A0:
- run `37476327384`
- artifact `11418978543`
- digest `sha256:bab338c55ae37bd322fcc20d838ed3b4250da959e2d7135525ec611da8f9c083`
- outcome `HIRA_V1_S73_A0_COUNTERFACTUAL_SAFETY_VETO_READY`.

## Frozen fresh authority

- seed **94001**
- TRAIN **768**
- DEV **192**
- **12** wholly fresh S73 domains
- exact S72 state/question/option overlap **0**
- K=4
- two semantic option views
- 24 epochs
- one DEV only.

Authority:
`src/nmd/v1_s73_authority.py`

Authority tests:
`tests/test_v1_s73_authority.py`

## Frozen controlled variable

Both arms share exactly:
- S69 query-gated identity interaction representation;
- S59 pairwise head, **32,832 params**;
- one pairwise training trajectory;
- one correction trajectory;
- S71 mean-only composer, **80 params**;
- S72 opponent-profile vector residual direction;
- one S73 linear safety predictor state, **9 params**;
- S66 TRAIN-only composer responsibility supervision;
- S73 TRAIN-only CE+cross-view-JS safety target;
- one encoder/state-once.

Reference:
- predictor trained/checkpointed but ignored at inference;
- emits exact S72 candidate for every item.

Treatment:
- predictor threshold fixed at **0.5**;
- accept => exact S72 candidate;
- veto => exact fused baseline;
- no interpolation.

Treatment parameter advantage:
**0**.

## Scientific isolation

At selected DEV:
- same correction state digest;
- same pairwise head state digest;
- same composer state digest;
- same veto predictor state digest;
- same selected epoch;
- same raw pairwise gold-pair accuracy/margin;
- same alpha distribution.

Only endpoint hard-veto usage differs.

## Matched trainer

`scripts/hira_v1_s73_train_dev.py`

Guards:
- `tests/test_v1_s73_matched_harness.py`
- `tests/test_v1_s73_train_dev_workflow.py`

Workflow:
`.github/workflows/hira-v1-s73-counterfactual-safety-veto-train-dev.yml`

Pre-DEV implementation exact head:
`f174213200346803e337bf4c144475abe9f8bd15`

Exact-head generic CI:
`37478346006`

Result:
- Python 3.10: PASS
- Python 3.12: PASS

## Authorization rule

Marker:
`research/HIRA-V1-S73-ENABLE-TRAIN-DEV`

The marker MUST remain absent until this staging commit itself passes generic CI on Python 3.10 and 3.12.

After authorization:
- exactly one fresh S73 TRAIN/DEV court;
- no retry after scientific exposure;
- no threshold/feature/class-weight/target sweep;
- no residual/representation/head/composer change;
- no second DEV;
- no external Laya/Jev evaluation.

Scientific failure is valid.
