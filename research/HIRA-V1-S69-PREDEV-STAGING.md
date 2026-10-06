# HIRA V1 S69 pre-DEV staging receipt

Status: **STAGED / FRESH TRAIN-DEV NOT AUTHORIZED**

Issue: #325  
PR: #326

Parent S68:
- merged main `4cfdbc6fe100715bd950c9a0e2f504f71c486e9c`
- scientific run `37389861004`
- artifact `11381470793`
- digest `sha256:0156bb57125f6b8c4f380649505803ed76cf49b8edf1c5bd4774fd8790386eaa`
- verdict **Case C**.

S69 A0:
- run `37417409926`
- artifact `11391117254`
- digest `sha256:40c2e13e6a40f9e0b2a51a4cdeb7a6819511da4dd29d9ce19f3d688ad0617383`
- outcome `HIRA_V1_S69_A0_QUERY_GATED_INTERACTION_REP_READY`.

## Frozen fresh authority

- seed **90001**
- TRAIN **768**
- DEV **192**
- **12** wholly fresh S69 domains
- K=4
- two semantic option views
- exact S68 state/question/option overlap **0**
- 24 epochs
- one DEV only.

Authority:
`src/nmd/v1_s69_authority.py`

Authority tests:
`tests/test_v1_s69_authority.py`

## Frozen controlled variable

Reference pairwise representation:
`normalize([identity, joint_context])`.

Treatment pairwise representation:
`x = joint_context + 16*(identity ⊙ joint_context)`
then `normalize([identity, normalize(x)])`.

Both:
- representation dimension 512
- representation trainable params 0
- exact S59 pairwise head 32,832 params
- bit-identical head initialization
- same gold-vs-distractor pairwise objective
- same rows/order
- same optimizer/LR/weight decay
- same shared correction trajectory.

Treatment parameter advantage: **0**.

## Downstream isolation

Both arms retain:
- S64 contextual gate, 60 params;
- S66 per-view counterfactual responsibility objective;
- **exact S59 reference representation as the gate context in both arms**.

Therefore the only scientific intervention is the representation consumed by the pairwise head.

## Matched trainer

`scripts/hira_v1_s69_train_dev.py`

Guards:
- `tests/test_v1_s69_matched_harness.py`
- `tests/test_v1_s69_train_dev_workflow.py`

Workflow:
`.github/workflows/hira-v1-s69-query-gated-interaction-train-dev.yml`

## Authorization rule

Marker:
`research/HIRA-V1-S69-ENABLE-TRAIN-DEV`

The marker MUST remain absent until this exact final pre-DEV staging head passes generic CI on both Python 3.10 and 3.12.

After authorization:
- exactly one fresh S69 TRAIN/DEV court;
- no retry;
- no second DEV;
- no formula/scale/normalization sweep;
- no reliability target reopening;
- no external Laya/Jev evaluation.

Scientific failure is valid.
