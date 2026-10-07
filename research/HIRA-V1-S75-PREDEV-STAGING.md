# HIRA V1 S75 pre-DEV staging receipt

Status: **STAGED / FRESH TRAIN-DEV NOT AUTHORIZED**

Issue: #337  
PR: #338

Parent S74:
- merged main `0edfb0a64823ccb7ef2cf5c4e77b9fb9696361be`
- scientific run `37487578737`
- artifact `11423428905`
- digest `sha256:cb2d8e2718d3aa426bf3ee44795d36c25e5960ca727fb8cc6280d12cfbefdb21`
- verdict **Case D**.

S75 A0:
- run `37578213725`
- artifact `11463278233`
- digest `sha256:fae3cc55fdbd8ab841da16b0f6ae113dc17d0186f844cbda26fab6a6a70c424e`
- outcome `HIRA_V1_S75_A0_TOKEN_LEVEL_BIDIRECTIONAL_BINDING_READY`.

Final pre-DEV implementation head:
`4769bb629f8f920fd1c8f1b7c41c3c05316cc0e6`

Exact-head generic CI:
`37578933767`
- Python 3.10: PASS
- Python 3.12: PASS

## Frozen fresh authority

- seed **96001**
- TRAIN **768**
- DEV **192**
- **12** wholly fresh S75 domains
- exact S74 state/question/option overlap **0**
- K=4
- two semantic option views
- 24 epochs
- one DEV only.

Authority:
`src/nmd/v1_s75_authority.py`

Authority tests:
`tests/test_v1_s75_authority.py`

## Frozen controlled variable

Reference:
- exact S69 query-gated identity interaction representation;
- representation dim **512**;
- representation params **0**.

Treatment:
- S75 Token-Level Bidirectional Evidence Binding Representation;
- signed state/query/option token binding occurs before query-token pooling;
- representation dim **512**;
- representation params **0**.

Both:
- exact S59 anti-symmetric pairwise head;
- **32,832 trainable params**;
- bit-identical initialization;
- same gold-vs-distractor softplus pairwise objective;
- same TRAIN rows/order;
- same optimizer/LR/weight decay;
- same correction/native trajectory;
- exact same S64/S66 bounded composition used only as downstream diagnostic;
- no teacher/pseudo-target/DEV-derived target;
- no pairwise gradient into representation/native.

Frozen TBER constants:
- temperature **0.10**
- identity interaction scale **16**
- norm epsilon **1e-12**.

Treatment parameter advantage:
**0**.

## A0 evidence

- neutral collapse to S69 max abs error: **0**
- nontrivial treatment-reference max abs difference: **0.1254134774**
- reference option permutation error: **0**
- treatment option permutation error: **0**
- K=3/7/255 PASS
- upstream gradient zero
- checkpoint replay error **0**
- fresh TRAIN/DEV exposed=false.

## Stop rule

After authorization:
- exactly one fresh S75 TRAIN/DEV court;
- no token-weight formula sweep;
- no temperature sweep;
- no pooling/normalization sweep;
- no interaction-scale sweep;
- no head width/rank change;
- no target/loss change;
- no retry after DEV exposure;
- no second DEV;
- no external Laya/Jev evaluation.

Scientific weakness is valid.
