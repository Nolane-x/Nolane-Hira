# HIRA V1 S74 pre-DEV staging receipt

Status: **STAGED / FRESH TRAIN-DEV NOT AUTHORIZED**

Issue: #335  
PR: #336

Parent S73:
- merged main `45ff1bc301972fc8aa04c1456200283f65616937`
- scientific run `37480212536`
- artifact `11420533826`
- digest `sha256:0f08d05bea219333495529c6ae2c54d65fd9a276dd5f783adff0b0ac3925982f`
- verdict **Case C**.

S74 A0:
- run `37484291103`
- artifact `11422648744`
- digest `sha256:68620a7c563e5a22325b4415b53020720e37ba05e8f2eb675fef5d83d4089093`
- outcome `HIRA_V1_S74_A0_DIRECT_SET_ARBITRATION_CORE_READY`.

Pre-DEV implementation head:
`01bfcb396c535fe82330aac721f9ba22326cfbb0`

Exact-head generic CI:
`37486015772`
- Python 3.10: PASS
- Python 3.12: PASS

## Frozen fresh authority

- seed **95001**
- TRAIN **768**
- DEV **192**
- **12** wholly fresh S74 domains
- exact S73 state/question/option overlap **0**
- K=4
- two semantic option views
- 24 epochs
- one DEV only.

Authority:
`src/nmd/v1_s74_authority.py`

Authority tests:
`tests/test_v1_s74_authority.py`

## Frozen controlled variable

Both arms share:
- exact S69 query-gated identity interaction representation;
- exact S59 pairwise head, **32,832 params**;
- one shared pairwise training trajectory;
- one shared correction trajectory;
- identical TRAIN rows/order;
- identical optimizer/LR/weight decay;
- identical selector;
- exact DSAC architecture, **257 params** per arm;
- bit-identical initialization;
- direct final logits;
- fixed objective `0.5*(CE_c+CE_p)+0.10*JS`;
- no teacher/pseudo-target/DEV target.

Reference:
- fused/native channels live;
- all eight relational channels mechanically zero.

Treatment:
- fused/native channels live;
- all eight S69/S59 relational channels live.

Treatment parameter advantage:
**0**.

## Isolation

At selected DEV:
- same correction state digest;
- same pairwise head state digest;
- same selected epoch;
- same raw pairwise gold-pair accuracy/margin;
- reference relational feature max abs exactly 0;
- treatment relational feature max abs > 1e-5.

DSAC states are expected to diverge because the controlled input differs.

## Stop rule

After authorization:
- exactly one fresh S74 TRAIN/DEV court;
- no hidden-width sweep;
- no channel sweep;
- no loss-weight sweep;
- no activation sweep;
- no fused-residual fallback;
- no target change;
- no optimizer/LR/WD change;
- no retry after DEV exposure;
- no second DEV;
- no external Laya/Jev evaluation.

Scientific weakness is valid.
