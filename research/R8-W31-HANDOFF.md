# R8-W31 handoff — qualified dual semantic adapter

Status: **TRAIN/DEV FROZEN / SEALED CONFIRM PRE-EXPOSURE**

Issue: #151  
Branch: `feat/r8-w31-qualified-dual-semantic-adapter`  
Base main: `fc906391fd55ffc6edcce60e383364801260701a`

## Why W31 exists

W30 closed `W30_REFERENCE_INADEQUATE`.

Two independent lessons are frozen:

1. authority semantics must be externally qualified before candidate training consumes the wave;
2. the single shared rank-8 residual bridge can improve DEV yet regress sealed F1/F2/composed severity.

W31 therefore changes both the evidence protocol and the candidate architecture.

## Frozen base

- A13 encoder frozen;
- exact W28 T0 frozen;
- W29 symmetric scorer remains baseline;
- W30 bridge checkpoint is not reused;
- relation refinement OFF;
- no factor-specific learned head;
- no primitive-specific learned head.

## Candidate

Two independent identity-initialized rank-8 residual maps operate after T0:

```
state:  z_s' = z_s + U_s(D_s(z_s))
schema: z_o' = z_o + U_o(D_o(z_o))
```

Capacity:
- state adapter: 2,048 parameters;
- schema adapter: 2,048 parameters;
- total trainable adapter parameters: 4,096;
- T0 projection remains frozen;
- all adapter layers bias-free;
- both up projections initialize to exact zero;
- initial logits must exactly equal unbridged W29 T0.

Runtime mode:
`dual_symmetric_semantic`

It is full-K, coarse-only, relation-delta-zero and typed state-once.

## Training contract

- seed: 3117
- epochs: 16
- batch: 32 logical cases
- optimizer: AdamW
- lr: 2e-4
- weight decay: 0.01
- grad clip: 1.0
- temperature: 0.07
- anchor coefficient: 0.35
- positive anchor margin threshold: 0.08
- trainable parameter count: exactly 4,096

Primary loss balances F0/F1/F2 equally.

Anchor preservation penalizes adapted decisions that destroy a correct high-margin frozen-T0 anchor.

DEV selection order:
1. worst-factor balanced accuracy;
2. worst-factor top-1;
3. composed severity;
4. lower invalid-vector rate;
5. earlier epoch.

## Evidence plan

### Qualification before any training
- QH, QI
- 96 cases/domain
- external DeBERTa/RoBERTa NLI panel only
- no HIRA candidate evaluation
- both must pass before TRAIN is enabled

### TRAIN
QJ/QK/QL/QM

### DEV
QN

### SEALED CONFIRM
QO/QP

All text and style banks are disjoint from W29/W30 and older authorities.

## Evidence firewall

Forbidden for fitting/selection:
- W29 EW/EX/EY/EZ;
- W30 FA/FB/FC/FD/FE/FF/FG;
- W28 and older authority rows;
- Banking77 final/test;
- typed final/test;
- Laya/JEV benchmark cells.

W30 aggregate metrics may motivate architecture only.

## Frozen outcomes

1. `W31_REFERENCE_QUALIFICATION_FAIL`
2. `W31_DUAL_ADAPTER_FAIL`
3. `HIRA_V0_TRANSFER_CORE_READY`

No partial promotion.

## Promotion gate

Per sealed domain:
- each factor top-1 >= .90
- each factor BA >= .88
- factor-vector top-1 >= .82
- composed severity >= .82
- invalid-vector <= .05
- option-order invariance = 1
- state-once = 1
- full-K = 1
- relation delta = 0
- probability mass error <= 1e-6

Relative to frozen unbridged T0:
- composed severity delta >= +.15
- worst-factor top-1 delta >= +.08
- no factor regression > .02

## Current implementation state

Reference qualification is complete and frozen:
- authority run: `36281931836`
- outcome: `W31_REFERENCE_QUALIFIED`
- QH: PASS
- QI: PASS
- qualification artifact: `r8-w31-reference-qualification`
- artifact ID: `10918919102`
- artifact digest: `sha256:badc9d6e91864519d7a3472b6b7cde313e42908611be38faa978a95b8f622d5e`
- no HIRA candidate evaluated;
- A13 not loaded;
- no W29/W30/older authority rows used.

Implemented:
- `src/nmd/w31_dual_semantic_adapter.py`
- runtime coarse mode `dual_symmetric_semantic`
- `src/nmd/w31_transfer_authority.py`
- `src/nmd/w31_transfer_cache.py`
- `src/nmd/w31_transfer_eval.py`
- `src/nmd/w31_transfer_core.py`
- `scripts/r8_w31_qualify.py`
- `scripts/r8_w31_train.py`
- qualification-gated TRAIN/DEV workflow
- exact capacity/identity/runtime/checkpoint/selection tests

Frozen TRAIN/DEV implementation details:
- QJ/QK/QL/QM TRAIN = 384 cases;
- QN DEV = 96 cases;
- exactly 4,096 trainable adapter parameters;
- factor-balanced F0/F1/F2 CE;
- T0-correct anchors with frozen baseline margin >= 0.08;
- anchor hinge preserves adapted signed margin >= 0.08;
- anchor coefficient = 0.35;
- DEV selection order is worst-factor BA, worst-factor top-1, composed severity, lower invalid-vector rate, earlier epoch.

Frozen TRAIN/DEV result:
- authority run: `36290117569`
- TRAIN: QJ/QK/QL/QM = 384 cases
- DEV: QN = 96 cases
- selected DEV epoch: 12
- checkpoint SHA256: `2a14479cc45b4e92c9ac76ce1535500b11778c8aa5b194b4ff53669f7f11054f`
- training artifact: `r8-w31-dual-adapter-training`
- artifact ID: `10922330271`
- artifact digest: `sha256:2e055d2050b3fc27c85bdb47aa0044947b5b33b27b76d5c4f96989286aef358c`
- anchor rate: 0.6102430555555556

QN frozen unbridged baseline:
- F0 top-1: 0.625
- F1 top-1: 0.7291666666666666
- F2 top-1: 0.7916666666666666
- factor-vector / composed severity: 0.3958333333333333
- invalid-vector rate: 0.3958333333333333

QN selected dual adapter:
- F0 top-1: 0.8645833333333334
- F1 top-1: 0.875
- F2 top-1: 0.8645833333333334
- factor-vector / composed severity: 0.6354166666666666
- invalid-vector rate: 0.010416666666666666
- probability mass max error: 1.1920928955078125e-07

Implemented after TRAIN/DEV freeze:
- `scripts/r8_w31_confirm.py`
- sealed QO/QP runtime + quality evaluator
- per-domain quality gate
- typed state-once/full-K/order-invariance/runtime gate
- pooled transfer gate against frozen unbridged T0
- `.github/workflows/r8-w31-sealed-confirm.yml`

Current boundary:
1. sealed evaluator unit/full CI must pass;
2. only then create `research/R8-W31-ENABLE-CONFIRM`;
3. first QO/QP exposure is one-way and cannot tune this checkpoint;
4. freeze authoritative outcome;
5. closure + FULL bundle;
6. only `HIRA_V0_TRANSFER_CORE_READY` may unblock calibration/OOD/high-K.

QO/QP remain sealed at this moment.
