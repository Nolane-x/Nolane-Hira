# R8-W31 handoff — qualified dual semantic adapter

Status: **PRE-QUALIFICATION / IMPLEMENTATION**

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

Implemented:
- `src/nmd/w31_dual_semantic_adapter.py`
- runtime coarse mode `dual_symmetric_semantic`
- exact capacity/identity/runtime/checkpoint tests

Next:
1. isolated W31 fresh authority generator;
2. qualification-only reference workflow;
3. train/dev cache and balanced+anchor trainer;
4. sealed confirm evaluator;
5. closure and full bundle.

Do not create a TRAIN authority marker until qualification passes.
