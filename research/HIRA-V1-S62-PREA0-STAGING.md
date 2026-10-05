# HIRA V1 S62 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #309

Parent S61 merged main:
`e6c7053a8d24c75b5fcf1db75c8aa9f30cea35e2`

Parent verdict:
**Case B**

Fresh S61:
- run `37302829277`
- artifact `11343045582`
- digest `sha256:6e99ba1a5fab92cb6031c41eea54636721b78e690236dfb39788066803785f74`.

## Frozen S62 scientific variable

Reference gate:
- exact S61 gold-CE supervision.

Treatment gate:
- TRAIN-only counterfactual reliability BCE.

Shared:
- exact S61 gate architecture/features;
- 5 params per gate;
- bit-identical initialization;
- alpha max 0.35;
- initial alpha 0.10;
- correction/head/native/cache unchanged by gate losses.

## Frozen reliability target

Probe:
- alpha_probe **0.35**
- tolerance **1e-8**.

Positive target only when fixed bounded probe:
- does not worsen paired gold CE;
- AND strictly lowers paired cross-view JS.

All other cases target 0.

A0 synthetic court includes:
- beneficial => 1;
- correctness harm + JS help => 0;
- CE help + stability harm => 0;
- both harm => 0;
- mixed batch => both classes present.

## A0 staged

Core:
`src/nmd/v1_reliability_supervised_adaptive_gate.py`

Tests:
`tests/test_v1_reliability_supervised_adaptive_gate.py`

Court:
`scripts/hira_v1_s62_a0_train_only_reliability_supervised_gate.py`

Workflow:
`.github/workflows/hira-v1-s62-a0-train-only-reliability-supervised-gate.yml`

Marker:
`research/HIRA-V1-S62-ENABLE-A0`

A0 proves:
- exact target semantics;
- target detached;
- no DEV target dependency;
- reference/treatment gate params 5 each;
- added treatment params 0;
- identical initialization;
- reference gold-CE gradients live;
- treatment reliability-BCE gradients live;
- both upstream gradient paths zero;
- K=3/7/255;
- bounded residual;
- full-K probability mass;
- one encoder/state-once;
- checkpoint replay exact.

## Authorization rule

The A0 marker MUST remain absent until this exact final staging head passes generic CI on Python 3.10 and 3.12.

A0 is mechanical only:
- no fresh S62 TRAIN/DEV;
- no S62 model selection;
- no external Laya/Jev evaluation.
