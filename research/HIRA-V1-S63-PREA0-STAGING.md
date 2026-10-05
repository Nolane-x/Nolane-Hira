# HIRA V1 S63 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #311

Parent S62 merged main:
`798f5c4d933635b36f4e0eaf76503f8213afedd9`

Parent fresh S62:
- run `37307658289`
- artifact `11344855774`
- digest `sha256:7812c348d707ea8675fc9370af573a87d9e5b1a0b10e55a1f335bf890ddb3305`
- verdict **Case B**.

## Frozen S63 scientific variable

Reference:
- exact S62 5-param reliability-supervised gate;
- exact four S61 scalar features;
- exact S62 reliability target/BCE.

Treatment:
- 61-param learned permutation-invariant set reliability gate;
- per-option normalized fused/pairwise geometry;
- shared 4 -> 8 tanh option encoder;
- mean+max pooling;
- concatenated exact four S61 scalar features;
- final representation dim 20;
- exact same S62 reliability target/BCE.

Added treatment parameters: **56**.

## Frozen treatment capacity

- `W_phi [8,4]`
- `b_phi [8]`
- `w_out [20]`
- `b_out scalar`
- total **61**.

Initialization:
- set-encoder seed **63063**
- output weights exactly zero
- output bias `-0.916290731874155`
- initial alpha exactly **0.10**
- alpha max **0.35**.

## Reliability authority retained exactly

- alpha probe **0.35**
- tolerance **1e-8**
- target positive iff probe paired CE is non-worse AND paired JS strictly improves
- TRAIN only
- no DEV target.

## A0 staged

Core:
`src/nmd/v1_learned_set_reliability_gate.py`

Tests:
- `tests/test_v1_learned_set_reliability_gate.py`
- `tests/test_v1_s63_a0_harness.py`
- `tests/test_v1_s63_a0_workflow.py`

Court:
`scripts/hira_v1_s63_a0_learned_set_reliability_gate.py`

Workflow:
`.github/workflows/hira-v1-s63-a0-learned-set-reliability-gate.yml`

A0 proves:
- exact 61-param treatment surface;
- exact tensor names/shapes;
- representation dims 4 -> 8 -> pool16 -> 20;
- deterministic init;
- exact initial alpha 0.10;
- permutation invariance;
- positive-scale/offset invariance;
- flat finite behavior;
- bounded residual;
- K=3/7/255;
- probability mass <=1e-6;
- exact S62 target inheritance;
- first-step output gradients live;
- first-step phi gradients zero by frozen zero-output initialization;
- isolated clone warm-step activates phi gradients;
- production initial state unchanged;
- no upstream gradient;
- deterministic checkpoint replay;
- no fresh S63 TRAIN/DEV.

## Authorization rule

Marker:
`research/HIRA-V1-S63-ENABLE-A0`

The marker MUST remain absent until this exact final pre-A0 staging head passes generic CI on Python 3.10 and 3.12.

A0 is mechanical only:
- no fresh S63 TRAIN/DEV;
- no model selection;
- no external Laya/Jev evaluation.
