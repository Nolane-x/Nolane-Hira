# HIRA V1 S64 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #315

Parent S63 merged main:
`4fcb58f0c35e2feca841b08f6f560c092ae600fd`

Parent fresh S63:
- run `37314015283`
- artifact `11347706059`
- digest `sha256:dcb69358dfa97f3644198534faa1c2f5f214b4f1773d79008eec13cafaf2a97e`
- verdict **Case B**.

## Frozen S64 scientific variable

Both arms use exactly **60 trainable params** and bit-identical initialization.

Reference:
- S63 decision-surface option features;
- four contextual channels forced exactly to zero.

Treatment:
- same surface features;
- four contextual channels from a fixed detached projection of the S59 512D state/query/option representation.

Only treatment receives contextual information.

## Frozen contextual path

Context source:
- exact S59 detached representation;
- no new encoder call;
- contains query-free identity + joint state/query/option context.

Projection:
- shape **[4,512]**
- seed **64064**
- row normalized
- buffer only
- zero trainable parameters.

Option input:
- surface 4 + context 4 = **8**.

Learned encoder:
- W_phi **[5,8]**
- b_phi **[5]**
- tanh
- mean+max pool => **10**.

Append exact four S61 scalar features:
- final representation **14**.

Output:
- w_out **[14]**
- b_out scalar.

Total:
**40 + 5 + 14 + 1 = 60 params per arm**.

Treatment parameter advantage: **0**.

Initialization:
- encoder seed **64164**
- w_out exact zero
- b_out `-0.916290731874155`
- alpha initial **0.10**
- alpha max **0.35**.

## Reliability authority retained exactly

- S62 alpha probe **0.35**
- target tolerance **1e-8**
- target positive iff paired probe CE non-worse AND paired JS strictly better
- TRAIN only
- no DEV target.

## A0 staged

Core:
`src/nmd/v1_contextual_reliability_gate.py`

Unit mechanics:
`tests/test_v1_contextual_reliability_gate.py`

A0 court:
`scripts/hira_v1_s64_a0_contextual_reliability_gate.py`

Guards:
- `tests/test_v1_s64_a0_harness.py`
- `tests/test_v1_s64_a0_workflow.py`

Workflow:
`.github/workflows/hira-v1-s64-a0-contextual-reliability-gate.yml`

A0 must prove:
- matched 60/60 trainable params
- added treatment params 0
- parameter init bit-identical
- context projection fixed/non-trainable
- reference context exact zero
- treatment context non-degenerate
- context sensitivity with fused/pair surfaces held fixed
- option permutation invariance
- decision-surface affine invariance
- flat finite mechanics
- alpha initial 0.10
- exact S62 target
- staged gradient path
- upstream gradient zero
- K=3/7/255
- bounded residual
- probability mass <=1e-6
- exact checkpoint replay
- one encoder/state-once
- no fresh S64 TRAIN/DEV.

## Authorization rule

Marker:
`research/HIRA-V1-S64-ENABLE-A0`

The marker MUST remain absent until this exact final pre-A0 staging head passes generic CI on Python 3.10 and 3.12.

A0 is mechanical only:
- no fresh S64 TRAIN/DEV
- no model selection
- no external Laya/Jev evaluation.
