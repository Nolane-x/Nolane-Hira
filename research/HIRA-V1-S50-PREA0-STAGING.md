# HIRA V1 S50 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #283  
PR: #284

## Parent

S49 merged main:
`4273fa0848093385374c0157ea7f3a1ce667b6e3`

S49 scientific closure:
**INVALID MATCHED COURT / INCONCLUSIVE**

Reason:
reference and treatment native runtime trajectories differed at all 24 epochs, violating the preregistered matched-native evidence hierarchy.

## Frozen S50 design

S50 eliminates branch-specific native divergence by construction:

1. one shared native authority trained to fixed epoch 24 on TRAIN only, with zero DEV exposure;
2. one frozen native checkpoint;
3. one immutable shared-native evidence cache;
4. two private readout branches over the same cache.

Reference private signature:
question-conditioned native relation signature.

Treatment private signature:
query-free state↔option identity.

Both retain the same raw query correction readout.

## Core staged

- `research/HIRA-V1-S50-CONTRACT.md`
- `research/HIRA-V1-S50-INTERPRETATION-PLAN.md`
- `src/nmd/v1_shared_native_private_readouts.py`
- `tests/test_v1_shared_native_private_readouts.py`

Core invariants:
- clone + detach + contiguous cache
- stable SHA-256 evidence digest
- source mutation isolation
- deterministic cache replay
- cache requires_grad false
- correction initialization bit-identical
- reference/treatment correction count 114,688
- treatment identity parameters 0
- branch-order replay exact
- K=3/7/255
- full-K probability mass.

## A0 staged

- `scripts/hira_v1_s50_a0_shared_native_forked_readouts.py`
- `tests/test_v1_s50_a0_harness.py`
- `.github/workflows/hira-v1-s50-a0-shared-native-forked-readouts.yml`

A0 uses 16 wholly new S50-A0 cases and the frozen M4 runtime.

Actual-runtime court requires:
- one native output authority
- shared cache digests including triadic + native relation evidence
- deterministic cache replay
- no live gradient graph in cache
- no native parameters in private optimizer
- bit-identical correction initialization
- exact branch-order replay
- fused replay from cached triadic evidence only
- live raw-query correction path
- treatment identity diagnostics
- probability mass validity.

## Authorization rule

A0 workflow is marker-gated on:

`research/HIRA-V1-S50-ENABLE-A0`

The marker MUST NOT exist until the exact final staging head passes generic CI on Python 3.10 and 3.12.

A0 is diagnostic only:
- no model selection
- no S50 TRAIN/DEV
- no external Laya/Jev evaluation
- no semantic tuning after exposure except a separately documented pre-semantic mechanical abort.
