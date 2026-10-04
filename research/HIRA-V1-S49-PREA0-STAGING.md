# HIRA V1 S49 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #281  
PR: #282

## Parent

S48 merged main:
`721b506688044f75b142e7a0f41a63cf74663674`

S48 scientific verdict:
**DOMINATED NEGATIVE** from run `37172187841`.

No S48 rerun is authorized.

## Frozen S49 variable

Reference:
question-conditioned native relation signature + raw-query private correction.

Treatment:
query-free state↔option identity + the same raw-query private correction.

Unchanged:
- native trainable **49,152**
- correction **114,688**
- total treatment **163,840**
- one encoder batch
- no second encoder
- native relation path
- raw query readout
- A/B/W shapes/init.

Identity adds **0 parameters** and its API accepts no question tensor.

## Staged S49-A0 courts

Core:
- K=3/7/255
- no-question API
- option permutation equivariance
- state-token permutation invariance
- option-token permutation invariance
- option-view permutation invariance
- masked-padding invariance
- distinct controlled identities
- raw query remains live after fixed identity
- probability mass
- checkpoint roundtrip.

Runtime/ownership:
- canonical M4 runtime
- one encoded batch/state-once
- actual identity cross-state-view diagnostics
- exact matched native initialization
- private correction -> native gradient 0
- native objective -> private gradient 0
- matched native one-step gradient/parameter/output identity
- deterministic W -> B -> A warm-start.

## Staged files

- `research/HIRA-V1-S49-CONTRACT.md`
- `research/HIRA-V1-S49-INTERPRETATION-PLAN.md`
- `src/nmd/v1_query_free_option_identity.py`
- `tests/test_v1_query_free_option_identity.py`
- `scripts/hira_v1_s49_a0_query_free_option_identity.py`
- `tests/test_v1_s49_a0_harness.py`
- `.github/workflows/hira-v1-s49-a0-query-free-option-identity.yml`

## Authorization rule

A0 workflow is marker-gated on:
`research/HIRA-V1-S49-ENABLE-A0`

The marker MUST remain absent until the exact final pre-A0 staging head passes generic CI on Python 3.10 and 3.12.

A0 is diagnostic only:
- no model selection
- no S49 TRAIN/DEV exposure
- no external Laya/Jev evaluation
- only a pre-semantic mechanical abort may authorize a mechanical A0 repair.
