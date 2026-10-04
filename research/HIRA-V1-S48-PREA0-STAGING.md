# HIRA V1 S48 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #279  
PR: #280

## Parent

S47 merged main:
`b48ce16faa721a79e998db0c7d585544e4281c9a`

S47 fresh scientific court:
- run `37168305687`
- artifact `11291161358`
- digest `sha256:9887903bcdc7e59907d0bde425389b7642bff2f91f2a7fe5e12555ad89c3db76`
- interpretation **Case C**
- ordinal DEV_READY false

## Frozen S48 variable

Reference:
- existing raw-query private correction.

Treatment:
- same A/B/W parameter shapes and initialization;
- same correction capacity **114,688**;
- same total treatment **163,840**;
- raw normalized query is projected through the centered option-signature difference span;
- only normalized quotient `q*` enters private A/B/W correction;
- raw query has no bypass.

Quotient:
`D_i = S_i - mean(S)`

`u = sum_i <q,D_i>D_i`

`q* = normalize(u)`

No learned quotient parameter.

## Staged courts

Core:
- exact S45/S48 correction capacity
- K=3/7/255
- option permutation equivariance
- orthogonal nuisance invariance
- relation-relevant direction preserved
- distinct relation direction separation
- deterministic zero-subspace behavior
- no raw-query bypass
- probability mass validity

A0 runtime:
- frozen M4 integrity
- one encoder batch/state-once
- matched native runtime
- native/correction gradient isolation
- matched native one-step parameter/output identity
- W -> B -> A warm-start
- actual quotient diagnostics

## Staged files

- `research/HIRA-V1-S48-CONTRACT.md`
- `research/HIRA-V1-S48-INTERPRETATION-PLAN.md`
- `src/nmd/v1_query_quotient_private_correction.py`
- `tests/test_v1_query_quotient_private_correction.py`
- `scripts/hira_v1_s48_a0_query_quotient_option_evidence.py`
- `tests/test_v1_s48_a0_harness.py`
- `.github/workflows/hira-v1-s48-a0-query-quotient-option-evidence.yml`

## Authorization rule

A0 workflow is marker-gated on:

`research/HIRA-V1-S48-ENABLE-A0`

The marker MUST NOT exist until the exact final staging head passes generic CI on Python 3.10 and 3.12.

A0 is diagnostic only:
- no model selection
- no fresh S48 TRAIN/DEV
- no post-A0 semantic tuning except mechanical repair if execution aborts before usable semantic outputs
- no external Laya/Jev evaluation.
