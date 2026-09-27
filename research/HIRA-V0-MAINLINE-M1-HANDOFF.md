# HIRA V0 MAINLINE M1 — reliability / OOD / abstention

Status: **M1-A MECHANISM IMPLEMENTED / FRESH AUTHORITY PRE-EXPOSURE**

Issue: #161  
Branch: `feat/hira-v0-mainline-m1-reliability`  
Base main: `17ad32104ac610908230c13d4b936a1b1ac3ecac`

## 1. Why M1 exists

M0 integrated a real Hira v0 typed model shell with exact W28 + W34 provenance.

M0 intentionally did not claim:
- calibrated confidence;
- OOD detection;
- abstention;
- selective acceptance.

M1 makes those first-class product behavior.

## 2. Non-negotiable separation

Calibration is not OOD.

OOD is not:
- 1 - max softmax;
- entropy;
- top-margin;
- temperature.

Decision confidence signals are diagnostics only.

Automatic ACCEPT requires three independent qualified authorities:

1. calibrated decision probabilities;
2. qualified OOD authority returning in-distribution;
3. qualified selective-risk threshold.

Missing or provisional authority fails closed.

## 3. M1-A mechanism

Implemented:

- `src/nmd/mainline_reliability.py`
- `ConfidenceDiagnostics`
- `HiraV0ReliabilityReceipt`
- `HiraV0ReliableDecision`
- `HiraV0ReliabilityPolicy`
- `HiraV0Session.decide_reliable()`
- M1 provisional manifest

Diagnostics:

- max probability;
- normalized entropy;
- top-1 / top-2 probability margin;
- option count.

These metrics never act as OOD authority.

## 4. Fail-closed action order

Final reliability action is one of:

- `accept`
- `abstain`
- `escalate`

Frozen M1-A order:

1. missing/unqualified OOD authority -> ESCALATE;
2. OOD authority detects explicit distribution shift -> ESCALATE;
3. missing/stale OOD calibration -> ESCALATE;
4. qualified OOD score above threshold -> ABSTAIN;
5. calibration authority not qualified -> ESCALATE;
6. probabilities not explicitly marked qualified-calibrated -> ESCALATE;
7. selective-risk policy not qualified -> ESCALATE;
8. qualified calibrated confidence below frozen threshold -> ABSTAIN;
9. only otherwise -> ACCEPT.

Thus high raw model confidence can never bypass OOD or calibration.

## 5. Typed semantics remain separate

`noul` is still a typed semantic decision primitive.

Reliability abstention is a control-plane result.

A model may produce an ordinary `noul` decision while reliability separately returns:
- accept;
- abstain;
- escalate.

The two meanings must never be conflated.

## 6. State-once contract

`HiraV0Session.decide_reliable()` first executes the ordinary typed decision through the existing session.

It then computes reliability from the resulting decision and external OOD authority inputs.

It does not call `compile_state` again.

M0 invariants remain:
- full-K;
- relation refinement OFF;
- W34 provisional transfer core;
- dynamic schema;
- one state encode per session.

## 7. Current manifest

M1-A manifest:

- version: `0.0-m1a`
- semantic front-end: provisional
- projection: frozen research base
- transfer core: provisional
- typed runtime: available
- reliability/OOD/abstention: **provisional**
- high-K: pending
- multilingual: pending
- production-ready: false

M1-A does not promote reliability to available.

## 8. Fresh authority still required

No M1 empirical reliability authority has been exposed yet.

M1-B must use wholly fresh evidence.

Required evidence separation:

- calibration TRAIN;
- calibration DEV;
- selective-risk CONFIRM;
- OOD DEV;
- OOD sealed CONFIRM.

No sealed row may be used for:
- calibration fitting;
- OOD-head fitting;
- threshold selection;
- candidate selection.

## 9. Candidate families

Calibration mechanism family may reuse code knowledge from W6c:
- primitive temperature: 3 trainable params;
- primitive temperature + semantic noul bias: 4 trainable params.

But W6c rows are not M1 authority.

OOD must be independent.

Candidate families may include:
- frozen-feature distance;
- energy-like semantic score;
- tiny dedicated OOD head.

No candidate may define OOD as `1-max_probability`.

## 10. Decision core freeze

During M1 reliability training:

- semantic front-end gradients: forbidden;
- W28 projection gradients: forbidden;
- W34 transfer gradients: forbidden;
- HIRA relation-core gradients: forbidden.

Only explicitly declared reliability/control parameters may train.

## 11. M1-A exit boundary

Mechanism code may merge after:
- fail-closed contracts pass;
- state-once test passes;
- confidence-as-OOD shortcut test passes;
- M0 regressions pass;
- full repository CI passes.

But M1 overall remains open until fresh calibration + OOD sealed authority is completed.

Next after M1-A mechanism merge or freeze:
- build fresh M1 authority namespace;
- fit tiny calibration candidate;
- build independent OOD candidate tournament;
- freeze thresholds on DEV;
- expose sealed CONFIRM once;
- only then promote reliability from provisional to available.


## 12. Fresh M1 authority namespace — implemented, unexposed

Implemented:
- `src/nmd/mainline_m1_authority.py`
- `tests/test_mainline_m1_authority.py`

Partitions:

Calibration/selective ID:
- CAL TRAIN: UA / UB / UC = 108 cases
- CAL DEV: UD = 36 cases
- SELECTIVE CONFIRM: UE = 36 cases, sealed

OOD:
- OOD TRAIN: UF / UG = 72 cases
- OOD DEV: UH = 36 cases
- OOD CONFIRM: UI = 36 cases, sealed

Every domain is exactly balanced across:
- choice
- score
- noul

Calibration ID rows also balance fresh confidence bands:
- strong
- mixed
- thin

OOD rows balance:
- topic mismatch
- foreign task
- no relevant evidence

Important:
- OOD rows have no fabricated gold decision;
- OOD labels are separate from typed-task labels;
- calibration and OOD state texts are disjoint;
- sealed partitions fail closed unless explicitly opened;
- exact text is regression-tested against W33/W34 authority and W6c reliability lexicons.

No M1 model inference, calibration fitting, OOD training, threshold selection, or sealed exposure has occurred yet.

## 13. Immediate next boundary

Before any M1 empirical exposure:

1. M1 mechanism + authority contracts must pass dedicated CI;
2. full repository Python 3.10/3.12 CI must pass;
3. freeze exact M0/W34 provenance;
4. implement frozen logit/feature cache so the decision core cannot receive gradients;
5. only then expose CAL TRAIN/DEV and OOD TRAIN/DEV;
6. select calibration candidate, OOD candidate and selective thresholds using DEV only;
7. freeze all selections;
8. expose UE and UI once;
9. reliability may become `available` only if sealed gates pass.
