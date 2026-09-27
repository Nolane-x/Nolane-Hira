# HIRA V0 MAINLINE M1 — reliability / OOD / abstention

Status: **M1-B DEV QUALIFICATION FAIL / M1-R2 PREREGISTRATION**

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


## 14. Frozen M1 TRAIN/DEV optimizer contract

Calibration tournament:
- candidates: control / primitive-temperature / primitive-temperature-noul-bias;
- trainable params: 0 / 3 / 4;
- epochs: 8;
- optimizer: Adam;
- lr: 0.01;
- weight decay: 0;
- seed: 11017;
- objective: soft-target NLL + soft Brier;
- DEV selection: lower soft ECE, lower soft NLL, lower soft Brier, higher hard accuracy, fewer params, earlier epoch.

OOD tournament:
- candidates: semantic-linear / semantic-confidence-linear;
- feature counts: 5 / 8;
- trainable params: 6 / 9;
- epochs: 60;
- optimizer: AdamW;
- lr: 0.02;
- weight decay: 0.001;
- seed: 11029;
- positive class = OOD;
- feature normalization fit on TRAIN only;
- threshold selected on DEV only;
- selection: higher AUROC, higher balanced accuracy, lower OOD false-accept, higher ID accept, simpler semantic-only candidate on exact tie.

Both OOD candidates include all four frozen semantic-geometry features:
1. state-question cosine;
2. state-option max cosine;
3. state-option mean cosine;
4. state-option spread.

The full candidate may additionally use confidence diagnostics.
No confidence-only OOD candidate exists.

Selective-risk policy:
- threshold selected on CAL DEV only;
- target selective accuracy: >= 0.90;
- minimum coverage: >= 0.25;
- selection maximizes coverage among thresholds meeting the target.

## 15. Frozen DEV qualification gate

After the first UA-UD/UF-UH empirical run, the selected candidate is allowed to expose UE/UI only if all of the following pass.

Calibration DEV:
- selected soft ECE <= 0.20;
- selected hard accuracy no worse than control by > 0.01;
- ECE improvement >= 0.01 OR control soft ECE already <= 0.15;
- probability mass error <= 1e-6.

Selective DEV:
- `meets_target = true`;
- selective accuracy >= 0.90;
- coverage >= 0.25.

OOD DEV:
- AUROC >= 0.80;
- balanced accuracy >= 0.75;
- OOD false-accept rate <= 0.25;
- selected candidate contains all mandatory semantic features.

If this DEV qualification fails:
- do not expose UE;
- do not expose UI;
- do not tune against sealed evidence;
- close or redesign M1 candidate using only aggregate TRAIN/DEV results.

## 16. Frozen sealed promotion gate

If DEV qualifies, freeze:
- calibration candidate and checkpoint;
- OOD candidate, feature normalization, checkpoint and threshold;
- selective confidence threshold;
- exact W28/W34 provenance.

Then expose UE and UI exactly once.

Calibration on sealed UE:
- soft ECE <= 0.15;
- probability mass error <= 1e-6;
- hard accuracy no worse than fresh uncalibrated control by > 0.01;
- ECE improvement >= 0.02 OR control ECE already <= 0.15.

OOD on sealed UE + UI:
- AUROC >= 0.85;
- balanced accuracy >= 0.80;
- OOD recall >= 0.80;
- OOD false-accept rate <= 0.20;
- ID accept rate >= 0.70.

Final selective reliability on sealed UE:
- accepted coverage >= 0.25;
- accepted hard accuracy >= 0.90;
- selective risk <= 0.10.

Final OOD safety on sealed UI:
- final policy ACCEPT rate on OOD <= 0.10.

Runtime/integrity:
- one state encode per case;
- full-K;
- relation delta = 0;
- no decision-core gradients/training;
- confidence is not OOD authority;
- UE/UI used for no fitting, threshold selection or candidate ranking.

Only if every sealed component passes may M1 produce:

`HIRA_V0_M1_RELIABILITY_READY`

Otherwise:

`HIRA_V0_M1_RELIABILITY_FAIL`

Reliability remains provisional after a failure.


## 17. First M1 TRAIN/DEV authority — frozen result

Authoritative run:

`36305969587`

Empirical head:

`bcc947e95d312e229a93b579356c7eaf6269f788`

Artifact:
- name: `hira-v0-mainline-m1-train-dev`
- ID: `10927625673`
- digest: `sha256:79ff7eb67c3ce82ed82562787d813b75d35e7eca1bed0211f01ddf1b09c31dca`

Exposure:
- UA/UB/UC calibration TRAIN: 108
- UD calibration DEV: 36
- UF/UG OOD TRAIN: 72
- UH OOD DEV: 36
- total state encodes: 252
- state encodes/case: 1.0
- decision-core trainable params: 0
- sealed rows used: false
- UE exposed: false
- UI exposed: false

### Calibration DEV

Control:
- hard accuracy: 0.6388888888888888
- soft ECE: 0.1296966220769617
- soft Brier: 0.23696935145805278
- soft NLL: 0.9593999683856964

Primitive temperature, epoch 7:
- params: 3
- hard accuracy: 0.6388888888888888
- soft ECE: 0.10072291601035327
- soft Brier: 0.2298705099096373
- soft NLL: 0.9521524227327771

Primitive temperature + noul bias, epoch 1:
- params: 4
- hard accuracy: 0.5833333333333334
- soft ECE: 0.0819705815778838
- soft Brier: 0.23640032479953435
- soft NLL: 0.9588056471612718

The original cross-candidate selection key chose the 4-param candidate because it prioritized ECE before the already-frozen hard-accuracy DEV gate.

The DEV gate correctly rejected that selection:
- calibration accuracy non-regression: FAIL
- calibration ECE absolute: PASS
- calibration ECE mechanism: PASS
- probability integrity: PASS

### OOD DEV

Selected candidate:

`semantic-linear`

- parameters: 6
- features:
  - state-question cosine
  - state-option max cosine
  - state-option mean cosine
  - state-option spread
  - normalized log K
- no confidence feature used
- selected epoch: 60
- threshold: 0.52
- AUROC: **0.9969135522842407**
- balanced accuracy: **0.9722222222222222**
- OOD recall: **0.9722222222222222**
- OOD false-accept: **0.027777777777777776**
- ID accept: **0.9722222222222222**

All frozen OOD DEV gates PASS.

The 9-param semantic+confidence candidate was weaker:
- AUROC 0.9567901492118835
- balanced accuracy 0.9305555555555556
- OOD false-accept 0.1388888888888889

This strengthens the separation hypothesis: semantic geometry is a better OOD authority here than raw decision confidence.

### Selective DEV

Original max-probability threshold policy:
- threshold: 0.51
- coverage: 0.2777777777777778
- accepted accuracy: 0.8
- selective risk: 0.2
- target >= 0.90: FAIL

### Frozen DEV verdict

`dev_qualification.pass = false`

Failures:
- calibration accuracy non-regression
- selective accuracy
- selective target

Passes:
- calibration ECE absolute
- calibration ECE mechanism
- calibration probability integrity
- selective coverage
- all OOD gates

Therefore:

**UE and UI remain sealed.**

`HIRA-V0-MAINLINE-M1-ENABLE-CONFIRM` MUST NOT be created for this candidate.

## 18. M1-R2 hypothesis — fresh selective correctness authority

The R1 failure localizes the remaining reliability problem.

OOD is already strong enough to freeze as a provisional M1 subcomponent.

The weak link is selective correctness:
- maximum probability is not sufficiently aligned with correctness;
- calibration selection must respect hard-decision non-regression before optimizing ECE.

M1-R2 changes only the reliability control layer.

### Frozen from M1-B R1

OOD candidate:
- `semantic-linear`
- 6 trainable parameters
- epoch 60
- threshold 0.52
- exact source artifact: `10927625673`
- source artifact digest: `sha256:79ff7eb67c3ce82ed82562787d813b75d35e7eca1bed0211f01ddf1b09c31dca`

It is not retrained or retuned in R2.

### Fresh calibration selection rule

Use wholly fresh R2 TRAIN/DEV.

First filter candidates by:
- hard accuracy regression versus fresh control <= 0.01;
- probability mass error <= 1e-6.

Only eligible candidates participate in calibration ranking.

Among eligible candidates:
1. lower soft ECE;
2. lower soft NLL;
3. lower soft Brier;
4. fewer parameters;
5. earlier epoch.

This prevents a lower-ECE calibrator from silently damaging the typed hard decision.

### New selective-risk head

Do not use max probability as the selective authority.

Train a tiny correctness predictor on frozen decision features.

Target:
- 1 if the frozen typed hard decision is correct;
- 0 otherwise.

Candidate families:
1. `semantic-risk-linear`
   - semantic geometry + normalized K
   - 6 parameters
2. `semantic-confidence-risk-linear`
   - semantic geometry + calibrated confidence diagnostics + normalized K
   - 9 parameters

The risk head is distinct from OOD:
- OOD predicts distribution mismatch;
- selective-risk predicts decision correctness conditional on being evaluated.

No decision-core parameter may train.

### Fresh R2 ID authority

R2 must not reuse UA-UD for fitting or selection.

Use wholly fresh domains:
- R2 TRAIN: UJ / UK / UL / UM
- R2 DEV: UN / UO

Each domain remains balanced over:
- choice
- score
- noul
- strong / mixed / thin evidence bands

R2 may continue using untouched UE as final selective/calibration sealed confirmation only after fresh R2 DEV passes.

UI remains untouched OOD sealed confirmation for the already-frozen semantic-linear OOD head.

### R2 DEV gate

Calibration:
- hard accuracy regression <= 0.01
- soft ECE <= 0.20
- ECE improvement >= 0.01 OR fresh control ECE <= 0.15
- probability mass error <= 1e-6

Selective risk:
- accepted accuracy >= 0.90
- coverage >= 0.25
- selective risk <= 0.10

Frozen OOD:
- retain exact R1 candidate/checkpoint/normalization/threshold
- no retuning against R2 ID DEV

Only if every R2 DEV gate passes may UE/UI be exposed.
