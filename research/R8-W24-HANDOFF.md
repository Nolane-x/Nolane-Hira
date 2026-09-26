# R8-W24 handoff — atomic severity-factor recoverability

Status: **PRE-DIAGNOSTIC. No DP/DQ/DR/DS HIRA/reference cache or W24 empirical result exists yet.**

Issue: #135

Branch:
`feat/r8-w24-atomic-severity-factors`

Base main:
`c7dd2a3b64e6874bf5961e20f24ec4b68a5b8649`

Read first:
1. this file;
2. `research/R8-W23-HANDOFF.md`;
3. `research/R8-W22-HANDOFF.md`;
4. issue #135.

## 0. HIRA identity

Nolane HIRA is a compact non-autoregressive typed decision engine.

W24 remains diagnostic:
- zero HIRA training;
- no production promotion;
- no typed-kernel integration;
- no K32/K64;
- wholly fresh authority only.

## 1. Why W24 exists

W23 closed with:
**`AUTHORITY_REFERENCE_UNRESOLVED`**

W23 cross-encoder pooled prototype:
- DeBERTa severity 64.063%, confidence 100.000%, joint 64.063%;
- RoBERTa severity 58.594%, confidence 95.833%, joint 55.208%.

The critical split:
- confidence is highly recoverable by strong NLI cross-encoders;
- severity remains weak and cross-model severity agreement is only 65.63–75.00%.

Therefore W24 does not swap in another reference model.
It tests whether the four-way severity class is semantically entangled.

## 2. Frozen HIRA base

A13:
- `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length 256;
- frozen.

W9 semantic projection:
- 256->128 bias-free;
- scorer SHA256 `8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102`;
- source run `36122220588`;
- freeze artifact `10858424139`.

Trainable parameters:
**0**.

## 3. Fresh W24 authority

Domains:
- DP — municipal recreation-facility access service assessment — seed `431101`;
- DQ — nonprofit appointment-scheduling service triage — seed `431107`;
- DR — university print-and-copy service incident assessment — seed `431119`;
- DS — public EV-charging information service triage — seed `431131`.

Per domain:
- severity S0-S3;
- confidence C0-C2 positive control;
- 8 fresh variants/S×C;
- 96 cases/domain.

Total:
**384 cases**.

Each logical HIRA case:
- exactly one isolated severity narrative;
- exactly one isolated confidence narrative;
- one logical state compile;
- one batched A13 query invocation;
- exactly two encoded query sequences.

Exposure:
**NONE.**

## 4. Frozen atomic severity factors

Three binary monotone factors:

### F0 — meaningful disruption

0:
trivial/local nuisance; ordinary task remains essentially normal.

1:
materially noticeable disruption requiring adaptation, workaround or operational follow-up.

### F1 — major functional loss

0:
important/core service function remains practically available.

1:
important/core function is materially unavailable or heavily impaired.

### F2 — immediate criticality

0:
ordinary or prompt handling is sufficient.

1:
delay is unacceptable; immediate intervention is required.

Gold vectors:
- S0 -> 000;
- S1 -> 100;
- S2 -> 110;
- S3 -> 111.

Exact deterministic decode:
- 000 -> S0;
- 100 -> S1;
- 110 -> S2;
- 111 -> S3;
- all other vectors -> UNDEFINED and incorrect.

No monotonic repair.
No nearest-vector projection.
No learned decoder.
No tie rescue.

## 5. Direct four-way control

The same isolated severity narrative is scored with a fresh four-way S0-S3 schema.

This direct control is fixed before exposure and is not allowed to select W24 factors or gates.

## 6. HIRA factor scoring

Each factor:
- absent/present candidates;
- fresh D0/D1/D2 definitions per side;
- exact frozen W9 symmetric bidirectional token MaxSim;
- arithmetic directional mean;
- arithmetic mean across D0/D1/D2;
- frozen scorer scale;
- two-option softmax.

No learned thresholds or calibration.

## 7. Primary independent factor reference

DeBERTa-v3 NLI:
- `cross-encoder/nli-deberta-v3-base`;
- revision `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`;
- weight SHA256 `d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa`.

Primary W24 factor scoring is directional NLI:
- premise = severity narrative;
- hypothesis = factor-option statement;
- entailment probability;
- higher absent/present entailment score wins.

This is frozen before exposure.

## 8. W23 cross-encoder control

Retain exact W23 bidirectional prototype scoring as a non-selecting control:
- DeBERTa-v3 NLI;
- RoBERTa NLI.

Direct four-way severity remains reported under this control.

## 9. Confidence positive control

Fresh confidence field uses the exact W23 cross-encoder methodology.

Per domain require:
- DeBERTa confidence >= .95;
- RoBERTa confidence >= .90;
- CE0/CE1 confidence agreement >= .90.

Failure:
`W24_REFERENCE_ENVIRONMENT_UNSTABLE`.

No HIRA severity interpretation on that domain.

## 10. Atomic reference adequacy

After the confidence positive control passes, require per domain:

Directional DeBERTa:
- F0 >= .92;
- F1 >= .92;
- F2 >= .92;
- balanced accuracy >= .90 each;
- full factor vector >= .85;
- composed severity >= .85;
- invalid vector rate <= .05;
- finite scores;
- probability-mass error <=1e-6.

Failure:
`W24_ATOMIC_REFERENCE_INADEQUATE`.

## 11. HIRA atomic adequacy

Interpret only on atomic-reference-adequate domains.

Require:
- F0 >= .80;
- F1 >= .80;
- F2 >= .80;
- full vector >= .62;
- composed severity >= .65;
- invalid vector rate <= .15;
- probability-mass error <=1e-6.

## 12. Causal factorization gain

Compare on identical fresh cases:

`HIRA atomic-composed severity - HIRA direct four-way severity`

Require:
**>= +0.12**

for the factor-interface interpretation.

## 13. Frozen domain classifications

### `ATOMIC_SEVERITY_INTERFACE_LIMIT`

Require:
- reference environment stable;
- atomic reference adequate;
- HIRA atomic adequate;
- causal gain >= +.12.

### `HIRA_ATOMIC_SEVERITY_GEOMETRY_LIMIT`

Require:
- reference environment stable;
- atomic reference adequate;
- HIRA atomic adequacy fails.

### `DIRECT_SEVERITY_ALREADY_ADEQUATE`

Require:
- reference environment stable;
- atomic reference adequate;
- HIRA direct severity >= .75;
- factorization gain < .05.

Otherwise:
`ATOMIC_SEVERITY_UNRESOLVED`.

Reference failures take precedence.

## 14. Cross-domain outcome

- same interpretable HIRA class >=3/4 ->
  `STABLE_ATOMIC_SEVERITY_LOCALIZATION`;
- reference environment unstable >=3/4 ->
  `REFERENCE_ENVIRONMENT_UNSTABLE`;
- atomic reference inadequate >=3/4 ->
  `ATOMIC_AUTHORITY_UNRESOLVED`;
- two incompatible interpretable classes >=2 each ->
  `MIXED_ATOMIC_SEVERITY_LOCALIZATION`;
- else ->
  `ATOMIC_SEVERITY_UNRESOLVED`.

## 15. Required metrics

HIRA direct:
- severity top1/MRR/margin/MAE.

HIRA factors:
- F0/F1/F2 top1/balanced accuracy/MRR/margin;
- full-vector accuracy;
- invalid-vector rate;
- composed severity top1/MAE;
- direct->composed transitions.

Primary reference:
- directional factor scores;
- factor accuracy/balanced accuracy;
- vector accuracy;
- invalid-vector rate;
- composed severity.

Controls:
- W23-style cross-encoder direct severity;
- confidence positive-control metrics.

Integrity:
- exact hashes;
- 384 cases;
- 96/domain;
- balanced S/C cells;
- exact factor mapping;
- one HIRA state compile/case;
- one HIRA A13 invocation/case;
- two HIRA query sequences/case;
- zero training/selection;
- no W23-W18 rows;
- no Banking77;
- no typed final/test;
- campaign cells = 0.

## 16. Permanent forbidden evidence

Never tune using:
- DL/DM/DN/DO;
- DH/DI/DJ/DK;
- DD/DE/DF/DG;
- CZ/DA/DB/DC;
- CV/CW/CX/CY;
- CR/CS/CT/CU;
- all older exposed authorities;
- Banking77 0-799;
- typed final/test;
- campaign cells.

## 17. Authorization boundary

Stable `ATOMIC_SEVERITY_INTERFACE_LIMIT`:
- later wholly fresh TRAIN/DEV/dual-CONFIRM may test atomic severity + deterministic composition;
- calibration/OOD remains mandatory.

Stable `HIRA_ATOMIC_SEVERITY_GEOMETRY_LIMIT`:
- no production integration;
- future work may diagnose/train severity geometry only on new TRAIN/DEV.

Reference/authority unresolved:
- do not tune HIRA.

W24 alone cannot:
- open K32/K64;
- alter W18-W23 verdicts;
- promote deterministic typed composition to production.

## 18. Current state

Completed:
- W23 frozen/merged;
- W23 issue closed;
- W24 issue #135 preregistered before exposure;
- W24 branch created from exact post-W23 main;
- fresh domains/seeds frozen;
- three atomic factors and exact monotone decoder frozen;
- DeBERTa factor-reference model/revision/hash frozen;
- W23 cross-encoder control frozen;
- reference/HIRA gates frozen;
- this handoff written before exposure.

Exposure:
**NONE.**

Next:
1. implement fresh W24 authority generator;
2. implement factor schemas and direct severity control;
3. implement freshness contracts;
4. implement two-field HIRA cache;
5. implement HIRA factor/direct evaluator;
6. implement directional DeBERTa factor reference;
7. implement W23 control + confidence positive control;
8. implement classifier/outcome;
9. add unit/contracts + pre-data workflow;
10. exact-head unit + repo CI;
11. only then enable empirical authority;
12. freeze result before merge.

A future AI must update this file after every meaningful W24 session.
