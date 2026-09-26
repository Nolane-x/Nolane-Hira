# R8-W24 handoff — atomic severity-factor recoverability

Status: **CLOSED DIAGNOSTIC. Frozen outcome: `STABLE_ATOMIC_SEVERITY_LOCALIZATION`; stable classification `HIRA_ATOMIC_SEVERITY_GEOMETRY_LIMIT` on 4/4 fresh domains.**

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
**DP/DQ/DR/DS permanently exposed by authority run `36227184733`.**

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


---

## 19. Authoritative W24 closure

Exact verdict-bearing head:
`1cdc31f6beaa7d195831f24ccf8bdde9889c2a1b`

Authority run:
`36227184733`

All authority jobs PASS:
- frozen unit/contracts/freshness;
- exact W9 provenance;
- fresh DP/DQ/DR/DS cache;
- frozen HIRA direct four-way and atomic-factor evaluation;
- frozen directional DeBERTa atomic-factor authority;
- frozen W23 DeBERTa/RoBERTa direct-severity + confidence controls;
- frozen classifier/outcome verification.

Artifacts:
- frozen W9 bundle: `10900309386`;
  digest `sha256:93d26c79f593297fc4453fbe36fcb80c6bc67f4df4dae23069ad60c5f63c7f8c`;
- fresh W24 cache: `10900374149`;
  digest `sha256:185f0e96866c9975e51980e357cd5f8a2981b5c82b1cfcbb6bca54eb67ee12fc`;
- authoritative W24 audit: `10902370888`;
  digest `sha256:40ac9d35e4824f8b06c476128b076dbeeb987650b35f013e3eb1030ba9227505`.

Integrity:
- 384 fresh cases;
- 96/domain;
- balanced S×C cells;
- exact frozen mapping S0->000, S1->100, S2->110, S3->111;
- one logical state compile/case;
- one batched A13 invocation/case;
- exactly two isolated HIRA query sequences/case;
- trainable parameters = 0;
- training = false;
- selection = false;
- exact frozen A13/W9/reference hashes;
- no W23-W18 rows;
- no Banking77;
- no typed final/test;
- campaign cells = 0.

## 20. Frozen verdict

Overall:

**`STABLE_ATOMIC_SEVERITY_LOCALIZATION`**

Stable classification:

**`HIRA_ATOMIC_SEVERITY_GEOMETRY_LIMIT`**

Per-domain:
- DP -> `HIRA_ATOMIC_SEVERITY_GEOMETRY_LIMIT`;
- DQ -> `HIRA_ATOMIC_SEVERITY_GEOMETRY_LIMIT`;
- DR -> `HIRA_ATOMIC_SEVERITY_GEOMETRY_LIMIT`;
- DS -> `HIRA_ATOMIC_SEVERITY_GEOMETRY_LIMIT`.

Reference-environment-stable domains:
**4 / 4**.

Atomic-reference-adequate domains:
**4 / 4**.

This is the first W18-W24 latent-severity phase in which the independent authority gate passes on all fresh domains and a HIRA-specific severity-geometry interpretation is therefore authorized.

## 21. Independent atomic-factor authority

Pooled directional DeBERTa atomic reference:

- F0 top1 **96.875%**;
  balanced accuracy **97.917%**;
- F1 top1 **100.000%**;
  balanced accuracy **100.000%**;
- F2 top1 **100.000%**;
  balanced accuracy **100.000%**;
- full factor-vector accuracy **96.875%**;
- deterministic composed severity **96.875%**;
- composed severity MAE **0.03125**;
- invalid-vector rate **0%**;
- probability-mass max error <= `1.1920928955078125e-07`.

Therefore the atomic severity authority itself is independently recoverable under the frozen primary reference.

## 22. Reference environment positive controls

Frozen W23-style DeBERTa control, pooled:
- direct severity **90.625%**;
- confidence **100.000%**;
- joint **90.625%**.

Frozen W23-style RoBERTa control, pooled:
- direct severity **87.500%**;
- confidence **100.000%**;
- joint **87.500%**.

Confidence positive-control agreement is **100%** on every domain.

The fresh W24 environment is therefore stable by the preregistered control.

## 23. Frozen HIRA result

Pooled direct four-way severity:
- top1 **26.563%**;
- MRR **57.161%**;
- MAE **0.96875**.

Pooled HIRA atomic factors:
- F0 top1 **33.594%**;
  balanced accuracy **55.729%**;
- F1 top1 **60.938%**;
  balanced accuracy **60.938%**;
- F2 top1 **25.000%**;
  balanced accuracy **50.000%**;
- full factor-vector accuracy **0%**;
- deterministic composed severity **0%**;
- composed severity MAE **3.0**;
- invalid factor-vector rate **100%**;
- factor probability-mass max error <= `1.1920928955078125e-07`.

Factorization gain:
**-26.563 percentage points**.

Direct->composed transitions:
- wrong -> right: **0 / 384**;
- right -> wrong: **102 / 384**.

Fresh HIRA confidence positive-control:
- top1 **60.417%**.

Every fresh domain fails HIRA atomic adequacy.

## 24. Scientific interpretation

The strongest defensible conclusion is:

> W24 independently establishes that the fresh severity task is recoverable when decomposed into preregistered atomic factors, while the frozen A13/W9 HIRA semantic geometry fails to recover those same factors. Because the reference environment is stable on 4/4 domains and the atomic authority reaches 96.875% composed severity, the repeated severity failure can now be localized to the frozen HIRA latent severity representation/scoring geometry rather than to authority ambiguity alone.

This is stronger than W18-W23 because authority inadequacy no longer blocks the HIRA interpretation.

The failure pattern is structural:
- F2 is at **25%** top1 with **50%** balanced accuracy;
- F0 is also weak;
- the predicted factor vector is invalid on **100%** of cases;
- deterministic composition collapses to **0%**, not merely a small regression from direct severity.

W24 does **not** prove:
- A13 alone is the sole bottleneck;
- W9 projection alone is the sole bottleneck;
- token MaxSim alone is the sole bottleneck;
- retraining will necessarily solve the issue;
- deterministic typed composition is production-ready.

It localizes the failure to the frozen HIRA severity semantic path as a whole.

## 25. Authorization after W24

W24 authorizes a narrower next research program:

**Diagnose which frozen HIRA severity-geometry component causes atomic-factor collapse.**

A valid next phase may use new TRAIN/DEV data because the geometry limit is now independently established, but must still reserve wholly fresh dual-CONFIRM domains before any production claim.

The next phase should isolate, in a controlled ablation:
1. A13 representation ceiling;
2. W9 256->128 projection damage;
3. symmetric token-MaxSim scorer/interface;
4. candidate definition/prototype aggregation;
5. factor calibration only after representation/scoring capacity is shown.

A strong next design should compare on fresh TRAIN/DEV:
- raw A13 semantic features;
- frozen W9 projection;
- lightweight newly trained projection;
- pooled/cross-token alternatives;
- matched direct vs atomic controls.

Promotion requires:
- TRAIN improvement;
- DEV replication;
- two sealed fresh CONFIRM authorities;
- no reuse of DP/DQ/DR/DS for model selection;
- explicit calibration/OOD before production;
- still no K32/K64 automatically.

## 26. Permanent exposure after W24

DP/DQ/DR/DS are permanently exposed.

Never reuse them for:
- projection/scorer training;
- architecture selection;
- factor wording/prototype selection;
- calibration;
- threshold/gate tuning;
- checkpoint/seed selection;
- production promotion.

All prior exposed authorities remain forbidden.

A future AI must read this closure before opening the next phase.
