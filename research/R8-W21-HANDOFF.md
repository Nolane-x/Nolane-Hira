# R8-W21 handoff — prototype-grounded latent semantic recoverability

Status: **PRE-DIAGNOSTIC. No DD/DE/DF/DG A13/reference cache or W21 empirical result exists yet.**

Issue: #129

Branch:
`feat/r8-w21-prototype-grounded-latent`

Base main:
`645cb4fc35232027f2ffdeeb391864c7fd45f994`

Read first:
1. this file;
2. `research/R8-W20-HANDOFF.md`;
3. `research/R8-W19-HANDOFF.md`;
4. `research/R8-W18-HANDOFF.md`;
5. issue #129.

## 0. HIRA identity

Nolane HIRA is a compact non-autoregressive typed decision engine.

Long-term target:
- state logically compiled once;
- dynamic semantic schemas and high-cardinality candidates;
- isolated fields where contextual contamination matters;
- typed choice/score/noul;
- semantic transfer;
- explicit calibration/OOD/reliability;
- deterministic composition only after latent semantics are independently established;
- small local-friendly trainable footprint;
- sealed fresh authorities and immutable negative evidence.

Scientific rules:
- freeze before exposure;
- no post-exposure wording/gate/prototype changes;
- reference is diagnostic only;
- negative/partial results remain negative/partial;
- no production promotion from descriptive gains.

## 1. Frozen semantic base

A13:
- `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length 256;
- frozen.

W9 projection:
- 256->128 bias-free;
- scorer SHA `8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102`;
- source run `36122220588`;
- freeze artifact `10858424139`.

W21 trainable parameters:
**0**.

## 2. Decisive history

W16:
- stable contextual-state contamination.

W17:
- field isolation repairs direct intent semantics but whole typed competence remains weak.

W18:
- deterministic composition works exactly when latent variables are correct;
- authority remains reference-inadequate.

W19:
- cumulative ordinal interface is not reference-adequate.

W20:
- merged main `645cb4fc35232027f2ffdeeb391864c7fd45f994`;
- verdict-bearing head `85d9496c31bbdbe7a93268a50c21c54bf06c9001`;
- authority run `36217155264`;
- frozen outcome `LATENT_BOUNDARY_UNRESOLVED`;
- 4/4 `W20_REFERENCE_INADEQUATE`.

W20 pooled HIRA:
- flat-local severity 71.88%;
- pairwise severity 61.46%;
- flat-local confidence 78.13%;
- pairwise confidence 75.00%;
- swap identity 1.0.

W20 MiniLM:
- pairwise severity 60.42%;
- pairwise confidence 70.31%;
- swap identity 1.0.

Therefore candidate ordering and cumulative decoding are not sufficient explanations.
Repeated weakness now centers on **abstract latent semantics**.

## 3. W21 scientific question

Can latent severity/confidence become independently recoverable when class meaning is grounded by multiple concrete prototypes rather than abstract class prose?

Primary comparison:
1. fresh abstract categorical baseline;
2. concrete prototype-grounded class scoring.

No training.

## 4. Fresh authority

Fresh domains:
- DD — regional archive access disruption assessment — seed `401101`;
- DE — community cooling-center service triage — seed `401107`;
- DF — research equipment booking incident assessment — seed `401119`;
- DG — local transit information service triage — seed `401131`.

Per domain:
- severity S0-S3;
- confidence C0-C2;
- 8 query variants per S×C;
- 96 query cases/domain;
- 384 total.

Each query case:
- one severity sequence;
- one confidence sequence;
- one batched A13 state invocation;
- exactly two encoded query sequences.

Exposure:
**NONE.**

## 5. Prototype banks

Per domain:

Severity:
- 4 classes;
- exactly 3 concrete prototypes/class;
- 12 prototypes.

Confidence:
- 3 classes;
- exactly 3 concrete prototypes/class;
- 9 prototypes.

Rules:
- natural concrete statements/scenarios;
- no canonical class labels as standalone values;
- no numeric latent IDs;
- no exact query sentence reuse;
- no exposed W18-W20 text;
- membership frozen before exposure;
- no prototype selection.

Schema-side prototype encoding/caching does not change the one-state-invocation query accounting.

## 6. Baseline — abstract categorical

Fresh W21 D0/D1/D2 definitions:
- four severity options;
- three confidence options.

Use exact W9 symmetric bidirectional token MaxSim and frozen scale.

## 7. Prototype candidate

For each class:
- score isolated query field against all 3 fixed class prototypes;
- exact W9 symmetric bidirectional token MaxSim;
- class logit = arithmetic mean of the 3 prototype logits;
- softmax over class logits.

No learned weights.
No top-k.
No prototype dropping.

MiniLM uses normalized pooled cosine query-to-prototype and the same arithmetic mean.

## 8. Stability diagnostics

Primary score always uses all 3 prototypes.

Descriptive only:
- three leave-one-prototype-out 2-of-3 subsets;
- prediction agreement across subsets.

Subset diagnostics cannot select or tune the primary candidate.

## 9. Frozen gates

Reference/domain:
- severity >=.90;
- confidence >=.90;
- joint >=.82;
- probability mass error <=1e-6.

Else:
`W21_REFERENCE_INADEQUATE`.

HIRA prototype adequacy:
- severity >=.80;
- confidence >=.85;
- joint >=.70;
- severity subset agreement >=.90;
- confidence subset agreement >=.92.

Causal gain over abstract:
- severity >= abstract +.12;
- confidence >= abstract +.08;
- joint >= abstract +.12;
- severity MAE improves >=.15.

## 10. Frozen classifications

`PROTOTYPE_GROUNDING_INTERFACE_LIMIT`:
reference adequate + HIRA adequate + causal gains pass.

`PROTOTYPE_LATENT_EXTRACTION_LIMIT`:
reference adequate + HIRA prototype adequacy fails.

`ABSTRACT_LATENT_INTERFACE_ADEQUATE`:
reference adequate + abstract HIRA severity >=.80 + confidence >=.85 + joint >=.70 + prototype joint gain <.08.

Otherwise:
`PROTOTYPE_LATENT_UNRESOLVED`.

Cross-domain:
- same eligible class >=3/4 -> `STABLE_PROTOTYPE_LATENT_LOCALIZATION`;
- incompatible eligible classes >=2 each -> `MIXED_PROTOTYPE_LATENT_LOCALIZATION`;
- otherwise `PROTOTYPE_LATENT_UNRESOLVED`.

## 11. Required metrics

Abstract:
- severity/confidence top1/MRR/margin/MAE;
- joint S+C.

Prototype:
- class top1/MRR/margin/MAE;
- joint S+C;
- per-prototype raw score anatomy;
- leave-one-out subset agreement;
- probability mass.

Integrity:
- 384 query cases;
- 96/domain;
- balanced latent cells;
- exactly 3 prototypes/class;
- query/prototype exact-sentence overlap = 0;
- one logical state compile/query;
- one A13 query invocation/query;
- two query sequences;
- zero training/selection;
- exact hashes;
- no W20/W19/W18 rows;
- no Banking77/final-test/campaign cells.

## 12. Authorization boundary

Stable prototype interface limit:
- later fresh TRAIN/DEV/dual-CONFIRM may combine prototype latent extraction with deterministic typed composition;
- retain abstract control;
- reliability/calibration remains mandatory.

Prototype extraction limit:
- no symbolic production integration;
- diagnose encoder/projection or latent-variable design.

Abstract adequate:
- do not replace interface just because prototypes are concrete.

Mixed/unresolved/reference inadequate:
- remain diagnostic.

No W21 result alone opens K32/K64.

## 13. Permanent forbidden evidence

Never tune using:
- CZ/DA/DB/DC;
- CV/CW/CX/CY;
- CR/CS/CT/CU;
- CK-CQ;
- CG-CJ;
- earlier exposed authorities;
- Banking77 0-799;
- typed final/test;
- campaign cells.

## 14. Current state

Completed:
- W20 frozen/merged;
- W20 issue closed;
- W21 issue #129 preregistered before exposure;
- W21 branch created from exact post-W20 main;
- fresh domains/seeds frozen;
- prototype mechanism/aggregator frozen;
- reference/HIRA gates frozen;
- this handoff created before exposure.

Exposure:
**NONE.**

Next:
1. implement fresh query + prototype authority;
2. implement exact-text freshness contracts;
3. implement two-field query cache + prototype schema cache;
4. implement abstract and prototype scoring;
5. implement leave-one-prototype-out diagnostics;
6. implement matched MiniLM reference;
7. implement frozen classifier;
8. add tests/pre-data workflow;
9. exact-head unit + repo CI;
10. only then enable empirical authority;
11. freeze result before merge.

A future AI must update this file after every meaningful W21 session.
