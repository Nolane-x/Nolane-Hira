# R8-W21 handoff — prototype-grounded latent semantic recoverability

Status: **CLOSED DIAGNOSTIC. Frozen outcome: `PROTOTYPE_LATENT_UNRESOLVED`; DD/DE/DF/DG are all `W21_REFERENCE_INADEQUATE`.**

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


---

## 15. Authoritative W21 closure

Exact verdict-bearing head:
`50b228efa3c7af9694ca50daa939aa7e794ff633`

Authority run:
`36218324517`

All authority jobs PASS:
- frozen unit/contracts/freshness;
- exact W9 upstream provenance;
- fresh DD/DE/DF/DG query + prototype cache;
- frozen HIRA abstract/prototype evaluation;
- pinned MiniLM matched prototype reference;
- leave-one-prototype-out diagnostics;
- frozen per-domain classifier/outcome.

Artifacts:
- frozen W9 bundle: `10898191900`;
  digest `sha256:871672d18551425e097eae3889e9e0a93906a6b6d3941cd010f73342def29a14`;
- fresh W21 cache: `10897854816`;
  digest `sha256:d0095d71ebe0777a2a343509c270d0e9c16f40d06ad99e04c513e2f8d59ca522`;
- authoritative W21 audit: `10897989700`;
  digest `sha256:ec521d615afd0a61c3c61ef02cc245a91f9a70509031527ebb92bbd4b9b4ef14`.

Integrity:
- 384 query cases;
- 96/domain;
- exactly 3 prototypes/class;
- one logical state compile/query;
- one batched A13 query invocation/query;
- exactly two isolated query sequences/query;
- query/prototype exact-sentence overlap = 0;
- trainable parameters = 0;
- training = false;
- selection = false;
- probability-mass max error <= `1.1920928955078125e-07` for HIRA;
- no W20/W19/W18 rows;
- no Banking77;
- no typed final/test;
- campaign cells = 0.

## 16. Frozen verdict

Overall:

**`PROTOTYPE_LATENT_UNRESOLVED`**

Stable classification:
`null`.

Per-domain:
- DD -> `W21_REFERENCE_INADEQUATE`;
- DE -> `W21_REFERENCE_INADEQUATE`;
- DF -> `W21_REFERENCE_INADEQUATE`;
- DG -> `W21_REFERENCE_INADEQUATE`.

Classification counts:
- `W21_REFERENCE_INADEQUATE`: 4/4.

No prototype-grounded production integration and no deterministic typed-kernel integration is authorized by W21.

## 17. Pooled HIRA result

Abstract categorical baseline:
- severity **39.063%**;
- confidence **52.083%**;
- joint S+C **20.052%**;
- severity MAE **0.7266**.

Prototype-grounded:
- severity **48.438%**;
- confidence **73.958%**;
- joint S+C **35.938%**;
- severity MAE **0.7031**;
- severity leave-one-out agreement **60.156%**;
- confidence leave-one-out agreement **63.542%**.

Descriptive gain over abstract:
- severity **+9.375 pp**;
- confidence **+21.875 pp**;
- joint **+15.885 pp**;
- severity MAE improvement only **0.0234**, far below the frozen 0.15 gate.

Thus concrete prototypes help some HIRA behavior, especially confidence/joint, but do not satisfy frozen adequacy/stability gates.

## 18. Pinned MiniLM reference result

Abstract:
- severity **51.563%**;
- confidence **69.792%**;
- joint **37.500%**.

Prototype-grounded:
- severity **53.906%**;
- confidence **71.875%**;
- joint **38.542%**;
- severity leave-one-out agreement **28.125%**;
- confidence leave-one-out agreement **34.375%**.

Frozen reference gates required:
- prototype severity >=90%;
- prototype confidence >=90%;
- prototype joint >=82%.

The reference misses all three adequacy gates by very large margins on pooled evidence, and every fresh domain is independently reference-inadequate.

## 19. Scientific interpretation

The strongest defensible conclusion is:

> Concrete prototype grounding improves HIRA over fresh abstract prose on several metrics, but the effect is not independently reference-adequate. The pinned MiniLM reference also fails badly on the same prototype authority, and prototype subset stability is weak. Therefore W21 does not establish either a prototype-grounding interface rescue or a HIRA latent-extraction limit.

Descriptively:
- HIRA confidence benefits strongly from concrete prototypes;
- HIRA severity improves modestly;
- joint extraction improves materially;
- prototype membership is not stable enough under leave-one-out perturbation;
- MiniLM's repeated inadequacy across W18-W21 now prevents the current single-reference gate from distinguishing HIRA-specific failure from authority/reference mismatch.

These observations are hypothesis-generating only.

## 20. Permanent exposure after W21

DD/DE/DF/DG are permanently exposed.

Never reuse them for:
- prototype selection;
- prototype wording redesign;
- reference model selection;
- gate tuning;
- A13/W9 tuning;
- calibration;
- production promotion.

All older forbidden evidence remains forbidden.

## 21. Authorized continuation

Because W21 is reference-inadequate/unresolved:
- do **not** promote prototype grounding;
- do **not** integrate deterministic typed composition;
- do **not** lower W21 reference gates;
- do **not** rewrite DD/DE/DF/DG prototypes and rerun;
- do **not** open K32/K64 from this result.

After four consecutive reference-inadequate latent-interface phases (W18-W21), the next diagnostic should stop changing only the HIRA-side interface.

A defensible next question is:

> Is the repeated unresolved status caused by a weak single-reference ceiling, by intrinsically ambiguous fresh latent authorities, or by HIRA itself?

A next phase should use wholly fresh domains and a preregistered independent reference panel, while keeping HIRA frozen and zero-training. It should distinguish:
1. authority/reference adequacy;
2. cross-reference agreement;
3. HIRA-vs-reference gap.

No reference model may be selected after exposure.

A future AI should read this frozen W21 closure before opening the next diagnostic.
