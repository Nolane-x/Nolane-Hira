# R8-W23 handoff — cross-encoder latent authority localization

Status: **CLOSED DIAGNOSTIC. Frozen outcome: `AUTHORITY_REFERENCE_UNRESOLVED`; DL/DM/DN/DO are all `W23_CROSS_ENCODER_REFERENCE_INADEQUATE`.**

Issue: #133

Branch:
`feat/r8-w23-cross-encoder-authority`

Base main:
`f2252b93c7b56e98d466efd9296597f3f6cfc2b0`

Read first:
1. this file;
2. `research/R8-W22-HANDOFF.md`;
3. `research/R8-W21-HANDOFF.md`;
4. issue #133.

## 0. HIRA identity

Nolane HIRA is a compact non-autoregressive typed decision engine.

Long-term target:
- compile state once;
- dynamic semantic schemas and high-cardinality candidates;
- isolated fields where contextual contamination matters;
- typed choice/score/noul;
- semantic transfer;
- explicit calibration/OOD/reliability;
- deterministic composition only after latent semantics are independently established;
- small local-friendly trainable footprint;
- sealed fresh authorities and immutable negative evidence.

W23 remains diagnostic:
- zero HIRA training;
- no production promotion;
- no typed-kernel integration;
- no K32/K64;
- fresh authority only.

## 1. Decisive history entering W23

W18:
- deterministic typed composition is exact when latent values are correct;
- blocked by reference-inadequate latent extraction.

W19:
- cumulative ordinal interface unresolved;
- 4/4 reference-inadequate.

W20:
- local pairwise boundary interface unresolved;
- 4/4 reference-inadequate.

W21:
- concrete prototypes improve HIRA descriptively;
- 4/4 reference-inadequate.

W22:
- merged main `f2252b93c7b56e98d466efd9296597f3f6cfc2b0`;
- verdict-bearing head `e27ca484c21c0fda400b44f3a25862f3a4050b0b`;
- authority run `36222318646`;
- frozen outcome `AUTHORITY_REFERENCE_UNRESOLVED`;
- 4/4 `W22_REFERENCE_PANEL_INADEQUATE`;
- panel-adequate domains 0/4;
- legacy-MiniLM-limit domains 0/4.

W22 pooled HIRA prototype:
- severity 51.56%;
- confidence 63.54%;
- joint 32.03%.

W22 pooled references:
- MiniLM 25.78 / 52.08 / 13.54%;
- MPNet 39.84 / 79.17 / 31.51%;
- E5 60.16 / 76.04 / 46.61%;
- BGE 52.34 / 83.33 / 44.53%.

Therefore W23 stops treating another bi-encoder swap as sufficient evidence.

## 2. Frozen HIRA base

A13:
- `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length 256;
- frozen.

W9 projection:
- 256->128 bias-free;
- scorer SHA256 `8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102`;
- source run `36122220588`;
- freeze artifact `10858424139`.

W23 HIRA trainable parameters:
**0**.

## 3. Fresh W23 authority

Fresh domains:
- DL — public museum reservation service incident assessment — seed `421101`;
- DM — neighborhood waste-collection information service triage — seed `421107`;
- DN — university room-booking service assessment — seed `421119`;
- DO — regional parcel-locker support triage — seed `421131`.

Per domain:
- severity S0-S3;
- confidence C0-C2;
- 8 fresh query variants per S×C;
- 96 query cases/domain;
- 384 total.

Each HIRA query:
- one severity sequence;
- one confidence sequence;
- one batched A13 query invocation;
- exactly two isolated query sequences.

Fresh prototype banks:
- severity: 4 classes × exactly 3 prototypes;
- confidence: 3 classes × exactly 3 prototypes;
- shared within domain;
- no query/prototype exact sentence overlap;
- no exact W18-W22 text reuse.

Exposure:
**DL/DM/DN/DO permanently exposed by authority run `36223939271`.**

Authoritative empirical cache, cross-encoder scores, bi-encoder controls and final W23 classification now exist and are frozen in the closure sections below.

## 4. Primary cross-encoder panel frozen before exposure

### CE0 — DeBERTa-v3 NLI

- repo: `cross-encoder/nli-deberta-v3-base`;
- revision: `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`;
- model.safetensors SHA256:
  `d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa`;
- sequence-pair NLI classifier.

### CE1 — RoBERTa NLI

- repo: `cross-encoder/nli-roberta-base`;
- revision: `1be0567456f0543475805e758725f151f283705a`;
- model.safetensors SHA256:
  `efc90996d2ed80123c26c9091c91385ffddc6d2fd0b2bacf3187fbd6c5b87953`;
- sequence-pair NLI classifier.

No model/revision/hash may change after exposure.

## 5. Frozen sentence-pair scoring

For every isolated query field and prototype:
1. score `(query, prototype)`;
2. score `(prototype, query)`;
3. find the frozen model config's entailment label;
4. softmax logits;
5. prototype score = arithmetic mean of bidirectional entailment probabilities;
6. class score = arithmetic mean of all 3 fixed prototype scores;
7. prediction = argmax class score.

No:
- learned weights;
- calibration fitting;
- top-k prototype selection;
- direction selection;
- post-exposure score transform.

For probability-mass integrity only:
- softmax the class scores.

## 6. Frozen W22 bi-encoder control

Retain exact W22 methodology as non-selecting control:
- MiniLM;
- MPNet;
- E5-small-v2;
- BGE-small-en-v1.5.

Exact revisions, hashes, pooling and prefixes remain frozen from W22.

Bi-encoder output cannot select W23:
- data;
- wording;
- cross-encoder panel;
- gates;
- HIRA parameters.

## 7. Cross-encoder individual adequacy

Per domain each CE model passes iff:
- prototype severity >= .80;
- prototype confidence >= .85;
- prototype joint >= .68;
- finite scores;
- normalized class probability mass error <= 1e-6.

## 8. Cross-encoder authority adequacy

Per domain require all:
- CE0 passes;
- CE1 passes;
- CE0/CE1 severity agreement >= .85;
- CE0/CE1 confidence agreement >= .90;
- consensus severity >= .82;
- consensus confidence >= .87;
- consensus joint >= .72.

Consensus:
- if CE0 and CE1 agree, use that class;
- if they disagree, prediction is undefined and counts incorrect;
- no HIRA/bi-encoder output may break ties.

Otherwise:
`W23_CROSS_ENCODER_REFERENCE_INADEQUATE`.

## 9. HIRA interpretation

Interpret HIRA only on cross-encoder-authority-adequate domains.

HIRA prototype adequacy:
- severity >= .75;
- confidence >= .80;
- joint >= .62;
- probability mass <=1e-6.

Authority adequate + HIRA adequate:
`HIRA_LATENT_AUTHORITY_ADEQUATE`.

Authority adequate + HIRA inadequate:
`HIRA_LATENT_GEOMETRY_LIMIT`.

## 10. Bi-encoder-family diagnostic

On a cross-encoder-authority-adequate domain:

`BIENCODER_REFERENCE_FAMILY_LIMIT = true` iff:
- fewer than 2/4 W22-style bi-encoders pass the frozen .80/.85/.68 individual gate.

This flag is orthogonal to HIRA classification.

## 11. Cross-domain outcome

- same HIRA classification >=3/4 ->
  `STABLE_CROSS_ENCODER_LOCALIZATION`;
- no stable HIRA class, but cross-encoder authority adequate >=3/4 and bi-encoder-family limit >=3/4 ->
  `STABLE_BIENCODER_REFERENCE_FAMILY_LIMIT`;
- cross-encoder authority inadequate >=3/4 ->
  `AUTHORITY_REFERENCE_UNRESOLVED`;
- otherwise ->
  `CROSS_ENCODER_REFERENCE_MIXED`.

## 12. Required metrics

HIRA:
- abstract severity/confidence/joint;
- prototype severity/confidence/joint;
- MRR/margin/MAE;
- probability mass.

Each cross-encoder:
- prototype severity/confidence/joint;
- MRR/margin/MAE;
- bidirectional entailment score anatomy;
- probability mass.

Cross-encoder panel:
- CE0/CE1 severity agreement;
- CE0/CE1 confidence agreement;
- consensus severity/confidence/joint;
- undefined/disagreement counts.

Bi-encoder control:
- exact W22 per-reference prototype metrics;
- per-domain individual pass counts.

Integrity:
- exact model revisions/hashes;
- 384 cases;
- 96/domain;
- 3 prototypes/class;
- zero exact query/prototype overlap;
- zero exact prior-authority overlap;
- one HIRA state compile/query;
- one HIRA A13 query invocation/query;
- two HIRA sequences/query;
- zero training/selection;
- no W22-W18 rows;
- no Banking77;
- no typed final/test;
- campaign cells = 0.

## 13. Authorization boundary

Stable `HIRA_LATENT_GEOMETRY_LIMIT`:
- no production integration;
- future work may diagnose/train latent semantic geometry only on new TRAIN/DEV authority.

Stable `HIRA_LATENT_AUTHORITY_ADEQUATE`:
- later fresh phase may test latent extraction + deterministic typed composition;
- retain controls;
- require TRAIN/DEV/dual-CONFIRM;
- calibration/OOD remains mandatory.

Stable bi-encoder-family limit:
- future diagnostic reference methodology may use preregistered cross-encoders;
- no retroactive relabeling of W18-W22.

Reference inadequate/mixed:
- remain diagnostic;
- no HIRA tuning.

## 14. Permanent forbidden evidence

Never tune with:
- DH/DI/DJ/DK;
- DD/DE/DF/DG;
- CZ/DA/DB/DC;
- CV/CW/CX/CY;
- CR/CS/CT/CU;
- older exposed authorities;
- Banking77 0-799;
- typed final/test;
- campaign cells.

## 15. Current state

Completed:
- W22 frozen and merged;
- W23 issue #133 preregistered before exposure;
- W23 implementation completed;
- exact-head unit + repo CI passed before exposure;
- fresh DL/DM/DN/DO cache materialized;
- CE0/CE1 cross-encoder evaluation completed;
- W22 bi-encoder control evaluation completed;
- authoritative audit completed and verified;
- final W23 verdict frozen below.

Exposure:
**DL/DM/DN/DO permanently exposed.**

Next:
1. merge this frozen W23 closure;
2. close issue #133;
3. open W24 only from post-W23 `main`;
4. preserve every W23 exposed row as forbidden evidence.

A future AI must update this file after every meaningful W23 session.


---

## 16. Authoritative W23 closure

Exact verdict-bearing head:
`d7004ac62df3133d8bdafff18f35e91247738844`

Authority run:
`36223939271`

Authority completed successfully:
- frozen unit/contracts/freshness PASS;
- exact W9 provenance PASS;
- fresh DL/DM/DN/DO cache PASS;
- two pinned NLI cross-encoders evaluated;
- frozen W22 four-bi-encoder control evaluated;
- frozen consensus/classifier PASS.

Artifacts:
- fresh W23 cache: `10900560385`;
  digest `sha256:f0e7d8dbceaf8c0e805080e2b97b1c06fffff44fb672419428cb26f6d709a444`;
- authoritative W23 audit: `10900272920`;
  digest `sha256:c6013dcbb3e3e0784d8c04e3258608d02caec7b8686f3a976a7ffd6cca40074f`;
- frozen W9 inputs: `10899903099`;
  digest `sha256:737ab1957e34bf3ec6b6f9ab4f0be7cf6006f8d16f4a352ede75dee2828f2a57`.

Integrity:
- 384 fresh cases;
- 96/domain;
- exactly 3 prototypes/class;
- one HIRA state compile/query;
- one batched HIRA A13 query invocation/query;
- exactly two isolated HIRA sequences/query;
- cross-encoder revisions/hashes frozen before exposure;
- W22 bi-encoder control frozen before exposure;
- trainable parameters = 0;
- training = false;
- selection = false;
- no W22-W18 rows;
- no Banking77;
- no typed final/test;
- campaign cells = 0.

## 17. Frozen verdict

Overall:

**`AUTHORITY_REFERENCE_UNRESOLVED`**

Stable HIRA classification:
`null`.

Cross-encoder-authority-adequate domains:
**0 / 4**.

Bi-encoder-family-limit domains:
**0 / 4** because that diagnostic is only authorized on cross-encoder-authority-adequate domains.

Per-domain:
- DL -> `W23_CROSS_ENCODER_REFERENCE_INADEQUATE`;
- DM -> `W23_CROSS_ENCODER_REFERENCE_INADEQUATE`;
- DN -> `W23_CROSS_ENCODER_REFERENCE_INADEQUATE`;
- DO -> `W23_CROSS_ENCODER_REFERENCE_INADEQUATE`.

No HIRA latent-geometry diagnosis is authorized.

## 18. Pooled HIRA result

Fresh abstract categorical control:
- severity **38.281%**;
- confidence **50.000%**;
- joint severity+confidence **19.271%**.

Prototype-grounded:
- severity **40.625%**;
- confidence **75.000%**;
- joint severity+confidence **30.990%**.

Relative to abstract control:
- severity **+2.344 pp**;
- confidence **+25.000 pp**;
- joint **+11.719 pp**.

Confidence again benefits substantially from prototype grounding.
Severity remains weak.

## 19. Cross-encoder result

### DeBERTa-v3 NLI

Pooled prototype:
- severity **64.063%**;
- confidence **100.000%**;
- joint **64.063%**.

Per-domain severity:
- DL **59.375%**;
- DM **62.500%**;
- DN **71.875%**;
- DO **62.500%**.

Per-domain confidence:
- DL/DM/DN/DO **100.000%**.

### RoBERTa NLI

Pooled prototype:
- severity **58.594%**;
- confidence **95.833%**;
- joint **55.208%**.

Per-domain severity:
- DL **56.250%**;
- DM **59.375%**;
- DN **62.500%**;
- DO **56.250%**.

Per-domain confidence:
- DL/DM/DN/DO **95.833%**.

Neither cross-encoder passes the frozen individual gate on any domain because severity remains below 80%.

## 20. Cross-encoder consensus

DL:
- severity agreement **65.625%**;
- confidence agreement **95.833%**;
- consensus severity **43.750%**;
- consensus confidence **95.833%**;
- consensus joint **41.667%**.

DM:
- severity agreement **68.750%**;
- confidence agreement **95.833%**;
- consensus severity **50.000%**;
- consensus confidence **95.833%**;
- consensus joint **47.917%**.

DN:
- severity agreement **75.000%**;
- confidence agreement **95.833%**;
- consensus severity **59.375%**;
- consensus confidence **95.833%**;
- consensus joint **56.250%**.

DO:
- severity agreement **65.625%**;
- confidence agreement **95.833%**;
- consensus severity **46.875%**;
- consensus confidence **95.833%**;
- consensus joint **43.750%**.

The cross-encoder authority therefore fails all four fresh domains.

## 21. Scientific interpretation

The strongest defensible conclusion is:

> W23 shows that the repeated authority failure is not specific to bi-encoder similarity. Two independently pinned NLI cross-encoders recover confidence nearly perfectly, yet both fail the preregistered severity criterion and disagree materially on severity. The unresolved variable is therefore now sharply localized to the current severity authority/interface, not to confidence and not merely to the legacy reference family.

This does **not** prove:
- HIRA severity geometry is intrinsically inadequate;
- the severity classes are human-ambiguous;
- any particular new wording would fix the authority;
- deterministic typed composition should enter production.

What W23 does establish:
- confidence semantic authority is substantially recoverable by stronger sentence-pair models;
- severity remains the dominant source of authority failure;
- replacing bi-encoders with NLI cross-encoders alone is insufficient;
- another blind reference-model swap is not justified.

## 22. Permanent exposure after W23

DL/DM/DN/DO are permanently exposed.

Never use them for:
- severity wording redesign;
- class-boundary redesign;
- prototype selection;
- reference-model selection;
- gate tuning;
- HIRA/A13/W9 tuning;
- calibration;
- production promotion.

All prior exposed authorities remain forbidden.

## 23. Authorized continuation

The next phase must isolate **severity authority recoverability** from HIRA geometry.

A defensible W24 should:
1. use wholly fresh domains;
2. keep confidence as a positive-control field, not the primary research target;
3. replace coarse four-way severity authority with independently preregistered **atomic severity factors** that compose deterministically into S0-S3;
4. require the independent reference to recover those atomic factors before interpreting HIRA;
5. preserve direct four-way severity classification as a matched control;
6. keep HIRA fully frozen and zero-training;
7. use at least one strong sentence-pair/cross-encoder reference plus the frozen W23 reference methodology as control;
8. forbid any post-exposure factor definition, composition rule, model, wording or gate change;
9. interpret HIRA only if factor authority is independently adequate;
10. still forbid production typed-kernel integration and K32/K64 until stable fresh localization exists.

The central W24 question should be:

> Is severity unrecoverable because the four-way class is semantically entangled, while a small set of atomic severity factors is independently recoverable and deterministically composable?

A future AI must read this frozen W23 closure before opening W24.
