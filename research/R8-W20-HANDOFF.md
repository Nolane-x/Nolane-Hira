# R8-W20 handoff — local pairwise latent-boundary recoverability

Status: **CLOSED DIAGNOSTIC. Frozen outcome: `LATENT_BOUNDARY_UNRESOLVED`; CZ/DA/DB/DC are all `W20_REFERENCE_INADEQUATE`.**

Issue: #127

Branch:
`feat/r8-w20-pairwise-latent-boundary`

Base main:
`95d0a9f72345dda566fd12c1b229fe3145101df5`

Read first when restoring:
1. this file;
2. `research/R8-W19-HANDOFF.md`;
3. `research/R8-W18-HANDOFF.md`;
4. `research/R8-W17-HANDOFF.md`;
5. issue #127.

## 0. HIRA identity

Nolane HIRA is a compact non-autoregressive typed decision engine.

Long-term target:
- compile/encode state once;
- preserve isolated semantic fields where contextual contamination matters;
- support dynamic semantic schemas and high-cardinality candidates;
- typed choice / score / noul primitives;
- semantic transfer plus explicit calibration/OOD authorities;
- deterministic composition only when its prerequisites are empirically established;
- small local-friendly trainable footprint;
- immutable negative results and sealed fresh authorities.

Scientific discipline:
- freeze data, wording, equations and gates before exposure;
- exposed rows are permanently forbidden;
- reference models are diagnostics only;
- PARTIAL/FAIL never becomes success post hoc;
- no production promotion from descriptive gains.

## 1. Frozen semantic base

A13:
- `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length 256;
- fully frozen.

W9 semantic projection:
- 256 -> 128 bias-free;
- scorer SHA `8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102`;
- source run `36122220588`;
- freeze artifact `10858424139`.

W20 trainable parameter count:
**0**.

## 2. Decisive history entering W20

W16:
- stable `CONTEXTUAL_STATE_CONTAMINATION`;
- flattened typed context damages intent token geometry before scoring.

W17:
- field isolation causally repairs much direct intent semantics;
- whole typed path remains weak.

W18:
- deterministic typed composition is internally correct when latent fields are correct;
- HARD composition 69.11%, SOFT 69.38% vs FIELD_DIRECT 45.31%;
- reference inadequacy prevents production promotion.

W19:
- merged main `95d0a9f72345dda566fd12c1b229fe3145101df5`;
- authority run `36214486773`;
- frozen outcome `LATENT_AXIS_UNRESOLVED`;
- CV/CW/CX/CY all `W19_REFERENCE_INADEQUATE`.

W19 pooled HIRA:
- flat severity 50.00%;
- flat confidence 48.61%;
- flat joint 24.31%;
- cumulative ordinal severity 23.96%;
- ordinal confidence 40.28%;
- ordinal joint 9.72%.

W19 MiniLM:
- ordinal severity 31.25%;
- ordinal confidence 50.00%;
- severity monotonicity 76.04%;
- confidence monotonicity 91.67%.

Key descriptive pattern:
- lower local boundaries are materially easier than upper ones;
- cumulative TRUE-count decoding is not independently adequate.

Therefore W20 must remain diagnostic.

## 3. W20 scientific question

Can ordered latent values be recovered through **direct locally contrastive adjacent-class comparisons**, with each boundary independently reference-adequate, before any global ordinal reconstruction?

W20 explicitly separates:
1. local boundary identifiability;
2. secondary global reconstruction.

The primary authority is local-boundary accuracy, not cumulative ordinal counting.

## 4. Fresh W20 authority

Wholly fresh domains:
- CZ — public-library service interruption triage — seed `391101`;
- DA — community energy service incident assessment — seed `391107`;
- DB — campus laboratory support disruption triage — seed `391119`;
- DC — municipal water-service issue assessment — seed `391131`.

Per domain:
- severity S0/S1/S2/S3;
- confidence C0/C1/C2;
- 8 fresh wording variants per S×C cell;
- 96 cases/domain.

Total:
- 384 cases.

Each logical case contains exactly two isolated semantic sequences:
1. severity field;
2. confidence field.

Representation accounting:
- one logical state compile/case;
- one batched A13 invocation/case;
- exactly two encoded sequences/case.

No intent candidate set is part of the primary W20 authority.

Exposure:
**NONE.**

No CZ/DA/DB/DC A13 cache exists.
No W20 MiniLM score exists.
No W20 classification exists.

## 5. Freshness rules

No exact W18/W19:
- field sentence;
- flat class definition;
- threshold definition;
- schema question wording

may be reused.

Field text must:
- express latent meaning naturally;
- avoid canonical class labels as structured values;
- avoid numeric level identifiers;
- avoid direct copying of pair-option definitions.

After first eligible cache, CZ/DA/DB/DC are permanently exposed.

## 6. Baseline A — matched flat categorical extraction

Severity:
- four fresh natural D0/D1/D2 definitions.

Confidence:
- three fresh natural D0/D1/D2 definitions.

Operator:
- exact frozen W9 projection;
- L2 normalization;
- symmetric bidirectional token MaxSim;
- arithmetic directional mean;
- arithmetic mean over D0/D1/D2;
- frozen scorer scale;
- softmax across the full class set.

## 7. Candidate B — local adjacent pairwise boundaries

Severity:
- `B_S01`: S0 vs S1;
- `B_S12`: S1 vs S2;
- `B_S23`: S2 vs S3.

Confidence:
- `B_C01`: C0 vs C1;
- `B_C12`: C1 vs C2.

For every boundary:
- two locally contrastive natural semantic alternatives;
- D0/D1/D2 views for each side;
- exact W9 semantic operator;
- no TRUE/FALSE threshold language;
- no learned threshold;
- no calibration fitting;
- no cumulative TRUE-count decode.

Primary boundary evaluation uses **only rows whose gold class is one of the two adjacent classes**.

## 8. Pairwise symmetry control

Every local boundary must be scored in both option orders.

Required integrity:
- swapping option order and remapping labels must preserve predicted class identity >= .99;
- mapped score/probability differences must remain numerically equivalent within tolerance;
- any failed symmetry control invalidates the affected boundary authority.

This protects the experiment from accidental candidate-position effects.

## 9. Secondary global reconstruction

Global reconstruction is descriptive only and can never rescue failed primary-boundary adequacy.

Severity pair deltas:
- `d01 = score(S1)-score(S0)`;
- `d12 = score(S2)-score(S1)`;
- `d23 = score(S3)-score(S2)`.

Utilities:
- `u0 = 0`;
- `u1 = d01`;
- `u2 = d01 + d12`;
- `u3 = d01 + d12 + d23`.

Confidence:
- `u0 = 0`;
- `u1 = d01`;
- `u2 = d01 + d12`.

Softmax over utilities gives descriptive class probabilities.

Report:
- global top1;
- MAE;
- joint S+C;
- probability mass;
- flat->pairwise wrong-to-right;
- flat->pairwise right-to-wrong.

## 10. Pinned reference

MiniLM:
- `sentence-transformers/all-MiniLM-L6-v2`;
- revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- weight SHA `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.

Reference evaluates exactly the same fresh:
- flat interface;
- adjacent pairwise interfaces;
- option-order swap controls.

Reference never enters HIRA inference or selection.

## 11. Frozen reference adequacy

Per domain, every boundary must pass:

Severity:
- B_S01 >= .90;
- B_S12 >= .90;
- B_S23 >= .90.

Confidence:
- B_C01 >= .90;
- B_C12 >= .90.

Also:
- option-order swap identity >= .99 each;
- finite scores;
- probability-mass error <= 1e-6.

Otherwise:
`W20_REFERENCE_INADEQUATE`.

## 12. Frozen HIRA pairwise adequacy

Per domain:
- mean severity boundary accuracy >= .82;
- every severity boundary >= .75;
- mean confidence boundary accuracy >= .85;
- every confidence boundary >= .80;
- option-order swap identity >= .99 for every boundary.

## 13. Frozen causal gain vs matched flat local slices

For each adjacent pair, evaluate the full flat baseline on the same local two-class gold slice.

Require:
- mean severity pairwise gain >= +.08;
- no severity boundary degrades by > .03;
- mean confidence pairwise gain >= +.05;
- no confidence boundary degrades by > .03.

## 14. Frozen domain classifications

### `LOCAL_PAIRWISE_INTERFACE_LIMIT`

Require:
- reference adequacy;
- HIRA pairwise adequacy;
- all causal-gain gates pass.

Interpretation:
recoverable local boundary semantics exist but the global flat interface hides them.

### `LATENT_BOUNDARY_EXTRACTION_LIMIT`

Require:
- reference adequacy;
- HIRA pairwise adequacy fails.

Interpretation:
even locally contrastive adjacent semantics remain insufficiently recoverable under frozen A13/W9 geometry.

### `FLAT_LOCAL_INTERFACE_ADEQUATE`

Require:
- reference adequacy;
- flat-local mean severity >= .82;
- flat-local mean confidence >= .85;
- pairwise mean gain < .05.

Interpretation:
pairwise framing is not the key missing interface.

Otherwise:
`LATENT_BOUNDARY_UNRESOLVED`.

Cross-domain:
- same non-unresolved/non-reference-inadequate class >=3/4 ->
  `STABLE_LATENT_BOUNDARY_LOCALIZATION`;
- two incompatible non-unresolved classes >=2 each ->
  `MIXED_LATENT_BOUNDARY_LOCALIZATION`;
- else ->
  `LATENT_BOUNDARY_UNRESOLVED`.

## 15. Required metrics

Flat:
- severity/confidence top1/MRR/margin;
- flat-local adjacent pair accuracy;
- joint S+C.

Pairwise:
- each boundary top1/MRR/margin/balanced accuracy;
- swap-order identity and numeric equivalence;
- mean severity boundary accuracy;
- mean confidence boundary accuracy.

Secondary reconstruction:
- severity/confidence global top1;
- MAE;
- joint S+C;
- probability mass;
- transition counts vs flat.

Integrity:
- exact model/scorer/reference hashes;
- one state compile/case;
- one A13 invocation/case;
- two isolated sequences/case;
- trainable params = 0;
- training = false;
- selection = false;
- no W19/W18 rows;
- no Banking77;
- no typed final/test;
- campaign cells = 0.

## 16. Authorization boundary

Stable `LOCAL_PAIRWISE_INTERFACE_LIMIT`:
- a later fresh phase may test pairwise latent extraction with deterministic typed composition;
- retain flat control;
- production remains gated by fresh TRAIN/DEV/dual-CONFIRM and reliability;
- still do not open K32/K64 automatically.

Stable `LATENT_BOUNDARY_EXTRACTION_LIMIT`:
- no symbolic production integration;
- diagnose encoder/projection semantic geometry.

Stable `FLAT_LOCAL_INTERFACE_ADEQUATE`:
- do not replace the flat interface merely because pairwise decomposition is cleaner.

Mixed/unresolved/reference inadequate:
- remain diagnostic;
- no rescue training.

## 17. Permanent forbidden evidence

Never use for W20 tuning:
- W19 CV/CW/CX/CY;
- W18 CR/CS/CT/CU;
- W17 CK/CL/CM/CN/CO/CP/CQ;
- W16 CG/CH/CI/CJ;
- all older exposed authorities;
- Banking77 0-799;
- typed final/test;
- public campaign cells.

## 18. Pre-exposure execution state (historical)

Completed:
- W19 closure merged to main;
- W19 issue closed;
- W20 issue #127 preregistered before exposure;
- branch created from exact post-W19 main;
- fresh domains/seeds frozen;
- local pairwise mechanism frozen;
- reference/HIRA gates frozen;
- this handoff created before exposure.

Exposure:
**NONE.**

Next:
1. implement fresh balanced W20 authority generator;
2. implement freshness authority;
3. implement isolated two-field cache;
4. implement flat + adjacent-pair schemas;
5. implement pairwise swap-order control;
6. implement HIRA evaluator;
7. implement pinned MiniLM matched reference;
8. implement frozen classifier/outcome;
9. add contracts + pre-data workflow;
10. run exact-head unit + repo CI;
11. only then enable empirical authority;
12. freeze results here before merge.

A future AI must update this file after every meaningful W20 session.


---

## 19. Authoritative W20 closure

Exact verdict-bearing head:
`85d9496c31bbdbe7a93268a50c21c54bf06c9001`

Authority run:
`36217155264`

All authority jobs PASS:
- frozen unit/contracts/freshness;
- exact W9 upstream provenance;
- fresh CZ/DA/DB/DC two-field cache;
- frozen HIRA flat/pairwise evaluation;
- pinned MiniLM matched pairwise reference;
- option-order swap controls;
- frozen per-domain classifier/outcome.

Artifacts:
- frozen W9 bundle: `10897728271`;
  digest `sha256:1518bc9a5e8fdf59a61e262865187da5dba18e0cf1f32633b048e016636b83c7`;
- fresh W20 cache: `10897043527`;
  digest `sha256:676c92fd104626670231f3f27d030926fd71d7c5952d7197c180d5f5cc3bea26`;
- authoritative W20 audit: `10897398301`;
  digest `sha256:23f7907f63dd127f6b03c827c1a7d37f2c50dd3f6a56b249477293f8fc35d245`.

Integrity:
- 384 cases;
- 96/domain;
- severity balance 24/class/domain;
- confidence balance 32/class/domain;
- one logical state compile/case;
- one batched A13 invocation/case;
- exactly two isolated sequences/case;
- trainable parameters = 0;
- training = false;
- selection = false;
- pairwise probability-mass max error <= `1.1920928955078125e-07`;
- HIRA option-order swap identity = 1.0 on every pooled boundary;
- reference option-order swap identity = 1.0 on every pooled boundary;
- no W19 rows;
- no W18 rows;
- no Banking77;
- no typed final/test;
- campaign cells = 0.

## 20. Frozen verdict

Overall:

**`LATENT_BOUNDARY_UNRESOLVED`**

Stable classification:
`null`.

Per-domain:
- CZ -> `W20_REFERENCE_INADEQUATE`;
- DA -> `W20_REFERENCE_INADEQUATE`;
- DB -> `W20_REFERENCE_INADEQUATE`;
- DC -> `W20_REFERENCE_INADEQUATE`.

Classification counts:
- `W20_REFERENCE_INADEQUATE`: 4/4.

No production latent-interface or deterministic typed-kernel integration is authorized by W20.

## 21. Pooled HIRA result

Matched flat local slices:

Severity:
- S0/S1 **59.375%**;
- S1/S2 **81.250%**;
- S2/S3 **75.000%**;
- mean **71.875%**.

Confidence:
- C0/C1 **64.063%**;
- C1/C2 **92.188%**;
- mean **78.125%**.

Local pairwise interface:

Severity:
- S0/S1 **71.875%**;
- S1/S2 **62.500%**;
- S2/S3 **50.000%**;
- mean **61.458%**.

Confidence:
- C0/C1 **81.250%**;
- C1/C2 **68.750%**;
- mean **75.000%**.

Thus pairwise-vs-flat-local mean change is:
- severity **-10.417 pp**;
- confidence **-3.125 pp**.

Boundary-specific changes are heterogeneous:
- S0/S1: **+12.500 pp**;
- S1/S2: **-18.750 pp**;
- S2/S3: **-25.000 pp**;
- C0/C1: **+17.188 pp**;
- C1/C2: **-23.438 pp**.

Secondary global pairwise reconstruction:
- severity top1 **28.125%**;
- confidence top1 **69.792%**;
- joint severity+confidence **19.531%**.

The global reconstruction is descriptive only and cannot rescue failed local/reference gates.

## 22. Pinned MiniLM reference result

Local pairwise boundaries:

Severity:
- S0/S1 **57.813%**;
- S1/S2 **65.625%**;
- S2/S3 **57.813%**;
- mean **60.417%**.

Confidence:
- C0/C1 **81.250%**;
- C1/C2 **59.375%**;
- mean **70.313%**.

Matched reference flat-local means:
- severity **65.104%**;
- confidence **71.875%**.

Frozen reference gate required **every individual local boundary >=90%** plus swap identity >=.99.

Swap integrity passes perfectly, but semantic boundary adequacy fails by a large margin on every severity boundary and on both confidence boundaries.

Therefore W20 cannot causally classify the HIRA pairwise interface.

## 23. Scientific interpretation

The strongest defensible conclusion is:

> Direct adjacent-class comparison does not solve the latent-boundary problem under the frozen W20 semantics. The failure is not HIRA-specific: the pinned independent MiniLM reference also fails the same local-boundary adequacy authority. Perfect option-order invariance rules out candidate ordering as the explanation. The experiment therefore does not establish either a HIRA latent-boundary extraction limit or a local-pairwise interface rescue.

Descriptively:
- local pairwise framing helps the lowest severity/certainty boundary in HIRA;
- the same framing materially damages the upper HIRA boundaries;
- abstract upper-boundary semantics remain difficult for the reference as well;
- cumulative ordinal decoding was not the sole cause of W19 failure;
- candidate-position artifacts are not responsible;
- direct abstract class/boundary prose is now the repeated common weakness across W18-W20.

These observations are hypothesis-generating only.

## 24. Permanent exposure after W20

CZ/DA/DB/DC are permanently exposed.

Never reuse them for:
- pair wording selection;
- exemplar/prototype selection;
- gate tuning;
- A13/W9 tuning;
- reference selection;
- calibration;
- production promotion.

All older forbidden evidence remains forbidden.

## 25. Authorized continuation

Because W20 is reference-inadequate/unresolved:
- do **not** integrate pairwise latent extraction into production;
- do **not** integrate the deterministic typed kernel on the basis of W20;
- do **not** lower the frozen reference gates;
- do **not** rewrite CZ/DA/DB/DC pair definitions and rerun;
- do **not** open K32/K64 from this result.

A next phase should stop asking whether another abstract wording/decode rule alone fixes the same authority.

A narrower defensible next question is:

> Can latent classes/boundaries become independently recoverable when semantic meaning is grounded by multiple concrete, lexically disjoint prototype exemplars rather than abstract class prose?

Any such continuation must:
1. use wholly fresh domains;
2. preregister query/prototype lexical families before exposure;
3. retain zero training in the primary diagnostic;
4. separate prototype retrieval adequacy from downstream typed composition;
5. retain a pinned independent reference adequacy gate;
6. keep W20 CZ/DA/DB/DC permanently forbidden;
7. forbid prototype/query exact-text overlap and label-token leakage;
8. keep deterministic composition diagnostic until latent extraction is independently established.

A future AI should read this frozen W20 closure before opening the next diagnostic.
