# R8-W11 handoff — production semantic-interface decomposition audit

Status: **CLOSED DIAGNOSTIC. Frozen outcome: `INTERFACE_DECOMPOSITION_UNRESOLVED`. No rescue mechanism is authorized.**

Issue: #109

Branch:
`feat/r8-w11-interface-decomposition`

Base main:
`1441c7265fff4a960251ccb1b745bb2c684a41be`

Read first when restoring:
1. this file;
2. `research/R8-W10-HANDOFF.md`;
3. `research/R8-W9-HANDOFF.md`;
4. `research/R8-W8-HANDOFF.md`;
5. issue #109.

## 0. Project identity

Nolane HIRA is a compact non-autoregressive typed decision engine.

Long-term goals:
- state encoded once/case;
- dynamic semantic schemas;
- changing/high-cardinality candidate sets;
- typed choice / score / noul primitives;
- small local-friendly parameter footprint;
- transferable semantic binding;
- robust structured decisions;
- explicit reliability integrity;
- falsifiable mechanisms, sealed authorities and immutable negative results.

## 1. Frozen semantic/runtime base

A13:
- `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length 256;
- frozen.

W9 semantic projection:
- scorer SHA `8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102`;
- selected W9 DEV epoch 5;
- source run `36122220588`;
- source freeze artifact `10858424139`.

Frozen HIRACore:
- SHA `d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588`.

Production competitive constants:
- candidate-relative IDF from current library;
- sibling common-mode subtraction from current library;
- salience threshold 0.5;
- salient minimum weight 0.5;
- pooled relation mode;
- forced full-K;
- adaptive budget false.

No W11 parameter is trainable.

## 2. Decisive history

W5:
- pooled matching/simple token probes/encoder scaling failed;
- late interaction + candidate-relative salience produced the strongest tiny semantic binder.

W6/W6b:
- production typed integration;
- strong internal competence but incomplete generalization/reliability.

W6c:
- calibration fixes confidence, not hard semantic competence.

W6d/e:
- strong overall typed scores do not guarantee reproducible held-out K64 generalization.

W6f:
- uniform/removing candidate-relative salience was much worse;
- relation reranking was net helpful in that authority;
- high-K pooled failure unresolved.

W6g:
- field-semantic weakness localized.

W6h:
- small residual semantic adapter failed replication;
- full projection retune stronger.

W6i:
- simple representation bridges/formats unresolved.

W6j:
- stable synthetic high-K target `COARSE_CONJUNCTION_LIMIT`.

W7:
- explicit factor smooth-AND branch failed;
- equal-data free-form retune much stronger.

W7b:
- synthetic free-form retune gains were strong but failed one preregistered replication gate;
- no stable typed-vs-pair attribution.

W7c:
- frozen Banking77 transfer rows 400–799:
  frozen/typed-only 1/400;
  pair-only/primary/replica 0/400;
  retuned paths were highly confident while wrong.
- verdict `PUBLIC_HIGH_K_TRANSFER_ABSENT`.

W8:
- fresh semantic transfer decomposition;
- stable `GENERAL_SEMANTIC_TRANSFER_LIMIT`;
- failure already exists at K4 before high cardinality dominates.

W9:
- direct semantic alignment objective;
- full projection retuning stronger than shared/asymmetric rank-16 bridges;
- verdict `SEMANTIC_ALIGNMENT_FAIL`.
- projection control improved low-K/alignment but did not rescue.

W10:
- exact authority head `c5e8846851d4f520771d402be132f03a25622807`;
- run `36126611084`;
- frozen outcome `REPRESENTATION_CEILING_UNRESOLVED`.

W10 pooled natural-definition top1:
- A0 raw A13 mean: K4 53.13%, K16 23.44%;
- A1 raw A13 symmetric MaxSim: 64.06%, 34.77%;
- P0 W6e projected symmetric: 72.27%, 48.44%;
- P1 W9 semantic projected symmetric: **79.30%, 54.30%**;
- S0 W6e production final: 46.88%, 22.27%;
- S1 W9 projection through production: **60.55%, 32.03%**;
- frozen MiniLM reference: **97.66%, 90.63%**.

W10 per-domain:
- BF `SCORING_INTERFACE_LIMIT`;
- BG `SCORING_INTERFACE_LIMIT`;
- BH unresolved;
- BI unresolved.

Only 2/4 crossed the frozen scoring-interface gates; stable target required >=3/4, so W10 remained unresolved.

W10 merged main:
`1441c7265fff4a960251ccb1b745bb2c684a41be`.

## 3. Why W11 exists

W10 substantially weakens a simple “A13 itself is the whole bottleneck” story:
- frozen reference shows the tasks have strong recoverable semantic structure;
- projection P0 often improves raw A13;
- W9 semantic projection P1 is much stronger again.

But P1 semantic geometry loses ~19–23 pp when passed through production scoring.

W10 could not say which production transformation causes that loss.

W11 therefore decomposes the exact cumulative path:

semantic pair geometry
-> directionality
-> question context
-> candidate-relative weighting
-> sibling competition
-> salient minimum
-> exact production coarse
-> HIRACore final.

This is diagnostic, not a scorer search.

## 4. Fresh W11 domains

- BJ university housing administration — seed 301801;
- BK agricultural equipment leasing — seed 301807;
- BL professional certification administration — seed 301819;
- BM airline baggage support — seed 301831.

Each:
- 16 latent intents;
- 4 independently written state utterances/intent;
- one natural definition;
- one terse label for descriptive use;
- 64 bases.

Total 256 bases.

Nested K:
4 / 8 / 16.

All operators use exactly the same:
- state;
- gold;
- candidate IDs;
- option ordering;
- K membership.

Natural definitions are the only primary classification view.

No prior exact text atom may appear.

## 5. Frozen reference ceiling

R0:
- `sentence-transformers/all-MiniLM-L6-v2`;
- revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- expected frozen `model.safetensors` SHA
  `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.

Diagnostic task-adequacy ceiling only.

Never enters HIRA or model selection.

## 6. Exact Q0-Q7 cumulative stage ladder

All Q stages use exact W9 semantic projection weights.

### Q0 symmetric state↔definition MaxSim

- state content tokens only;
- option content tokens only;
- 128-d projected + L2 normalized;
- option->state MaxSim mean;
- state->option MaxSim mean;
- mean of both directional means.

This equals the conceptual W10 P1 starting point.

### Q1 production directionality

- option->state MaxSim mean only;
- no question tokens;
- no IDF;
- no common-mode subtraction;
- no salient-min term.

### Q2 question context

- context = state content + question content;
- option->context MaxSim mean;
- still unweighted;
- no common-mode;
- no salient-min.

### Q3 candidate-relative IDF

- Q2 similarities;
- exact production candidate-relative IDF over option tokens;
- IDF-weighted mean coverage;
- no common-mode;
- no salient-min.

### Q4 sibling common-mode subtraction

- exact salience-weighted sibling-token common mode per option/context token;
- subtract before context MaxSim;
- exact IDF-weighted mean coverage;
- no salient-min.

### Q5 salient-min aggregation

Q4 plus:
- salience threshold 0.5;
- exact empty-salient fallback;
- `weighted_mean + 0.5 * min_coverage`.

This must be the manually decomposed production raw score.

### Q6 actual production CompetitiveCoarseScorer

- exact library scorer;
- exact W9 semantic projection/log scale;
- same cached tokens.

Q5 and Q6 must be rank-equivalent.
Q5->Q6 is implementation validation, not a scientific stage.

### Q7 frozen HIRACore final

- Q6 exact coarse override;
- frozen HIRACore;
- pooled relation;
- forced full-K;
- adaptive budget false.

## 7. Mandatory metrics

Per domain/stage/K:
- top1;
- top5 where K>=8;
- MRR;
- mean gold margin.

Every adjacent transition:
- correct->wrong count/rate;
- wrong->correct count/rate;
- mean rank delta;
- mean margin delta.

Also:
- first-loss stage for every Q0-correct/Q7-wrong base;
- first-rescue stage for every Q0-wrong/Q7-correct base;
- first-loss histogram;
- Q5/Q6 top1 identity rate;
- Q5/Q6 complete rank-order identity rate;
- Q7 probability mass max error;
- state encodes/base.

Reference:
- R0 K4/K8/K16 top1/MRR/margin.

## 8. Adequacy precondition

A domain is stage-classifiable only if:
- R0 K4 >=.75;
- Q0 K4 >=.70;
- Q0 K16 >=.45.

Otherwise:
`SEMANTIC_BASELINE_INADEQUATE`.

This prevents attributing damage downstream when the starting semantic geometry is already weak.

## 9. Frozen adjacent-stage rule

For an adequate domain an adjacent transformation is damaging iff:
- K4 output <= input - .10;
- K16 output <= input - .08;
- correct->wrong rate >=.10;
- wrong->correct rate <= correct->wrong rate - .04.

The earliest damaging stage determines the simple stage class:
- Q0->Q1 `DIRECTIONALITY_LOSS`;
- Q1->Q2 `QUESTION_CONTEXT_LOSS`;
- Q2->Q3 `IDF_WEIGHTING_LOSS`;
- Q3->Q4 `COMMON_MODE_SUBTRACTION_LOSS`;
- Q4->Q5 `SALIENT_MIN_COVERAGE_LOSS`;
- Q6->Q7 `RELATION_RERANKING_LOSS`.

Q5->Q6 must be numerically/rank equivalent.

If multiple adjacent stages independently meet the damage rule before/after first loss:
`MULTI_STAGE_INTERFACE_LOSS`.

If no single stage meets it but:
- Q7 K4 <= Q0 K4 -.15;
- Q7 K16 <= Q0 K16 -.10;
then:
`DISTRIBUTED_INTERFACE_LOSS`.

Otherwise:
`INTERFACE_DECOMPOSITION_UNRESOLVED`.

## 10. Cross-domain outcome

Stable:
same non-unresolved/non-inadequate class on >=3/4 BJ/BK/BL/BM.

Mixed:
two incompatible classes each occur on >=2 domains.

Otherwise:
unresolved.

Overall:
- `STABLE_INTERFACE_LOCALIZATION`;
- `MIXED_INTERFACE_LOCALIZATION`;
- `INTERFACE_DECOMPOSITION_UNRESOLVED`.

Thresholds and precedence are frozen before exposure.

## 11. Exposed / forbidden evidence

Never reuse for W11:
- all exposed W6b-W10 diagnostic/CONFIRM rows;
- W8 AU/AV/AW/AX;
- W9 AY/AZ/BA/BB/BC/BD/BE;
- W10 BF/BG/BH/BI;
- Banking77 0–799;
- typed final/test;
- public campaign cells.

After W11, BJ/BK/BL/BM become exposed permanently.

## 12. Scientific boundary

No training in W11.

Do not:
- tune projection;
- tune IDF;
- alter salience threshold;
- alter min-coverage weight;
- alter relation mode;
- search question wording after exposure;
- reorder stages after exposure;
- use R0 inside HIRA;
- choose a subset of domains after exposure;
- reinterpret W10.

Historical results matter:
- W6f says crude salience deletion was harmful;
- W6f says relation was net helpful there.
W11 may still localize a stage under this new natural-semantic setting, but must not erase those older findings.

## 13. Authorization after W11

If one stable stage:
- W12 may test one bounded fresh mechanism for that stage.

If stable `DISTRIBUTED_INTERFACE_LOSS`:
- W12 must test a jointly simplified interface, not overfit one micro-stage.

If mixed/unresolved:
- no rescue;
- prior-art + architecture reassessment first.

## 14. Current state

Completed:
- W10 empirical closure frozen in its handoff;
- PR #108 merged;
- issue #107 closed;
- W11 issue #109 preregistered;
- W11 branch created from exact post-W10 main;
- this handoff created before any BJ/BK/BL/BM exposure.

No W11 fresh A13/reference embeddings exist.

Remaining:
1. implement fresh BJ-BM generator;
2. implement state-once paired cache;
3. implement Q0-Q7 exact decomposition;
4. implement R0 frozen reference;
5. implement transition/first-loss metrics;
6. implement frozen classifier library/tests;
7. exact W9 checkpoint provenance loader;
8. pre-data unit/CI gate;
9. gated authority workflow;
10. only then expose BJ-BM;
11. freeze exact result here before merge.

## 15. Handoff discipline

Every meaningful session must append:
- exact branch head;
- exact runs/conclusions;
- artifacts/digests;
- exposure state;
- exact frozen metrics/classification once exposed;
- completed/remaining work;
- forbidden next moves.

A future AI must be able to continue without chat memory.


---

## 16. Pre-authority implementation freeze

No BJ/BK/BL/BM A13 or MiniLM reference embeddings existed before the validation below.

Exact full pre-authority implementation head:
`2a9f04b6f05105ee2f29a19e3c5c67bcc7a55d7c`.

Implemented:
- `src/nmd/interface_decomposition_authority.py` fresh BJ-BM paired generator;
- `src/nmd/interface_decomposition_cache.py` one-state-encode/base paired cache;
- `src/nmd/interface_decomposition.py` frozen stage classifiers/outcome;
- `src/nmd/interface_decomposition_eval.py` exact Q0-Q7 cumulative evaluator;
- `scripts/r8_w11_build_cache.py` sealed A13 cache builder with full prior-authority text freshness checks;
- `scripts/r8_w11_evaluate.py` exact W9 checkpoint + pinned reference evaluator;
- `tests/test_interface_decomposition.py`;
- `.github/workflows/r8-w11-unit.yml`.

Frozen evaluator details:
- Q0: W9 projected symmetric state↔definition MaxSim;
- Q1: option→state only;
- Q2: add exact question content;
- Q3: exact production candidate-relative IDF;
- Q4: exact production common-mode subtraction;
- Q5: exact production weighted mean + salient-min aggregation;
- Q6: actual `CompetitiveCoarseScorer`;
- Q7: frozen HIRACore pooled-relation final.
- Q5/Q6 complete rank-order identity is fail-closed at every K.
- transition damage gate rates use exactly the preregistered 128 natural-definition decisions/domain: K4 + K16;
- K8 remains descriptive for transition gates and is included in first-loss/first-rescue histograms.

Pinned R0:
- `sentence-transformers/all-MiniLM-L6-v2`;
- revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- required exact weight SHA `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`;
- evaluator fails closed on mismatch.

Pre-data validation:
- W11 unit run `36130334763`: **PASS**;
- repository CI run `36130334745`: **PASS**;
- exact head for both: `2a9f04b6f05105ee2f29a19e3c5c67bcc7a55d7c`.

No fresh empirical result exists at this point.
No W11 stage classification exists at this point.
The next commit may add only the gated authority orchestration; scientific thresholds/stage order are frozen.


---

## 17. Authoritative W11 closure

Exact authority head:
`12ca94f4304a5bc7a2bff6981bcd5e38b41e1f26`

Authority run:
`36130752815` — **PASS**.

All gated stages passed:
`unit -> exact W9 provenance -> fresh BJ/BK/BL/BM state-once cache -> pinned MiniLM reference + Q0-Q7 evaluation`.

Artifacts:
- fresh interface cache:
  - `10860944743`;
  - `sha256:49446233f16312f901c64cde216ee21209abfb25dcc6d0dc842d30a066e58f8c`.
- exact frozen W9 checkpoint bundle:
  - `10861888246`;
  - `sha256:afbadf77f41a2cbb4e3795b0b6fd938735f4a6d9a332b264dca12436f0b1e114`.
- authoritative interface audit:
  - `10861914056`;
  - `sha256:6610f0b36cd8d661ce8b8e20427e516227a7ddbb175146036753c78453af33a6`.

Integrity:
- 256 fresh base states;
- 1,536 paired views;
- state encodes/base = 1.0;
- probability mass max error = `1.7891579773277044e-07`;
- MiniLM pinned weight SHA exactly
  `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`;
- Q5/Q6 top1 identity = 100% at K4/K8/K16;
- Q5/Q6 complete rank-order identity = 100% at K4/K8/K16;
- no training;
- no W8/W9/W10 diagnostic reuse;
- no Banking77 reuse;
- no typed final/test rows;
- campaign cells 0.

### Frozen outcome

**`INTERFACE_DECOMPOSITION_UNRESOLVED`**

No stable stage reaches the preregistered >=3/4 domain rule.

Classification counts:
- `COMMON_MODE_SUBTRACTION_LOSS`: 1 domain;
- `SALIENT_MIN_COVERAGE_LOSS`: 1 domain;
- `QUESTION_CONTEXT_LOSS`: 1 domain;
- one domain unresolved.

Therefore:
- no stage-specific W12 rescue is authorized;
- no distributed-interface rescue is authorized;
- thresholds/stage precedence remain unchanged.

### Domain anatomy

#### BJ — university housing administration

Classification:
**`COMMON_MODE_SUBTRACTION_LOSS`**

Adequacy:
- R0 K4 98.44%;
- Q0 K4 85.94%;
- Q0 K16 64.06%.

Production final:
- Q7 K4 71.88%;
- Q7 K16 50.00%.

Frozen damaging stage:
- Q3 -> Q4 common-mode subtraction.

Gate transition rates:
- correct -> wrong: 22.66%;
- wrong -> correct: 12.50%.

Other notable transitions:
- Q0 -> Q1 directionality c->w 10.16%, w->c 1.56%, but it does not satisfy the independent frozen K4/K16 loss requirements;
- Q4 -> Q5 salient-min c->w 7.81%, w->c 7.81%.

#### BK — agricultural equipment leasing

Classification:
**`SALIENT_MIN_COVERAGE_LOSS`**

Adequacy:
- R0 K4 98.44%;
- Q0 K4 84.38%;
- Q0 K16 62.50%.

Production final:
- Q7 K4 56.25%;
- Q7 K16 42.19%.

Frozen damaging stage:
- Q4 -> Q5 salient-min coverage.

Gate transition:
- correct -> wrong 19.53%;
- wrong -> correct 0.78%.

Common-mode is highly disruptive at the individual-decision level here:
- Q3 -> Q4 c->w 25.00%;
- w->c 23.44%;
but its large two-way churn prevents the frozen one-direction damage classification.

#### BL — professional certification administration

Classification:
**`QUESTION_CONTEXT_LOSS`**

Adequacy:
- R0 K4 89.06%;
- Q0 K4 89.06%;
- Q0 K16 64.06%.

Production final:
- Q7 K4 81.25%;
- Q7 K16 62.50%.

Frozen damaging stage:
- Q1 -> Q2 question-context addition.

Gate transition:
- correct -> wrong 15.63%;
- wrong -> correct 2.34%.

The later common-mode step is net rescuing on many decisions:
- Q3 -> Q4 c->w 10.94%;
- w->c 22.66%.

#### BM — airline baggage support

Classification:
**`INTERFACE_DECOMPOSITION_UNRESOLVED`**

Adequacy:
- R0 K4 98.44%;
- Q0 K4 79.69%;
- Q0 K16 60.94%.

Production final:
- Q7 K4 75.00%;
- Q7 K16 37.50%.

No single adjacent transformation meets every frozen damage gate.

Notable churn:
- Q3 -> Q4 c->w 17.19%, w->c 9.38%;
- Q4 -> Q5 c->w 11.72%, w->c 3.13%.

The total K16 degradation is real, but the preregistered distributed-loss rule is not satisfied together with the absence of a qualifying adjacent stage strongly enough to create a stable cross-domain target.

### Pooled stage trajectory — natural definitions

Frozen MiniLM task ceiling R0:
- K4 **96.09%**;
- K8 **92.97%**;
- K16 **87.89%**.

W9 semantic projection starting geometry Q0:
- K4 **84.77%**;
- K8 **75.39%**;
- K16 **62.89%**.

Q1 — drop reverse state->schema coverage:
- K4 81.25%;
- K8 71.48%;
- K16 58.20%.

Q2 — add production question context:
- K4 76.17%;
- K8 66.02%;
- K16 53.91%.

Q3 — add candidate-relative IDF:
- K4 79.69%;
- K8 69.14%;
- K16 57.42%.

Q4 — add common-mode subtraction:
- K4 78.13%;
- K8 66.41%;
- K16 55.08%.

Q5 — add salient-min coverage:
- K4 **69.92%**;
- K8 **57.42%**;
- K16 **47.27%**.

Q6 — actual production CompetitiveCoarseScorer:
- identical ranking to Q5 at all K.

Q7 — frozen HIRACore final:
- K4 **71.09%**;
- K8 **57.03%**;
- K16 **48.05%**.

Thus pooled Q0 -> Q7:
- K4: -13.67 pp;
- K16: -14.84 pp.

Relation/HIRACore is slightly net positive pooled:
- Q6 K4 69.92% -> Q7 71.09%;
- Q6 K16 47.27% -> Q7 48.05%.

Candidate-relative IDF is also net positive pooled:
- Q2 K4 76.17% -> Q3 79.69%;
- Q2 K16 53.91% -> Q3 57.42%.

The largest pooled single-step drop is Q4 -> Q5 salient-min:
- K4 -8.20 pp;
- K16 -7.81 pp;
but the frozen adjacent damage threshold requires >=10 pp K4 and >=8 pp K16 within a domain, and that mechanism is not stable across >=3 domains.

### Scientific interpretation

W11 does **not** support one universal production-stage culprit.

Instead it exposes strong domain-dependent interaction:
- question context can be harmful in one semantic family;
- common-mode subtraction can be harmful in another and net rescuing in another;
- salient-min can be sharply harmful in one domain and moderately harmful elsewhere;
- IDF is pooled beneficial;
- HIRACore relation is pooled slightly beneficial;
- the exact production scorer decomposition is validated by Q5/Q6 100% rank identity.

This explains why W10 saw a large pooled P1 -> S1 interface loss but could not get a 3/4-domain stable `SCORING_INTERFACE_LIMIT` classification. The loss is not generated by one stable micro-stage under the current interface.

The correct conclusion is heterogeneity/interaction, **not** permission to delete question context, common-mode subtraction or salient-min globally.

Historical W6f remains compatible:
- crude salience removal was harmful;
- relation reranking was net helpful there.
W11 likewise sees IDF/relation pooled benefits while showing context-dependent damage from later competitive transforms.

### Permanent forbidden evidence after W11

BJ/BK/BL/BM are now exposed.

Never use them for:
- training;
- mechanism selection;
- IDF/salience tuning;
- question wording search;
- threshold tuning;
- min-coverage tuning;
- relation selection;
- seed selection;
- choosing a preferred W11 stage.

All earlier exposed/forbidden evidence remains forbidden.

### Authorized next research boundary

Because W11 is unresolved:

**Do not train a stage-specific rescue.**

The next phase must be a **prior-art + architecture reassessment** of why a fixed competitive scoring interface produces domain-dependent rank transformations.

A valid next diagnostic may investigate a higher-level invariant, for example:
- whether competition should be conditioned on semantic uncertainty/evidence rather than always applied;
- whether question/context tokens should be gated by candidate-independent semantic relevance;
- whether coverage penalties need normalization against semantic confidence;
- whether the production path should preserve a candidate-independent semantic anchor and use competition only as a bounded residual.

But none of these is authorized as a mechanism yet.

Before training anything:
1. review relevant retrieval/reranking, late-interaction, robust set scoring and mixture/gating prior art;
2. derive one architecture-level invariant that explains BJ/BK/BL/BM heterogeneity without tuning on them;
3. preregister a wholly fresh diagnostic for that invariant;
4. create a new handoff before fresh exposure.

A future AI should read W11 first, then W10, W9 and W8.
