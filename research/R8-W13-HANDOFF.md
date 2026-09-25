# R8-W13 handoff — paraphrase-consistency semantic reliability audit

Status: **CLOSED DIAGNOSTIC. Frozen outcome: `SEMANTIC_CONSISTENCY_UNRESOLVED`. No rescue mechanism is authorized.**

Issue: #113

Branch:
`feat/r8-w13-semantic-consistency`

Base main:
`5a3c45aebde8a74271a61193c7e64233df9723ff`

Read first when restoring:
1. this file;
2. `research/R8-W12-HANDOFF.md`;
3. `research/R8-W11-HANDOFF.md`;
4. `research/R8-W10-HANDOFF.md`;
5. `research/R8-W9-HANDOFF.md`;
6. issue #113.

## 0. Project identity

Nolane HIRA is a compact non-autoregressive typed decision engine, not a next-token language model.

Long-term target:
- encode state once;
- support dynamic semantic schemas;
- handle changing/high-cardinality candidate sets;
- typed choice / score / noul primitives;
- small local-friendly trainable footprint;
- transferable semantic binding;
- robust high-K structured decisions;
- calibrated/reliable probabilities;
- sealed fresh authorities and immutable negative results.

## 1. Frozen semantic/runtime base

A13:
- `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length 256;
- fully frozen.

W9 semantic projection:
- scorer SHA `8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102`;
- selected epoch 5;
- source run `36122220588`;
- source freeze artifact `10858424139`.

Frozen HIRACore:
- SHA `d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588`.

No W13 parameter is trainable.

## 2. Decisive history entering W13

W5:
- pooled/bilinear/simple probes failed;
- late interaction + competitive salience became the semantic production path.

W6-W6e:
- typed integration/reliability became strong internally;
- fresh-domain K64 remained unstable.

W6f-W6i:
- salience removal was harmful;
- relation reranking was not the root cause;
- local residual adapters and naive factorized/canonical interfaces did not rescue.

W6j:
- stable synthetic high-K diagnosis `COARSE_CONJUNCTION_LIMIT`.

W7:
- explicit factor smooth-AND failed;
- equal-data free-form retune was much stronger.

W7b:
- typed+pair free-form retuning produced high synthetic K64 but failed frozen replication gates.

W7c:
- Banking77 rows 400-799 showed essentially absent external high-K transfer with severe overconfidence.

W8:
- stable `GENERAL_SEMANTIC_TRANSFER_LIMIT` at low K.

W9:
- shared/asymmetric low-rank semantic bridges failed;
- full 256->128 semantic projection retune was the strongest signal.

W10:
- candidate-independent W9 projection geometry P1 reached ~79.3% K4 / 54.3% K16 pooled;
- same projection through production S1 fell to ~60.6% / 32.0%;
- result unresolved under 3/4 domain rule.

W11:
- decomposed production interface;
- different domains localized to question-context, common-mode and salient-min losses;
- no stable micro-stage culprit.

W12:
- tested scalar normalized-margin reliability;
- valid authority head `87af9fecb9a340dd868d085f4c97fc8026a3a4da`;
- run `36137418586`;
- pooled semantic anchor A 80.86% K4 / 60.55% K16;
- production final F 66.41% / 40.63%;
- BN/BO -> `GLOBAL_ANCHOR_DOMINANCE`;
- BP/BQ unresolved;
- frozen outcome `ANCHOR_RESIDUAL_UNRESOLVED`.

Thus the semantic anchor is repeatedly valuable, but one scalar top1-top2/MAD margin is not a stable reliability representation.

## 3. W13 scientific question

W13 asks:

**Does cross-paraphrase semantic ranking consistency provide a more transferable label-free reliability signal than scalar score margin?**

Core invariant:

> if several meaning-equivalent schema definitions independently produce the same semantic ranking/top choice, that agreement may indicate trustworthy candidate-independent semantic evidence.

W13 is diagnostic only.

No:
- training;
- calibration;
- checkpoint selection;
- architecture promotion;
- post-exposure threshold tuning.

## 4. Prior-art motivation

The lane is motivated by:
- retrieval query-variation robustness findings showing rankings can be unstable under paraphrases;
- paraphrastic consistency work treating meaning-preserving formulation agreement as a model reliability property;
- confidence-aware/adaptive reranking work arguing that refinement should not be applied identically to every ranking state.

W13 does not claim novelty for any of these concepts.

The HIRA-specific question is whether this reliability principle appears under the exact frozen semantic anchor / production competition split already exposed by W10-W12.

## 5. Fresh W13 domains

Fresh/disjoint:

- BR municipal parking permit administration — seed 321001;
- BS continuing-education enrollment services — seed 321013;
- BT household insurance claim administration — seed 321019;
- BU grocery delivery subscription support — seed 321031.

Each:
- exactly 16 intents;
- 4 state variants/intent;
- 3 frozen natural-definition paraphrases/intent: D0/D1/D2;
- 64 bases/domain.

Total:
- 256 bases;
- K4/K8/K16;
- 3 semantic schema views/K;
- 2,304 views.

BR/BS/BT/BU become permanently exposed at first eligible W13 cache materialization.

## 6. Frozen candidate identity

For one base:
- one deterministic 16-intent master membership;
- nested K4 < K8 < K16;
- same candidate IDs/order across D0/D1/D2;
- same gold index across paraphrases;
- same state text across all views.

No view-specific candidate sampling.

## 7. Frozen semantic operators

A0:
- D0 candidate-independent symmetric MaxSim.

A1:
- D1 same operator.

A2:
- D2 same operator.

Each:
- exact frozen A13 content tokens;
- exact W9 256->128 projection;
- L2 normalized tokens;
- definition->state MaxSim mean;
- state->definition MaxSim mean;
- arithmetic mean of both directions;
- no question;
- no IDF;
- no sibling common-mode;
- no salient-min;
- no HIRACore.

E:
- candidate score = arithmetic mean of A0/A1/A2 raw scores;
- no learned weights/temperature/calibration.

F:
- D0 only;
- exact production CompetitiveCoarseScorer with W9 projection;
- exact production question context / IDF / common-mode / weighted mean + 0.5 salient-min;
- exact frozen pooled HIRACore final;
- forced full-K;
- adaptive budget false.

## 8. Label-free consistency strata

For each base/K:
- STRICT = A0/A1/A2 all choose same top1 candidate ID;
- MAJORITY = exactly two choose same top1;
- SPLIT = all three choose different top1 IDs.

Primary binary split:
- STRICT;
- NON_STRICT = MAJORITY + SPLIT.

No labels enter strata.

Descriptive stability:
- pairwise Spearman rank correlation for A0/A1, A0/A2, A1/A2;
- mean pairwise rank stability;
- top3-union size;
- E winner vote count.

## 9. Frozen guards

`G_strict_ensemble`:
- STRICT -> E;
- NON_STRICT -> F.

`G_margin_countmatched`:
- for each domain/K, N = STRICT case count;
- rank all 64 cases by W12 normalized canonical A0 margin;
- exactly highest N use E;
- rest use F;
- tie break base_id.

This is the equal-coverage scalar-margin control.

`G_majority_ensemble`:
- STRICT+MAJORITY -> E;
- SPLIT -> F;
- descriptive only.

## 10. Reference ceiling

Pinned diagnostic-only reference:
- `sentence-transformers/all-MiniLM-L6-v2`;
- revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- expected weight SHA `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.

Reference:
- state vs D0 definition cosine;
- task adequacy only;
- never enters HIRA/guards/training.

## 11. Frozen adequacy

Domain must satisfy:
- R0 K4 >= .75;
- E K4 >= .70;
- E K16 >= .45;
- STRICT coverage >= .25 at K4 and K16.

Else:
`CONSISTENCY_BASELINE_INADEQUATE`.

## 12. Frozen classifications

### PARAPHRASE_CONSISTENCY_RELIABLE

Require:
- adequacy passes;
- STRICT E >= .85 K4 and >= .70 K16;
- STRICT E - NON_STRICT E >= .15 at both K4/K16;
- STRICT F <= E-.10 K4 and <= E-.08 K16;
- pooled STRICT K4+K16 E->F c->w - w->c >= .08;
- NON_STRICT F >= E-.03 at both K4/K16;
- G_strict_ensemble >= F+.05 K4 and >= F+.04 K16;
- G_strict_ensemble >= G_margin_countmatched+.03 at K4 or K16.

### MULTIVIEW_ANCHOR_DOMINANCE

Require:
- adequacy;
- E >= F+.08 K4 and K16;
- F >=.05 worse than E on both STRICT and NON_STRICT for at least one primary K;
- primary consistency-gated rule does not pass.

### PRODUCTION_VALUE_ON_INCONSISTENCY

Require:
- adequacy;
- F >= E+.05 at K4 or K16;
- NON_STRICT pooled K4+K16 w->c - c->w >= .08;
- STRICT F regression <.05 at K4/K16.

### PARAPHRASE_CONSISTENCY_UNINFORMATIVE

Require:
- adequacy;
- STRICT E - NON_STRICT E <.15 at both K4/K16.

Else:
`SEMANTIC_CONSISTENCY_UNRESOLVED`.

## 13. Cross-domain outcome

Same non-unresolved/non-inadequate class on >=3/4:
`STABLE_SEMANTIC_CONSISTENCY_LOCALIZATION`.

Two incompatible classes each on >=2:
`MIXED_SEMANTIC_CONSISTENCY_LOCALIZATION`.

Else:
`SEMANTIC_CONSISTENCY_UNRESOLVED`.

No rule changes after exposure.

## 14. Authorization boundary

If stable `PARAPHRASE_CONSISTENCY_RELIABLE`:
- a future W14 may test one bounded consistency gate;
- fresh TRAIN/DEV/dual-CONFIRM;
- anchor-only / production-only / scalar-margin / consistency controls.

If stable `MULTIVIEW_ANCHOR_DOMINANCE`:
- W14 targets anchor-preserving production redesign, not routing.

If stable `PRODUCTION_VALUE_ON_INCONSISTENCY`:
- W14 may preserve production only where semantic views disagree, with fresh causal controls.

If mixed/unresolved/uninformative/inadequate:
- no rescue training.

## 15. Forbidden evidence

Never reuse for W13 selection/tuning:
- all W6b-W12 exposed authority rows;
- W8 AU/AV/AW/AX;
- W9 AY/AZ/BA/BB/BC/BD/BE;
- W10 BF/BG/BH/BI;
- W11 BJ/BK/BL/BM;
- W12 BN/BO/BP/BQ;
- Banking77 rows 0-799;
- typed final/test;
- public campaign cells.

## 16. Current state

Completed:
- W12 merged to main `5a3c45aebde8a74271a61193c7e64233df9723ff`;
- issue #113 preregistered before W13 exposure;
- branch created from exact post-W12 main;
- this handoff created before W13 exposure.

No BR/BS/BT/BU cache exists.
No W13 empirical metric/classification exists.

Remaining:
1. freeze exact D0/D1/D2 generator;
2. implement state-once multiview cache;
3. implement A0/A1/A2/E/F evaluator;
4. implement consistency strata and rank-stability metrics;
5. implement count-matched margin control;
6. implement frozen classifiers;
7. add exact MiniLM adequacy reference;
8. add unit contracts/workflow;
9. freeze prior-art note;
10. run pre-data unit/CI;
11. only then enable authority;
12. freeze exact result here before merge.

## 17. Handoff discipline

At every meaningful update record:
- exact branch head;
- runs;
- artifacts/digests;
- exposure state;
- exact scientific mutations before exposure;
- exact classification after exposure;
- completed/remaining work;
- forbidden next moves.

A future AI must be able to continue from this file without chat memory.


## 18. Pre-data implementation update

Implemented before any BR/BS/BT/BU empirical exposure:

- `src/nmd/semantic_consistency.py`
  - STRICT / MAJORITY / SPLIT label-free top1 agreement;
  - full-rank pairwise Spearman stability;
  - equal-coverage count-matched scalar-margin control;
  - frozen domain classifiers;
  - frozen cross-domain outcome.

- `src/nmd/semantic_consistency_authority.py`
  - BR/BS/BT/BU fresh domain generator;
  - 16 intents/domain;
  - 4 state variants/intent;
  - three prewritten meaning-equivalent definitions D0/D1/D2;
  - exact paired nested K4/K8/K16 identity.

- `src/nmd/semantic_consistency_cache.py`
  - state-once A13 cache;
  - 9 views/base;
  - paraphrase identity enforcement;
  - nested candidate membership/order enforcement.

- `src/nmd/semantic_consistency_eval.py`
  - A0/A1/A2 candidate-independent W9 semantic anchors;
  - equal-weight E ensemble;
  - exact D0 production F path;
  - consistency strata;
  - guard/control evaluation;
  - rank stability / top3 union / vote diagnostics.

- `scripts/r8_w13_build_cache.py`
  - exact frozen A13 verification;
  - exact-text freshness firewall through W12;
  - 256-base / 2,304-view receipt.

- `scripts/r8_w13_evaluate.py`
  - exact W9 semantic projection/HIRACore provenance;
  - pinned MiniLM adequacy reference;
  - frozen W13 classifier/outcome.

- `tests/test_semantic_consistency.py`
  - paired paraphrase/nested-K contracts;
  - consistency category contracts;
  - rank-stability contracts;
  - deterministic count-matched margin control;
  - classifier semantics;
  - state-once cache.

- `.github/workflows/r8-w13-unit.yml`
  - branch-scoped pre-data compile/test gate.

- `research/R8-W13-PRIOR-ART.md`
  - frozen pre-data motivation.

Draft PR:
- #114.

### Pre-data evaluability amendment

Before any W13 cache/reference exposure, issue #113 comment froze one additional adequacy condition:

- NON_STRICT coverage >= .10 at K4;
- NON_STRICT coverage >= .10 at K16.

Reason:
STRICT-vs-NON_STRICT causal reliability is not evaluable if essentially every case is STRICT.

No other gate, precedence, domain, seed, paraphrase, score operator or guard changed.

### Current exposure state

**No BR/BS/BT/BU A13 cache exists.**
**No MiniLM W13 score exists.**
**No W13 classification exists.**

Authority workflow remains intentionally absent until the exact implementation head is green under W13 unit + repository CI.


---

## 19. Authoritative W13 closure

Exact empirical head:
`a47f368d57de811f1b8f38beff39db973b9ef89b`

Authority run:
`36141078081`

All jobs PASS:
- frozen W13 unit contracts;
- exact W9 checkpoint provenance;
- fresh BR/BS/BT/BU state-once cache;
- pinned MiniLM reference;
- frozen consistency evaluator/classifier.

Artifacts:
- W9 frozen checkpoint bundle: `10867295064`,
  digest `sha256:9ee1c020e5d6bc87cfa4ce3c21e127e7a5a2c3778363d8e7995731de144fe07b`;
- fresh W13 cache: `10867091010`,
  digest `sha256:a8d248375c4d1e915e15e1fc2a310179490be0d8c728a29b1fa1deef91ae7018`;
- authoritative W13 audit: `10866893030`,
  digest `sha256:2e516d35c55408756ac3126d0e0501d1dcdd50286c2e6a066357597466d690ac`.

Integrity:
- 256 fresh bases;
- 2,304 D0/D1/D2 × K4/K8/K16 views;
- state encodes/base = 1.0;
- probability mass max error = `1.4487886801362038e-07`;
- no training;
- no W8-W12 authority rows;
- no Banking77 rows;
- no typed final/test;
- campaign cells = 0;
- exact pinned MiniLM revision/SHA verified.

### Frozen outcome

**`SEMANTIC_CONSISTENCY_UNRESOLVED`**

Stable classification:
`null`.

Classification counts:
- `MULTIVIEW_ANCHOR_DOMINANCE`: 1.

Per-domain:
- BR -> `CONSISTENCY_BASELINE_INADEQUATE`;
- BS -> `CONSISTENCY_BASELINE_INADEQUATE`;
- BT -> `CONSISTENCY_BASELINE_INADEQUATE`;
- BU -> `MULTIVIEW_ANCHOR_DOMINANCE`.

The frozen >=3/4 cross-domain stability gate is not met.

No W14 rescue/training mechanism is authorized.

## 20. Pooled semantic anatomy

Single paraphrase anchors:
- A0: 80.08% K4 / 47.27% K16;
- A1: 62.50% / 43.75%;
- A2: 87.11% / 70.70%.

Equal-weight multiview semantic ensemble E:
- K4 **85.16%**;
- K8 **76.17%**;
- K16 **63.28%**.

Production final F:
- K4 **57.42%**;
- K8 **40.63%**;
- K16 **28.52%**.

Thus pooled E -> F:
- K4 **-27.73 pp**;
- K8 **-35.55 pp**;
- K16 **-34.77 pp**.

Pinned MiniLM task ceiling:
- K4 **92.97%**;
- K16 **64.84%**.

The multiview candidate-independent anchor nearly reaches the pinned reference at K16 while the production path collapses far below it.

## 21. Why the preregistered STRICT consistency hypothesis does not pass

STRICT top1 agreement is highly precise, but its K16 coverage is too low on three fresh domains:

BR:
- STRICT coverage K4 46.875%;
- STRICT coverage K16 **23.438%**;
- frozen minimum K16 coverage = 25%;
- STRICT E K4/K16 = 100% / 100%;
- therefore domain classification = `CONSISTENCY_BASELINE_INADEQUATE`.

BS:
- STRICT coverage K4 48.438%;
- K16 **21.875%**;
- STRICT E K4 93.55%;
- STRICT E K16 92.86%;
- classification = inadequate.

BT:
- STRICT coverage K4 50.0%;
- K16 **20.313%**;
- STRICT E K4/K16 = 100% / 100%;
- classification = inadequate.

BU:
- STRICT coverage K4 57.813%;
- K16 **26.563%**;
- E K4 89.06%;
- E K16 68.75%;
- F K4 59.38%;
- F K16 32.81%;
- classification = `MULTIVIEW_ANCHOR_DOMINANCE`.

The K16 coverage gate was frozen before data and **must not be lowered** to 20%, 21% or 23% after seeing these results.

## 22. STRICT vs NON_STRICT anatomy

### BR
STRICT:
- E K4/K16 = 100% / 100%;
- F = 70.0% / 33.33%;
- pooled primary E->F c->w 42.22%;
- w->c 0%.

NON_STRICT:
- E K4/K16 = 58.82% / 51.02%;
- F = 55.88% / 22.45%.

Rank-stability correlation with E correctness:
- Spearman = 0.5139.

Vote-count correlation with E correctness:
- 0.5630.

### BS
STRICT:
- E K4 93.55%;
- E K16 92.86%;
- F K4 70.97%;
- F K16 35.71%.

NON_STRICT:
- E K4 87.88%;
- E K16 56.0%;
- F K4 24.24%;
- F K16 6.0%.

Here production is catastrophically worse even when semantic views are not STRICT, which directly violates the desired “production is neutral/helpful on inconsistency” routing law.

### BT
STRICT:
- E K4/K16 = 100% / 100%;
- F = 81.25% / 76.92%.

NON_STRICT:
- E = 65.63% / 47.06%;
- F = 40.63% / 35.29%.

### BU
STRICT:
- E K4/K16 = 100% / 100%;
- F = 75.68% / 41.18%.

NON_STRICT:
- E = 74.07% / 57.45%;
- F = 37.04% / 29.79%.

Across the authority, STRICT agreement is indeed associated with strong semantic correctness, but production is often harmful in both STRICT and NON_STRICT cases rather than providing a stable fallback specifically for inconsistency.

## 23. Frozen guard/control results

Pooled `G_strict_ensemble`:
- K4 69.53%;
- K16 40.63%.

Pooled equal-coverage `G_margin_countmatched`:
- K4 66.80%;
- K16 35.94%.

STRICT consistency therefore beats the scalar-margin control pooled, but the superiority is not stable per-domain enough to satisfy the frozen causal classification.

Descriptive-only `G_majority_ensemble`:
- K4 **85.16%**;
- K16 **56.25%**.

This is a strong hypothesis-generating result, but it was explicitly frozen as descriptive-only before exposure and cannot be promoted post hoc into the W13 primary classifier.

Do not reclassify W13 using majority agreement.

## 24. Scientific interpretation

W13 substantially strengthens one broad conclusion:

**candidate-independent multiview semantic evidence is much stronger than the current production path on these fresh natural semantic domains.**

It also shows:
- exact 3/3 paraphrase top1 consensus has very high precision;
- rank stability and vote count correlate positively with semantic correctness in several domains;
- the equal-weight multiview ensemble E itself is materially stronger than any single weak paraphrase view;
- production often destroys correct rankings even outside the STRICT subset.

But W13 falsifies the exact preregistered claim that **STRICT 3/3 agreement has enough coverage and conditional residual structure to serve as a stable production router**.

The most interesting new clue is the descriptive majority-vote result:
- requiring only 2/3 agreement appears to preserve much more semantic coverage;
- the majority guard remains far stronger than F;
- but W13 cannot use it for scientific promotion because it was not the primary frozen classification.

A future diagnostic may use this clue only on wholly fresh data.

## 25. Permanent forbidden evidence after W13

BR/BS/BT/BU are now exposed permanently.

Never use them for:
- training;
- paraphrase generation/search;
- vote-threshold selection;
- ensemble-weight selection;
- routing/gating;
- margin threshold selection;
- production mixing weights;
- architecture selection;
- calibration;
- seed choice.

All earlier exposed evidence remains forbidden.

## 26. Authorized continuation boundary

Because W13 is unresolved:

**No rescue training is authorized.**

A legitimate next phase may run a wholly fresh diagnostic testing whether **2-of-3 semantic agreement / multiview ensemble dominance** is a stable reliability invariant.

Such a diagnostic must:
1. use new domains and new paraphrase wording;
2. preregister majority-consensus coverage and precision gates before exposure;
3. compare majority consistency directly against equal-coverage scalar margin;
4. distinguish “reliability signal” from “multiview anchor is simply globally better”;
5. retain a pinned external adequacy reference;
6. keep A13/W9 projection/HIRACore frozen;
7. remain no-training until a stable target is found.

It must not:
- lower W13 STRICT coverage gates;
- reuse BR/BS/BT/BU;
- promote `G_majority_ensemble` from W13 itself;
- train a router from W13;
- claim external/general superiority.

A future AI should read W13 first, then W12/W11/W10/W9.


---

## 19. Authoritative W13 closure

Exact empirical head:
`a47f368d57de811f1b8f38beff39db973b9ef89b`

Authority run:
`36141078081`

All jobs PASS:
- unit;
- exact W9 upstream provenance;
- fresh BR/BS/BT/BU state-once cache;
- frozen multiview evaluator/reference/classifier.

Artifacts:
- frozen W9 checkpoints `10867295064`, digest `sha256:9ee1c020e5d6bc87cfa4ce3c21e127e7a5a2c3778363d8e7995731de144fe07b`;
- W13 cache `10867091010`, digest `sha256:a8d248375c4d1e915e15e1fc2a310179490be0d8c728a29b1fa1deef91ae7018`;
- W13 audit `10866893030`, digest `sha256:2e516d35c55408756ac3126d0e0501d1dcdd50286c2e6a066357597466d690ac`.

Integrity:
- 256 fresh bases;
- 2,304 paired schema/cardinality views;
- state encodes/base = 1.0;
- no training;
- probability mass max error = `1.4487886801362038e-07`;
- no W8-W12 rows;
- no Banking77 reuse;
- no typed final/test rows;
- campaign cells = 0;
- pinned MiniLM reference verified exactly.

### Frozen outcome

**`SEMANTIC_CONSISTENCY_UNRESOLVED`**

Stable classification:
`null`.

Classification counts:
- `MULTIVIEW_ANCHOR_DOMINANCE`: 1;
- `CONSISTENCY_BASELINE_INADEQUATE`: 3.

The >=3/4 stable-target rule is not met.

No W14 rescue mechanism is authorized from W13.

## 20. Pooled semantic anatomy

Frozen multiview ensemble E:
- K4 **85.156%**;
- K8 **76.172%**;
- K16 **63.281%**.

Production final F:
- K4 **57.422%**;
- K8 **40.625%**;
- K16 **28.516%**.

Thus E -> F pooled loss:
- K4 **-27.734 pp**;
- K8 **-35.547 pp**;
- K16 **-34.766 pp**.

Frozen MiniLM reference:
- K4 **92.969%**;
- K8 **79.297%**;
- K16 **64.844%**.

Notably, at K16 the candidate-independent HIRA multiview ensemble E is already close to the pinned external semantic reference:
- E 63.281%;
- R0 64.844%.

This does not authorize replacing production with E, but it is strong evidence that semantic geometry itself can be much better than the current production final path.

### Frozen diagnostic guards

`G_strict_ensemble`:
- K4 69.531%;
- K8 56.250%;
- K16 40.625%.

Count-matched scalar-margin control `G_margin_countmatched`:
- K4 66.797%;
- K8 48.047%;
- K16 35.938%.

Descriptive `G_majority_ensemble`:
- K4 **85.156%**;
- K8 **72.656%**;
- K16 **56.250%**.

STRICT guard beats the equal-coverage margin control pooled:
- K4 +2.734 pp;
- K8 +8.203 pp;
- K16 +4.688 pp.

This is useful descriptive evidence that cross-paraphrase agreement carries information beyond the old scalar-margin family. However the frozen primary classifier cannot use a pooled-only advantage.

## 21. Per-domain frozen classifications

### BR — municipal parking permit administration

Classification:
**`CONSISTENCY_BASELINE_INADEQUATE`**

The semantic task itself is adequate:
- R0 K4 90.625%;
- E K4 78.125%;
- E K16 62.500%.

But frozen STRICT coverage gate fails at K16:
- STRICT K4 46.875%;
- STRICT K16 **23.438% < 25%**.

Do not lower the 25% gate after exposure.

Descriptive reliability is nevertheless strong:
- STRICT E K4 100%;
- STRICT E K16 100%;
- NON_STRICT E K4 58.824%;
- NON_STRICT E K16 51.020%;
- STRICT F K4 70.0%;
- STRICT F K16 33.333%.

STRICT guard:
- 76.563% K4;
- 40.625% K16.

Count-matched margin guard:
- 65.625% K4;
- 31.250% K16.

### BS — continuing-education enrollment services

Classification:
**`CONSISTENCY_BASELINE_INADEQUATE`**

Adequate semantic performance:
- R0 K4 95.313%;
- E K4 90.625%;
- E K16 64.063%.

STRICT coverage:
- K4 48.438%;
- K16 **21.875% < 25%**.

STRICT E:
- K4 93.548%;
- K16 92.857%.

NON_STRICT E:
- K4 87.879%;
- K16 56.0%.

Despite very high STRICT correctness, the preregistered coverage condition makes this domain non-evaluable for the primary consistency classifier.

### BT — household insurance claim administration

Classification:
**`CONSISTENCY_BASELINE_INADEQUATE`**

- R0 K4 93.750%;
- E K4 82.813%;
- E K16 57.813%.

STRICT coverage:
- K4 50.0%;
- K16 **20.313% < 25%**.

STRICT E:
- K4 100%;
- K16 100%.

NON_STRICT E:
- K4 65.625%;
- K16 47.059%.

Again the descriptive signal is exceptionally strong, but coverage is below the frozen threshold.

### BU — grocery delivery subscription support

Classification:
**`MULTIVIEW_ANCHOR_DOMINANCE`**

All adequacy gates pass:
- R0 K4 92.188%;
- E K4 89.063%;
- E K16 68.750%;
- STRICT coverage K4 57.813%;
- STRICT coverage K16 26.563%;
- NON_STRICT coverage K4 42.188%;
- NON_STRICT coverage K16 73.438%.

STRICT E:
- K4 100%;
- K16 100%.

Production F on STRICT:
- K4 75.676%;
- K16 41.176%.

This domain satisfies broad multiview-anchor dominance rather than the confidence-gated residual rule.

## 22. Scientific interpretation

W13 does not establish a stable production routing law, because the frozen evaluability gate fails on BR/BS/BT.

But it produces the strongest reliability evidence seen so far:

1. **Three independent semantic paraphrases agreeing on top1 is highly predictive of correctness.**
   - STRICT E is 93.5-100% at K4 and 92.9-100% at K16 on the reported fresh domains.
2. **The signal is materially stronger than scalar margin at equal coverage.**
   - pooled STRICT guard beats count-matched margin at K4/K8/K16.
3. **The candidate-independent multiview ensemble is dramatically stronger than production.**
   - E 85.16/76.17/63.28 versus F 57.42/40.63/28.52.
4. **The problem is coverage, not obvious reliability quality.**
   - on three domains only ~20-23% of K16 cases are STRICT, just below the preregistered 25% minimum.

This distinction is critical.

Do **not** reinterpret W13 as `PARAPHRASE_CONSISTENCY_RELIABLE`.
The authority was designed to demand both:
- strong conditional accuracy;
- enough support/coverage to establish the rule.

It only clearly demonstrates the first.

### What W13 falsifies

- Scalar margin is not the only useful reliability family; consistency contains additional signal.
- A simple STRICT-only router is still insufficient as a stable architecture rule because coverage is not robust enough under the frozen threshold.
- Production is not required to achieve strong low-K semantic performance: candidate-independent multiview semantics substantially outperform it.

### What W13 does not prove

- that three paraphrases should be required in production;
- that majority agreement is production-safe;
- that the 25% coverage threshold should be relaxed;
- that E should replace HIRA production;
- that multiview consistency generalizes outside these internal fresh domains.

## 23. Permanent exposed evidence

BR/BS/BT/BU are now permanently exposed.

Never use them to:
- tune number/content of paraphrases;
- lower STRICT coverage thresholds;
- choose STRICT vs MAJORITY routing;
- tune ensemble weights;
- tune production mixing;
- train a reliability gate;
- select seeds/checkpoints;
- design post-hoc classification rules.

All earlier exposed evidence remains forbidden.

## 24. Authorized next research boundary

Because W13 is unresolved/inadequate:

**No rescue training is authorized.**

However, W13 provides a sharper diagnostic target than W12.

The next phase may test, on wholly fresh domains, a **continuous or graded multiview semantic-consistency representation** that does not require exact 3/3 top1 agreement as its only reliable region.

A valid next diagnostic should preregister before exposure:
- continuous pairwise rank agreement;
- top-k set overlap;
- vote strength;
- score-vector agreement;
- semantic ensemble confidence;
- exact comparison to count-matched scalar margin;
- fixed coverage bins or continuous monotonicity tests;
- no learned gate;
- no threshold selected from W13.

The core question should be:

> Can semantic consistency provide a stable reliability ordering across most cases, not merely a very accurate but low-coverage STRICT subset?

If yes on fresh data, only then may a later phase test a learned/bounded gate.

A future AI should read W13 first, then W12/W11/W10/W9.
