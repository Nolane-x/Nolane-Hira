# R8-W13 handoff — paraphrase-consistency semantic reliability audit

Status: **PRE-DIAGNOSTIC. No BR/BS/BT/BU A13/reference cache or W13 classification exists yet.**

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
