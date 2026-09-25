# R8-W12 handoff — semantic-anchor / competitive-residual reliability audit

Status: **CLOSED DIAGNOSTIC. Frozen outcome: `ANCHOR_RESIDUAL_UNRESOLVED`. No rescue mechanism is authorized.**

Issue: #111

Branch:
`feat/r8-w12-anchor-residual-audit`

Base main:
`bcaaf006e3fcce054de62f5ed1276d468982c3d2`

Read first when restoring:
1. this file;
2. `research/R8-W11-HANDOFF.md`;
3. `research/R8-W10-HANDOFF.md`;
4. `research/R8-W9-HANDOFF.md`;
5. `research/R8-W8-HANDOFF.md`;
6. issue #111.

## 0. Project identity

Nolane HIRA is a compact non-autoregressive typed decision engine.

Long-term goals:
- state encoded once/case;
- dynamic semantic schemas;
- changing/high-cardinality candidate sets;
- typed choice / score / noul;
- small local-friendly parameter footprint;
- transferable semantic binding;
- robust structured decisions;
- explicit probability/reliability integrity;
- sealed authorities and immutable negative results.

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

No W12 parameter is trainable.

## 2. Decisive history entering W12

W5 established late interaction + competitive salience as the production semantic scorer.

W6-W6e showed strong internal typed competence but unstable held-out high-K generalization.

W6f-W6i ruled against simple salience removal, relation blame, local field adapters, and naive canonical/factorized interfaces.

W6j localized a synthetic high-K `COARSE_CONJUNCTION_LIMIT`.

W7 explicit factor smooth-AND failed; equal-data free-form retuning was much stronger.

W7b free-form typed+pair retuning produced high synthetic K64 but failed a preregistered replication gate.

W7c external Banking77 transfer was essentially absent and highly overconfident.

W8 stably localized `GENERAL_SEMANTIC_TRANSFER_LIMIT` at low K.

W9 contrastive low-rank shared/asymmetric bridges failed; full projection semantic retuning was the strongest signal.

W10 representation-ceiling audit:
- P1 W9 semantic projected symmetric MaxSim pooled 79.30% K4 / 54.30% K16;
- S1 same projection through production 60.55% / 32.03%;
- MiniLM reference 97.66% / 90.63%;
- outcome `REPRESENTATION_CEILING_UNRESOLVED`;
- BF/BG suggested `SCORING_INTERFACE_LIMIT`, BH/BI unresolved.

W11 decomposed the production interface:
- Q0 semantic anchor 84.77% K4 / 62.89% K16;
- Q7 production final 71.09% / 48.05%;
- BJ -> common-mode loss;
- BK -> salient-min loss;
- BL -> question-context loss;
- BM unresolved;
- outcome `INTERFACE_DECOMPOSITION_UNRESOLVED`.

Therefore no stage-specific rescue is authorized.

## 3. Prior-art reassessment

W12 does not claim novelty for late interaction, score fusion, residual ranking, or uncertainty gating.

Relevant prior art:
- ColBERTv2 — lightweight late interaction / token-level semantic evidence.
- TRIAL — token relations and token importance can materially change late-interaction relevance.
- AcuRank — uncertainty-aware adaptive reranking avoids identical refinement policy for every ranking state.
- BRIGHT — mismatched rerankers can reduce retrieval quality.
- FaLCon — recent anchor-constrained coarse-to-fine retrieval with bounded corrective evidence and uncertainty-gated consensus.

The HIRA-specific invariant under test is:

**preserve a candidate-independent semantic anchor when it is confident; treat production competition as a residual correction whose usefulness may depend on anchor uncertainty.**

## 4. W12 purpose

W12 is no-training diagnostic.

It asks whether W11's domain-dependent micro-stage losses collapse into a higher-level stable pattern:

- high-confidence semantic anchor decisions are systematically damaged by production competition;
- low-confidence anchor decisions are neutral or helped by production competition.

If stable, a future W13 may test a bounded/gated residual mechanism on wholly fresh data.

## 5. Fresh domains

- BN public library account services — seed 311901;
- BO vehicle inspection administration — seed 311907;
- BP workplace benefits administration — seed 311919;
- BQ veterinary appointment support — seed 311931.

Each:
- 16 latent intents;
- 4 independent states/intent;
- natural definition + terse label;
- 64 bases.

Total 256 bases.

Nested K4/K8/K16.

BN/BO/BP/BQ become permanently exposed after first authority.

## 6. Frozen operators

### A — semantic anchor

Exact W11 Q0 with W9 semantic projection:
- state content only;
- definition content only;
- projected 128-d;
- L2 normalized;
- bidirectional symmetric MaxSim mean;
- no question;
- no IDF;
- no common-mode;
- no salient-min;
- no HIRACore.

### C — production coarse

Exact CompetitiveCoarseScorer with W9 projection:
- question context;
- candidate-relative IDF;
- sibling common-mode subtraction;
- weighted mean + 0.5 salient-min;
- exact production log scale.

### F — production final

Frozen HIRACore on exact C coarse override:
- pooled relation;
- forced full-K;
- adaptive budget false.

## 7. Anchor confidence

For A score vector:
- raw margin = top1 - top2;
- robust spread = median absolute deviation from score median;
- normalized margin `m = raw_margin / max(spread, 1e-6)`.

No gold label enters confidence.

Within each domain and K independently:
- HIGH = top 25% margin, exactly 16 cases;
- LOW = bottom 25%, exactly 16;
- MIDDLE = other 32;
- tie break by base_id.

## 8. Diagnostic guards

`G_high_anchor`:
- HIGH -> A;
- otherwise -> F.

`G_low_anchor_control`:
- LOW -> A;
- otherwise -> F.

These are diagnostic counterfactuals only.

No threshold search.

## 9. Required metrics

Per domain/K:
- A/C/F top1/top5/MRR/margins;
- A->C and A->F c->w, w->c, rank/margin delta;
- HIGH/LOW/MIDDLE A/F accuracy and transition rates;
- G_high_anchor / G_low_anchor_control top1/MRR;
- gain vs A/F;
- Spearman(anchor margin, A-correct);
- Spearman(anchor margin, F-minus-A correctness delta);
- state encodes/base;
- F probability mass error.

Reference:
- exact frozen MiniLM from W10/W11;
- pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- expected weight SHA `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`;
- adequacy only, never HIRA scoring.

## 10. Frozen domain classifiers

### CONFIDENCE_GATED_RESIDUAL_SIGNAL

Require:
- R0 K4 >=.75;
- A K4 >=.70;
- A K16 >=.45;
- HIGH A >=.80 K4 and >=.65 K16;
- HIGH F <= A-.10 K4 and <= A-.08 K16;
- pooled HIGH K4+K16 c->w - w->c >=.08;
- LOW F no worse than A by >.03 at K4/K16;
- G_high_anchor >= F+.05 K4 and >= F+.04 K16;
- G_high_anchor beats G_low_anchor_control by >=.03 K4 or K16.

### GLOBAL_ANCHOR_DOMINANCE

Require:
- adequacy passes;
- A >= F+.08 at K4 and K16;
- production is >=.05 worse than A in both HIGH and LOW for at least one primary K;
- confidence-gated rule does not pass.

### COMPETITIVE_RESIDUAL_VALUE

Require:
- adequacy passes;
- F >= A+.05 at K4 or K16;
- LOW w->c - c->w >=.08 pooled K4+K16;
- HIGH F regression <.05 at both K4/K16.

### ANCHOR_MARGIN_UNRELIABLE

Require:
- adequacy passes;
- HIGH A - LOW A <.15 at both K4 and K16.

Else:
`ANCHOR_RESIDUAL_UNRESOLVED`.

## 11. Cross-domain outcome

Same non-unresolved class on >=3/4:
`STABLE_ANCHOR_RESIDUAL_LOCALIZATION`.

Two incompatible classes each on >=2:
`MIXED_ANCHOR_RESIDUAL_LOCALIZATION`.

Else:
`ANCHOR_RESIDUAL_UNRESOLVED`.

No post-exposure threshold change.

## 12. Authorization boundary

If stable `CONFIDENCE_GATED_RESIDUAL_SIGNAL`:
- W13 may test one bounded/gated residual mechanism;
- fresh TRAIN/DEV/dual CONFIRM;
- anchor-only and production-only controls;
- no prior-domain reuse.

If stable `GLOBAL_ANCHOR_DOMINANCE`:
- W13 must redesign production around preserved anchor, not merely add a router.

If stable `COMPETITIVE_RESIDUAL_VALUE`:
- retain competitive corrections and investigate why earlier domains lost anchor geometry.

If mixed/unresolved/margin unreliable:
- no rescue training.

## 13. Permanent forbidden evidence

Never use as W12 fresh data or future tuning:
- all W6b-W11 exposed authority rows;
- W8 AU/AV/AW/AX;
- W9 AY/AZ/BA/BB/BC/BD/BE;
- W10 BF/BG/BH/BI;
- W11 BJ/BK/BL/BM;
- Banking77 0–799;
- typed final/test;
- public campaign cells.

## 14. Current state

Completed:
- W11 closure merged main `bcaaf006e3fcce054de62f5ed1276d468982c3d2`;
- issue #111 preregistered;
- W12 branch created from exact post-W11 main;
- this handoff created before any W12 empirical exposure.

No BN/BO/BP/BQ A13/reference cache exists.
No W12 empirical result exists.

Remaining:
1. implement fresh BN-BQ authority generator;
2. implement anchor/residual classifier library;
3. implement state-once cache;
4. implement A/C/F evaluator and confidence strata;
5. implement counterfactual guards;
6. implement exact pinned reference evaluator;
7. add unit contracts;
8. add pre-data unit workflow;
9. run unit/CI;
10. only then add/enable authority;
11. freeze exact authority result here before merge.

## 15. Handoff discipline

At every meaningful session update record:
- exact branch head;
- exact runs;
- artifacts/digests;
- exposure status;
- exact classification if exposed;
- completed/remaining work;
- forbidden next moves.

A future AI must be able to continue without chat memory.


## 16. Pre-data implementation update

Implemented before any BN/BO/BP/BQ empirical exposure:
- `src/nmd/anchor_residual.py`
  - label-free normalized anchor margin;
  - deterministic per-domain/per-K 16/32/16 quartiles;
  - frozen domain classifiers;
  - frozen cross-domain outcome;
  - Spearman helper for diagnostic correlation.
- `src/nmd/anchor_residual_authority.py`
  - BN/BO/BP/BQ fresh domain generator;
  - 16 latent intents/domain;
  - 4 state variants/intent;
  - paired natural-definition / terse-label views;
  - nested K4/K8/K16 identity.
- `src/nmd/anchor_residual_cache.py`
  - state-once cached A13/schema representation;
  - W12-specific schema identity;
  - paired view/nested-K validation.
- `src/nmd/anchor_residual_eval.py`
  - A semantic anchor;
  - C actual production CompetitiveCoarseScorer;
  - F frozen HIRACore final;
  - HIGH/LOW/MIDDLE transitions;
  - G_high_anchor / G_low_anchor_control;
  - confidence/correctness correlations.
- `scripts/r8_w12_build_cache.py`
  - exact frozen A13 verification;
  - exact-text freshness firewall including W11 atoms.
- `scripts/r8_w12_evaluate.py`
  - exact W9 projection/HIRA provenance;
  - pinned MiniLM adequacy reference;
  - frozen W12 classifications/outcome.
- `tests/test_anchor_residual.py`.
- `.github/workflows/r8-w12-unit.yml`.
- `research/R8-W12-PRIOR-ART.md`.

Infrastructure bugs caught before authority:
1. definition-only guard summaries originally requested absent label rows; fixed before exposure.
2. generated cache-builder source contained one escaped newline token; fixed before exposure.

No A13 W12 cache exists.
No MiniLM W12 reference scores exist.
No W12 classification exists.

Current pre-data branch head at this update:
`26758fce46efec42529515e49956b097fab37bf5`.

Pre-data runs in progress at this update:
- W12 unit run `36134088126` on code head `80a226be3e59ce3d3d2bf115af66be57b4fc2bb2`;
- repository CI run `36134194851` on latest head `26758fce46efec42529515e49956b097fab37bf5`.

Authority workflow remains intentionally absent until pre-data validation is green.


## 17. First authority invalidation and post-exposure infrastructure boundary

First authority attempt:
- run `36134781906`;
- exact head `0244954b657a32a4a4722d5a5e74cfc24d611d56`.

Authority progression:
- unit: PASS;
- upstream W9 provenance: PASS;
- cache: PASS;
- evaluate: FAIL before audit/classification emission.

Exposure occurred:
- marker `R8_W12_BN_BQ_A13_EXPOSURE_BEGIN`;
- BN/BO/BP/BQ A13 cache was materialized;
- 256 bases / 1,536 views;
- state encode calls = 256;
- prior exact-text overlap = [];
- training_performed = false;
- cache artifact `10863541450`;
- artifact digest `sha256:b06a09e02db71bfa73c4fcf88e0b4d4bb9edc5bdb7c075e240a4e79079e00a86`;
- internal cache SHA `9eab23c3b448a46f83bc4071c28a7973d995cdb85c265688cd74779c4935b022`.

The evaluate failure was a pure summary-plumbing exception:

`NameError: name 'views' is not defined`

inside `_summary(...)` while summarizing definition-only guard records.

No authoritative metrics, classifications, stable target, or outcome were emitted before the crash.

Post-exposure repair boundary:
- only `src/nmd/anchor_residual_eval.py` summary API and `tests/test_anchor_residual.py` were changed;
- no A/C/F scorer equations changed;
- no semantic projection/HIRACore weights changed;
- no confidence equation/quartile rule changed;
- no MiniLM reference changed;
- no classifier threshold/precedence changed;
- no fresh-domain text/IDs/seeds changed;
- no gold labels were used to design the repair.

Diff from authority head `0244954...` to repaired head `d5dc3d3...` is limited to:
1. adding an explicit `views` argument to the generic summary helper;
2. calling guard summaries with `views=("definition",)`;
3. adding a unit contract for the definition-only summary.

Repaired head:
`d5dc3d3b7e1436845b29bf68fe47651aea27548d`.

Validation on repaired head:
- W12 unit run `36135836895`: PASS;
- repository CI run `36135841981`: PASS.

Scientific status:
- BN/BO/BP/BQ are already exposed and remain permanently forbidden for any scientific tuning;
- first authority run is invalid as a verdict-bearing authority;
- a rerun is allowed only as an exact-protocol infrastructure recovery;
- no scientific mutation may occur before or during the rerun;
- the rerun must preserve every frozen gate and use the exact deterministic BN/BO/BP/BQ generator.

No W12 scientific verdict exists yet.


---

## 18. Authoritative W12 recovery closure

Valid authority head:
`87af9fecb9a340dd868d085f4c97fc8026a3a4da`

Valid authority run:
`36137418586`

All authority jobs PASS:
- unit: PASS;
- exact W9 upstream provenance: PASS;
- deterministic BN/BO/BP/BQ cache: PASS;
- frozen evaluator/reference/classifier: PASS.

This is the valid verdict-bearing authority after the first run's summary-only infrastructure failure documented above.

### Artifacts

Frozen W9 checkpoint bundle:
- artifact `10865053462`;
- digest `sha256:e8b182f27559b69f278db496a73e11c45e22ad4cdcf5f494294ad6ab4808d947`.

Fresh W12 cache:
- artifact `10864518393`;
- digest `sha256:35c39ebb242c2ab7baeea300505a0568efb8737f9e85ee225ca9354e8c6eb7c2`.

Authoritative W12 audit:
- artifact `10864783524`;
- digest `sha256:01a0a96d92a2e6282282ccf897ff007c6b92bde687bf3dcba4dadc61452e2e88`.

Integrity:
- 256 bases;
- 1,536 paired views;
- state encodes/base = 1.0;
- probability mass max error = `1.7912691419041948e-07`;
- no training;
- no W8/W9/W10/W11 rows;
- no Banking77 rows;
- no typed final/test rows;
- campaign cells = 0;
- exact MiniLM reference revision/SHA verified.

### Frozen overall outcome

**`ANCHOR_RESIDUAL_UNRESOLVED`**

Stable classification:
`null`.

Classification counts:
- `GLOBAL_ANCHOR_DOMINANCE`: 2;
- unresolved: 2.

The frozen >=3/4 cross-domain stability boundary is not met.

No W13 rescue mechanism is authorized from W12.

## 19. Per-domain frozen classifications

### BN — public library account services

Classification:
**`GLOBAL_ANCHOR_DOMINANCE`**

Adequacy:
- R0 MiniLM K4: 96.875%;
- anchor A K4: 89.063%;
- anchor A K16: 68.750%.

Production final F:
- K4: 57.813%;
- K16: 35.938%.

Anchor loss through production:
- K4: -31.25 pp;
- K16: -32.81 pp.

HIGH-confidence anchor stratum:
- A K4 100.0% -> F 62.5%;
- A K16 87.5% -> F 56.25%;
- pooled K4+K16 c->w 37.5%;
- w->c 3.125%.

LOW stratum:
- A K4 75.0% -> F 62.5%;
- A K16 18.75% -> F 12.5%.

Anchor margin has useful correctness correlation:
- Spearman(margin, A-correct) = 0.4683.

However production is not merely selectively harmful to HIGH; it is broadly worse, satisfying `GLOBAL_ANCHOR_DOMINANCE`.

### BO — vehicle inspection administration

Classification:
**`GLOBAL_ANCHOR_DOMINANCE`**

Adequacy:
- R0 K4: 100.0%;
- A K4: 84.375%;
- A K16: 59.375%.

F:
- K4: 67.188%;
- K16: 40.625%.

HIGH:
- A K4 93.75% -> F 81.25%;
- A K16 75.0% -> F 68.75%.

LOW:
- A K4 68.75% -> F 62.5%;
- A K16 31.25% -> F 18.75%.

Spearman(margin, A-correct) = 0.3800.

Again, the evidence supports broad anchor superiority rather than a stable high-confidence-only routing law.

### BP — workplace benefits administration

Classification:
**`ANCHOR_RESIDUAL_UNRESOLVED`**

Adequacy:
- R0 K4: 96.875%;
- A K4: 76.563%;
- A K16: 56.250%.

F:
- K4: 73.438%;
- K16: 40.625%.

HIGH:
- A K4 93.75% -> F 93.75%;
- A K16 81.25% -> F 50.0%.

LOW:
- A K4 62.5% -> F 62.5%;
- A K16 25.0% -> F 37.5%.

LOW pooled K4+K16 transitions:
- c->w 21.875%;
- w->c 28.125%.

This domain contains genuine low-confidence residual value at K16 while simultaneously damaging HIGH K16, but it does not satisfy the frozen cross-K/domain rule for `CONFIDENCE_GATED_RESIDUAL_SIGNAL`.

### BQ — veterinary appointment support

Classification:
**`ANCHOR_RESIDUAL_UNRESOLVED`**

Adequacy:
- R0 K4: 98.438%;
- A K4: 73.438%;
- A K16: 57.813%.

F:
- K4: 67.188%;
- K16: 45.313%.

HIGH:
- A K4 75.0% -> F 68.75%;
- A K16 62.5% -> F 62.5%.

LOW:
- A K4 62.5% -> F 56.25%;
- A K16 37.5% -> F 25.0%.

The preregistered HIGH adequacy/damage conditions and conditional residual-value rules do not hold.

## 20. Pooled anatomy

Frozen MiniLM reference:
- K4 **98.05%**;
- K8 **92.97%**;
- K16 **87.11%**.

Semantic anchor A:
- K4 **80.86%**;
- K8 **71.09%**;
- K16 **60.55%**.

Production coarse C:
- K4 **66.02%**;
- K8 **50.78%**;
- K16 **41.41%**.

Production final F:
- K4 **66.41%**;
- K8 **53.52%**;
- K16 **40.63%**.

Thus A -> F pooled:
- K4: **-14.45 pp**;
- K8: **-17.58 pp**;
- K16: **-19.92 pp**.

Counterfactual `G_high_anchor`:
- K4 69.92%;
- K8 57.03%;
- K16 44.92%.

Counterfactual `G_low_anchor_control`:
- K4 67.97%;
- K8 54.30%;
- K16 41.80%.

The HIGH-anchor guard is better than final production and slightly better than the LOW-anchor control, but it remains far below anchor-only A:
- A K4 80.86% vs guard 69.92%;
- A K16 60.55% vs guard 44.92%.

Therefore a quartile router does not preserve enough of the semantic anchor to qualify as a stable architecture rule.

## 21. Scientific interpretation

W12 confirms a higher-level pattern that W10/W11 repeatedly hinted at:

**the candidate-independent semantic anchor is often substantially stronger than the full competitive production path.**

But W12 falsifies the specific stronger hypothesis that the preregistered normalized-margin quartiles yield one stable confidence-gated residual law across fresh domains.

Evidence is heterogeneous:
- BN/BO: production competition is broadly harmful -> `GLOBAL_ANCHOR_DOMINANCE`;
- BP: production can help some low-confidence K16 cases while damaging high-confidence K16;
- BQ: neither broad dominance nor a clean conditional residual rule is strong enough.

Therefore:
- do not build a W13 learned router from these quartiles;
- do not globally delete the competitive path based only on BN/BO;
- do not reinterpret BP as proof of confidence gating;
- do not tune quartile thresholds/margin formula on exposed BN/BO/BP/BQ.

The robust result across W10-W12 is weaker but important:

> candidate-independent semantic geometry deserves preservation as a first-class signal; the current competitive production path can destroy it substantially, and the damage/value tradeoff is domain- and confidence-structure-dependent.

The unresolved piece is how to estimate **when** competition is trustworthy using an invariant that transfers across domains.

## 22. Permanent exposed evidence after W12

BN/BO/BP/BQ are permanently exposed.

Never use them for:
- training;
- router/gate fitting;
- confidence-threshold selection;
- confidence-formula search;
- production/anchor mixing weights;
- candidate selection;
- calibration;
- seed choice;
- architecture selection.

The first invalid authority cache and the valid recovery authority both count as exposure. All prior forbidden evidence remains forbidden.

## 23. Authorized next research boundary

Because W12 is unresolved:

**No rescue training is authorized.**

The next phase must be a fresh diagnostic/prior-art reassessment of confidence and semantic evidence quality.

A valid continuation should test a higher-level reliability representation without fitting on W12, for example:
- semantic agreement across independent candidate-independent views rather than one margin;
- local neighborhood consistency / rank stability under harmless schema paraphrases;
- disagreement between semantic anchor and competitive path as an uncertainty signal;
- score-distribution geometry beyond a single top1-top2/MAD scalar;
- whether anchor-only dominance itself is stable under fresh natural semantic domains when the reference ceiling is strong.

It must not:
- retune W12 quartiles;
- learn a gate on BN/BO/BP/BQ;
- introduce a rescue model before a stable reliability target exists;
- use MiniLM as HIRA supervision/candidate;
- reopen exposed authorities.

A future AI should read W12 first, then W11/W10/W9.
