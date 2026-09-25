# R8-W12 handoff — semantic-anchor / competitive-residual reliability audit

Status: **PRE-DIAGNOSTIC. No BN/BO/BP/BQ A13/reference cache or W12 localization result exists yet.**

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
