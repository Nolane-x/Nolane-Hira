# R8-W14 handoff — continuous multiview semantic-reliability audit

Status: **PRE-DIAGNOSTIC. No BV/BW/BX/BY A13/reference cache or W14 classification exists yet.**

Issue: #115

Branch:
`feat/r8-w14-continuous-reliability`

Base main:
`b09b708ce9d7d5468fcef11d9e0b52645c1d8d4c`

Read first when restoring:
1. this file;
2. `research/R8-W13-HANDOFF.md`;
3. `research/R8-W12-HANDOFF.md`;
4. `research/R8-W11-HANDOFF.md`;
5. `research/R8-W10-HANDOFF.md`;
6. issue #115.

## 0. Project identity

Nolane HIRA is a compact non-autoregressive typed decision engine, not a next-token LLM.

Long-term target:
- state encoded once;
- dynamic semantic schemas;
- changing/high-cardinality candidate sets;
- typed choice / score / noul;
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
- freeze run `36122220588`;
- artifact `10858424139`.

Frozen HIRACore:
- SHA `d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588`.

No W14 parameter is trainable.

## 2. Decisive history entering W14

W5:
- simple pooled/bilinear probes failed;
- late interaction + competitive salience became production semantic scorer.

W6-W6e:
- typed integration/reliability became strong internally;
- held-out high-K generalization remained unstable.

W6f-W6i:
- salience removal, relation blame, local residual adapters and naive factorization did not solve the problem.

W6j:
- stable synthetic `COARSE_CONJUNCTION_LIMIT`.

W7:
- explicit factor smooth-AND failed;
- equal-data free-form retune was much stronger.

W7b:
- strong synthetic free-form retune but preregistered replication failed.

W7c:
- Banking77 frozen transfer essentially absent.

W8:
- stable `GENERAL_SEMANTIC_TRANSFER_LIMIT` at low K.

W9:
- low-rank semantic bridges failed;
- full projection semantic retune was strongest signal.

W10:
- P1 semantic geometry 79.30% K4 / 54.30% K16;
- production with same projection 60.55% / 32.03%;
- outcome unresolved, with 2/4 scoring-interface signal.

W11:
- production interface decomposed;
- domain-dependent losses at question context/common-mode/salient-min;
- no stable micro-stage culprit.

W12:
- anchor A 80.86% K4 / 60.55% K16;
- production F 66.41% / 40.63%;
- normalized-margin quartile routing unresolved.

W13:
- exact head `a47f368d57de811f1b8f38beff39db973b9ef89b`;
- authority run `36141078081`;
- frozen outcome `SEMANTIC_CONSISTENCY_UNRESOLVED`;
- E 85.16% K4 / 76.17% K8 / 63.28% K16;
- F 57.42% / 40.63% / 28.52%;
- STRICT semantic correctness often ~93-100%;
- but BR/BS/BT STRICT K16 coverage was only 23.44/21.88/20.31%, below frozen 25%;
- BU alone classified `MULTIVIEW_ANCHOR_DOMINANCE`;
- STRICT guard beat count-matched scalar-margin control pooled.

Therefore W14 does not lower W13 coverage gates and does not train a router.

## 3. W14 purpose

Test whether a continuous, full-coverage, label-free multiview consistency score can stably order semantic reliability across fresh domains.

Core question:

**Can semantic consistency provide a transferable reliability ordering over most cases, not merely a highly accurate low-coverage STRICT subset?**

## 4. Fresh W14 domains

- BV municipal waste-collection services — seed 331101;
- BW university transcript administration — seed 331111;
- BX home internet account support — seed 331123;
- BY community recreation membership services — seed 331133.

Each:
- 16 latent intents;
- 4 state variants/intent;
- 3 independent meaning-equivalent natural definitions D0/D1/D2;
- 64 bases.

Total:
- 256 bases;
- 2,304 schema/cardinality views.

Nested K4/K8/K16.

BV/BW/BX/BY become permanently exposed after first eligible authority.

## 5. Frozen semantic operators

A0/A1/A2:
- exact W13 candidate-independent semantic anchor with each paraphrase;
- W9 semantic projection;
- bidirectional symmetric token MaxSim.

E:
- arithmetic mean of A0/A1/A2 candidate scores.

F:
- exact D0 current production path using W9 projection + CompetitiveCoarseScorer + frozen HIRACore.

## 6. Continuous reliability features

For one base/K:

### V — vote strength

`V = (vmax - 1) / 2`

where vmax is maximum top1 vote count among A0/A1/A2.

Range:
- 0 split;
- .5 majority;
- 1 strict.

### S — full-rank stability

Three pairwise Spearman correlations over candidate order.

`S = clip((mean_rho + 1)/2, 0, 1)`.

### O — top3 overlap

Three pairwise Jaccard similarities among top min(3,K) candidate-ID sets.

`O = mean(pairwise_jaccard)`.

### R — frozen reliability

`R = (V + S + O) / 3`.

No learned weights.
No labels.
No post-exposure changes.

## 7. Fixed tertiles

Within each domain/K independently, rank descending by R:
- HIGH = first 21;
- MIDDLE = next 22;
- LOW = last 21;
- ties by base_id.

Full coverage is guaranteed.

## 8. Scalar-margin control

For each domain/K:
- exact W12 normalized A0 top1-top2/MAD margin;
- choose top 21 as HIGH_MARGIN;
- tie by base_id.

## 9. Frozen guards

`G_consistency_high`:
HIGH -> E, otherwise F.

`G_margin_high`:
HIGH_MARGIN -> E, otherwise F.

`G_consistency_nonlow`:
HIGH/MIDDLE -> E, LOW -> F.
Descriptive only.

## 10. Reference adequacy

Pinned MiniLM:
- `sentence-transformers/all-MiniLM-L6-v2`;
- revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- weight SHA `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.

Reference is adequacy only and never enters HIRA or R.

## 11. Frozen adequacy

Per domain:
- R0 K4 >=.75;
- E K4 >=.70;
- E K16 >=.45.

No consistency-coverage gate because tertiles have fixed support.

## 12. Frozen classes

### CONTINUOUS_CONSISTENCY_RELIABLE

Require:
- adequacy;
- HIGH E >=.85 K4 and >=.70 K16;
- HIGH E - LOW E >=.15 at K4 and K16;
- Spearman(R,E-correct) >=.25 K4 and >=.20 K16;
- HIGH F <= E-.10 K4 and <= E-.08 K16;
- LOW F >= E-.03 K4 and K16;
- G_consistency_high >= F+.05 K4 and >= F+.04 K16;
- G_consistency_high >= G_margin_high+.03 at K4 or K16.

### CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE

Require:
- adequacy;
- E >= F+.08 K4/K16;
- F <= E-.05 in both HIGH and LOW at K4 or K16;
- reliable class above does not pass.

### PRODUCTION_VALUE_LOW_RELIABILITY

Require:
- adequacy;
- LOW pooled K4+K16 w->c - c->w >=.08 for E->F;
- HIGH F regression <.05 at K4/K16;
- F >= E+.05 at K4 or K16 OR G_consistency_high >= E-.03 at both K.

### CONTINUOUS_CONSISTENCY_UNINFORMATIVE

Require:
- adequacy;
- HIGH-LOW E gap <.10 at K4/K16;
- abs Spearman(R,E-correct)<.15 at K4/K16.

Else:
`CONTINUOUS_RELIABILITY_UNRESOLVED`.

## 13. Cross-domain outcome

Same non-unresolved class >=3/4:
`STABLE_CONTINUOUS_RELIABILITY_LOCALIZATION`.

Two incompatible non-unresolved classes >=2 each:
`MIXED_CONTINUOUS_RELIABILITY_LOCALIZATION`.

Else:
`CONTINUOUS_RELIABILITY_UNRESOLVED`.

No threshold/weight/binning change after exposure.

## 14. Authorization boundary

W14 is diagnostic only.

Stable continuous consistency:
- W15 may test one tiny bounded reliability-gated anchor/residual mechanism;
- fresh TRAIN/DEV/dual-CONFIRM;
- E-only/F-only/scalar-margin/consistency controls.

Stable anchor dominance:
- W15 must test anchor-preserving production redesign, not a router.

Stable low-reliability production value:
- W15 may retain production only as bounded low-reliability correction.

Unresolved/mixed/uninformative:
- no rescue training.

## 15. Permanent forbidden evidence

Never reuse:
- W6b-W13 exposed rows;
- W12 BN/BO/BP/BQ;
- W13 BR/BS/BT/BU;
- Banking77 0-799;
- typed final/test;
- public campaign cells.

## 16. Claim discipline

Do not:
- tune R weights;
- optimize tertile thresholds;
- add/drop features after exposure;
- use labels in R;
- tune paraphrases after exposure;
- tune W9 projection/production/HIRACore;
- use MiniLM inside HIRA;
- promote pooled-only evidence;
- claim rescue from diagnostic guards.

## 17. Current state

Completed:
- W13 authoritative unresolved closure merged main `b09b708ce9d7d5468fcef11d9e0b52645c1d8d4c`;
- issue #115 preregistered;
- W14 branch created from exact post-W13 main;
- this handoff created before any W14 empirical exposure.

No BV/BW/BX/BY cache exists.
No W14 empirical result exists.

Remaining:
1. implement continuous reliability core V/S/O/R and tertiles;
2. implement fresh BV-BY generator;
3. implement state-once cache;
4. implement A0/A1/A2/E/F evaluator;
5. implement count-matched margin control and guards;
6. implement frozen classifiers;
7. implement pinned reference;
8. add unit contracts and pre-data workflow;
9. run unit/CI;
10. only then enable authority;
11. freeze exact result here before merge.

## 18. Handoff discipline

Every meaningful session must update:
- exact branch head;
- exact runs/artifacts;
- exposure state;
- completed/remaining work;
- exact scientific mutations before exposure;
- exact metrics/verdict after exposure;
- forbidden next moves.

A future AI must be able to continue without chat memory.


## 19. Pre-data implementation update

Implemented before any BV/BW/BX/BY empirical exposure:

- `src/nmd/continuous_reliability.py`
  - frozen V/S/O/R equations;
  - deterministic full-coverage 21/22/21 tertiles;
  - count-matched scalar-margin control;
  - frozen domain classifications;
  - frozen cross-domain outcomes.

- `src/nmd/continuous_reliability_authority.py`
  - fresh BV/BW/BX/BY generator;
  - 16 intents/domain;
  - 4 state variants/intent;
  - three independent D0/D1/D2 natural definitions;
  - paired nested K4/K8/K16 identity;
  - deterministic candidate membership/order.

- `tests/test_continuous_reliability.py`
  - generator shape/identity;
  - V/S/O/R contracts;
  - deterministic tertiles/margin control;
  - classifier semantics;
  - cross-domain stability semantics.

- `.github/workflows/r8-w14-unit.yml`
  - branch-scoped compile/test gate;
  - no cache construction;
  - no MiniLM;
  - no empirical authority.

- `research/R8-W14-PRIOR-ART.md`
  - frozen pre-exposure prior-art boundary;
  - explicitly disclaims novelty for paraphrase consistency, self-consistency, rank stability and uncertainty aggregation.

Current branch head at this update:
`657fc87be8ebb7780e49fa48adad5417dff73a5f`.

Current validation:
- earlier handoff-only CI run `36142990256`: PASS;
- exact-head W14 unit `36143367092`: queued at this update;
- exact-head repository CI `36143366824`: queued at this update.

Exposure state:
**No BV/BW/BX/BY A13 cache exists.**
**No W14 MiniLM score exists.**
**No W14 empirical metric/classification exists.**

Authority workflow does not exist and must remain absent until:
1. exact-head W14 unit passes;
2. exact-head repository CI passes;
3. state-once cache/evaluator/reference stack is implemented and independently unit-tested;
4. this handoff is updated with the exact pre-authority head.
