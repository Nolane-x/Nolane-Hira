# R8-W14 handoff — continuous multiview semantic-reliability audit

Status: **CLOSED DIAGNOSTIC. Frozen outcome: `STABLE_CONTINUOUS_RELIABILITY_LOCALIZATION`; stable target: `CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE`.**

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


## 20. Full pre-authority execution stack frozen

Completed before any BV/BW/BX/BY empirical exposure:

- state-once cache:
  `src/nmd/continuous_reliability_cache.py`;
- frozen A0/A1/A2/E/F evaluator:
  `src/nmd/continuous_reliability_eval.py`;
- sealed fresh cache builder:
  `scripts/r8_w14_build_cache.py`;
- frozen W9 provenance + MiniLM evaluator:
  `scripts/r8_w14_evaluate.py`;
- cache state-once unit contract;
- complete W14 pre-data compile/test workflow.

Exact validated code head:
`0e9cc13c52d2cdfd9dfdf8126304580b4dec09e0`.

Validation:
- W14 unit run `36143756919`: PASS;
- repository CI run `36143756938`: PASS;
- Python 3.10: PASS;
- Python 3.12: PASS.

Scientific contracts remain exactly those preregistered in issue #115:
- V/S/O definitions unchanged;
- R = equal arithmetic mean unchanged;
- 21/22/21 tertiles unchanged;
- scalar-margin count-matched control unchanged;
- domain seeds/text generation unchanged;
- classifier thresholds/precedence unchanged;
- W9 projection/HIRACore provenance unchanged;
- MiniLM reference unchanged.

Freshness firewall now includes W13 BR/BS/BT/BU atoms.

Exposure state at this freeze:
**No BV/BW/BX/BY A13 cache exists.**
**No W14 MiniLM score exists.**
**No W14 empirical metric/classification exists.**

Authority may be enabled only after the doc-only freeze head is green.


---

## 21. Authoritative W14 closure

Exact empirical head:
`401bcd266fd4a465db8bd492533833db8a6fa76b`

Authority run:
`36144650389`

All authority jobs PASS:
- frozen unit gate;
- exact W9 checkpoint provenance;
- fresh BV/BW/BX/BY state-once cache;
- continuous reliability evaluator;
- pinned MiniLM adequacy reference;
- frozen classifier and cross-domain outcome.

First eligible exposure marker:
`R8_W14_BV_BY_A13_EXPOSURE_BEGIN`

Exposure began during authority run `36144650389`.
BV/BW/BX/BY are permanently exposed from this point onward.

### Artifacts

Frozen W9 checkpoint bundle:
- artifact `10868692915`;
- digest `sha256:009cf70d5592a33dfd9fc2758ce8ec8b4414ed280fd2d8ea30a9a03ff2eb00c8`.

Fresh W14 cache:
- artifact `10868639335`;
- artifact digest `sha256:b59bb98ed8e64d3477e50f209e11aa757a4d46a5b611e618269b0522e1bc064c`;
- internal cache SHA `58eb78e0f50359012c8fc459bc7cdf6b4bf765314e1a06d838bd68703f620be3`;
- all-text-atom SHA `54f1e85315c8593d08c3c2d3ff2f595570a86338c2abda03963cdb692157f474`;
- base-ID SHA `77d6a72818d52d7f02ff06eda2392ab72a76b3d57622c4dcc49424341d415042`;
- case-ID SHA `185b9abb156a0742e48955cff3cca4ff68a00baf536820eb124ccd09b68b5cdd`.

Authoritative W14 audit:
- artifact `10869068957`;
- digest `sha256:782661ec37590ce7fea1f7dd555970967006885a66fe4335424541ba94bd2d65`.

### Integrity

- 256 fresh bases;
- 2,304 paired schema/cardinality views;
- BV/BW/BX/BY 64 bases each;
- state encodes/base = 1.0;
- probability mass max error = `2.141459845006466e-07`;
- no training;
- no W8-W13 exposed rows;
- no Banking77 reuse;
- no typed final/test rows;
- campaign cells = 0;
- exact A13 revision/SHA verified;
- exact W9 projection/HIRACore provenance verified;
- exact pinned MiniLM revision/SHA verified;
- prior exact-text overlap = [].

## 22. Frozen overall outcome

**`STABLE_CONTINUOUS_RELIABILITY_LOCALIZATION`**

Stable classification:

**`CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE`**

Classification counts:
- `CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE`: **4/4**.

This is a stable W14 target.

However, this is **not** evidence that the preregistered continuous reliability score R is a successful router.
The stronger and more stable result is that the candidate-independent multiview semantic ensemble E broadly dominates the production final path F across fresh domains, including both HIGH and LOW reliability regions.

## 23. Pooled W14 anatomy

Frozen multiview semantic ensemble E:
- K4 **94.922%**;
- K8 **90.234%**;
- K16 **84.766%**.

Current production final F:
- K4 **66.797%**;
- K8 **53.906%**;
- K16 **40.234%**.

Thus E -> F pooled degradation:
- K4 **-28.125 pp**;
- K8 **-36.328 pp**;
- K16 **-44.531 pp**.

Pinned MiniLM reference R0:
- K4 **95.313%**;
- K8 **87.891%**;
- K16 **86.719%**.

A striking result:
- E K4 94.922% vs R0 95.313%;
- E K8 90.234% vs R0 87.891%;
- E K16 84.766% vs R0 86.719%.

On this fresh authority, HIRA's frozen candidate-independent multiview semantic representation is close to the pinned semantic reference at K4/K16 and slightly above it at K8.

This does not establish broad external superiority; MiniLM is only a frozen diagnostic reference and the authority is internal synthetic/naturalistic fresh data.

### Diagnostic guards

`G_consistency_high`:
- K4 75.000%;
- K8 65.625%;
- K16 52.344%.

Equal-coverage scalar-margin control `G_margin_high`:
- K4 75.000%;
- K8 64.063%;
- K16 53.516%.

Descriptive `G_consistency_nonlow`:
- K4 **85.156%**;
- K8 **78.125%**;
- K16 **69.141%**.

The HIGH-only continuous consistency guard does **not** stably beat the scalar-margin control:
- K4 tie;
- K8 +1.56 pp;
- K16 -1.17 pp.

Therefore the R score is not promoted as a superior confidence router.

The non-LOW descriptive guard is much stronger, but it remains a post-defined descriptive guard under the frozen protocol and cannot be promoted into a mechanism from W14.

## 24. Per-domain frozen classifications

### BV — municipal waste-collection services

Classification:
**`CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE`**

Adequacy:
- E K4 96.875%;
- E K16 87.500%.

Production:
- F K4 67.188%;
- F K16 42.188%.

HIGH:
- E K4 100% -> F 76.19%;
- E K16 90.48% -> F 52.38%.

LOW:
- E K4 90.48% -> F 66.67%;
- E K16 85.71% -> F 47.62%.

LOW pooled K4+K16 transition:
- E->F c->w 35.71%;
- w->c 4.76%.

R correctness correlation:
- K4 0.2629;
- K16 0.1049.

The production loss is broad, not confined to LOW reliability.

### BW — university transcript administration

Classification:
**`CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE`**

- E K4 96.875%;
- E K16 78.125%;
- F K4 76.563%;
- F K16 46.875%.

HIGH:
- E K4 100% -> F 76.19%;
- E K16 90.48% -> F 47.62%.

LOW:
- E K4 95.24% -> F 66.67%;
- E K16 52.38% -> F 42.86%.

R correlation:
- K4 0.2003;
- K16 0.4297.

Even where R has useful K16 ordering, F remains materially below E in both HIGH and LOW regions.

### BX — home internet account support

Classification:
**`CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE`**

- E K4 96.875%;
- E K16 90.625%;
- F K4 65.625%;
- F K16 37.500%.

HIGH:
- E K4 95.24% -> F 71.43%;
- E K16 85.71% -> F 42.86%.

LOW:
- E K4 95.24% -> F 57.14%;
- E K16 95.24% -> F 33.33%.

R is actually weakly anti-correlated with E correctness here:
- K4 -0.0491;
- K16 -0.0987.

This directly falsifies a universal interpretation of R as the correct reliability coordinate, while strongly supporting broad anchor dominance.

### BY — community recreation membership services

Classification:
**`CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE`**

- E K4 89.063%;
- E K16 82.813%;
- F K4 57.813%;
- F K16 34.375%.

HIGH:
- E K4 100% -> F 71.43%;
- E K16 85.71% -> F 61.90%.

LOW:
- E K4 85.71% -> F 57.14%;
- E K16 85.71% -> F **4.76%**.

LOW pooled:
- c->w 57.14%;
- w->c 2.38%.

This is especially strong evidence that production competition can catastrophically destroy already-good candidate-independent semantics even in cases classified as lower R.

## 25. Scientific interpretation

W14 materially changes the architecture target.

W12 asked whether one scalar margin could tell us when production should override the anchor: unresolved.

W13 asked whether exact 3/3 paraphrase agreement could define a reliable region: descriptively strong, but support failed the frozen K16 coverage gate.

W14 removed the coverage problem by using a continuous full-coverage score.

The result is not a successful router.

Instead, the more stable fact is stronger:

> **Across four wholly fresh domains, the multiview candidate-independent semantic anchor E is broadly superior to the existing competitive production final path F, including in both HIGH and LOW continuous-consistency tertiles.**

This reaches the frozen 4/4 stability criterion.

Therefore the next architecture question is no longer primarily “when should we trust competition?”

It is:

> **How should HIRA preserve the multiview candidate-independent semantic anchor as the primary decision signal, while allowing typed/relation reasoning to contribute only as a bounded non-destructive residual?**

This is an anchor-preserving production redesign problem.

### What W14 supports

- candidate-independent semantic evidence is now a stable first-class architecture target;
- three-view semantic aggregation can be dramatically stronger than the current production interface;
- current competitive production transforms can destroy correct semantic ranking broadly;
- this is reproducible on 4/4 fresh W14 domains.

### What W14 does not support

- the frozen R score as a stable router;
- learning a confidence gate from BV/BW/BX/BY;
- deleting all typed/relation reasoning;
- promoting E alone to final production;
- using three paraphrases as an unquestioned production requirement;
- broad external superiority claims.

HIRACore/relation remained historically net-positive in W6f/W11 after coarse scoring. The redesign should preserve useful typed reasoning while preventing destructive coarse replacement.

## 26. Permanent forbidden evidence after W14

BV/BW/BX/BY are permanently exposed.

Never use them for:
- training;
- architecture selection;
- mixing-weight selection;
- residual bound tuning;
- paraphrase-count selection;
- reliability-feature tuning;
- threshold/tertile tuning;
- checkpoint/seed selection;
- calibration.

All previous exposed authorities remain forbidden.

## 27. Authorized W15 boundary

W14's stable target authorizes a fresh **anchor-preserving production redesign** experiment.

W15 must **not** be a confidence router.

A valid W15 should compare, on wholly fresh TRAIN/DEV/dual-CONFIRM data:

1. current production control F;
2. multiview semantic anchor control E;
3. an anchor-preserving residual architecture where:
   - E is the primary coarse semantic score;
   - typed/relation or competitive evidence may add a bounded residual;
   - the residual cannot arbitrarily replace/reorder strong semantic geometry;
   - no W14 data selects residual scale/bounds;
4. equal-data/equal-step controls;
5. frozen A13;
6. preferably frozen HIRACore in the primary architecture comparison unless the experiment explicitly preregisters otherwise.

W15 must preregister:
- exact residual equation;
- parameter budget;
- whether paraphrases are runtime schema inputs or compressed into schema-side cached representations;
- fresh domains/seeds;
- TRAIN/DEV/dual CONFIRM;
- K4/K8/K16 and high-K follow-up;
- semantic retention gates;
- typed primitive gates;
- external/public validation boundary.

Do not use BV/BW/BX/BY to choose any W15 hyperparameter.

A future AI should read W14 first, then W13/W12/W11/W10.
