# R8-W15 handoff — anchor-preserving bounded residual production redesign

Status: **PRE-DATA MECHANISM AUTHORITY. No BZ/CA/CB/CC/CD/CE/CF A13 cache, training result, DEV result or CONFIRM result exists yet.**

Issue: #117

Branch:
`feat/r8-w15-anchor-preserving-residual`

Base main:
`cbed464e16f9f44c8ddf0f46a257db47f5ff6595`

Read first when restoring:
1. this file;
2. `research/R8-W14-HANDOFF.md`;
3. `research/R8-W13-HANDOFF.md`;
4. `research/R8-W12-HANDOFF.md`;
5. `research/R8-W11-HANDOFF.md`;
6. `research/R8-W10-HANDOFF.md`;
7. issue #117.

---

## 0. Project identity

Nolane HIRA is a compact non-autoregressive typed decision engine, not a next-token LLM.

Long-term target:
- encode state exactly once/case;
- support dynamic semantic schemas and changing candidate sets;
- typed choice / score / noul primitives;
- strong semantic transfer to unseen schema categories;
- robust high-cardinality decisions;
- calibrated/reliable probabilities;
- small local-friendly trainable footprint;
- sealed fresh authorities and immutable negative results;
- eventual external/public validation only after internal fresh replication.

The research standard is falsification-first:
- frozen gates never move after exposure;
- PARTIAL/FAIL is never promoted because it is close;
- exposed CONFIRM/diagnostic rows are permanently forbidden;
- internal synthetic/naturalistic authority is never called broad external superiority.

---

## 1. Frozen encoder/runtime

A13:
- model: `microsoft/xtremedistil-l6-h256-uncased`;
- revision: `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA256: `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length: 256;
- fully frozen.

W9 semantic projection / scorer source:
- `CompetitiveCoarseScorer` 256 -> 128 bias-free projection + scalar scale;
- source scorer SHA: `8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102`;
- selected W9 DEV epoch: 5;
- W9 authority run: `36122220588`;
- W9 freeze artifact: `10858424139`.

Frozen HIRACore:
- 422,159 params;
- SHA: `d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588`.

W15 primary comparison freezes:
- A13;
- W9 projection;
- production competitive scorer weights;
- HIRACore.

Only six residual scale parameters are trainable in each trainable W15 candidate.

---

## 2. Decisive research history

### W5

Simple pooled/bilinear/capacity probes did not solve semantic binding.

Late interaction + candidate-relative salience + soft sibling competition established the production semantic scorer.

W5i showed a fresh forward competitive control itself could strongly rescue its internal authority. The listwise mechanism was not proven causal.

### W6 / W6b

Production typed integration succeeded and preserved state-once execution.

W6b showed strong internal typed competence but only partial reliability/generalization.

### W6c

Tiny temperature calibration genuinely improved confidence calibration but did not repair hard semantic generalization.

Conclusion:
calibration and semantic competence are separate.

### W6d / W6e

Held-out fresh-domain K64 generalization did not reproducibly rescue.

W6e falsified the idea that full-core joint adaptation alone solved the problem.

### W6f

Candidate-relative salience removal was dramatically worse.

Relation reranking was net helpful.

Therefore salience deletion and relation deletion were not justified.

### W6g / W6h / W6i

W6g localized fresh-domain field-semantic weakness.

W6h's local residual adapter failed reproducibility.

W6i showed naive canonical/factorized representation bridges were worse and left the representation bridge unresolved.

### W6j

Stable internal architecture diagnosis:
`COARSE_CONJUNCTION_LIMIT`.

### W7

Explicit factorized smooth-AND conjunction failed:
`CONJUNCTIVE_COARSE_FAIL`.

Equal-data free-form scorer retuning was much stronger.

### W7b

Free-form retuning was strong on fresh internal synthetic domains but failed one preregistered replication gain gate:
`FREEFORM_RETUNE_NONREPLICATING`.

No gate was relaxed.

### W7c

Frozen public Banking77 transfer was essentially absent:
- frozen 1/400;
- typed-only 1/400;
- primary/replica 0/400;
- retuned checkpoints could be highly confident while wrong.

Verdict:
`PUBLIC_HIGH_K_TRANSFER_ABSENT`.

### W8

Stable low-K fresh semantic localization:
`GENERAL_SEMANTIC_TRANSFER_LIMIT`.

Natural definitions helped terse labels but did not rescue.

### W9

Direct semantic alignment with rank-16 shared/asymmetric residual bridges failed:
`SEMANTIC_ALIGNMENT_FAIL`.

Strongest positive signal:
full 32,768-param projection semantic retuning.

### W10

Representation-ceiling decomposition:
- raw A13 symmetric MaxSim pooled K4/K16: 64.06% / 34.77%;
- frozen W6e projected: 72.27% / 48.44%;
- W9 semantic projected P1: 79.30% / 54.30%;
- same projection through production S1: 60.55% / 32.03%;
- MiniLM reference: 97.66% / 90.63%.

Outcome:
`REPRESENTATION_CEILING_UNRESOLVED`.

This weakened an A13-only ceiling explanation and exposed large production-interface loss.

### W11

Production path Q0->Q7 decomposed.

Fresh domains localized different damaging micro-stages:
- question context;
- common-mode subtraction;
- salient-min;
- unresolved.

Outcome:
`INTERFACE_DECOMPOSITION_UNRESOLVED`.

No single micro-stage culprit.

### W12

Candidate-independent semantic anchor A was materially stronger than production F.

Simple normalized top1-top2/MAD confidence quartiles failed to produce one stable routing law:
`ANCHOR_RESIDUAL_UNRESOLVED`.

BN/BO showed broad anchor dominance; BP/BQ unresolved.

### W13

Three independently worded semantic definitions D0/D1/D2.

Multiview E strongly outperformed production F:
- E 85.16 / 76.17 / 63.28 at K4/K8/K16;
- F 57.42 / 40.63 / 28.52.

STRICT 3/3 top1 agreement was extremely accurate but failed preregistered K16 support/coverage in 3/4 domains.

Outcome:
`SEMANTIC_CONSISTENCY_UNRESOLVED`.

### W14 — decisive authority entering W15

Exact empirical head:
`401bcd266fd4a465db8bd492533833db8a6fa76b`.

Authority run:
`36144650389`.

Merge main:
`cbed464e16f9f44c8ddf0f46a257db47f5ff6595`.

Frozen outcome:
**`STABLE_CONTINUOUS_RELIABILITY_LOCALIZATION`**

Stable target:
**`CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE`**

All four fresh domains BV/BW/BX/BY independently hit the same class.

Pooled:
- multiview E K4: **94.922%**;
- E K8: **90.234%**;
- E K16: **84.766%**;
- production F K4: **66.797%**;
- F K8: **53.906%**;
- F K16: **40.234%**;
- MiniLM reference K4: 95.313%;
- reference K8: 87.891%;
- reference K16: 86.719%.

Thus current production destroyed approximately:
- 28.13 pp at K4;
- 36.33 pp at K8;
- 44.53 pp at K16.

W14's continuous reliability score R was not a superior router.

The stable 4/4 result is broader:

> candidate-independent multiview semantic evidence is a first-class primary signal and the current production path broadly damages it, including both HIGH and LOW reliability regions.

W14 therefore authorizes W15 as an **anchor-preserving production redesign**, not a confidence router.

---

## 3. W15 hypothesis

The current production path treats competitive scoring as the coarse authority and lets HIRACore relation deltas modify it.

W14 suggests the direction should be inverted:

> Multiview semantic E should be the coarse authority. Competitive and relation evidence may contribute only as bounded residual corrections.

The mechanism is intentionally tiny.

The scientific question:

**Can a six-parameter bounded residual retain the fresh multiview semantic geometry while recovering useful typed/relation evidence?**

A separate equal-parameter unbounded residual control tests whether explicit magnitude bounding is causally important.

---

## 4. Multiview typed schema

Every option for every typed decision has exactly three prewritten meaning-equivalent natural definitions:
- D0;
- D1;
- D2.

For one decision:
- state is encoded exactly once;
- D0/D1/D2 option schemas are encoded on the schema side;
- exact W9 frozen projection is used;
- each view uses bidirectional symmetric token MaxSim.

Call raw candidate score vectors:
- A0;
- A1;
- A2.

Multiview semantic anchor:
`E = (A0 + A1 + A2) / 3`.

No labels enter E.

No generated paraphrase occurs after exposure.

---

## 5. Anchor logits

Frozen production semantic scale:
`s = CompetitiveCoarseScorer.scale()`.

Primary anchor logits:
`L = s * E`.

D0 anchor logits:
`L0 = s * A0`.

---

## 6. Competitive residual

C is the exact D0 current `CompetitiveCoarseScorer` output.

Raw residual:
`r_c = C - L0`.

Then center over valid candidates:
`r_c = r_c - mean_valid(r_c)`.

Interpretation:
candidate-relative production correction beyond the D0 semantic baseline.

---

## 7. Relation residual

Run frozen HIRACore using:
- `coarse_override=L`;
- forced full-K;
- adaptive budget false.

Take its per-option `relation_delta`.

Center:
`r_r = relation_delta - mean_valid(relation_delta)`.

Because full-K is forced, every option receives relation evaluation and the candidate budget cannot delete the anchor winner.

---

## 8. Robust anchor spread

For valid L:
- `m = median(L)`;
- `g = median(abs(L-m))`;
- clamp `g >= 1e-3`.

g is label-free and per-decision.

---

## 9. Bounded residual equation

For primitive t in:
- choice;
- score;
- noul.

Trainable raw scalars:
- theta_c[t];
- theta_r[t].

Mapping:
`beta_c[t] = 0.25 * sigmoid(theta_c[t])`
`beta_r[t] = 0.25 * sigmoid(theta_r[t])`.

Exactly six trainable parameters.

Final:

`Z = L
   + g * beta_c[t] * tanh(r_c/g)
   + g * beta_r[t] * tanh(r_r/g)`.

Frozen guarantee:
- each residual source contributes at most .25*g/candidate;
- both sources combined contribute at most .5*g/candidate.

This does not mathematically freeze all ranks, but it prevents arbitrarily large competitive/relation replacement of the semantic anchor.

Initialization:
- theta = -2.197224577;
- sigmoid(theta)=.1;
- initial beta=.025.

Near-anchor initialization retains usable gradient.

---

## 10. Equal-parameter unbounded control

Same six theta parameters and same beta mapping.

`Z_unbounded = L + beta_c[t]*r_c + beta_r[t]*r_r`.

No tanh or g cap.

All data/objectives/optimizer steps equal to bounded candidates.

This is the causal control for the bound itself.

---

## 11. W15 candidates

0. `production-frozen-control`
- exact current D0 production path;
- 0 trainable.

1. `multiview-anchor-control`
- logits L only;
- 0 trainable.

2. `unbounded-residual-control`
- E primary;
- 6 trainable scalars;
- unbounded residual equation.

3. `bounded-residual-primary`
- E primary;
- 6 trainable scalars;
- bounded equation.

4. `bounded-residual-replica`
- same architecture/data;
- independent optimizer seed;
- cannot replace primary after CONFIRM.

No post-exposure candidates.

---

## 12. Fresh W15 domains

Wholly fresh from W5-W14.

TRAIN:
- BZ museum membership and ticket-account services — seed 341201;
- CA residential solar installation support — 341207;
- CB maritime port credential administration — 341219;
- CC meal-kit subscription services — 341227.

DEV:
- CD commercial fleet maintenance contracts — 342331.

Untouched CONFIRM:
- CE community arts grant administration — 343441;
- CF language-exchange program administration — 344557.

Counts:
- TRAIN 96/domain = 384 cases / 1,920 typed decisions;
- DEV-CD 96 / 480 decisions;
- CONFIRM-CE 96 / 480;
- CONFIRM-CF 96 / 480.

Diagnosis cardinality/domain:
- 32 K4;
- 32 K8;
- 32 K16.

Each case:
1. diagnosis choice K4/K8/K16;
2. response choice K4;
3. needs_review noul K2;
4. risk ordinal score K4;
5. urgency ordinal score K4.

Every option in all five primitives has frozen D0/D1/D2 natural definitions.

---

## 13. Typed reliability targets

Reuse frozen target-mass semantics:
- verified .90;
- provisional .75;
- uncertain .60.

These define target distributions for the typed objective.

They are not post-hoc calibration.

---

## 14. Training

Only six theta parameters are optimized.

Typed objective:
- hard CE 1.0;
- teacher KL .5;
- hard Brier .1;
- soft-target Brier .5;
- ordinal MAE .2 for score primitives.

No:
- pair margin;
- contrastive alignment loss;
- projection gradient;
- HIRACore gradient;
- A13 gradient.

Trainable candidates:
- 8 epochs;
- AdamW;
- lr .03;
- weight decay 0;
- no scheduler;
- no warmup;
- no AMP;
- grad clip 1.0;
- deterministic epoch shuffle;
- same 384 train cases and equal optimizer case steps.

Seeds:
- unbounded 2303;
- bounded primary 2311;
- bounded replica 2323.

---

## 15. DEV selection

Each trainable candidate independently freezes on CD.

Lexicographic:
1. diagnosis K16 final top1;
2. K16 anchor-correct retention;
3. overall typed accuracy;
4. diagnosis K4 final top1;
5. lower score MAE;
6. lower soft-target ECE;
7. earlier epoch.

No cross-candidate checkpoint selection.

CE/CF remain sealed until all three trainable candidates are independently frozen.

---

## 16. Required metrics

Overall:
- typed accuracy;
- choice/noul/score accuracy;
- hard Brier;
- soft Brier;
- raw ECE diagnostic;
- soft-target ECE;
- score MAE;
- probability mass max error;
- state encodes/case.

Diagnosis K4/K8/K16:
- final top1/top5/MRR/margin;
- anchor E top1/top5/MRR;
- current production F top1/top5/MRR;
- E-correct -> final-wrong;
- E-wrong -> final-correct;
- anchor retention;
- anchor rescue.

Residual:
- beta_c/beta_r per primitive;
- raw and bounded residual magnitude;
- tanh saturation;
- residual/g ratio;
- bound guarantee.

Integrity:
- exact model/checkpoint/cache SHAs;
- six-param count;
- equal optimizer steps;
- primary/replica seed independence;
- no forbidden overlap.

---

## 17. Frozen gates

### Primary competence — both CE and CF

- overall >= .85;
- choice >= .85;
- noul >= .80;
- score >= .80;
- diagnosis K4 >= .88;
- K8 >= .82;
- K16 >= .72;
- anchor retention K4 >= .95;
- anchor retention K16 >= .90;
- probability mass error <= 1e-6;
- state encodes/case = 1.0.

### Gain vs production — both

- overall +.08;
- K4 +.12;
- K8 +.15;
- K16 +.20.

### Preserve anchor — both

- K4 >= E-.03;
- K8 >= E-.04;
- K16 >= E-.05.

### Residual value vs anchor — both

Define non-diagnosis typed accuracy over:
- response;
- needs_review;
- risk;
- urgency.

Require:
- non-diagnosis accuracy >= E+.02;
- K16 anchor-wrong rescue >= .08;
- K16 anchor-correct damage <= .10.

### Replica — both

- overall >= .82;
- K16 >= .65;
- K16 anchor retention >= .85;
- K16 >= E-.08;
- integrity pass.

### Bound causal vs unbounded — both

Require:
- bounded retention K16 >= unbounded+.05 OR bounded K16 >= unbounded+.05;
- bounded overall >= unbounded-.02;
- bounded non-diagnosis accuracy >= unbounded-.02.

---

## 18. Frozen verdicts

### BOUNDED_ANCHOR_RESIDUAL_RESCUE

Primary competence + production gain + anchor preservation + residual value pass both;
replica passes both;
bound causal passes both.

### ANCHOR_RESIDUAL_RESCUE_NO_BOUNDING_CAUSAL

Primary and replica rescue pass both;
bound causal fails;
unbounded control independently passes primary competence + anchor preservation on both.

### ANCHOR_ONLY_PRODUCTION_REDESIGN

On both:
- E overall >=.82;
- E diagnosis K4 >=.85;
- E K16 >=.68;
- E K16 >= F+.20;
- trainable residual candidates fail residual-value gate OR bounded primary is >.02 below E overall.

### ANCHOR_PRESERVING_PARTIAL

No full outcome, but bounded primary both:
- K16 >= production+.15;
- K16 >= E-.08;
- retention K16 >=.82;
- overall >= production+.05.

### ANCHOR_PRESERVING_FAIL

Otherwise.

No new verdict after exposure.

---

## 19. High-K / external boundary

W15 ends at K16 because W14's stable target is through K16.

Even a positive W15 does not authorize broad external claims.

Positive W15 -> W16 must replicate exact frozen architecture on wholly fresh K32/K64 authorities.

Only after high-K fresh replication may a separately preregistered public/external transfer authority open.

---

## 20. Permanent forbidden evidence

Never use for W15 selection/training:
- all W6b-W14 exposed authority rows;
- W12 BN/BO/BP/BQ;
- W13 BR/BS/BT/BU;
- W14 BV/BW/BX/BY;
- Banking77 0-799;
- typed final/test;
- public campaign cells.

After first CE/CF materialization:
CE/CF are permanently forbidden.

---

## 21. Claim discipline

Do not:
- tune .25 cap after exposure;
- tune tanh/spread equation after exposure;
- change paraphrases after exposure;
- unfreeze A13/projection/HIRACore;
- select the replica after CONFIRM;
- relax gates after close result;
- call three paraphrases the final UX;
- claim external superiority from W15.

---

## 22. Current state

Completed:
- W14 merged main `cbed464e16f9f44c8ddf0f46a257db47f5ff6595`;
- W14 issue #115 closed;
- W15 branch created from exact post-W14 main;
- issue #117 preregistered with full architecture/data/gates before exposure;
- this handoff is being created before any W15 empirical exposure.

Exposure:
**NONE.**

No BZ/CA/CB/CC/CD/CE/CF A13 cache exists.
No W15 training exists.
No DEV checkpoint exists.
No CE/CF materialization exists.

Next:
1. add prior-art boundary note;
2. implement six-param anchor-preserving residual module;
3. implement unit contracts for bound/equivariance/parameter count/near-anchor init;
4. implement fresh W15 authority generator;
5. implement state-once multiview typed cache;
6. implement train/eval core;
7. add sealed train/dev builder;
8. add candidate trainer;
9. add independent DEV freezer;
10. add sealed dual-CONFIRM evaluator;
11. add pre-data unit workflow;
12. run exact-head unit + repository CI;
13. only then enable authority;
14. freeze exact result here before merge.

A future AI must update this file after every meaningful session.


---

## 23. Pre-data implementation update — full execution stack

Scientific exposure remains:

**NONE.**

No BZ/CA/CB/CC/CD/CE/CF A13 cache has been materialized.
No W15 model has been trained.
No DEV-CD checkpoint has been selected.
No CONFIRM-CE/CF row has been materialized.

Implemented on `feat/r8-w15-anchor-preserving-residual`:

### Core
- `src/nmd/anchor_preserving_residual.py`
  - exactly six primitive-specific residual scalars;
  - robust midpoint-MAD anchor spread;
  - centered competitive and relation residuals;
  - bounded and equal-parameter unbounded paths;
  - structural source cap <= .25*g and combined cap <= .5*g.
- core commit: `4d1f741f914ad7e4e45006e56efaefc2baee5357`.

### Core contracts
- `tests/test_anchor_preserving_residual.py`
  - exact six-param budget;
  - beta=.025 near-anchor initialization;
  - exact zero-residual anchor identity;
  - residual shift invariance;
  - source/combined bound;
  - equal-parameter unbounded control can exceed cap;
  - candidate permutation equivariance;
  - gradients reach every primitive-specific scalar;
  - masked candidates do not affect valid statistics.
- float-only test tolerance repair after first pre-data CI failure:
  `7347ba1e4eeec54d3dda2a862b483a8c7c0a7c87`.
  This changed test tolerance only, not mechanism/scoring/data/gates.

### Fresh authority
- `src/nmd/anchor_preserving_residual_authority.py`
  - BZ/CA/CB/CC TRAIN;
  - CD DEV;
  - CE/CF sealed CONFIRM;
  - 16 natural intents/domain;
  - six independent state variants/intent;
  - exactly 96 cases/domain;
  - diagnosis K4/K8/K16 balanced 32/32/32;
  - five typed decisions/case;
  - every option has frozen D0/D1/D2;
  - D0 is exact production criterion;
  - confidence targets .90/.75/.60.
- commit `9a79eb9210c46fc9ce497bd5abb54258d72f6b39`.
- confirm materialization was removed from unit tests; default seal is tested without invoking capability.

### State-once multiview cache
- `src/nmd/anchor_preserving_residual_cache.py`
  - state compiled exactly once/case;
  - D0/D1/D2 compiled schema-side only;
  - exact option IDs/order preserved across views;
  - D0 stores production question/option token artifacts;
  - custom K4/K8/K16 validator;
  - five-decision typed target validation.
- commit `533e08200dc681196ad4a61a0eb585c0bce03936`.

### Train/eval/verdict core
- `src/nmd/anchor_preserving_residual_training.py`
  - exact W14 `_q0_symmetric` semantic operator reused;
  - E and D0 anchor logits use frozen scorer scale;
  - exact production C and F references;
  - relation residual computed from frozen HIRACore on anchor coarse override, forced full-K;
  - final residual logits do not run production again;
  - only mixer parameters receive gradient;
  - typed W6/W7 loss weights reused;
  - mandatory typed/calibration/K metrics;
  - anchor retention/rescue/damage;
  - residual magnitude/saturation/bound diagnostics;
  - DEV-CD selection key;
  - frozen gates and verdicts from issue #117.
- commit `071b8cdb593fa28e5246c5596a71cdb49b480886`.

### Download-free execution tests
- `tests/test_anchor_preserving_residual_authority.py`;
- `tests/test_anchor_preserving_residual_execution.py`;
- `tests/test_anchor_preserving_residual_execution_contracts.py`.
- one-case execution uses only local `TrainableSemanticEncoder` for shape/state-once testing, never as authority evidence.
- unit checks include:
  - state encode exactly once;
  - D0/D1/D2 identity;
  - E/F/residual shape/integrity;
  - bound diagnostics;
  - candidate set/seeds/epochs;
  - CE/CF capability literal confined to confirm script;
  - train/dev builder cannot import confirm capability;
  - exact W15 text atoms disjoint from all prior W5-W14 manifests before exposure.

### Authority scripts
Implemented but **not yet executed**:
- `scripts/r8_w15_build_cache.py`
  - exact A13 SHA verification;
  - full W5-W14 exact-text freshness firewall;
  - BZ-CC TRAIN + CD DEV only;
  - no confirm capability.
- `scripts/r8_w15_train_candidate.py`
  - exact W9 HIRA/scorer SHA verification;
  - controls epoch 0;
  - residual candidates exactly six trainable params;
  - 3,072 optimizer case-steps each.
- `scripts/r8_w15_freeze_candidates.py`
  - requires all five candidate receipts;
  - verifies exact shared cache/base provenance;
  - independent DEV selection;
  - primary/replica seed independence;
  - emits `all_candidates_frozen_before_confirm=true`.
- `scripts/r8_w15_confirm.py`
  - sole W15 file allowed to contain `allow_confirm=True`;
  - exact freeze boundary required;
  - materializes CE then CF only post-freeze;
  - evaluates all five frozen paths;
  - computes the frozen W15 verdict;
  - records CE/CF cache hashes and integrity flags.

### CI
- `.github/workflows/r8-w15-unit.yml`
  - pre-data branch-scoped gate;
  - compiles core + all execution scripts;
  - runs all W15 tests;
  - does not download A13;
  - does not materialize CE/CF;
  - authority workflow does not exist yet.

Earlier green pre-data checkpoint:
- W15 unit run on head `7e29deda89720c836e36ae82e8b00cd1cce33e1b`: PASS;
- repository CI on same head: PASS.

The complete-stack exact-head unit/CI after scripts + freshness guard is currently being validated.

Current latest implementation head at this handoff update:
`410d54794052763fb63b668f0d546349e6129327`.

### Remaining pre-exposure sequence

1. obtain W15 unit PASS on exact complete-stack head;
2. obtain repository CI PASS on the same exact head;
3. record exact run IDs here and in issue #117;
4. only then add/enable W15 authority workflow;
5. authority chain must be:
   `unit -> exact W9 provenance + BZ-CC/CD cache -> five candidates -> independent DEV-CD freeze -> CE/CF once -> frozen verdict`;
6. after CE/CF exposure, no scientific mutation is allowed;
7. freeze exact metrics/artifacts/verdict in this file;
8. merge only a clean, frozen result.

A future AI must not treat any current implementation test as empirical evidence.


---

## 24. Exact pre-authority freeze

Scientific exposure remains:

**NONE.**

Exact complete-stack implementation head:
`b46540f51963e8b0b48431fcb3d2647670565e8a`.

Pre-authority validation on that exact head:
- W15 unit run `36149189449`: PASS;
- duplicate branch unit run `36149186160`: PASS;
- repository CI `36149194396`: PASS;
- CI Python 3.10: PASS;
- CI Python 3.12: PASS.

All W15 execution components are present before exposure:
- frozen six-parameter bounded residual mechanism;
- equal-parameter unbounded control;
- fresh BZ/CA/CB/CC TRAIN + CD DEV generator;
- state-once multiview typed cache;
- exact W9 semantic projection / HIRACore provenance checks;
- five candidate paths;
- independent DEV-CD selection;
- sealed CE/CF generation capability;
- frozen verdict implementation;
- full W5-W14 exact-text freshness firewall.

No BZ/CA/CB/CC/CD A13 cache has been built.
No candidate has trained.
No DEV-CD checkpoint has been selected.
No CE/CF row has been materialized.

The next commit may add only the orchestration workflow. Once the first eligible BZ-CC/CD cache job begins empirical materialization:
- equations are frozen;
- .25 residual cap is frozen;
- optimizer/seeds are frozen;
- candidate set is frozen;
- domain text/seeds are frozen;
- DEV selector is frozen;
- CE/CF gates and verdicts are frozen;
- no scientific mutation is permitted.

Authority must consume exact W9 freeze artifact:
- source run `36122220588`;
- artifact `10858424139` (`r8-w9-freeze`);
- digest `sha256:6f0eee6ae58d489f0486372f0311e0f420065a0d9499185af7904377ae6a4712`;
- projection semantic HIRA SHA `d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588`;
- projection semantic scorer SHA `8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102`.

Required authority chain remains exactly:

`unit -> W9 provenance + BZ-CC/CD cache -> five independently evaluated/trained candidates -> DEV-CD freeze -> CE/CF one-time materialization -> frozen verdict`.
