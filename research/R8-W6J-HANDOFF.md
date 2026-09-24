# R8-W6j handoff — high-cardinality decision-decomposition audit

Status: **CLOSED DIAGNOSTIC. Authoritative outcome: `STABLE_HIGH_CARDINALITY_ARCHITECTURE`; stable target: `COARSE_CONJUNCTION_LIMIT`. W7 architectural phase is authorized.**

Issue: #95

Branch: `feat/r8-w6j-high-cardinality-decomposition`

Base main at branch creation:
`7ce1d29c0236778d3b889c3dd8c061cb2019a865`

---

## 0. Project identity — what Nolane HIRA is

Nolane HIRA is a compact, non-autoregressive, typed decision system.

It is **not** intended to be a tiny general-purpose next-token LLM.

Its core production objective is to turn one encoded state plus a dynamic semantic decision schema into typed decisions while preserving:

- one state encode per case;
- arbitrary/dynamic logical options rather than a fixed classifier head;
- choice, score and noul primitives;
- high-cardinality option selection;
- calibrated probabilities/reliability;
- small trainable parameter budgets;
- explicit empirical provenance and falsifiable promotion gates.

The long-term target is a small decision architecture that can generalize semantic decision rules across changing domains and schemas without requiring a giant autoregressive model.

The scientific rule of this repository is strict:
**internal synthetic/fresh authorities may establish mechanism evidence, but may not be turned into broad real-world or public-benchmark superiority claims.**

---

## 1. Frozen semantic encoder

A13 remains the frozen encoder throughout the current R8 line:

- model: `microsoft/xtremedistil-l6-h256-uncased`;
- revision: `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA-256: `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length: 256;
- fully frozen unless a future explicitly preregistered phase changes that boundary.

Do not enlarge or fine-tune A13 inside W6j.

---

## 2. Current production architecture entering W6j

### Competitive coarse scorer

Promoted production semantic binder:

- `CompetitiveCoarseScorer`;
- 256 -> 128 bias-free projection;
- scalar learned logit scale;
- exactly 32,769 trainable parameters;
- candidate-relative IDF/salience;
- forward token-context MaxSim;
- within-option common-mode subtraction;
- weighted mean coverage + 0.5 minimum salient-token coverage.

This component came from W5g/W5h/W5i evidence.

### HIRACore

Legacy core parameter count:
- 422,159 parameters.

Current high-level path:

`state-once memory -> full-K competitive coarse logits -> candidate selection/budget -> relation-to-state delta on selected candidates -> full-K logits -> softmax / typed value`.

Important architectural fact:

- the HIRACore relation stage lets each selected option query the same state/segment memory;
- selected candidates do **not** directly self-attend to or reason over one another;
- unselected candidate logits remain exactly their coarse logits;
- candidate-set coupling comes mainly from competitive coarse IDF/salience, selection, and final normalization.

### Integrated typed runtime

`NolaneHira` supports:

- `choice`;
- `score`;
- `noul`;

with dynamic schemas and one state compilation per case.

---

## 3. Decisive evidence before W6j

This section is intentionally compact but sufficient for a new AI to avoid repeating dead ends.

### W5 semantic binding sequence

W5a/W5b/W5c/W5d/W5e showed that:
- pooled/simple token matching and capacity scaling alone do not solve semantic routing;
- A13 -> A22 capacity increase did not rescue the task;
- raw encoder capacity was not the main bottleneck.

W5f late interaction produced a major jump.

W5g contrastive salience:
- untouched CONFIRM overall 50.52%;
- K128 47.92%;
- K255 41.67%;
- verdict `CONTRASTIVE_SALIENCE_PARTIAL`.

W5h balanced competitive binding:
- introduced soft within-option competition/common-mode subtraction;
- verdict `BALANCED_BINDING_PARTIAL`.

W5i:
- fresh forward competitive control itself achieved strong rescue-scale behavior;
- listwise reverse mechanism did not earn causal promotion;
- conclusion: production should integrate the forward competitive binder, not the listwise experiment.

### W6 integration

W6 Phase A integrated the competitive scorer into production while preserving backward compatibility and state-once behavior.

This phase was an integration proof, not a broad superiority claim.

### W6b typed/reliability fresh authority

Selected scorer-only candidate:
- 32,769 trainable parameters.

Untouched CONFIRM:
- overall 81.50%;
- choice 80.31%;
- score 88.75%;
- K64 60%;
- noul 69.375%;
- hard Brier 0.3283;
- soft ECE 0.16373.

Verdict:
`PRODUCTION_COMPETITIVE_PARTIAL`.

It narrowly missed noul and soft-ECE absolute gates.

### W6c calibration

A tiny three-temperature calibrator strongly improved calibration but did not repair hard semantic generalization.

Fresh-domain CONFIRM:
- overall 45.729%;
- K64 43.75%;
- soft ECE 0.09721;
- score MAE worsened.

Verdict:
`RELIABILITY_CALIBRATION_PARTIAL`.

Scientific conclusion:
confidence calibration is real and fixable, but is not the dominant hard-decision bottleneck.

### W6d multi-domain generalization

Joint HIRA+scorer adaptation reached:
- overall 89.167%;
- K64 54.167%.

The preregistered K64 gate was 55%, so the result missed by exactly 1/48 cases and remained:
`GENERALIZATION_FAIL`.

This suggested domain robustness may be a full typed-core property rather than a scorer-only property.

### W6e replication

Two untouched domains L/M failed to replicate the W6d near-rescue at K64.

Typical result:
- overall high (~88-89%);
- K64 only ~31-42% depending checkpoint/domain.

Verdict:
`JOINT_GENERALIZATION_REPLICATION_FAIL`.

The bottleneck became specifically high-cardinality diagnosis rather than general typed competence.

### W6f high-K localization

No training.

Found:
- relation reranking is net helpful;
- removing candidate-relative salience is dramatically worse;
- gold often remains inside coarse top5;
- errors are dominated by near-neighbor / one-field mismatches.

All pooled checkpoint classifications:
`UNRESOLVED_HIGH_K_FAILURE`.

### W6g second-order localization

Fresh Q/R/S.

All four semantic roles across the frozen checkpoints localized as:
`FIELD_SEMANTIC_COLLAPSE`.

Outcome:
`STABLE_SECOND_ORDER_LOCALIZATION`.

This authorized exactly one fresh field-semantic rescue lane.

### W6h field-semantic rescue

Tested:
- frozen control;
- 32,769-param projection retune;
- 8,192-param zero-init semantic residual adapter.

Untouched CONFIRM Y/Z:

Y K64:
- frozen 47.917%;
- projection retune 56.250%;
- adapter 45.833%.

Z K64:
- frozen 37.500%;
- projection retune 64.583%;
- adapter 62.500%.

The adapter was not reproducible across both domains.

Verdict:
`FIELD_SEMANTIC_FAIL`.

Important negative:
the frozen control already had ~95% full structured one-field pair accuracy, showing that the W6g isolated-value probe and the W6h training proxy were not equivalent measurements.

### W6i representation bridge

Fresh AA/AB/AC; no training.

Authority:
- exact empirical head `6993efdefa962ee6fca50f44f6d3dc6c39532b0e`;
- run `36022251442`;
- all jobs PASS;
- PR #94 merged as main commit `7ce1d29c0236778d3b889c3dd8c061cb2019a865`.

Artifacts:
- cache `10818615287`;
- audit `10818157410`;
- frozen checkpoints `10816673873`.

Cross-checkpoint result:
`REPRESENTATION_BRIDGE_UNRESOLVED`.

Checkpoint-local result:
- W6h projection-retune showed `STRUCTURED_PAIR_PROXY_MISMATCH` on AA/AB;
- W6e joint-primary did not;
- semantic adapter did not.

No stable cross-checkpoint causal classification.

Also:
- canonical tagged representation was worse than production text;
- factorized-mean was much worse;
- factorized-min was much worse.

Therefore:
`rescue_lane_authorized = false`.

This is the direct reason W6j must operate one architectural level higher.

---

## 4. Exposed / forbidden evidence

The following must never be reused for training, selection, threshold tuning, or mechanism search in W6j:

- W6b CONFIRM;
- W6c CONFIRM;
- W6d CONFIRM F;
- W6e CONFIRM L/M;
- W6f diagnostic N/O/P;
- W6g diagnostic Q/R/S;
- W6h CONFIRM Y/Z;
- W6i diagnostic AA/AB/AC;
- typed final/test rows;
- public campaign cells.

W6h T/U/V/W TRAIN and DEV-X are also not W6j diagnostic rows.

Historical artifacts may be loaded only as frozen checkpoint provenance when explicitly listed in the W6j protocol.

---

## 5. W6j purpose

W6j asks a higher-level question:

**Why does local semantic competence fail to compose into stable K64 global decisions?**

It does not assume that the next fix is:
- another adapter;
- more IDF tuning;
- a larger encoder;
- a different formatting string;
- another relation head patch.

Instead it decomposes the present architecture into falsifiable failure modes.

---

## 6. Frozen W6j hypotheses

### H1 — pairwise multiplicity hazard

Even if one gold-vs-negative comparison is fairly reliable, 63 opportunities for a hard negative to beat gold can make global top1 much weaker.

This would mean the architecture needs explicit error control / staged elimination / structured decomposition rather than merely higher average pair accuracy.

### H2 — set-context rank reversal

The same gold/wrong pair may reverse ordering between K2 and K64 because the competitive scorer is set-relative.

This would indicate global set composition is not context-consistent.

### H3 — coarse conjunction limit

The actual global wrong winner may already beat gold in isolated full-structured K2.

Then K64 is exposing a semantic/conjunction error that exists before global competition.

### H4 — relation decomposition damage

The coarse stage may be correct but the relation delta may destabilize it.

This is distinct from W6f's aggregate observation that relation was net helpful: W6j will test the exact global-winner path and an oracle-perfect coarse intervention.

### H5 — mixed / unresolved

Different checkpoints/domains may support different explanations.

If this happens, no architectural rescue is authorized from W6j.

---

## 7. Fresh W6j domains

Preregistered:

- AD — cryogenic telescope maintenance; seed 241301;
- AE — underground transit ventilation; seed 241307;
- AF — autonomous coastal monitoring; seed 241319.

Each:
- 64 base states;
- four semantic roles;
- one gold four-field signature;
- one deterministic K64 master set;
- fresh values/templates/roles/IDs.

Total:
- 192 base states.

No W6j A13 cache exists yet.

---

## 8. Frozen checkpoints for W6j

No training.

### W6e joint-primary

- HIRA SHA `d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588`;
- scorer SHA `6d5a7f2d3ed63ecd756181b1cb54e4704f68e5f74983a897f0a68fb4d1d63d2e`;
- artifact `10805567861`.

### W6h projection-retune

- same frozen HIRA;
- scorer SHA `39235c425d22d01adcea47a7d9dca3330d022488191e5d0ea33bfbb4ee8af9e2`;
- artifact `10816055074`.

### W6h semantic-adapter

- same frozen HIRA;
- scorer SHA `9d34e50152706b7a2164b78427abf27dd98966938c3bcce58f71e36163e03469`;
- artifact `10815347443`.

---

## 9. W6j diagnostic probes

### A. Gold-vs-every-negative K2 sweep

For each base, score gold against each of the 63 negatives independently using exact production structured text.

Record:
- coarse/final pair win;
- margins;
- changed-role count;
- pair-loss count per base;
- all-63-pairs-win certificate.

The certificate is diagnostic only because it uses gold identity.

### B. Global-winner counterfactual

For nested K8/K16/K32/K64:
- find actual global wrong winner;
- evaluate the exact gold-vs-winner pair in K2;
- measure rank reversal.

### C. Multiplicity anatomy

Measure:
- global failure conditional on >=1 genuine K2 loss;
- global failure conditional on all K2 wins;
- pair-loss-count distribution;
- candidate-distance distribution;
- all-pair certificate gap vs K64 top1.

No independence assumption is allowed in the primary gate.

### D. Coarse vs relation path

Measure K64:
- coarse correct -> final correct;
- coarse correct -> final wrong;
- coarse wrong -> final correct;
- coarse wrong -> final wrong;
- relation rescue/damage.

### E. Oracle-conjunction relation stress

Diagnostic oracle:
`oracle_score = 4 - HammingDistance(candidate, gold)`.

Gold is uniquely top1 by construction.

Feed that coarse vector through frozen HIRACore relation machinery.

If the relation stage breaks a perfect oracle coarse ranking at nontrivial rate, that is direct evidence of relation decomposition damage.

Never treat this oracle as a production candidate.

---

## 10. Frozen W6j classifications

### PAIRWISE_MULTIPLICITY_LIMIT

Requires:
- K64 final top1 < 0.65;
- >=80% of K64 global errors contain >=1 genuine K2 pair loss;
- all-pair-win but K64-fail rate <0.10;
- winner reversal rate <0.15;
- relation damage among coarse-correct cases <0.10.

### SET_CONTEXT_RANK_REVERSAL

Requires:
- K64 final top1 <0.65;
- winner reversal rate >=0.25;
- all-pair-win but K64-fail rate >=0.15;
- K2-final -> K64-final sign reversal for the exact gold/global-winner pair on >=20% of K64 errors.

### COARSE_CONJUNCTION_LIMIT

Requires:
- K64 final top1 <0.65;
- >=70% of K64 errors have the eventual winner already beating gold in K2 coarse;
- relation rescue <+0.10 absolute top1;
- oracle-conjunction final top1 >=0.95.

### RELATION_DECOMPOSITION_DAMAGE

Requires:
- stable coarse-correct measurement;
- coarse-correct -> final-wrong >=0.15;
- oracle coarse -> final error >=0.05;
- relation damage exceeds relation rescue by >=0.05.

### MIXED_HIGH_CARDINALITY_ARCHITECTURE

Stable domains/checkpoints disagree on mechanism.

### HIGH_CARDINALITY_DECOMPOSITION_UNRESOLVED

No classification is stable.

Cross-checkpoint stability:
- W6e joint-primary + at least one W6h-trained checkpoint must agree;
- agreement on at least two of AD/AE/AF;
- third checkpoint may not show an opposite dominant mechanism.

---

## 11. Scientific boundary

W6j is diagnostic only.

Do not:
- train on AD/AE/AF;
- retune thresholds after exposure;
- enlarge A13;
- add another semantic adapter;
- modify IDF/salience;
- change relation architecture during this lane;
- reuse exposed earlier rows;
- claim a production rescue from an oracle diagnostic;
- populate public campaign cells;
- claim broad superiority.

If W6j localizes a stable architectural target:
- open **W7** as a genuinely new architectural phase;
- preregister fresh authority data and a mechanism specific to that target before implementation results are seen.

If W6j is unresolved/mixed:
- stop synthetic high-K rescue expansion;
- move to external/public task-decomposition research before redesigning HIRA.

---

## 12. Current implementation state

Completed in this session:
- W6i authoritative closure was frozen;
- PR #94 merged;
- issue #93 closed;
- W6j issue #95 created;
- draft PR #96 opened;
- branch `feat/r8-w6j-high-cardinality-decomposition` created from exact main;
- this handoff created and kept current;
- `src/nmd/high_cardinality_decomposition_authority.py` implemented:
  - fresh AD/AE/AF domains;
  - 64 bases/domain;
  - exact deterministic master K64;
  - exact distance mix: 12 one-field, 20 two-field, 15 three-field, 16 four-field negatives;
  - nested K8/K16/K32/K64 identities;
- `src/nmd/high_cardinality_decomposition.py` implemented:
  - frozen architectural gates;
  - no post-hoc mechanism precedence;
  - mixed classification when multiple gates pass;
  - cross-checkpoint stability contract;
- `src/nmd/high_cardinality_decomposition_eval.py` implemented:
  - base-centric cache;
  - one state encode/base;
  - one K64 schema compilation/base;
  - nested-view reconstruction from master artifacts;
  - exhaustive 63 K2 gold-vs-negative comparisons;
  - global-winner reversal metrics;
  - relation rescue/damage metrics;
  - oracle-conjunction relation probe;
  - probability-mass integrity recording;
- `scripts/r8_w6j_build_cache.py` implemented with exact frozen A13 provenance and prior-value leakage checks;
- `scripts/r8_w6j_evaluate.py` implemented with exact W6e/W6h checkpoint receipt/SHA checks;
- generator, classifier and evaluator unit tests added;
- pre-diagnostic unit workflow added.

Pre-data protocol note:
- if multiple mechanism gates pass, classify as `MIXED_HIGH_CARDINALITY_ARCHITECTURE`; never choose a convenient precedence after seeing results;
- W6e joint-primary must itself be stable on >=2 fresh domains and at least one W6h-trained checkpoint must agree before a W7 architecture lane can be authorized.

Current implementation head before this handoff update:
- `28a6d914315717eacc43ecb4ae7bd72a709e564e`.

Earlier generator/classifier heads have passed unit CI. The exact current full-stack head is still in pre-diagnostic unit/CI validation.

Not yet completed:
- exact current full-stack unit/CI must pass;
- gated W6j A13 diagnostic authority workflow is not yet enabled;
- no AD/AE/AF A13 cache has been exposed;
- no W6j checkpoint evaluation has occurred;
- no W6j empirical classification exists.

No W6j empirical data exists yet.

---

## 13. Immediate next steps for another AI/session

1. Implement `src/nmd/high_cardinality_decomposition_authority.py` with AD/AE/AF fresh generators and deterministic nested K sets.
2. Add strict tests for:
   - 64 bases/domain;
   - exactly 64 master candidates/base;
   - exact candidate identity across K2/nested/full-set contexts;
   - minimum hard-negative distance counts;
   - pairwise domain freshness/disjointness;
   - no overlap with W5-W6i values/templates/roles.
3. Implement a cache that encodes state exactly once/base and stores production schema/token artifacts for all required views.
4. Implement exhaustive gold-vs-63 K2 evaluation without changing checkpoint parameters.
5. Implement global winner reversal and multiplicity metrics.
6. Implement oracle coarse -> frozen HIRACore relation probe.
7. Centralize frozen classifications in library code and unit-test threshold semantics before any A13 cache exists.
8. Add gated authority workflow:
   `unit -> exact frozen checkpoint provenance + fresh AD/AE/AF cache -> evaluator -> cross-checkpoint stability`.
9. Do not mutate classification gates after first eligible empirical cache/evaluation exposure.
10. After result, update this handoff with exact head, run, artifacts, metrics and frozen outcome before merging.

---

## 14. Claim discipline

The most important instruction for continuation:

A strong number is not enough.

- PARTIAL is not RESCUE.
- checkpoint-local evidence is not cross-checkpoint stability.
- synthetic/fresh internal evidence is not public generalization.
- an oracle diagnostic is not a deployable mechanism.
- a post-exposure threshold change invalidates causal interpretation.

Preserve negative results. They are part of the project state.


---

## 15. Authoritative W6j closure

Exact empirical authority head:

`af33754719b1f70e08f9c01991ea41bfd00c496a`

Authority run:

`36072141944`

All jobs PASS:

`unit -> exact frozen upstream provenance + fresh AD/AE/AF cache -> fixed-checkpoint architectural evaluator`.

### Artifacts

- fresh cache: `10838263710`
  - digest `sha256:112d60d6fd8cfbb8e4320e242e959ad58550663e3093c2bdca183d22b46b9b23`;
  - internal cache SHA `62db02fc466d8e70b6fde9054771defbf0d615397e713be37b3b4f315575d59c`;
- frozen checkpoint bundle: `10838857136`
  - digest `sha256:975fc537e01079356d728cc96ac82241244cf534f5cb24e2d2be61c8fcef6405`;
- authoritative decomposition audit: `10838774135`
  - digest `sha256:3e511325acc1abc9a9fe287224add3d2daeaf97d5033da63bc74e6b6a282ffff`.

Fresh cache integrity:

- 192 bases / 768 nested views;
- AD/AE/AF: 64 bases each;
- state encode calls: 192;
- state encodes/base: 1.0;
- case ID SHA `b884262217d74e639f8bca676de4a5ef3d9f2155e9d9c5f5e9b9463e0b82e1ad`;
- semantic-view SHA `e919eafca4bccb94c2b661944b48930411f03fb15fa4f5f12faf429462685172`;
- all-value SHA `a45c7bf0ba45456823043188513cf9cbac24f1c85a260b41760059a380b1f3d0`;
- prior-value overlap: 0;
- no training;
- no W6b-W6i exposed rows;
- no typed final/test rows;
- campaign cells 0.

Probability mass max error:

`2.384185791015625e-07`.

### Frozen checkpoint stability

#### W6e joint-primary

Stable on **all AD/AE/AF** as:

**`COARSE_CONJUNCTION_LIMIT`**

Pooled:

- K8 coarse/final top1: 61.458% / 69.792%;
- K16: 56.250% / 58.333%;
- K32: 43.750% / 44.792%;
- K64: **30.729% / 32.813%**;
- 129 K64 final errors;
- 100% of K64 errors have at least one genuine K2 pair loss;
- eventual K64 wrong winner already beats gold in isolated K2 **coarse** on **91.473%** of K64 errors;
- winner K2-final -> K64-final reversal: only **3.101%**;
- all-63-pairs-win case rate: 15.625%;
- among those certificate cases, K64 failure rate: **0%**;
- relation net top1 gain: +2.083 pp;
- oracle-conjunction final top1: **99.479%**;
- oracle relation damage: 0.521%.

Interpretation:
the wrong global winner is usually already semantically preferred by the learned coarse scorer before large-set competition. Full-set rank reversal is rare. The relation stage is not the primary failure and preserves an oracle-perfect coarse ranking almost exactly.

#### W6h projection-retune

Per-domain:

- AD -> `COARSE_CONJUNCTION_LIMIT`;
- AE -> mixed: both pairwise multiplicity and coarse conjunction gates pass;
- AF -> mixed: both pairwise multiplicity and coarse conjunction gates pass.

Checkpoint stability therefore remains:

`HIGH_CARDINALITY_DECOMPOSITION_UNRESOLVED`.

Pooled:

- K64 coarse/final: 48.958% / 50.521%;
- global-error pair-loss rate: 96.842%;
- eventual winner beats gold in K2 coarse: 89.474%;
- winner reversal: 7.368%;
- all-pair-win but K64-fail conditional rate: 4.615%;
- oracle final top1: 99.479%;
- relation net gain: +1.563 pp.

Important:
projection retuning materially improves absolute K64 and reduces pair-loss count, but does not change the core causal story enough to produce an opposing stable mechanism.

#### W6h semantic adapter

Stable on AD/AF as:

**`COARSE_CONJUNCTION_LIMIT`**

AE is mixed, not an opposite stable class.

Pooled:

- K64 coarse/final: 39.063% / 42.188%;
- global-error pair-loss rate: 97.297%;
- eventual winner beats gold in K2 coarse: 94.595%;
- winner reversal: 5.405%;
- all-pair-win but K64-fail conditional rate: 5.769%;
- oracle final top1: 99.479%;
- relation net gain: +3.125 pp.

### Authoritative cross-checkpoint outcome

**`STABLE_HIGH_CARDINALITY_ARCHITECTURE`**

Stable causal target:

**`COARSE_CONJUNCTION_LIMIT`**

`rescue_lane_authorized = true`.

This meets the frozen cross-checkpoint rule because:

- W6e joint-primary is stable as coarse-conjunction limit on all three fresh domains;
- W6h semantic-adapter independently agrees on at least two domains;
- projection-retune does not produce a stable opposing class.

### What W6j falsifies

The current evidence argues strongly against these being the dominant explanation:

1. **set-context rank reversal**
   - winner reversal remains only about 3-7% pooled;
   - the eventual wrong winner usually already beats gold in K2 coarse.

2. **relation decomposition as the main bottleneck**
   - relation is small/net helpful pooled;
   - an oracle-perfect conjunction coarse vector survives relation with ~99.48% final top1.

3. **pure large-K multiplicity as the sole cause**
   - multiplicity is real in some projection-retune domains;
   - however the stable cross-checkpoint target remains coarse conjunction;
   - most global errors are traceable to genuine local coarse semantic losses rather than otherwise-correct pair ordering being broken only by K64 context.

### Architectural conclusion

The next architecture must target **learned conjunctive coarse evidence**.

Do not spend W7 on:
- another calibration-only mechanism;
- another relation reranker;
- IDF removal;
- another unstructured residual adapter;
- canonical-tag text formatting alone;
- frozen mean/min factorization;
- a larger A13 encoder.

The key question is now:

**Can HIRA represent and train an explicit conjunction over independently evidenced schema factors so that a candidate must match all required semantic conditions, while preserving state-once execution and a small parameter budget?**

W7 must answer that question on wholly fresh data and must include an equal-data free-form scorer control.

---

## 16. Authorized next phase: W7

W7 is a genuinely new architectural phase, not W6j tuning.

W7 may test a production-compatible **explicit conjunctive evidence interface** with fresh authority data.

Minimum scientific requirements:

- A13 remains frozen;
- HIRACore remains frozen in the primary mechanism comparison so the causal question stays coarse-only;
- current free-form competitive scorer must be a same-data control;
- conjunctive candidate must expose explicit schema factors rather than infer synthetic gold structure from labels;
- state is encoded exactly once/case;
- factor/schema encodings are schema-side and separately accounted;
- no AD/AE/AF row may be used for W7 training, DEV selection or CONFIRM;
- use at least two untouched W7 CONFIRM domains;
- require reproducible K64 gain on both, not pooled-only rescue;
- record pair-loss counts by Hamming/semantic distance;
- preserve probability integrity and typed-interface compatibility;
- do not promote a W7 mechanism unless its gain survives fresh free-form control and both untouched domains.

W6j itself is diagnostic only. Its `COARSE_CONJUNCTION_LIMIT` label does not prove that any particular conjunction architecture will work.

---

## 17. Handoff state after W6j

W6j empirical research is complete.

Before another AI continues implementation:

1. merge PR #96 only after closure docs and repository CI are clean;
2. close issue #95 after merge;
3. create W7 from the resulting main commit;
4. create a new `research/R8-W7-HANDOFF.md` immediately;
5. preregister W7 architecture, candidates, fresh domains, train/DEV/CONFIRM split, parameter budgets, objective, selection rule and rescue gates **before** any W7 empirical cache is exposed;
6. preserve every W6j artifact/hash/result above;
7. never train or tune using AD/AE/AF.

The next AI should read this file first, then issue #95 / PR #96, then the eventual W7 handoff.
