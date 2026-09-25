# R8-W8 handoff — semantic/schema transfer decomposition audit

Status: **CLOSED DIAGNOSTIC. Authoritative outcome: `STABLE_SEMANTIC_TRANSFER_LOCALIZATION`; stable target: `GENERAL_SEMANTIC_TRANSFER_LIMIT`.**

Issue: #103

Branch:
`feat/r8-w8-semantic-transfer`

Base main:
`7438c92a6c5d12db818a024d80e6c86bb67a1ae0`

Read first when restoring:
1. this file;
2. `research/R8-W7C-HANDOFF.md`;
3. `research/R8-W7B-HANDOFF.md`;
4. `research/R8-W7-HANDOFF.md`;
5. `research/R8-W6J-HANDOFF.md`;
6. issue #103.

## 0. What HIRA is

Nolane HIRA is a compact, non-autoregressive typed decision system, not a next-token LLM.

Long-term goals:
- encode state once/case;
- accept dynamic semantic schemas;
- handle large changing candidate sets;
- support choice / score / noul typed primitives;
- remain small/local-friendly;
- preserve probability/reliability integrity;
- support semantic binding and structured decisions;
- make every mechanism claim falsifiable;
- preserve negative results.

## 1. Frozen semantic encoder/runtime

A13:
- `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length 256;
- fully frozen.

Production semantic coarse scorer:
- competitive late interaction;
- 256 -> 128 bias-free projection;
- scalar logit scale;
- 32,769 params;
- candidate-relative IDF/salience;
- within-option common-mode subtraction;
- weighted mean + salient minimum coverage.

HIRACore:
- 422,159 params;
- frozen in W8.

W8 execution:
- competitive coarse;
- pooled relation;
- forced full-K;
- adaptive budget false;
- no calibration;
- no training.

## 2. Decisive research history

W5:
- simple pooled matching / token probes / encoder scaling failed;
- late interaction and competitive salience produced the current semantic scorer.

W6/W6b:
- production integration + typed authority;
- strong overall competence but incomplete K64/reliability.

W6c:
- calibration helps confidence, not semantic hard decisions.

W6d/e:
- high overall typed competence does not guarantee fresh-domain high-K generalization.

W6f:
- removing candidate-relative salience is much worse;
- relation reranking is net helpful;
- pooled high-K failure unresolved.

W6g:
- fresh second-order localization -> FIELD_SEMANTIC_COLLAPSE.

W6h:
- small residual semantic adapter did not reproduce.

W6i:
- representation bridge unresolved;
- canonical/factorized text interfaces did not rescue.

W6j:
- stable architectural target COARSE_CONJUNCTION_LIMIT;
- wrong K64 winner usually already beats gold in isolated K2 coarse;
- relation preserves an oracle-perfect coarse ranking ~99.48%.

W7:
- explicit schema-factor smooth-AND architecture;
- verdict CONJUNCTIVE_COARSE_FAIL;
- equal-data free-form retune was much stronger.

W7b:
- factorial typed-only / pair-only / typed+pair;
- exact empirical head `6abc16b579b19e6cae75770e69c7da388194ce78`;
- run `36107487032`;
- verdict FREEFORM_RETUNE_NONREPLICATING.
- Primary K64 = 79.17% on both AS/AT.
- Replica K64 = 81.25% / 72.92%.
- Replication failed because AS gain over stronger frozen baseline was +27.08 pp vs frozen +30 pp gate.
- Pair-only was weak; typed-only carried much gain; no stable attribution class.

W7c external transfer:
- exact head `6e90c6eee154de92e7727acd5daec47370dc62e7`;
- run `36111726339`;
- pinned Banking77 rows 400–799;
- verdict PUBLIC_HIGH_K_TRANSFER_ABSENT.
- frozen W6e = 1/400;
- typed-only = 1/400;
- pair-only = 0/400;
- typed+pair primary = 0/400;
- typed+pair replica = 0/400.
- retuned primary mean confidence ~0.950 at 0/400;
- replica ~0.907 at 0/400.
- full-K/state-once/probability integrity all passed.

W7c therefore proves that strong internal W7b gains do not currently transfer to an unseen real-world intent-label interface.

## 3. Exposed / forbidden evidence

Never use for W8 training, selection, threshold tuning, wording search, mechanism search or fresh authority:
- W6b CONFIRM;
- W6c CONFIRM;
- W6d F;
- W6e L/M;
- W6f N/O/P;
- W6g Q/R/S;
- W6h Y/Z;
- W6i AA/AB/AC;
- W6j AD/AE/AF;
- W7 AL/AM;
- W7b AS/AT;
- Banking77 rows 0–399 exposed by W4a;
- Banking77 rows 400–799 exposed by W7c;
- typed final/test rows;
- public campaign cells.

Historical checkpoints may be loaded only as exact frozen provenance.

## 4. W8 purpose

W8 asks:

**Where does semantic transfer break between raw state language and an unseen schema?**

It distinguishes:
1. state-language representation failure;
2. terse-label/schema representation failure;
3. state↔schema alignment failure;
4. synthetic structured-format dependence;
5. cardinality amplification after low-K semantic transfer is already good.

W8 is diagnostic only and does not train a new model.

## 5. Frozen checkpoints

Primary cross-checkpoint set:
- frozen-w6e-control;
- typed-only-retune;
- typed-plus-pair-primary;
- typed-plus-pair-replica.

Source:
- W7b freeze run `36107487032`;
- artifact `10852555298`;
- digest `sha256:e586cdb2ddea0185ff6663690868fec586a98b7eb617804eb4ea92ed8f45a6f4`.

Pair-only may be included as an optional diagnostic row but does not participate in stable-target authorization.

## 6. Fresh W8 domains

Fresh naturalistic synthetic domains:
- AU household utility support, seed 271501;
- AV clinic appointment administration, seed 271507;
- AW parcel-delivery customer service, seed 271519;
- AX consumer device warranty support, seed 271531.

Each:
- 64 base states;
- 16 latent intent IDs;
- fresh state templates;
- fresh semantic vocabulary;
- four frozen schema renderings/intent;
- no overlap with prior authority IDs/values/templates.

Total:
256 base states.

## 7. Paired representation ladder

Every base uses identical:
- state text;
- latent gold intent ID;
- negative intent IDs;
- candidate ordering;
- nested K membership.

Only schema text changes.

V0 terse-label:
short real-world-style intent name only.

V1 natural-definition:
full natural definition, independently written from state template.

V2 structured-criterion:
same semantic content rendered in the structured criterion style closest to W7/W7b.

V3 lexical-bridge:
V1 plus one frozen semantically valid lexical anchor sharing one key state-family concept.
This is diagnostic ceiling only.

No view may contain a gold bit, domain ID or copied full state.

## 8. Cardinality ladder

Nested from each 16-intent master set:
- K4;
- K8;
- K16.

Same state, gold and candidate IDs.

W8 intentionally does not use K64: first determine whether transfer fails before high cardinality.

## 9. Mandatory metrics

Per checkpoint/domain/view/K:
- coarse top1;
- final top1;
- top5 for K>=8;
- MRR;
- gold margin to best negative;
- relation rescue/damage;
- probability mass max error;
- state encodes/base.

Paired transitions:
- V0 wrong -> V1 right;
- V1 wrong -> V2 right;
- V1 wrong -> V3 right;
- V2 right -> V0 wrong;
- K4 right -> K16 wrong;
- K4 wrong -> K16 wrong.

Report per-domain and pooled.

## 10. Frozen mechanism classifiers

SCHEMA_LABEL_INTERFACE_LIMIT:
- V0 K4 <.55;
- V1-V0 >=+.20;
- V1 >=.65;
- V2-V1 <+.15;
- V3-V1 <+.15.

SYNTHETIC_FORMAT_DEPENDENCE:
- V2-V1 >=+.20;
- V2 >=.70;
- V0 <.65 and V1 <.65.

STATE_SCHEMA_ALIGNMENT_LIMIT:
- V3-V1 >=+.20;
- V3 >=.70;
- V2 does not independently meet synthetic-format dependence.

GENERAL_SEMANTIC_TRANSFER_LIMIT:
- max(V0,V1,V2,V3) K4 <.60.

CARDINALITY_AMPLIFICATION_AFTER_TRANSFER:
- best natural transferable view among V1/V3 has K4 >=.75;
- same view K16 <= K4-.20;
- coarse-wrong accounts for >=70% of K16 final errors.

MIXED_SEMANTIC_TRANSFER_FAILURE:
different stable domains/checkpoints support incompatible classes.

SEMANTIC_TRANSFER_UNRESOLVED:
no class stable.

Cross-checkpoint stable target:
- frozen W6e plus >=1 W7b retuned checkpoint agree;
- agreement on >=3/4 AU/AV/AW/AX;
- no other retuned checkpoint gives stable opposite class on >=3 domains.

Overall:
- STABLE_SEMANTIC_TRANSFER_LOCALIZATION;
- MIXED_SEMANTIC_TRANSFER_LOCALIZATION;
- SEMANTIC_TRANSFER_UNRESOLVED.

## 11. Scientific boundaries

Do not:
- train on W8;
- tune view wording after exposure;
- change thresholds after exposure;
- use Banking77;
- scale/fine-tune A13;
- unfreeze HIRACore;
- add retrieval/examples;
- call V3 production;
- claim real-world superiority.

If stable target:
W9 may test one fresh mechanism specifically for that target.

If mixed/unresolved:
stop local mechanism invention and do representation/prior-art reassessment.

## 12. Current state

Completed:
- W7c closure merged to main `7438c92a6c5d12db818a024d80e6c86bb67a1ae0`;
- issue #103 preregistered;
- branch `feat/r8-w8-semantic-transfer` created from exact post-W7c main;
- this handoff created before any W8 empirical exposure.

No W8 AU/AV/AW/AX data has been materialized or encoded.

Implemented pre-diagnostic:
- src/nmd/semantic_transfer_authority.py:
  - fresh AU/AV/AW/AX domains;
  - 16 latent intents/domain;
  - 4 state variants/intent -> 64 bases/domain;
  - V0/V1/V2/V3 paired schema ladder;
  - nested K4/K8/K16 identities.
- src/nmd/semantic_transfer.py:
  - frozen per-domain classifiers;
  - no post-hoc precedence;
  - >=3/4 domain checkpoint stability;
  - cross-checkpoint stability contract.
- src/nmd/semantic_transfer_cache.py:
  - one state encode/base;
  - paired schema-token cache;
  - candidate/gold identity validation;
  - nested K relative-order validation.
- src/nmd/semantic_transfer_eval.py:
  - frozen competitive scorer + HIRACore evaluation;
  - coarse/final ranks, margins, MRR, relation rescue/damage;
  - K4 representation transitions;
  - K4->K16 paired transitions;
  - classifier metric aggregation.
- scripts/r8_w8_build_cache.py:
  - exact A13 provenance;
  - exact prior text-atom freshness check;
  - 256-base / 3,072-view cache receipt.
- scripts/r8_w8_evaluate.py:
  - exact W7b freeze/checkpoint SHA validation;
  - four frozen checkpoint evaluation;
  - cross-checkpoint stability output.
- tests/test_semantic_transfer.py;
- tests/test_semantic_transfer_cache.py;
- .github/workflows/r8-w8-unit.yml.

Pre-data transition-stage semantics were frozen in issue #103 before any A13 exposure.

Exact full implementation head:
705cf7a84629be0ed7f20037df5abf34d199ff7e

Exact W8 unit run:
36113431426 — PASS.

Repository CI for the same implementation head:
36113434791 — still running at this handoff update.

No AU/AV/AW/AX A13 cache exists yet.
No W8 empirical localization exists yet.

Not yet completed:
- exact implementation repo CI completion;
- gated W8 diagnostic authority workflow;
- fresh A13 cache exposure;
- fixed-checkpoint evaluation;
- authoritative W8 outcome.

## 13. Immediate continuation

1. Implement AU/AV/AW/AX generator with explicit latent intents and four renderings.
2. Lock exact semantic identity and nested K tests.
3. Implement classifiers in library code before A13 exposure.
4. Add base-centric state-once cache.
5. Load exact W7b frozen checkpoints only.
6. Evaluate all paired views without training.
7. Add cross-checkpoint stability.
8. Run pre-data unit/CI.
9. Enable authority only after contracts pass.
10. Freeze exact run/artifact/result here before merge.

## 14. Handoff discipline

At every meaningful session update append:
- exact branch head;
- runs/conclusions;
- artifacts/digests;
- exposure status;
- exact metrics/classification if exposed;
- remaining work;
- forbidden next moves.

A future AI should be able to continue without chat memory.


---

## 15. Authoritative W8 closure

Exact empirical head:
`8f30d2b2f1d7b81f7e48ed43c9a89b1e46ead8c6`

Authority run:
`36113828435`

All jobs PASS:

`unit -> fresh AU/AV/AW/AX cache + exact frozen W7b provenance -> fixed-checkpoint semantic-transfer evaluator`.

Artifacts:
- frozen W7b checkpoints `10854865013`, digest `sha256:a307474a92c62ee6adc629f65a41e072994130dcb73b87f38e9522c7377d9504`;
- semantic-transfer cache `10854766296`, digest `sha256:370892853e5889b7055ddc2e2198f99183956c121244760f7bd2f5c92c9a0dcd`;
- semantic-transfer audit `10854675985`, digest `sha256:de128f2111e71abea5682a65e62e9ba053d5d05e5fcc72869efe4ba44935baa7`.

Integrity:
- 256 fresh base states;
- 3,072 paired representation/cardinality views;
- one state encode/base;
- no training;
- no Banking77 reuse;
- no prior exposed authority rows;
- no typed final/test;
- campaign cells 0;
- probability mass max error `1.7171259969472885e-07`.

### Frozen outcome

**`STABLE_SEMANTIC_TRANSFER_LOCALIZATION`**

Stable classification:

**`GENERAL_SEMANTIC_TRANSFER_LIMIT`**

`mechanism_lane_authorized = true`.

Cross-checkpoint stability holds because:
- frozen W6e control and all three W7b retuned checkpoints classify as `GENERAL_SEMANTIC_TRANSFER_LIMIT` on AU/AW/AX;
- AV is unresolved for all four rather than an opposing stable mechanism;
- therefore the same class agrees across 3/4 fresh domains with no stable opposite class.

### Pooled K4/K16 anatomy

#### Frozen W6e control
- V0 terse-label K4: 37.89%;
- V1 natural-definition K4: 50.78%;
- V2 structured-criterion K4: 40.63%;
- V3 lexical-bridge K4: 39.06%;
- best natural K4: 50.78% (V1);
- best natural K16: 22.66%;
- K16 coarse-wrong share among final errors: 97.47%.

#### Typed-only retune
- V0 K4: 36.33%;
- V1 K4: 51.95%;
- V2 K4: 45.31%;
- V3 K4: 48.83%;
- best natural K4: 51.95%;
- best natural K16: 25.39%;
- K16 coarse-wrong share: 97.38%.

#### Typed+pair primary
- V0 K4: 38.67%;
- V1 K4: 55.86%;
- V2 K4: 50.00%;
- V3 K4: 49.22%;
- best natural K4: 55.86% (V1);
- best natural K16: 28.91%;
- K16 coarse-wrong share: 96.15%.

#### Typed+pair replica
- V0 K4: 39.06%;
- V1 K4: 54.69%;
- V2 K4: 51.17%;
- V3 K4: 54.30%;
- best natural K4: 54.69% (V1);
- best natural K16: 26.56%;
- K16 coarse-wrong share: 97.87%.

### Domain stability anatomy

All four checkpoints:
- AU -> `GENERAL_SEMANTIC_TRANSFER_LIMIT`;
- AV -> `SEMANTIC_TRANSFER_UNRESOLVED`;
- AW -> `GENERAL_SEMANTIC_TRANSFER_LIMIT`;
- AX -> `GENERAL_SEMANTIC_TRANSFER_LIMIT`.

This is unusually stable across checkpoints. W7b retuning changes absolute accuracy somewhat but does not change the failure class.

### What W8 falsifies

The current evidence does **not** support these as the primary stable explanation:

1. **terse-label interface only**
   - V1 natural definitions help, but pooled best natural K4 remains only ~51-56%;
   - the gain is not enough to cross the preregistered schema-label rescue conditions.

2. **synthetic structured-format dependence**
   - V2 does not produce the required +20 pp recovery over V1;
   - on pooled metrics V2 is generally similar to or worse than V1.

3. **lexical bridge as the main missing piece**
   - V3 does not produce the required +20 pp recovery over V1;
   - it often remains around the same low-K accuracy range.

4. **cardinality-only amplification after adequate semantic transfer**
   - low-K semantic transfer is already below the required adequacy threshold;
   - K16 degradation is real, but it happens on top of an already weak K4 semantic interface.

### Scientific meaning

The stable bottleneck has moved one level deeper than the W6/W7 high-cardinality story.

The frozen semantic system does **not** first become correct at low cardinality and then fail only because of candidate count. On fresh AU/AW/AX semantics, even K4 accuracy stays below 60% across every tested representation view and every frozen checkpoint.

Natural definitions improve over terse labels, but not enough. Structured criterion formatting does not rescue. A lexical bridge does not rescue. Retuned scorers that were very strong on W7b synthetic authorities do not materially change this class.

The most defensible interpretation is therefore:

**the current A13 + shared projection/late-interaction semantic geometry does not transfer robustly to unseen semantic categories even before high cardinality becomes the dominant difficulty.**

This is a representation/semantic-generalization problem, not merely a ranking-budget problem.

### Permanent forbidden evidence after W8

AU/AV/AW/AX are now exposed.

Never use them for:
- training;
- DEV selection;
- schema wording search;
- prompt search;
- threshold tuning;
- checkpoint selection;
- calibration;
- mechanism selection;
- seed selection.

Keep all earlier forbidden evidence frozen as listed above.

### Authorized next direction

W8 authorizes one fresh mechanism lane because the target is stable.

That lane should target **general semantic transfer**, not:
- another K64-specific ranking trick;
- IDF/salience adjustment;
- relation reranking;
- explicit AND aggregation;
- calibration-only repair;
- schema text formatting tricks.

A sensible W9 must directly test whether a small semantic-transfer mechanism can improve unseen low-K semantics while keeping:
- A13 frozen unless a separately preregistered architecture question explicitly changes that;
- HIRACore frozen in the primary causal comparison;
- state-once execution;
- small parameter budget;
- wholly fresh domains;
- at least two untouched CONFIRM domains;
- an equal-data existing free-form control;
- no reuse of AU/AV/AW/AX.

Because W7c showed catastrophic real-world transfer failure, W9 should also require at least one external/public frozen evaluation only **after** internal fresh mechanism rescue succeeds. Do not use public data for mechanism selection.

## 16. Current continuation state

W8 empirical work is complete.

Before opening W9:
1. merge PR #104 only after this closure document and CI are clean;
2. close issue #103;
3. branch W9 from the resulting main commit;
4. create `research/R8-W9-HANDOFF.md` immediately;
5. preregister W9 mechanism, fresh train/DEV/CONFIRM domains, parameter budget, controls, gates and external-validation boundary before any empirical exposure;
6. never reuse AU/AV/AW/AX.

A future AI should read this file first, then W7c, W7b, W7 and W6j handoffs.
