# R8-W8 handoff — semantic/schema transfer decomposition audit

Status: **PRE-DIAGNOSTIC. No W8 AU/AV/AW/AX empirical cache or localization result exists yet.**

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

Not yet implemented:
- fresh W8 paired-domain generator;
- representation-view contracts;
- nested K identities;
- state-once/token cache;
- fixed-checkpoint evaluator;
- frozen classification library/tests;
- W8 unit workflow;
- W8 diagnostic authority workflow.

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
