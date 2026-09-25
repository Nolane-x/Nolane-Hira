# R8-W7c handoff — frozen external Banking77 transfer audit

Status: **CLOSED EXTERNAL TRANSFER AUTHORITY. Frozen verdict: `PUBLIC_HIGH_K_TRANSFER_ABSENT`. Banking77 rows 400–799 are exposed and forbidden for tuning.**

Issue: #101

Branch:
feat/r8-w7c-banking77-transfer

Base main:
d9af78ae92e2ca7ca3eb57347889676bfc41c83a

Read first when restoring:
1. this file;
2. research/R8-W7B-HANDOFF.md;
3. research/R8-W7-HANDOFF.md;
4. research/R8-W6J-HANDOFF.md;
5. issue #101.

## 0. Project identity

Nolane HIRA is a compact non-autoregressive typed decision system.

Long-term goals:
- encode state once per case;
- dynamic schemas and changing candidate sets;
- typed choice / score / noul;
- small local-friendly model;
- semantic binding and high-cardinality decisions;
- explicit probability/reliability integrity;
- fresh sealed authorities and immutable negative results.

## 1. Frozen semantic encoder

A13 remains exact and frozen:
- microsoft/xtremedistil-l6-h256-uncased
- revision 4226d9e4d2c08703e5cb0491b479bfc6a1607181
- weight SHA 5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880
- max length 256.

## 2. Decisive history entering W7c

W6j localized the stable high-K failure to COARSE_CONJUNCTION_LIMIT.

W7 tested an explicit factor-evidence smooth-AND branch and closed:
CONJUNCTIVE_COARSE_FAIL.

W7b factorially separated:
- typed-only retuning;
- pair-only retuning;
- typed+pair primary;
- typed+pair replica.

Exact W7b empirical head:
6abc16b579b19e6cae75770e69c7da388194ce78

Authority run:
36107487032

Frozen W7b verdict:
FREEFORM_RETUNE_NONREPLICATING.

Important W7b metrics:
- primary K64 79.17% on AS and AT;
- replica K64 81.25% AS / 72.92% AT;
- replication failed because AS gain over its stronger frozen baseline was +27.08 pp, below the preregistered +30 pp gate;
- no stable attribution class.

W7b merged main:
d9af78ae92e2ca7ca3eb57347889676bfc41c83a.

AS/AT are exposed permanently and forbidden for tuning.

## 3. Why W7c exists

W7b does not authorize production promotion.

W7c asks only:

**Does the frozen W7b high-K improvement transfer at all to external/public high-cardinality intent classification?**

No model is trained or selected in W7c.

## 4. Public task and freshness boundary

Dataset:
mteb/banking77

Revision:
18072d2685ea682290f7b8924d94c62acc19c0b2

Split:
test

Historical W4a already exposed rows 0–399 with the old W3 model and achieved 0/400.

Therefore W7c does not reuse that slice.

Frozen W7c selection:
- zero-based rows 400–799;
- exactly 400 examples.

This is public transfer evidence, not an untouched architecture-confirmation benchmark.

## 5. Frozen schema/protocol

- label space: all 77 unique label_text values from the pinned test split;
- label order: sorted;
- option semantics: underscores replaced with spaces;
- option IDs: intent-000 through intent-076;
- question: Which banking intent does message express?;
- state: canonical JSON with field message.

Execution:
- competitive coarse scorer;
- pooled relation mode;
- full K=77 forced budget;
- adaptive budget false;
- one state encode per case;
- one cached schema;
- no candidate pruning;
- no retrieval;
- no Banking77 training rows;
- no calibration;
- no prompt search.

## 6. Frozen checkpoints

Authority source:
W7b freeze artifact 10852555298 from run 36107487032.

Artifact digest:
sha256:e586cdb2ddea0185ff6663690868fec586a98b7eb617804eb4ea92ed8f45a6f4

Evaluate every frozen path:
1. frozen-w6e-control
2. typed-only-retune
3. pair-only-retune
4. typed-plus-pair-primary
5. typed-plus-pair-replica

Do not select among them using W7c rows.

## 7. Metrics

Per path:
- accuracy
- macro F1
- hard Brier
- NLL
- raw 15-bin ECE
- mean confidence
- AURC
- probability mass max error
- state encode calls/case
- candidate budget min/max
- tail mass max
- descriptive CI CPU p50/p95.

No calibrated result is produced.

## 8. Frozen classification

PUBLIC_HIGH_K_TRANSFER_SIGNAL:
- primary gain over frozen >= +0.10;
- replica gain over frozen >= +0.10;
- primary and replica accuracy >=0.25;
- state-once/probability integrity pass.

PUBLIC_HIGH_K_TRANSFER_WEAK:
- full signal fails;
- at least one primary/replica gain >=+0.05;
- neither primary nor replica is more than .05 below frozen.

PUBLIC_HIGH_K_TRANSFER_ABSENT:
otherwise.

This classification cannot override W7b's frozen verdict.

## 9. Claim discipline

Never:
- populate laya.app.banking77_full from rows 400–799;
- present this slice as direct Laya/Jev parity;
- tune after exposure;
- change labels/prompt/slice after exposure;
- pick a better candidate after exposure;
- alter W7b's verdict.

If transfer signal appears, only then broaden frozen external evaluation to more public tasks.

## 10. Current state

Completed:
- W7b closure merged;
- issue #101 preregistered;
- W7c branch created from exact post-W7b main;
- this handoff created before external row exposure.

Implemented pre-exposure:
- src/nmd/banking77_transfer.py:
  - deterministic rows 400:800 loader;
  - exact 77-label schema;
  - competitive full-K production runtime evaluation;
  - accuracy/F1/Brier/NLL/ECE/AURC/integrity metrics;
  - frozen transfer classifier.
- scripts/r8_w7c_banking77_transfer.py:
  - exact W7b freeze/checkpoint SHA verification;
  - exact pinned A13 verification;
  - five-path frozen evaluation;
  - no scorecard population.
- tests/test_banking77_transfer.py:
  - exact slice identity;
  - frozen candidate identity;
  - SIGNAL/WEAK/ABSENT gate semantics.
- .github/workflows/r8-w7c-unit.yml pre-exposure unit gate.

Exact implementation head before this handoff-only update:
203f30a6c97f2dfd4a5a018588ecae7e630dce7c

Current validation:
- W7c unit run 36111320621 is the exact-head validation run;
- repository CI run 36111320689 is the exact-head CI run;
- no external authority workflow exists yet;
- no Banking77 rows 400–799 have been requested or exposed.

Not yet completed:
- wait for exact-head unit/CI PASS;
- add gated external authority workflow;
- execute one frozen public transfer run;
- freeze result into this handoff.

## 11. Immediate next work

1. Implement deterministic rows 400:800 Banking77 loader.
2. Implement competitive full-K external evaluator.
3. Verify W7b freeze artifact/checkpoint hashes.
4. Add unit tests that do not load Banking77 labels/results.
5. Add branch-scoped gated workflow.
6. Only after unit/CI green, enable external authority.
7. Once exposure begins, mutate no scientific protocol.
8. Freeze exact result into this file before merge.

## 12. Handoff discipline

At every meaningful session update record:
- exact branch head;
- runs;
- artifacts/digests;
- exposure status;
- exact metrics/classification;
- remaining work;
- forbidden next moves.

A future AI must be able to continue without chat memory.


---

## 13. Authoritative W7c closure

Exact authority head:
`6e90c6eee154de92e7727acd5daec47370dc62e7`

Authority run:
`36111726339`

All jobs PASS:

`unit -> exact W7b freeze provenance -> frozen external Banking77 rows 400:800 evaluation`.

Pre-exposure validation:
- exact implementation unit run `36111320621`, attempt 2: PASS;
- exact implementation repository CI `36111320689`: PASS;
- authority unit: PASS;
- authority upstream provenance: PASS.

The explicit exposure marker was reached only after those gates:

`R8_W7C_BANKING77_ROWS_400_799_EXPOSURE_BEGIN`.

### Artifacts

Authoritative result:
- artifact `10852844687`;
- name `r8-w7c-banking77-transfer`;
- digest `sha256:f16f88806df51d84e92f181f482925e0e3520356117c258485adf9d0e01b3b69`.

Repackaged exact W7b freeze used by W7c:
- artifact `10852739904`;
- digest `sha256:7d6eb076d80be5d20da1b89035ca34791f3c424d60169d688b0d228e2e816cdb`.

Original W7b freeze provenance remained:
- source run `36107487032`;
- artifact `10852555298`;
- digest `sha256:e586cdb2ddea0185ff6663690868fec586a98b7eb617804eb4ea92ed8f45a6f4`.

### External protocol integrity

Dataset:
- `mteb/banking77`;
- revision `18072d2685ea682290f7b8924d94c62acc19c0b2`;
- split `test`;
- zero-based rows 400–799;
- 400 examples;
- 77 labels;
- no reuse of W4a first-400 rows.

For every checkpoint:
- full K=77;
- one state encode/case;
- adaptive budget false;
- no pruning;
- no Banking77 task training;
- no retrieval;
- no calibration fit;
- no W7c checkpoint selection;
- no Laya scorecard cell populated;
- no Jev scorecard cell populated;
- probability mass max error <2.5e-7;
- tail mass 0.

W7b's frozen verdict was not overridden.

## 14. Frozen external result

Authoritative verdict:

**`PUBLIC_HIGH_K_TRANSFER_ABSENT`**

### Frozen W6e control

- accuracy: **0.25% = 1/400**;
- macro F1: 0.000634;
- hard Brier: 1.35245;
- NLL: 6.96353;
- raw ECE: 0.56180;
- mean confidence: 0.56430;
- AURC: 0.99954;
- CI CPU p50: 11.41 ms/case;
- CI CPU p95: 13.19 ms/case.

### Typed-only retune

- accuracy: **0.25% = 1/400**;
- macro F1: 0.000634;
- hard Brier: 1.68647;
- NLL: 8.77279;
- raw ECE: 0.82354;
- mean confidence: 0.82493;
- AURC: 0.99998;
- gain vs frozen: **0.00 pp**.

### Pair-only retune

- accuracy: **0.00% = 0/400**;
- macro F1: 0;
- hard Brier: 1.56488;
- NLL: 7.44498;
- raw ECE: 0.75071;
- mean confidence: 0.75071;
- AURC: 1.0;
- gain vs frozen: **-0.25 pp**.

### Typed+pair primary

- accuracy: **0.00% = 0/400**;
- macro F1: 0;
- hard Brier: 1.90431;
- NLL: 11.07716;
- raw ECE: 0.95018;
- mean confidence: **0.95018**;
- AURC: 1.0;
- gain vs frozen: **-0.25 pp**.

### Typed+pair replica

- accuracy: **0.00% = 0/400**;
- macro F1: 0;
- hard Brier: 1.84128;
- NLL: 11.23850;
- raw ECE: 0.90736;
- mean confidence: **0.90736**;
- AURC: 1.0;
- gain vs frozen: **-0.25 pp**.

### Frozen classifier anatomy

- frozen accuracy 0.0025;
- primary accuracy 0.0;
- replica accuracy 0.0;
- typed-only gain 0.0;
- pair-only gain -0.0025;
- primary gain -0.0025;
- replica gain -0.0025.

All runtime-integrity gates pass.

Therefore neither SIGNAL nor WEAK transfer criteria are remotely close.

## 15. Scientific meaning

W7c is a decisive negative external-transfer result.

The strong W7/W7b synthetic high-K numbers do **not** transfer to held-out Banking77 intent classification.

The failure is more severe than merely missing a high-cardinality threshold:
- primary and replica are both 0/400;
- typed-only is only 1/400;
- pair-only is 0/400;
- the more aggressively retuned checkpoints become extremely confident while remaining essentially always wrong.

This directly falsifies the working idea that the W7b free-form retune recipe can currently be treated as a general semantic high-cardinality solution.

The result also changes how earlier internal results must be interpreted:

- W6j's `COARSE_CONJUNCTION_LIMIT` remains valid for the synthetic authority family on which it was diagnosed.
- W7's explicit conjunction mechanism still failed.
- W7b's synthetic typed+pair gains were real on AS/AT but domain/protocol specific.
- Those gains do not establish cross-domain natural-language schema transfer.

### Strong new bottleneck hypothesis

The next architecture research should target **schema/semantic transfer**, not K64 ranking mechanics in isolation.

The current scorer can become highly competent on synthetic structured schema families, yet fail to align:
- unconstrained natural user utterances;
- terse external intent labels;
- unseen real-world semantic categories.

The very high confidence of wrong retuned checkpoints is especially important. This suggests adaptation may sharpen the learned synthetic decision geometry without learning a transferable semantic interface.

This is an interpretation to test, not a promoted causal result.

## 16. What W7c does NOT mean

Do not claim:
- HIRA is universally incapable of Banking77;
- competitive late interaction is universally broken;
- high cardinality alone causes the failure;
- W7b training damaged every domain;
- direct Laya/Jev comparison from rows 400–799.

The W7c slice differs from the official first-400 Laya application cell, so it must not be used as that scorecard.

W4a first-400 was already a separate clean negative result with the old W3 specialist head.

## 17. Permanent forbidden evidence after W7c

Banking77 test rows 400–799 are now exposed.

Never use them for:
- training;
- DEV selection;
- prompt/schema wording search;
- scorer selection;
- calibration;
- threshold tuning;
- architecture choice;
- seed choice.

Rows 0–399 were already exposed by W4a.

Thus rows 0–799 must be treated as non-fresh for future research.

## 18. Authorized next research direction

Per the preregistered W7c boundary, because transfer is ABSENT:

**do not broaden to easier public benchmarks merely to collect positive cells.**

Return to architecture/generalization research.

The next lane should diagnose a new question on wholly fresh, non-Banking77 data:

**Where does semantic transfer break between raw state language and unseen schema labels/definitions?**

A useful fresh diagnostic phase should separate:
1. state-language representation failure;
2. terse-label/schema representation failure;
3. state↔schema alignment failure;
4. synthetic-format dependence;
5. cardinality-independent semantic failure.

It should include low-K and high-K versions of the same unseen semantics so that semantic transfer can be separated from candidate count.

Do not train another scorer before this localization is complete.

## 19. Handoff state after W7c

W7c empirical work is complete.

Before another AI continues:
1. freeze this closure on PR #102;
2. merge only after closure-doc CI is clean;
3. close issue #101;
4. create the next diagnostic lane from resulting main;
5. create its handoff before any fresh data exposure;
6. preserve the W7c artifact/run/hashes above;
7. never reuse Banking77 rows 0–799 as fresh authority.

The next AI should read this file first, then W7b, W7, and W6j handoffs.
