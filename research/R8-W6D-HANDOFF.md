# R8-W6d handoff — held-out domain generalization

Status: **PRE-AUTHORITY FROZEN. CONFIRM-F sealing repaired before empirical execution; no W6d empirical verdict is valid yet.**

Issue: #83

## Frozen premise

W6c closed as `RELIABILITY_CALIBRATION_PARTIAL`.

Fresh W6c CONFIRM showed:
- frozen W6b production control overall 45.729%;
- choice 43.49%;
- score 43.23%;
- noul 55.21%;
- diagnosis K64 43.75%;
- soft-target ECE 0.24553.

Three primitive temperatures reduced soft-ECE to 0.09721 but did not change hard predictions.

Therefore W6d targets held-out semantic/domain generalization, not calibration.

## Frozen upstream

A13 stays frozen.

Initialize all W6d paths from the exact W6b DEV-selected production checkpoint:
- HIRA SHA `925f74094ac4ae583c015ea2a0be32ec692d885ec64dcf9cf3d94b64be0ccf42`;
- scorer SHA `50abb2e8136599bcaf5c41d61036e3c335a7589c6244536cc0292dcea15b1ef0`;
- artifact `10792488690`.

Forbidden:
- W6b CONFIRM rows;
- W6c CONFIRM rows;
- typed final/test rows;
- public campaign cells.

## Six domains

A/B/C/D are source domains.
E is held-out DEV.
F is untouched held-out CONFIRM.

All six domains use pairwise-disjoint field values, role wording and templates.

Same typed task in every domain:
- diagnosis choice;
- response choice;
- needs_review noul;
- risk score;
- urgency score.

## Equal-budget causal comparison

Single-source TRAIN:
- A only;
- 384 states / 1,920 decisions;
- 8 cases per K x severity x confidence stratum.

Multi-source TRAIN:
- A/B/C/D;
- 96 states per source = 384 total;
- 2 cases per joint stratum per source.

The A portion of multi-source is a stratified semantic subset of the single-source A corpus. This reduces sampling confounding while preserving exactly equal total case/step budgets.

Held-out DEV E:
- 192 / 960;
- 4 cases per joint stratum.

Held-out CONFIRM F:
- 192 / 960;
- 4 cases per joint stratum;
- sealed until all DEV checkpoints freeze.

Seeds:
- A 181149;
- B 181151;
- C 181153;
- D 181157;
- DEV E 182251;
- CONFIRM F 183353;
- training 1201.

## Preregistered paths

0. `frozen-w6b-control`: 0 trainable params.
1. `single-source-scorer-only`: 32,769 params, A only.
2. `multi-source-scorer-only`: 32,769 params, A/B/C/D.
3. `multi-source-joint`: 454,928 params, A/B/C/D.

Candidate 1 vs 2 is the primary diversity test.
Candidate 2 vs 3 localizes remaining domain sensitivity to scorer-only vs full typed core.

No cross-candidate winner selection is allowed before CONFIRM. Each trainable path freezes its own DEV-E checkpoint, then all four paths are evaluated once on domain F.

## Current implementation

Implemented:
- six-domain authority generator;
- equal-budget source/cache builder;
- three trainable paths plus frozen production control;
- independent DEV-E checkpoint freezing;
- post-freeze CONFIRM-F evaluator;
- full gated GitHub Actions authority;
- leakage/balance/freshness/parameter/step-parity contracts.

Before any authority cache is generated:
1. unit tests must prove counts/balance;
2. A/B/C/D/E/F values/templates must be disjoint;
3. TRAIN/DEV/CONFIRM gold semantics must be disjoint;
4. multi-source A must be a real stratified subset of single-source A semantics;
5. W6d values must not overlap W6b/W6c/W5 authority values;
6. CONFIRM must remain sealed.

No W6d empirical authority is valid before these gates pass.


## Pre-authority CONFIRM-F seal repair

A protocol audit found that an early unit-test version explicitly called
`generate_w6d_confirm(allow_confirm=True)` to inspect balance and semantic
disjointness. That violates the preregistered rule that domain-F authority rows
must not be materialized before all DEV-E checkpoints freeze.

This was repaired before an eligible empirical authority executed:
- unit tests no longer materialize domain F;
- CONFIRM-F balance is checked from frozen static count contracts;
- F lexical/template reservation is checked statically from domain specs;
- source/DEV semantic disjointness is tested without F generation;
- a repository-level regression contract requires exactly one
  `allow_confirm=True` capability, in `scripts/r8_w6d_confirm.py`;
- build, train, freeze, source modules and unit tests are forbidden from
  carrying that capability.

Any authority run/head before this repair is non-authoritative even if it did
not reach data execution.

The final eligible authority must therefore begin at or after commit
`a53346d4f0d76badfad99ca06b4e800e83a122a2`.

No model formula, optimizer, authority seed, domain vocabulary, DEV ordering,
gate, or verdict rule changed in this seal repair.


## Pre-data authority orchestration repair

The first post-seal authority head `98a992e7e591b83ef704e557c2b1ba4bf8367875`
failed in the **unit gate before upstream/cache execution** because the authority
workflow referenced a nonexistent regression file:
`tests/test_competitive_production.py`.

This was an orchestration error only:
- compile passed;
- no W6d source/DEV cache was built by that run;
- no candidate trained;
- no DEV checkpoint frozen;
- CONFIRM-F remained ungenerated.

The workflow now executes the actual production regression set used by W6:
- `tests/test_competitive_coarse_runtime.py`;
- `tests/test_runtime.py`;
- `tests/test_token_relation_modes.py`;
- `tests/test_training.py`;
- `tests/test_typed_decisions.py`;
- plus `tests/test_typed_competitive_cache.py` and both W6d suites.

The standalone W6d unit workflow uses the same regression set so this class of
workflow drift is caught before future authority execution.

No data/model/seed/gate/verdict change was made.
