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


## Authoritative closure

Exact empirical head:
- `eb72c1da5c91b511bf9ce727a1c639b06bcb2359`.

Authority run:
- `35987408790`;
- all jobs PASS;
- chain: unit -> upstream provenance/cache -> four preregistered paths -> independent DEV-E freeze -> untouched CONFIRM-F.

Untouched CONFIRM-F:
- seed `183353`;
- 192 states / 960 decisions;
- generated only after every DEV-E checkpoint froze;
- state encodes/case = 1.0;
- probability integrity PASS;
- W6b CONFIRM rows used = false;
- W6c CONFIRM rows used = false;
- typed final/test rows used = false.

### Frozen control

`frozen-w6b-control`:
- overall **40.208%**;
- choice **30.208%**;
- noul **66.146%**;
- score **37.240%**;
- diagnosis K64 **6.25%**;
- hard Brier **0.74910**;
- soft-target ECE **0.22282**;
- score MAE **0.75852**.

### Primary causal comparison

`single-source-scorer-only`:
- overall **84.792%**;
- choice **69.531%**;
- noul **99.479%**;
- score **92.708%**;
- diagnosis K64 **25.00%**;
- hard Brier **0.22452**;
- soft-target ECE **0.09738**;
- score MAE **0.15985**.

`multi-source-scorer-only`:
- overall **84.271%**;
- choice **70.313%**;
- noul **91.667%**;
- score **94.531%**;
- diagnosis K64 **27.083%**;
- hard Brier **0.25611**;
- soft-target ECE **0.07859**;
- score MAE **0.16150**.

Same-budget scorer-only comparison:
- equal cases: true;
- equal optimizer steps: true;
- equal trainable parameters: true;
- overall delta multi - single: **-0.521 pp**;
- K64 delta: **+2.083 pp**;
- choice delta: **+0.781 pp**;
- score delta: **+1.823 pp**.

Therefore the preregistered diversity gates fail. Domain diversity applied only to the 32,769-parameter scorer does **not** causally rescue held-out semantic generalization.

### Joint localization diagnostic

`multi-source-joint`:
- 454,928 trainable parameters;
- overall **89.167%**;
- choice **82.813%**;
- noul **93.229%**;
- score **93.490%**;
- diagnosis K64 **54.167%**;
- hard Brier **0.21205**;
- soft-target ECE **0.14365**;
- score MAE **0.13817**.

Relative to multi-source scorer-only:
- overall gain **+4.896 pp**;
- K64 gain **+27.083 pp**.

The joint path passes the frozen overall/choice/score/noul competence gates and has a very large K64 gain, but misses the absolute K64 >=55% competence gate:
- observed 26/48 = **54.167%**;
- required at least 27/48 = **56.25%** under this finite sample;
- margin to the preregistered threshold corresponds to **one CONFIRM-F K64 case**.

This is not promoted to rescue. The frozen verdict remains:

**`GENERALIZATION_FAIL`**

The result is nevertheless highly diagnostic: the remaining domain sensitivity is not confined to the 32,769-parameter competitive projection. Joint adaptation of the typed HIRA core and scorer recovers a large amount of held-out performance.

### Preserved artifacts

- CONFIRM artifact ID `10802909682`;
- CONFIRM artifact digest `sha256:011c25e500dc5a8913b904ca0b43422a09789dce494444198af4a2097b02325a`;
- frozen checkpoint artifact ID `10802804564`;
- frozen checkpoint artifact digest `sha256:7ab1fce8732d5f0c5b84b54149e1918470544a0bb94429167edcd1bad8e46b34`.

## Scientific interpretation

W6d falsifies the narrow hypothesis that equal-budget multi-domain diversity in the scorer alone is sufficient.

It provides strong evidence that domain robustness is a property of the **full typed decision path**, because joint HIRA+scorer adaptation improves held-out K64 by 27.08 points over scorer-only while also improving overall accuracy.

Because CONFIRM-F is exposed, do not:
- lower the 55% K64 gate;
- retune on domain F;
- add epochs based on F;
- reuse F rows or values in the next lane;
- call the near-miss a rescue.

The next lane should be a wholly fresh held-out-domain replication of the **joint-adaptation hypothesis**, with new domains/seeds/lexicons and preregistered gates. If joint adaptation reproduces competence there, promote full-core domain-robust training; if not, investigate the residual high-K binding failure without using exposed W6d rows.
