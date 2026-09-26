# R8-W17 handoff — field-isolated state representation authority

Status: **PRE-DATA MECHANISM AUTHORITY. No CK/CL/CM/CN/CO/CP/CQ A13 cache or W17 result exists yet.**

Issue: #121

Branch:
`feat/r8-w17-field-isolated-state`

Base main:
`0a84fd45643ad76459fc5d23369a31ee123a985d`

Read first:
1. this file;
2. `research/R8-W16-HANDOFF.md`;
3. `research/R8-W15-HANDOFF.md`;
4. issue #121.

## 0. Project identity

Nolane HIRA is a compact non-autoregressive typed decision engine.

Long-term target:
- one logical state compilation;
- dynamic schemas;
- typed choice / score / noul;
- strong fresh semantic transfer;
- high-cardinality decisions;
- reliable/calibrated outputs;
- small local-friendly footprint;
- sealed empirical authorities.

Scientific discipline:
- no gate changes after exposure;
- exposed data permanently forbidden;
- PARTIAL/FAIL is never promoted post hoc;
- internal authority does not imply external superiority.

## 1. Frozen base

A13:
- microsoft/xtremedistil-l6-h256-uncased;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- SHA `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length 256;
- frozen.

W9 projection-semantic scorer:
- SHA `8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102`;
- freeze run `36122220588`;
- artifact `10858424139`.

HIRACore:
- frozen;
- used only by descriptive current-production D0 control.

W17 trainable parameter count:
**0**.

No optimizer.
No checkpoint selection.

## 2. Decisive history entering W17

W5 established late-interaction semantic binding.

W6-W7 established typed runtime and internal high-K behavior but did not prove broad fresh transfer.

W7c showed public Banking77 transfer was absent.

W8 localized low-K semantic-transfer weakness.

W9 found full projection semantic retuning stronger than tiny bridges.

W10-W12 showed candidate-independent semantic geometry often exceeded the full production path.

W13 showed strict paraphrase consistency was high-quality but sparse.

W14 established 4/4 `CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE`:
E 94.92 / 90.23 / 84.77% K4/K8/K16.

W15 anchor-preserving residual failed because the W14 anchor itself collapsed in a fresh typed regime:
`ANCHOR_PRESERVING_FAIL`.

W16 directly matched bare vs decorated state rendering and closed with:
- outcome `STABLE_REGIME_TRANSFER_LOCALIZATION`;
- stable target `CONTEXTUAL_STATE_CONTAMINATION`;
- 4/4 CG/CH/CI/CJ.

W16 pooled SYM:
- R0 bare 90.63 / 86.33 / 77.73%;
- R1 decorated 41.80 / 25.39 / 17.58%;
- R2 decorated contextual embeddings + prefix-only scoring 47.27 / 28.13 / 18.75%.

Because R2 kept exact R0 prefix token IDs but did not recover, W16 localized damage before scoring: full-sequence contextualization changed the intent-bearing A13 token representations.

Exact W16 empirical head:
`af83c92e7f25c70f044abf38f9a64ff5503746d2`

Authority:
`36202648911`

Audit artifact:
`10893240677`
digest `sha256:3306a48b5c9081ab400a88d89ea74134ef9f162c1197470789255dccb07d174b`.

## 3. W17 hypothesis

Preserve one logical typed state, but prevent unrelated semantic fields from self-attending into each other before the semantic anchor is computed.

Field-isolated candidate:
- one logical compile;
- one A13 invocation;
- batch of three independent sequences;
- field identity preserved.

Fields:
1. intent;
2. severity;
3. confidence.

Primitive relevance:
- diagnosis -> intent;
- response -> severity;
- needs_review -> severity + confidence;
- risk -> severity;
- urgency -> severity + confidence.

No learned field weights.

## 4. Candidate paths

### FULL_SINGLE

One flattened sequence containing all three fields.
One A13 invocation, one sequence.
All primitives use the full-state tokens.

### FULL_TRIPLICATE_CONTROL

One A13 invocation with three identical full-state sequences.
Each primitive uses one full copy.

Purpose:
equal sequence/batch compute control for FIELD_ISOLATED.

Required:
- prediction identity vs FULL_SINGLE = 1.0;
- max semantic-logit difference <=1e-5.

### FIELD_ISOLATED

One A13 invocation with exactly:
- intent sequence;
- severity sequence;
- confidence sequence.

Primitive-relevant token sets are selected by the frozen mapping above.

### PRODUCTION_D0_FULL

Exact current full-state D0 competitive scorer + frozen HIRACore.
Descriptive control only.

## 5. Semantic operator

Every option has D0/D1/D2.

Exact W9 projection.
L2 normalize.
D2S MaxSim mean.
S2D MaxSim mean.
SYM=.5*(D2S+S2D).

Multiview E = arithmetic mean D0/D1/D2.

Same operator for FULL_SINGLE, FULL_TRIPLICATE and FIELD_ISOLATED.

## 6. Fresh authority

TRAIN/descriptive:
- CK community garden plot administration — 361201;
- CL residential heat-pump support — 361207;
- CM campus media-equipment lending — 361219;
- CN ferry commuter account services — 361227.

DEV/descriptive:
- CO neighborhood workshop membership services — 362331.

Untouched CONFIRM:
- CP household appliance recycling pickup services — 363441;
- CQ public marina berth permit administration — 364557.

Each domain:
- 16 diagnosis intents;
- 6 state variants/intent;
- 96 cases;
- diagnosis K4/K8/K16 =32/32/32;
- five typed decisions;
- D0/D1/D2 for every option.

TRAIN 384 cases / 1920 decisions.
DEV 96 / 480.
Each CONFIRM 96 / 480.

No W17 fitting or DEV selection exists.

## 7. Typed gold semantics

Diagnosis:
- intent field only.

Response:
- severity only.

Needs_review:
- severity critical OR confidence uncertain.

Risk:
- severity.

Urgency:
- severity + confidence.

Confidence target mass remains:
- verified .90;
- provisional .75;
- uncertain .60.

## 8. Frozen gates

Integrity both CP/CQ:
- one logical state compile/candidate/case;
- FULL_SINGLE one invocation / one sequence;
- FULL_TRIPLICATE one invocation / three sequences;
- FIELD_ISOLATED one invocation / three sequences;
- triplicate prediction identity 1.0;
- triplicate max semantic-logit diff <=1e-5;
- probability error <=1e-6;
- no training;
- no forbidden overlap.

FIELD_ISOLATED competence both:
- overall >=.82;
- choice >=.82;
- noul >=.80;
- score >=.80;
- response >=.80;
- needs_review >=.80;
- risk >=.80;
- urgency >=.75;
- diagnosis K4 >=.85;
- K8 >=.78;
- K16 >=.65.

Gain vs FULL_SINGLE both:
- overall +.18;
- non-diagnosis +.12;
- K4 +.20;
- K8 +.25;
- K16 +.25.

Reference adequacy both:
- pinned MiniLM relevant-field diagnosis K4 >=.80;
- K16 >=.70.

## 9. Frozen verdicts

`FIELD_ISOLATED_TYPED_RESCUE`:
all integrity + reference + competence + causal gates pass both.

`FIELD_ISOLATED_TYPED_PARTIAL`:
no full rescue, but both:
- integrity pass;
- overall >= full+.12;
- non-diagnosis >= full+.08;
- K16 >= full+.20;
- K16 >=.55.

`FIELD_ISOLATION_CONTROL_INVALID`:
triplicate equivalence / field provenance / invocation accounting fails either CONFIRM.

`FIELD_ISOLATED_TYPED_FAIL`:
otherwise.

No new verdict after exposure.

## 10. Compute/accounting discipline

FIELD_ISOLATED is **not free**:
- it processes three encoder sequences in one batch instead of one full sequence.

W17 must report:
- logical compile count;
- encoder invocation count;
- encoded sequence count;
- active tokens per field/primitive.

Do not call this same-cost as FULL_SINGLE.

FULL_TRIPLICATE exists only to isolate batch/sequence-count effects.

## 11. Product boundary

W17 assumes fields are structurally known.

It does not solve automatic free-form field extraction.

A positive W17 would prove the value of representation separation under known fields, not a complete natural-language ingestion solution.

## 12. Prior-art boundary

See `research/R8-W17-PRIOR-ART.md`.

External work motivates field preservation only.
It does not select gates or mechanism details.

## 13. Forbidden evidence

Never use:
- W12 BN-BQ;
- W13 BR-BU;
- W14 BV-BY;
- W15 BZ-CF;
- W16 CG-CJ;
- Banking77 0-799;
- typed final/test;
- campaign cells.

After first CP/CQ materialization they become permanently forbidden.

## 14. Current state

Completed:
- W16 merged main `0a84fd45643ad76459fc5d23369a31ee123a985d`;
- W17 issue #121 preregistered before exposure;
- W17 branch created from exact post-W16 main;
- candidate representations, field mapping, fresh domains, gates and verdicts frozen in issue #121;
- this handoff created before W17 empirical exposure.

Exposure:
**NONE.**

No CK-CQ A13 cache exists.
No CP/CQ row has been materialized.
No W17 result exists.

Next:
1. freeze prior-art note;
2. implement fresh structured typed authority;
3. implement full/triplicate/isolated state cache;
4. implement primitive-relevant multiview semantic evaluator;
5. implement exact production D0 descriptive control;
6. implement typed metrics + gates/verdict;
7. implement pinned reference;
8. add freshness firewall and CONFIRM seal;
9. add unit + repository CI;
10. only after green pre-data head enable authority;
11. freeze exact result here before merge.

A future AI must update this file after every meaningful session.


---

## 15. Pre-authority implementation freeze

Complete scientific/execution stack head:
`161628fb355439ab2fe93584f52e4e9919e04158`

Exact validation:
- W17 unit run `36206223777`: **PASS**;
- repository CI run `36206223796`: **PASS** on Python 3.10 and Python 3.12.

Scientific exposure at this freeze:
**NONE.**

No CK/CL/CM/CN/CO A13 cache has been materialized.
No CP/CQ row has been materialized.
No W17 empirical metric or verdict exists.

Implemented and frozen:
- fresh structured typed authority CK-CQ;
- full/triplicate/field-isolated representation cache;
- exact primitive->field mapping;
- exact W9 multiview semantic evaluator;
- exact production D0+HIRACore descriptive control;
- typed accuracy/Brier/ECE/score/K metrics;
- triplicate equivalence control;
- zero-parameter pre-confirm freeze;
- sealed CP/CQ confirm evaluator;
- pinned MiniLM relevant-field reference;
- frozen rescue/partial/fail/control-invalid verdict implementation;
- exact-text freshness firewall through W16;
- download-free execution tests.

Pre-data failures repaired before exposure:
1. typed target probabilities were tuples and are now materialized as immutable float tensors in cache, matching established typed-cache contracts;
2. verdict fixture was aligned with the preregistered primitive-field provenance integrity gate.

Neither repair changed:
- field schema;
- candidate paths;
- fresh data;
- gates;
- thresholds;
- model weights;
- parameter budget.

Only authority orchestration may now be added.

Authority chain is frozen as:

`unit -> exact W9 provenance + CK-CN/CO cache -> zero-parameter preconfirm freeze -> one-time CP/CQ materialization -> frozen evaluator/reference/verdict`

Once the CK-CN/CO cache job begins:
- all equations are frozen;
- all field mappings are frozen;
- all fresh wording/seeds are frozen;
- all competence/gain/reference thresholds are frozen;
- no scientific mutation is allowed.

