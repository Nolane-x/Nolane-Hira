# R8-W27 handoff — compositional F2 authority qualification

Status: **CLOSED REFERENCE DIAGNOSTIC. Frozen outcome: `F2_DIRECT_PACKAGING_LIMIT`. Atomic U/C authority qualifies on 4/4 fresh domains; direct F2 fails its diagnostic gate on 4/4.**

Issue: #142

Branch:
`feat/r8-w27-compositional-f2-authority`

Base main:
`6d234863f0fe1d03b58f294321aca1cc247d2a87`

Read first:
1. this file;
2. `research/R8-W26-HANDOFF.md`;
3. `research/R8-W25-HANDOFF.md`;
4. `research/R8-W24-HANDOFF.md`.

## 0. Scope

W27 is reference-only.

It does not:
- train HIRA;
- materialize A13;
- run W9/P0;
- train T0/T1;
- open K32/K64;
- make a production claim.

The dormant W25 projection-rescue recipe remains unchanged and untouched.

## 1. Why W27 exists

W24 independently localized a real HIRA atomic-severity geometry limit.

W25 produced a strong replicated projection-retune signal, but official verdict was
`W25_REFERENCE_INADEQUATE`.

W26 moved reference qualification before HIRA execution and correctly stopped with
`W26_REFERENCE_QUALIFICATION_FAIL`.

Repeated reference anatomy:
- F0 is relatively stable;
- F1 is usually recoverable;
- F2 immediate-criticality is the recurring weak point.

W27 asks:

> Is F2 reference instability caused by packaging two independently recoverable semantic facts into one direct label?

## 2. F2 decomposition

Two preregistered evidence atoms:

### U — delay-intolerance

U=1:
intervention must begin now; a meaningful delay is not acceptable.

U=0:
a short response delay is acceptable; ordinary or prompt handling remains possible.

### C — immediate serious consequence of delay

C=1:
waiting creates a serious near-term consequence.

C=0:
waiting does not create an immediate serious consequence.

Frozen composition:

`F2 = U AND C`

No learned decoder.
No probabilistic threshold fit.
No post-hoc rule change.
No nearest valid-state repair.

The four evidence cells are:
- U0 C0;
- U1 C0;
- U0 C1;
- U1 C1.

Only U1 C1 is F2-positive.

## 3. Fresh authority

Fresh reference-only domains:

- EJ — municipal senior-transport scheduling incident review — seed 461101;
- EK — nonprofit food-pantry pickup coordination review — seed 461107;
- EL — university exam-access support incident review — seed 461119;
- EM — regional public-works dispatch support review — seed 461131.

Per domain:
- 96 cases;
- 24 cases/evidence cell;
- deterministic hidden U/C ground truth;
- F2 positive = 24;
- F2 negative = 72;
- all reporting includes balanced accuracy and positive/negative recall.

Total:
384 reference-only cases.

No W27 row is ever eligible for HIRA training/selection.

## 4. Surface construction

Each narrative is composed from:
1. fresh domain context;
2. one fresh timing clause encoding U;
3. one fresh consequence clause encoding C;
4. one neutral connective.

Timing and consequence clause libraries are frozen before exposure.

Narratives must not contain:
- U/C/F2 IDs;
- binary labels;
- class numbers;
- "positive"/"negative";
- direct copies of reference hypotheses.

Exact query/schema sentence overlap:
0.

No exact W26 or earlier narrative/schema reuse.

## 5. Reference panel

Reuse exact pinned W23/W26 reference models.

CE0 — DeBERTa:
- repo `cross-encoder/nli-deberta-v3-base`;
- revision `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`;
- weight SHA256 `d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa`.

CE1 — RoBERTa:
- repo `cross-encoder/nli-roberta-base`;
- revision `1be0567456f0543475805e758725f151f283705a`;
- weight SHA256 `efc90996d2ed80123c26c9091c91385ffddc6d2fd0b2bacf3187fbd6c5b87953`.

Scoring:
- premise = W27 narrative;
- hypothesis = atom/direct-F2 alternative;
- score = entailment probability;
- option score panel = arithmetic mean CE0/CE1.

No calibration.
No model-specific weighting.
No model selection.
No reference fitting.

## 6. Reference questions

Three binary tasks are scored independently:

### U
- U0: a short delay remains acceptable;
- U1: action must begin immediately.

### C
- C0: waiting does not create an immediate serious consequence;
- C1: waiting creates a serious near-term consequence.

### Direct F2 diagnostic
- F20: the situation is not an immediate critical no-delay event;
- F21: the situation is an immediate critical no-delay event.

Primary authority uses U/C and deterministic composition.
Direct F2 is diagnostic only and cannot invalidate an otherwise qualified compositional authority.

## 7. Frozen atomic qualification gates

Every EJ/EK/EL/EM domain must pass.

Panel consensus:
- U top1 >= .94;
- U balanced accuracy >= .94;
- C top1 >= .94;
- C balanced accuracy >= .94;
- composed F2 balanced accuracy >= .94;
- composed F2 positive recall >= .90;
- composed F2 negative recall >= .95;
- probability-mass error <= 1e-6.

Each individual CE floor:
- U balanced accuracy >= .86;
- C balanced accuracy >= .86;
- composed F2 balanced accuracy >= .86.

CE0/CE1 prediction agreement:
- U >= .90;
- C >= .90.

Failure on any domain:
`W27_ATOMIC_REFERENCE_INADEQUATE`.

## 8. Direct-F2 diagnostic gate

Evaluated only if atomic qualification passes all four domains.

Direct panel:
- balanced accuracy >= .88/domain;
- positive recall >= .80/domain;
- negative recall >= .90/domain;
- CE0/CE1 direct prediction agreement >= .85/domain.

If atomic authority passes but direct F2 fails one or more domains:

`F2_DIRECT_PACKAGING_LIMIT`

If both atomic authority and direct F2 pass all domains:

`F2_AUTHORITY_FULLY_QUALIFIED`

Precedence:
1. `W27_ATOMIC_REFERENCE_INADEQUATE`
2. `F2_DIRECT_PACKAGING_LIMIT`
3. `F2_AUTHORITY_FULLY_QUALIFIED`

No PARTIAL upgrade.

## 9. Required reporting

Per model + panel, per domain:
- U top1;
- U balanced accuracy;
- U positive/negative recall;
- C top1;
- C balanced accuracy;
- C positive/negative recall;
- direct F2 top1;
- direct F2 balanced accuracy;
- direct positive/negative recall;
- U agreement;
- C agreement;
- direct agreement;
- deterministic composed F2 top1;
- composed F2 balanced accuracy;
- composed positive/negative recall;
- probability-mass max error.

Integrity:
- 96 cases/domain;
- exactly 24/evidence cell/domain;
- exact model revision/hashes;
- zero A13;
- zero HIRA;
- zero T0/T1;
- no W26 rows;
- no W25 or older rows;
- no Banking77;
- no typed final/test;
- campaign cells = 0.

## 10. Authorization boundary

`W27_ATOMIC_REFERENCE_INADEQUATE`:
- no projection-rescue replication;
- redesign authority only on wholly fresh future rows.

`F2_DIRECT_PACKAGING_LIMIT`:
- authorizes a later fresh projection-rescue replication to use U/C atomic authority with deterministic F2 composition;
- direct F2 remains diagnostic;
- T0/T1 recipe must stay exactly frozen from W25.

`F2_AUTHORITY_FULLY_QUALIFIED`:
- also authorizes later fresh projection-rescue replication;
- compositional authority remains primary because it is causally more explicit.

W27 alone never authorizes HIRA-v0 integration.

## 11. Permanent forbidden evidence

Never fit/select using:
- EA/EB;
- DT-DZ;
- DP-DS;
- DL-DO;
- DH-DK;
- DD-DG;
- CZ-DC;
- CV-CY;
- CR-CU;
- all older exposed authorities;
- Banking77 0-799;
- typed final/test;
- campaign cells.

After first W27 empirical run:
EJ/EK/EL/EM become permanently exposed reference-only evidence.

## 12. Current state

Completed:
- W26 merged post-closure main `6d234863f0fe1d03b58f294321aca1cc247d2a87`;
- W26 issue #140 closed;
- W27 issue #142 preregistered before exposure;
- W27 branch created from exact post-W26 main;
- fresh EJ-EM authority generator implemented;
- U/C/direct-F2 reference scorer implemented;
- deterministic U AND C composition implemented;
- frozen classifier/contracts implemented;
- exact pre-exposure unit + repo CI passed;
- EJ-EM empirical reference-only qualification completed;
- final W27 outcome frozen below.

Exposure:
**EJ/EK/EL/EM permanently exposed reference-only by run `36234616900`.**

Next:
1. merge this frozen W27 closure;
2. close issue #142;
3. resume projection-rescue replication only in a wholly fresh later phase using compositional U/C authority as the primary F2 authority;
4. never reuse EJ-EM for model/reference/gate selection.

A future AI must update this file after every meaningful W27 session.


---

## 13. Authoritative W27 closure

Pre-exposure implementation head:
`95cf76548ec2523e13c419ed914ab54e5172d3b6`

Pre-exposure gates:
- W27 unit `36234385510`: PASS;
- repository CI `36234388843`: PASS on Python 3.10 and 3.12.

Exact empirical head:
`64e3b47bcfa734a89e7aabdcff4b10bf62a55664`

Reference-only qualification run:
`36234616900`

Execution/integrity:
**PASS**.

Artifact:
- ID `10903812772`;
- digest `sha256:9db82e233a5354d11bec352de47d89a1aa81e5b8c6bcb0592ed32415381701ec`.

Pinned weight verification:
- DeBERTa SHA `d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa`;
- RoBERTa SHA `efc90996d2ed80123c26c9091c91385ffddc6d2fd0b2bacf3187fbd6c5b87953`.

Integrity:
- 384 cases;
- 96/domain;
- exactly 24/evidence cell/domain;
- zero A13 materialization;
- zero HIRA materialization;
- zero T0/T1 execution;
- no reference output used as HIRA training target;
- no W26 or older exposed rows;
- no Banking77;
- no typed final/test;
- campaign cells = 0.

## 14. Frozen verdict

Official outcome:

**`F2_DIRECT_PACKAGING_LIMIT`**

Atomic-qualified domains:
**4 / 4**.

Direct-F2-qualified domains:
**0 / 4**.

Per-domain atomic authority:
- EJ: PASS;
- EK: PASS;
- EL: PASS;
- EM: PASS.

Per-domain direct F2 diagnostic:
- EJ: FAIL;
- EK: FAIL;
- EL: FAIL;
- EM: FAIL.

This authorizes a later fresh projection-rescue replication to use U/C atomic authority with deterministic F2 composition.
It does not authorize HIRA-v0 integration by itself.

## 15. Panel atomic authority

Consensus U:
- EJ **100.000%** top1 / **100.000%** balanced accuracy;
- EK **100.000%** / **100.000%**;
- EL **100.000%** / **100.000%**;
- EM **100.000%** / **100.000%**.

Consensus C:
- every domain **98.958%** top1 / **98.958%** balanced accuracy;
- negative recall **100.000%**;
- positive recall **97.917%**.

Deterministic composed F2 = U AND C:
- every domain **98.958%** top1;
- every domain **97.917%** balanced accuracy;
- positive recall **95.833%**;
- negative recall **100.000%**;
- probability-mass max error <= `1.1920928955078125e-07`.

CE0/CE1 agreement:
- U: EJ **98.958%**, EK/EL/EM **100.000%**;
- C: **93.750%** on every domain.

All frozen atomic gates pass on all four domains.

## 16. Direct F2 diagnostic failure

Consensus direct F2:

EJ:
- top1 **90.625%**;
- balanced accuracy **86.806%**;
- positive recall **79.167%**;
- negative recall **94.444%**;
- CE agreement **88.542%**.

EK:
- top1 **91.667%**;
- balanced accuracy **88.889%**;
- positive recall **83.333%**;
- negative recall **94.444%**;
- CE agreement **83.333%**.

EL:
- top1 **90.625%**;
- balanced accuracy **86.806%**;
- positive recall **79.167%**;
- negative recall **94.444%**;
- CE agreement **85.417%**.

EM:
- top1 **89.583%**;
- balanced accuracy **84.722%**;
- positive recall **75.000%**;
- negative recall **94.444%**;
- CE agreement **84.375%**.

The frozen direct-F2 gate therefore fails all four fresh domains.

## 17. Individual-reference anatomy

DeBERTa:
- U is ~99-100% balanced accuracy;
- C is **100%** balanced accuracy on all domains;
- deterministic composed F2 is **100%** balanced accuracy on all domains;
- direct F2 balanced accuracy ranges **94.444-96.528%**.

RoBERTa:
- U is **100%** balanced accuracy on all domains;
- C is **93.750%** balanced accuracy on all domains;
- deterministic composed F2 is **91.667%** balanced accuracy on all domains;
- direct F2 balanced accuracy ranges only **73.611-76.389%**.

The cross-family disagreement is therefore not about the underlying timing evidence U.
It is modest for consequence evidence C and becomes much larger when the two facts are packaged into one direct immediate-criticality label.

## 18. Scientific interpretation

The strongest defensible W27 conclusion is:

> The F2 authority problem is primarily an interface/packaging problem, not a failure of the underlying semantic evidence. Two independently pinned cross-encoder references recover delay-intolerance and immediate serious consequence well enough that their deterministic conjunction produces a highly stable F2 authority on all four fresh domains. Direct F2 wording, however, reintroduces substantial cross-model disagreement and positive-recall loss.

This resolves the specific blocker that stopped W26.

W27 establishes:
- U is independently reference-stable;
- C is independently reference-stable enough for the frozen gate;
- deterministic U AND C composition is reference-stable;
- direct F2 should no longer be the primary authority for projection-rescue replication.

W27 does NOT establish:
- W25 T0/T1 rescue replicates;
- A13/W9 redesign is production-ready;
- HIRA-v0 semantic core is ready;
- K32/K64 should open.

## 19. Permanent exposure after W27

EJ/EK/EL/EM are permanently exposed reference-only evidence.

Never reuse them for:
- projection training;
- DEV/checkpoint selection;
- authority wording/model/gate selection;
- calibration;
- production promotion.

All prior exposed evidence remains forbidden.

## 20. Authorized continuation

A later fresh phase may now resume the dormant W25 projection-rescue hypothesis with these constraints:

1. wholly fresh TRAIN/DEV/dual-CONFIRM data;
2. T0/T1 recipe unchanged from W25;
3. P0 exact frozen W9 control;
4. primary F2 authority = independently scored U/C atoms + deterministic U AND C composition;
5. direct F2 remains diagnostic only;
6. reference U/C authority must qualify before HIRA TRAIN/DEV/CONFIRM is spent;
7. both T0 primary and T1 replica must pass both sealed CONFIRM domains;
8. no W27 rows may be used for fitting or selection;
9. only replicated rescue may authorize a later HIRA-v0 semantic-core integration phase.

A future AI must read this frozen W27 closure before opening the next phase.
