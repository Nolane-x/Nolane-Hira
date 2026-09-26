# R8-W27 handoff — compositional F2 authority qualification

Status: **PRE-EXPOSURE PREREGISTRATION. No EJ-EM reference score exists. No HIRA/A13 materialization is authorized in W27.**

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
- W27 issue #142 created;
- W27 branch created from exact post-W26 main;
- scientific decomposition/gates/outcomes frozen in this handoff.

Exposure:
**NONE.**

Next:
1. implement fresh W27 authority generator;
2. implement two-atom + direct-F2 reference scorer;
3. implement deterministic F2 composition and metrics;
4. implement frozen classifier;
5. add contracts/unit workflow;
6. exact-head unit + repo CI;
7. only then enable EJ-EM empirical qualification;
8. freeze result before merge.

A future AI must update this file after every meaningful W27 session.
