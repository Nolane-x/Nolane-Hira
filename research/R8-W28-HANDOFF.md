# R8-W28 handoff — compositional-authority projection-rescue replication

Status: **CLOSED. Frozen outcome: `REPLICATED_COMPOSITIONAL_PROJECTION_RESCUE`. EN-EV are permanently exposed; T0/T1 both reach 100% on sealed EU/EV under fully adequate compositional reference authority.**

Issue: #145

Branch:
`feat/r8-w28-compositional-projection-replication`

Base main:
`a3bd5f0df630c6162590e5bc1cc72cdb906c3262`

Read first:
1. this file;
2. `research/R8-W27-HANDOFF.md`;
3. `research/R8-W25-HANDOFF.md`;
4. `research/R8-W24-HANDOFF.md`.

## 0. Goal

W28 is the first full fresh replication of the W25 projection-retune hypothesis after W27 established that direct F2 is a packaging problem and that primary F2 authority should instead come from independently scored U/C atoms with deterministic composition.

Scientific question:

> Does the exact W25 32,768-parameter projection retune replicate on wholly fresh TRAIN/DEV/dual-CONFIRM when the primary severity authority is independently qualified before HIRA and F2 is validated compositionally?

W28 may authorize a later HIRA-v0 semantic-core integration phase only if the full replicated rescue gate passes.

## 1. Frozen base

A13:
- `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length 256;
- frozen.

P0:
- exact W9 projection-semantic-control;
- 256->128 bias-free;
- scorer SHA256 `8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102`;
- source run `36122220588`;
- source artifact `10858424139`.

## 2. Exact W25 projection-rescue recipe

T0/T1 are copied unchanged:

- one shared 256->128 bias-free projection;
- initialize from exact W9 projection;
- A13 frozen;
- 32,768 trainable parameters;
- semantic operator = symmetric bidirectional token MaxSim;
- D0/D1/D2 arithmetic mean;
- objective = equal sum of F0/F1/F2 binary CE;
- temperature 0.07;
- AdamW;
- lr 3e-4;
- weight decay .01;
- 8 epochs;
- batch size 32;
- 384 TRAIN cases -> 12 optimizer steps/epoch;
- exactly 96 optimizer steps;
- no scheduler;
- no warmup;
- no AMP;
- grad clip 1.0;
- T0 seed 2519;
- T1 seed 2539.

No W25/W27 result may alter these values.

## 3. Severity semantics

Candidate HIRA factors remain exactly:
- F0 meaningful disruption;
- F1 major functional loss;
- F2 immediate criticality.

Severity mapping remains:
- S0 -> 000;
- S1 -> 100;
- S2 -> 110;
- S3 -> 111.

F2 gold is still binary for HIRA training/evaluation.

Reference authority for F2 is no longer direct.

Primary F2 reference authority:
- U = delay-intolerance;
- C = immediate serious consequence of delay;
- `F2_reference = U AND C`.

Direct-F2 reference scoring is diagnostic only.

## 4. Fresh partitions

### Reference qualification only

EN:
municipal emergency-housing intake support review
seed `471101`.

EO:
nonprofit medication-delivery coordination review
seed `471107`.

96 cases/domain, 192 total.

EN/EO are reference-only.
They are never used for HIRA training/DEV/checkpoint selection.

If qualification fails, stop before EP-EV materialization.

### HIRA TRAIN

EP:
university laboratory-access support review
seed `472201`.

EQ:
regional waste-transfer scheduling support
seed `472207`.

ER:
public cultural-event registration incident review
seed `472219`.

ES:
community utility-install appointment support
seed `472231`.

Total TRAIN:
384 cases.

### HIRA DEV

ET:
municipal records-transfer support review
seed `473307`.

96 cases.

### sealed CONFIRM

EU:
public prescription-collection coordination review
seed `474401`.

EV:
regional access-pass support incident review
seed `474409`.

192 total.

EU/EV may materialize only after both T0/T1 DEV-freeze receipts exist.

## 5. Fresh authority construction

Per domain:
- 4 severity classes;
- 24 fresh narratives/class;
- 96 cases/domain.

Each severity class carries a deterministic hidden evidence tuple:

S0:
- F0=0;
- F1=0;
- U=0;
- C=0;
- F2=0.

S1:
- F0=1;
- F1=0;
- U=1;
- C=0;
- F2=0.

S2:
- F0=1;
- F1=1;
- U=0;
- C=1;
- F2=0.

S3:
- F0=1;
- F1=1;
- U=1;
- C=1;
- F2=1.

This mapping is frozen before exposure.

Each narrative must express all relevant operational facts without including:
- factor IDs;
- severity labels;
- binary labels;
- class numbers;
- direct reference hypotheses.

No exact W27/W26/W25 wording reuse.
No exact query/schema sentence overlap.

## 6. Reference panel

Exact pinned W23-W27 cross-encoders.

CE0 — DeBERTa:
- repo `cross-encoder/nli-deberta-v3-base`;
- revision `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`;
- weight SHA256 `d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa`.

CE1 — RoBERTa:
- repo `cross-encoder/nli-roberta-base`;
- revision `1be0567456f0543475805e758725f151f283705a`;
- weight SHA256 `efc90996d2ed80123c26c9091c91385ffddc6d2fd0b2bacf3187fbd6c5b87953`.

Reference tasks:
- F0 direct atomic;
- F1 direct atomic;
- U atomic;
- C atomic;
- direct F2 diagnostic only.

Panel option score:
arithmetic mean CE0/CE1 entailment probabilities.

No calibration.
No model-specific weighting.
No model selection after exposure.
No reference output is ever used as a T0/T1 training target.

## 7. Qualification gate — EN/EO

Both domains must pass before any EP-EV materialization.

Panel consensus per domain:
- F0 top1 >= .94;
- F0 BA >= .92;
- F1 top1 >= .94;
- F1 BA >= .92;
- U top1 >= .94;
- U BA >= .94;
- C top1 >= .94;
- C BA >= .94;
- composed F2 BA >= .94;
- composed F2 positive recall >= .90;
- composed F2 negative recall >= .95;
- probability-mass error <= 1e-6.

Individual CE floor:
- F0 BA >= .85;
- F1 BA >= .85;
- U BA >= .86;
- C BA >= .86;
- composed F2 BA >= .86.

CE0/CE1 agreement:
- F0 >= .88;
- F1 >= .88;
- U >= .90;
- C >= .90.

Failure on EN or EO:

**`W28_REFERENCE_QUALIFICATION_FAIL`**

Then stop before HIRA.

## 8. HIRA candidates

P0:
- exact frozen W9 projection;
- 0 trainable.

T0:
- exact W25 recipe;
- primary;
- seed 2519.

T1:
- exact W25 recipe;
- replica;
- seed 2539.

No Q0/Q1 in W28.

## 9. DEV freeze

T0/T1 independently freeze on ET.

Lexicographic:
1. composed severity;
2. vector accuracy;
3. minimum factor balanced accuracy;
4. lower invalid rate;
5. lower factor CE;
6. earlier epoch.

Only after both receipts are frozen may EU/EV materialize.

## 10. CONFIRM reference adequacy

Primary reference remains F0/F1/U/C with F2 = U AND C.

Per EU/EV panel:
- F0 top1 >= .92;
- F0 BA >= .90;
- F1 top1 >= .92;
- F1 BA >= .90;
- U top1 >= .92;
- U BA >= .92;
- C top1 >= .92;
- C BA >= .92;
- composed F2 BA >= .92;
- composed positive recall >= .88;
- composed negative recall >= .94;
- probability-mass <= 1e-6.

Each CE floor:
- F0 BA >= .82;
- F1 BA >= .82;
- U BA >= .84;
- C BA >= .84;
- composed F2 BA >= .84.

CE agreement:
- F0 >= .85;
- F1 >= .85;
- U >= .88;
- C >= .88.

Failure on either domain:

**`W28_CONFIRM_REFERENCE_INADEQUATE`**

Reference precedence is absolute.

## 11. T0 full rescue gate

On BOTH EU and EV:
- F0 top1 >= .85;
- F1 top1 >= .85;
- F2 top1 >= .85;
- every factor BA >= .82;
- factor vector >= .70;
- composed severity >= .75;
- invalid <= .12;
- composed gain vs P0 >= +.25;
- probability mass <= 1e-6.

## 12. T1 replica gate

On BOTH EU and EV:
- every factor top1 >= .82;
- every factor BA >= .80;
- factor vector >= .65;
- composed severity >= .70;
- invalid <= .15;
- gain vs P0 >= +.20;
- probability mass <= 1e-6.

## 13. Frozen outcomes

Precedence:

1. `W28_REFERENCE_QUALIFICATION_FAIL`
2. `W28_CONFIRM_REFERENCE_INADEQUATE`
3. `REPLICATED_COMPOSITIONAL_PROJECTION_RESCUE`
4. `COMPOSITIONAL_PROJECTION_RESCUE_NONREPLICATING`

### REPLICATED_COMPOSITIONAL_PROJECTION_RESCUE

Require:
- EN/EO qualification pass;
- EU/EV reference adequacy pass;
- T0 full rescue pass both;
- T1 replica pass both.

### COMPOSITIONAL_PROJECTION_RESCUE_NONREPLICATING

Reference qualification and CONFIRM authority pass,
but replicated rescue does not.

No PARTIAL upgrade.

## 14. Authorization boundary

Only `REPLICATED_COMPOSITIONAL_PROJECTION_RESCUE` authorizes the next phase to treat the 32,768-param projection-retune mechanism as the first candidate HIRA-v0 semantic-core component.

That later phase still must verify:
- typed integration;
- state-once runtime;
- calibration;
- OOD/null;
- reliability;
- fresh dual CONFIRM;
- latency/RAM.

W28 alone does not open K32/K64 or claim Laya/JEV superiority.

## 15. Integrity requirements

Must prove:
- exact A13 hash;
- exact W9 projection hash;
- exact CE0/CE1 hashes;
- 96 cases/domain;
- balanced 24/severity/domain;
- deterministic hidden F0/F1/U/C/F2 mapping;
- one A13 severity encode/case;
- T0/T1 exactly 32,768 params;
- T0/T1 exactly 96 optimizer steps;
- no reference outputs as training targets;
- EN/EO reference-only;
- no EP-EV materialization before EN/EO qualification PASS;
- no EU/EV materialization before T0/T1 DEV-freeze receipts;
- no W27 or older rows;
- no Banking77;
- no typed final/test;
- campaign cells 0.

## 16. Permanent forbidden evidence

Never fit/select using:
- EJ-EM;
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

## 17. Current state

Completed:
- W27 merged main `a3bd5f0df630c6162590e5bc1cc72cdb906c3262`;
- W27 issue #142 closed;
- W28 issue #145 created;
- W28 branch created from exact post-W27 main;
- W28 partitions/seeds/reference/rescue gates frozen in this handoff.

Exposure:
**EN/EO permanently exposed reference-only by run `36237568223`; EP-ET TRAIN/DEV and EU/EV sealed CONFIRM permanently exposed by authority run `36240250948`.**

Next:
1. merge this frozen W28 closure;
2. close issue #145;
3. begin HIRA-v0 semantic-core integration only from post-W28 `main`;
4. treat the rescued 32,768-param projection as a candidate component, not a finished model;
5. keep every EN-EV row permanently forbidden from future tuning/selection.

A future AI must read the frozen W28 closure before opening the HIRA-v0 integration phase.


---

## 18. Frozen W28 reference qualification

Pre-exposure scientific head:
`a5f19a204bf98795821730e614af6d8fa13f4c6f`

Pre-exposure gates:
- W28 unit `36237376033`: PASS;
- repository CI `36237378103`: PASS on Python 3.10 and 3.12.

Qualification trigger head:
`44e7e52029937a07cb9d45390f1ffb53a78119a6`

Reference-only qualification run:
`36237568223`

Artifact:
- ID `10904409537`;
- digest `sha256:c4b9735062cdc2c051b7e84b2b1f7f401cae49dc70eda527a0e660a4bff270a8`.

Frozen qualification outcome:

**`W28_REFERENCE_QUALIFIED`**

Per-domain:
- EN: PASS;
- EO: PASS.

No A13/HIRA/T0/T1 was materialized or executed during qualification.

## 19. Qualification metrics

Panel consensus on both EN and EO:

- F0 top1 **100%**, balanced accuracy **100%**;
- F1 top1 **100%**, balanced accuracy **100%**;
- U top1 **100%**, balanced accuracy **100%**;
- C top1 **100%**, balanced accuracy **100%**;
- deterministic composed F2 = U AND C:
  - top1 **100%**;
  - balanced accuracy **100%**;
  - positive recall **100%**;
  - negative recall **100%**;
- probability-mass max error <= `1.1920928955078125e-07`.

Cross-model agreement:

EN:
- F0 **100%**;
- F1 **96.875%**;
- U **100%**;
- C **100%**.

EO:
- F0 **100%**;
- F1 **94.792%**;
- U **100%**;
- C **100%**.

All frozen qualification gates pass.

## 20. Direct-F2 diagnostic

Direct F2 remains weak despite perfect atomic authority.

EN direct F2 panel:
- top1 **71.875%**;
- balanced accuracy **52.083%**;
- positive recall **12.500%**;
- negative recall **91.667%**;
- CE agreement **73.958%**.

EO direct F2 panel:
- top1 **72.917%**;
- balanced accuracy **52.778%**;
- positive recall **12.500%**;
- negative recall **93.056%**;
- CE agreement **68.750%**.

This independently replicates W27's packaging diagnosis and further justifies using compositional U/C as the primary F2 authority.

## 21. Authorization after qualification

Because EN/EO qualification passed:

- EP-ES TRAIN and ET DEV may now materialize;
- exact W25 T0/T1 recipe may execute;
- T0/T1 must freeze independently on ET;
- EU/EV remain sealed until both freeze receipts exist;
- reference outputs remain forbidden as HIRA training targets;
- every rescue/CONFIRM gate remains exactly as preregistered above.

The full HIRA rescue authority has now completed and is frozen in the closure sections below.


---

## 22. Authoritative W28 closure

Exact authoritative HIRA head:
`53ecb6e0e8afb12dac53a8b175b50c5b62322dd1`

Authoritative HIRA run:
`36240250948`

Repository CI on exact authority head:
`36240252645` — PASS.

All authority jobs PASS:
- unit/contracts;
- frozen qualification provenance;
- frozen W9 provenance;
- EP-ES TRAIN + ET DEV cache;
- independent T0/T1 training + DEV freeze;
- sealed EU/EV materialization only after both candidate freeze receipts;
- EU/EV compositional reference confirmation;
- final P0/T0/T1 audit.

Frozen outcome:

**`REPLICATED_COMPOSITIONAL_PROJECTION_RESCUE`**

This is the first W18-W28 phase that satisfies:
- independently qualified reference authority before HIRA;
- fresh TRAIN/DEV;
- independent primary + replica projection training;
- sealed dual-CONFIRM;
- adequate reference on both CONFIRM domains;
- primary rescue on both domains;
- replica rescue on both domains.

## 23. Frozen candidate result

### P0 — frozen W9 projection

Pooled EU+EV:
- F0 top1 **77.083%**, balanced accuracy **61.806%**;
- F1 top1 **72.917%**, balanced accuracy **72.917%**;
- F2 top1 **31.771%**, balanced accuracy **54.514%**;
- factor-vector **21.875%**;
- composed severity **21.875%**;
- composed MAE **1.99479**;
- invalid vector **56.771%**.

Per domain:
- EU composed severity **21.875%**;
- EV composed severity **21.875%**.

### T0 — primary rescued projection

DEV ET:
- selected epoch **8**;
- F0/F1/F2 top1 **100%**;
- factor-vector **100%**;
- composed severity **100%**;
- invalid vector **0%**.

Checkpoint SHA256:
`1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f`

Sealed EU+EV pooled:
- F0 top1 **100%**, BA **100%**;
- F1 top1 **100%**, BA **100%**;
- F2 top1 **100%**, BA **100%**;
- factor-vector **100%**;
- composed severity **100%**;
- composed MAE **0**;
- invalid vector **0%**.

Per domain:
- EU composed severity **100%**;
- EV composed severity **100%**.

### T1 — independent replica

DEV ET:
- selected epoch **8**;
- F0/F1/F2 top1 **100%**;
- factor-vector **100%**;
- composed severity **100%**;
- invalid vector **0%**.

Checkpoint SHA256:
`5ecf0914067c657df27e9044d33b26f5cc05b27a8cf288e2718780dcbac07d19`

Sealed EU+EV pooled:
- F0 top1 **100%**, BA **100%**;
- F1 top1 **100%**, BA **100%**;
- F2 top1 **100%**, BA **100%**;
- factor-vector **100%**;
- composed severity **100%**;
- composed MAE **0**;
- invalid vector **0%**.

Per domain:
- EU composed severity **100%**;
- EV composed severity **100%**.

Both T0 and T1 satisfy their frozen rescue gates on both sealed CONFIRM domains.

## 24. CONFIRM reference authority

Panel consensus pooled EU+EV:
- F0 top1 **100%**, BA **100%**;
- F1 top1 **100%**, BA **100%**;
- U top1 **100%**, BA **100%**;
- C top1 **100%**, BA **100%**;
- deterministic composed F2 top1 **100%**, BA **100%**;
- composed F2 positive recall **100%**;
- composed F2 negative recall **100%**;
- probability-mass max error <= `1.1920928955078125e-07`.

EU and EV each independently pass the frozen CONFIRM reference gate.

Direct F2 remains diagnostic-only and weak:

EU:
- direct F2 top1 **71.875%**;
- BA **52.083%**;
- positive recall **12.5%**;
- negative recall **91.667%**.

EV:
- direct F2 top1 **73.958%**;
- BA **53.472%**;
- positive recall **12.5%**;
- negative recall **94.444%**.

This independently replicates the W27 direct-packaging diagnosis.

## 25. Authority artifacts

Authority-run artifacts:

- frozen W9:
  `10904898794`;
  digest `sha256:395c1e9b54aab4b2e6e81de87289fb849c5326e86a0f49f5ac20d50eddc6d86f`;

- frozen qualified-reference receipt:
  `10905593029`;
  digest `sha256:8e1380b72742cf0ba6652497d897aa65e2b4c27bbffdeb815da878c9804e1937`;

- TRAIN/DEV cache:
  `10905418812`;
  digest `sha256:6c3e3c9f526c535e77f9827de822b7aaf6be0b0f6aac1f8e4c1dcb7bff67fba1`;

- T0:
  `10905153620`;
  digest `sha256:e47020da54c45f896ebb957f852e09dc3cb1719caa14ef2ae1af605927c57258`;

- T1:
  `10905513796`;
  digest `sha256:2fe64d1517a992914d46838aab0295ead8a05fde124b1d0c772335741de0d745`;

- sealed EU/EV cache:
  `10905219898`;
  digest `sha256:c4c68f98ee5acb02f16b73f07aea3cdd0fee6a28fa514614ba87ff1dcce61732`;

- authoritative audit:
  `10905439885`;
  digest `sha256:c026ff144eb6b22049f103d3b1751d37da76175588b5dca25e2f479d5ade13fa`.

Original qualification artifact:
- `10904409537`;
- digest `sha256:c4b9735062cdc2c051b7e84b2b1f7f401cae49dc70eda527a0e660a4bff270a8`.

## 26. Scientific interpretation

The strongest defensible conclusion is:

> The frozen W9 256->128 projection is not adequate for the fresh atomic severity geometry, but the exact W25 32,768-parameter projection-retuning mechanism reproducibly repairs the semantic geometry under independently qualified compositional authority. Two independently seeded replicas reach 100% factor-vector and composed-severity accuracy on two sealed fresh CONFIRM domains.

This establishes a **mechanism rescue**, not a complete HIRA model.

W28 supports:
- A13 contains sufficient token-level information for this fresh atomic severity family;
- the frozen W9 projection is a major bottleneck for this family;
- a very small trainable projection can recover the needed geometry;
- the result is reproducible across independent seeds and fresh domains;
- direct F2 should remain a diagnostic surface while U/C composition is the primary authority.

W28 does **not** establish:
- general semantic intelligence;
- arbitrary-domain transfer;
- typed decision integration;
- calibrated uncertainty;
- OOD/null behavior;
- high-K behavior;
- state-once runtime correctness;
- production latency/RAM;
- Laya/JEV superiority;
- HIRA v0 completion.

## 27. Authorization after W28

W28 authorizes the next phase to begin **HIRA-v0 semantic-core integration**.

The rescued projection may now be treated as the first candidate trainable semantic-core component.

The next phase should integrate, under a new fresh authority:
1. frozen A13 encoder;
2. rescued 256->128 semantic projection mechanism;
3. state-once semantic compilation;
4. typed atomic/structured decision interface;
5. deterministic composition where independently justified;
6. calibrated probability output;
7. null/OOD abstention path;
8. runtime accounting for latency/RAM and query reuse.

It must not simply replay W28 severity data.

A strong integration phase should preserve a sealed fresh validation boundary and verify:
- semantic transfer;
- typed correctness;
- probability mass;
- calibration;
- OOD/null;
- state compile once + many queries;
- inference speed/memory.

Only after that should HIRA-v0 be called a real integrated model rather than a rescued component.

## 28. Permanent exposure after W28

EN/EO reference qualification, EP-ES TRAIN, ET DEV and EU/EV CONFIRM are permanently exposed.

Never reuse EN-EV for:
- projection/objective/lr/epoch/seed tuning;
- architecture selection;
- calibration;
- OOD threshold tuning;
- typed integration selection;
- checkpoint selection;
- production promotion.

All prior forbidden evidence remains forbidden.
