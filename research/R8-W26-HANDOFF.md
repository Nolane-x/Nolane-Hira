# R8-W26 handoff — qualified atomic-authority projection-rescue replication

Status: **PRE-DIAGNOSTIC. No EA-EI encoder/reference materialization exists.**

Issue: #140

Branch:
`feat/r8-w26-qualified-projection-replication`

Base main:
`816befddab2cb5f4eeec054f2b316605a0090451`

Read first:
1. this file;
2. `research/R8-W25-HANDOFF.md`;
3. `research/R8-W24-HANDOFF.md`;
4. `research/R8-W10-HANDOFF.md`;
5. issue #140.

## 0. HIRA identity

Nolane HIRA is a compact non-autoregressive typed decision engine.

W26 remains a controlled semantic-core replication phase:
- A13 frozen;
- no full HIRA production promotion;
- no K32/K64;
- no typed final/test;
- no public benchmark claim;
- no W25 CONFIRM reuse.

## 1. Why W26 exists

W24 closed with:
`STABLE_ATOMIC_SEVERITY_LOCALIZATION`

Stable classification:
`HIRA_ATOMIC_SEVERITY_GEOMETRY_LIMIT`.

W25 then tested a projection-retune hypothesis.

W25 official verdict:
**`W25_REFERENCE_INADEQUATE`**.

W25 nevertheless produced a strong hypothesis-generating signal:
- P0 frozen W9 projection: ~1% composed severity on DY/DZ;
- T0 primary: 78.13% / 79.17%;
- T1 replica: 79.17% / 81.25%;
- T0/T1 passed their frozen HIRA-side rescue gates on both sealed domains;
- single DeBERTa reference failed its preregistered authority gate, so rescue promotion was correctly blocked.

W26 asks:

> Does the exact W25 projection-retune mechanism replicate on wholly fresh data when the atomic authority itself is prequalified by a frozen two-cross-encoder panel before HIRA TRAIN/DEV/CONFIRM execution?

W26 is not allowed to tune from W25 DY/DZ.

## 2. Frozen base

A13:
- `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length 256;
- frozen.

P0:
- exact W9 projection-semantic-control;
- 256->128 bias-free;
- scorer/projection SHA256 `8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102`;
- source run `36122220588`;
- source freeze artifact `10858424139`.

## 3. Frozen projection-rescue recipe

T0/T1 copy W25 exactly:

- one shared 256->128 bias-free projection;
- initialize from exact W9 projection;
- A13 frozen;
- 32,768 trainable params;
- semantic operator = symmetric bidirectional token MaxSim;
- D0/D1/D2 arithmetic mean;
- equal sum of F0/F1/F2 binary CE;
- fixed temperature 0.07;
- AdamW;
- lr `3e-4`;
- weight decay `.01`;
- 8 epochs;
- batch size 32;
- 384 TRAIN cases -> exactly 12 optimizer steps/epoch;
- exactly 96 optimizer steps;
- no scheduler;
- no warmup;
- no AMP;
- grad clip 1.0;
- T0 seed 2519;
- T1 seed 2539.

No W25 result may alter any value above.

## 4. Atomic severity semantics

Factors:
- F0 — meaningful disruption;
- F1 — major functional loss;
- F2 — immediate criticality.

Gold mapping:
- S0 -> 000;
- S1 -> 100;
- S2 -> 110;
- S3 -> 111.

Decoder:
- 000 -> S0;
- 100 -> S1;
- 110 -> S2;
- 111 -> S3;
- every other vector -> UNDEFINED and incorrect.

No:
- monotone repair;
- nearest-vector projection;
- learned decoder;
- calibration fit;
- tie rescue.

## 5. Fresh partitions

### Reference qualification only

EA:
municipal community-room reservation incident assessment
seed `451101`.

EB:
nonprofit volunteer-shift coordination incident assessment
seed `451107`.

Per domain:
- 4 severity classes;
- 24 fresh narratives/class;
- 96 cases/domain.

Total:
192 cases.

EA/EB are NEVER used for:
- T0/T1 training;
- DEV selection;
- HIRA candidate selection;
- checkpoint selection.

If qualification fails, W26 stops before EC-EI materialization.

### HIRA TRAIN

EC:
university campus-shuttle support incident assessment
seed `452201`.

ED:
regional recycling-pickup support triage
seed `452207`.

EE:
public arts-program registration service assessment
seed `452219`.

EF:
community broadband-install scheduling support
seed `452231`.

Total TRAIN:
384 cases.

### HIRA DEV

EG:
municipal document-delivery service assessment
seed `453307`.

Total DEV:
96 cases.

### Sealed CONFIRM

EH:
public pharmacy pickup coordination support
seed `454401`.

EI:
regional mobility-pass support incident assessment
seed `454409`.

Total CONFIRM:
192 cases.

EH/EI may materialize only after both T0/T1 freeze receipts exist.

## 6. Freshness contract

All EA-EI:
- fresh narrative wording;
- fresh factor D0/D1/D2 wording;
- fresh factor reference hypotheses;
- no exact W25 wording reuse;
- no exact prior-authority query/schema reuse.

Narratives may not contain:
- S0/S1/S2/S3 labels;
- F0/F1/F2 IDs;
- binary labels;
- class numbers;
- direct copies of reference hypotheses.

Exact query/schema sentence overlap:
**0**.

## 7. Reference panel

Reuse exact W23 pins.

### CE0 — DeBERTa

- repo `cross-encoder/nli-deberta-v3-base`;
- revision `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`;
- weight SHA256 `d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa`.

### CE1 — RoBERTa

- repo `cross-encoder/nli-roberta-base`;
- revision `1be0567456f0543475805e758725f151f283705a`;
- weight SHA256 `efc90996d2ed80123c26c9091c91385ffddc6d2fd0b2bacf3187fbd6c5b87953`.

Atomic scoring:
- premise = query severity narrative;
- hypothesis = factor-option statement;
- entailment probability.

Panel option score:
arithmetic mean of CE0 and CE1 entailment probabilities.

No:
- calibration;
- model-specific weighting;
- model selection after exposure;
- reference outputs as T0/T1 training targets.

## 8. Reference qualification gate

EA and EB must BOTH pass.

Panel per domain:
- every factor top1 >= .94;
- every factor balanced accuracy >= .92;
- vector >= .90;
- composed severity >= .90;
- invalid vector <= .03;
- probability mass error <=1e-6.

Individual floor for EACH CE:
- every factor top1 >= .85;
- every factor BA >= .85;
- vector >= .75;
- composed >= .75;
- invalid <= .10.

CE0/CE1 prediction agreement:
- >= .88 for every factor.

Failure on EA or EB:

**`W26_REFERENCE_QUALIFICATION_FAIL`**

If qualification fails:
- stop workflow before EC-EI materialization;
- no T0/T1 training;
- no HIRA conclusion;
- freeze negative result.

## 9. HIRA candidates

### P0

Exact frozen W9 projection.
0 trainable.

### T0

Exact W25 projection-retune recipe.
Seed 2519.
Primary.

### T1

Exact W25 recipe.
Seed 2539.
Replica.

No Q0/Q1 in W26.

## 10. DEV freeze

T0/T1 independently freeze on EG.

Lexicographic:
1. composed severity;
2. vector accuracy;
3. minimum factor balanced accuracy;
4. lower invalid rate;
5. lower factor CE;
6. earlier epoch.

Only after both receipts are frozen:
EH/EI may materialize.

Primary and replica remain independent.
Never substitute T1 for T0.

## 11. CONFIRM reference adequacy

Same frozen panel on EH/EI.

Per domain panel:
- every factor top1 >= .92;
- every factor BA >= .90;
- vector >= .85;
- composed >= .85;
- invalid <= .05;
- probability mass error <=1e-6.

Each individual CE:
- every factor top1 >= .82;
- composed >= .72.

CE0/CE1 prediction agreement:
- >= .85 every factor.

Failure on either EH or EI:

**`W26_CONFIRM_REFERENCE_INADEQUATE`**

Reference precedence is absolute.

## 12. P0 reporting

On each EH/EI:
- factor top1;
- factor balanced accuracy;
- MRR;
- margin;
- vector accuracy;
- invalid rate;
- composed severity;
- composed MAE;
- probability mass.

## 13. T0 full rescue gates

On BOTH EH and EI:
- every factor top1 >= .85;
- every factor BA >= .82;
- vector >= .70;
- composed >= .75;
- invalid <= .12;
- composed gain vs P0 >= +.25;
- probability mass <=1e-6.

## 14. T1 replica gates

On BOTH EH and EI:
- every factor top1 >= .82;
- every factor BA >= .80;
- vector >= .65;
- composed >= .70;
- invalid <= .15;
- composed gain vs P0 >= +.20;
- probability mass <=1e-6.

## 15. Frozen outcomes

Precedence:

1. `W26_REFERENCE_QUALIFICATION_FAIL`
2. `W26_CONFIRM_REFERENCE_INADEQUATE`
3. `REPLICATED_PROJECTION_RESCUE`
4. `PROJECTION_RESCUE_NONREPLICATING`

### REPLICATED_PROJECTION_RESCUE

Require:
- EA/EB qualification pass;
- EH/EI reference adequate;
- T0 full rescue pass both;
- T1 replica pass both.

### PROJECTION_RESCUE_NONREPLICATING

Require:
- reference qualification pass;
- EH/EI reference adequate;
- full replicated rescue does not pass.

No PARTIAL upgrade.

## 16. Authorization boundary

Only `REPLICATED_PROJECTION_RESCUE` authorizes the next phase to treat the mechanism as the first candidate HIRA-v0 semantic-core component.

That next phase would still require:
- wholly fresh typed TRAIN/DEV;
- state-once runtime verification;
- reliability;
- calibration;
- OOD/null;
- dual sealed CONFIRM.

W26 alone does NOT:
- produce production HIRA;
- open K32/K64;
- claim Laya/JEV superiority.

Reference failure:
- no HIRA-v0 integration authorization.

## 17. Required integrity

Must prove:
- exact A13 SHA;
- exact W9 scorer/projection SHA;
- exact CE0/CE1 revisions + weight hashes;
- 96 cases/domain;
- balanced 24/class;
- one A13 severity encode/case;
- zero reference outputs used as T0/T1 targets;
- T0/T1 32,768 params each;
- T0/T1 exactly 96 optimizer steps;
- EA/EB reference-only;
- no EC-EI materialization before EA/EB qualification PASS;
- no EH/EI materialization before both T0/T1 DEV-freeze receipts;
- no W25 rows;
- no W24-W18 rows;
- no Banking77;
- no typed final/test;
- campaign cells 0.

## 18. Permanent forbidden evidence

Never fit/select using:
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

After exposure:
- EA/EB remain reference-qualification-only and forbidden for HIRA selection;
- EC-EF TRAIN exposed;
- EG DEV exposed;
- EH/EI sealed CONFIRM exposed.

## 19. Current state

Completed:
- W25 frozen and merged main `816befddab2cb5f4eeec054f2b316605a0090451`;
- W25 issue #138 closed;
- W26 issue #140 preregistered before exposure;
- W26 branch created from exact post-W25 main;
- data partitions frozen;
- reference panel frozen;
- T0/T1 recipe frozen;
- qualification/CONFIRM/reference/rescue gates frozen;
- this handoff created before any W26 materialization.

Exposure:
**NONE.**

Next:
1. implement fresh EA-EI authority generator;
2. implement W26 partitioned cache;
3. implement two-reference panel scorer + consensus;
4. implement qualification classifier/gate;
5. implement machine stop before EC-EI if EA/EB fail;
6. port exact W25 T0/T1 trainer without recipe changes;
7. implement EG DEV freeze;
8. implement sealed EH/EI gate;
9. implement final P0/T0/T1 + reference evaluator;
10. implement frozen outcome classifier;
11. add unit/contracts + pre-exposure workflow;
12. exact-head unit + repo CI;
13. only then expose EA/EB qualification;
14. proceed only if qualification PASS.

A future AI must update this file after every meaningful W26 session.
