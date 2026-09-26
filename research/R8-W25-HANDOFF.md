# R8-W25 handoff — atomic severity projection-capacity localization

Status: **PRE-DIAGNOSTIC. No DT-DZ A13/reference materialization exists.**

Issue: #138

Branch:
`feat/r8-w25-atomic-severity-geometry`

Base main:
`3484dedcdfde1e85c9411023a694ca04c08fdae6`

Read first:
1. this file;
2. `research/R8-W24-HANDOFF.md`;
3. `research/R8-W10-HANDOFF.md`;
4. `research/R8-W9-HANDOFF.md`;
5. issue #138.

## 0. Why W25 exists

W24 is the first W18-W24 severity phase with an independently adequate fresh authority.

Frozen W24 outcome:
`STABLE_ATOMIC_SEVERITY_LOCALIZATION`.

Stable classification:
`HIRA_ATOMIC_SEVERITY_GEOMETRY_LIMIT` on 4/4 DP/DQ/DR/DS.

W24 independent atomic reference:
- F0 96.875%;
- F1 100%;
- F2 100%;
- factor vector 96.875%;
- composed severity 96.875%.

Frozen HIRA:
- direct severity 26.563%;
- F0 33.594%;
- F1 60.938%;
- F2 25%;
- vector 0%;
- composed severity 0%;
- invalid vector 100%.

Therefore W25 may now diagnose the frozen severity geometry itself.

Important prior evidence:
- W9 found full projection semantic retuning stronger than small residual bridges, but not a rescue on generic intent transfer;
- W10 found W9 projection semantic geometry can be strong on generic intent data and that production scoring can later damage it;
- W25 must not assume the projection is automatically the culprit.

## 1. Frozen base

A13:
- `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length 256;
- frozen in every W25 path.

Frozen W9 semantic projection:
- 256->128 bias-free;
- scorer/projection SHA256 `8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102`;
- source run `36122220588`;
- source freeze artifact `10858424139`.

## 2. Fresh W25 data

TRAIN:
- DT municipal street-light service incident assessment — seed 441101;
- DU nonprofit meal-delivery coordination triage — seed 441107;
- DV university equipment-loan service assessment — seed 441119;
- DW regional ferry-ticket support triage — seed 441131.

DEV:
- DX community permit-inspection scheduling service — seed 442207.

Sealed CONFIRM:
- DY public clinic transport-booking support — seed 443311;
- DZ regional library-material transfer service — seed 443323.

Per domain:
- four severity classes;
- 24 independent narratives/class;
- 96 cases/domain.

Counts:
- TRAIN 384;
- DEV 96;
- DY 96;
- DZ 96.

Primary W25 lane is severity-only.
One logical state compile and one A13 severity encode/case.

Exposure:
**NONE.**

DY/DZ must not materialize until all trainable candidates are DEV-frozen.

## 3. Frozen severity semantics

Gold:
- S0 -> 000;
- S1 -> 100;
- S2 -> 110;
- S3 -> 111.

Factors:
- F0 meaningful disruption;
- F1 major functional loss;
- F2 immediate criticality.

All narratives and D0/D1/D2 factor-option wording are new.
No exact W24 wording reuse.

Decode only:
- 000 -> S0;
- 100 -> S1;
- 110 -> S2;
- 111 -> S3;
- other vector -> UNDEFINED.

No repair/projection to nearest monotone vector.

## 4. Candidates

### A0 — raw A13 semantic MaxSim

0 trainable.

- raw 256-d content tokens;
- L2 normalize;
- symmetric bidirectional token MaxSim;
- mean directions;
- mean D0/D1/D2;
- binary factor softmax.

### P0 — frozen W9 semantic baseline

0 trainable.

Exact frozen W9 projection, then same semantic operator as A0.

### Q0 — raw A13 linear probe

Diagnostic only.

Input:
content-token masked mean of raw A13 256-d severity tokens, then L2 normalize.

Three scalar logistic heads:
- 771 trainable params total.

### Q1 — W9-projected linear probe

Diagnostic only.

Input:
apply the exact W9 projection tokenwise, L2 normalize each projected token, masked-mean the projected tokens, then L2 normalize the pooled 128-d vector.

Three scalar logistic heads:
- 387 trainable params total.

### T0 — atomic projection-retune primary

Trainable:
one shared 256->128 bias-free projection,
32,768 params.

A13 frozen.
Initialize the trainable projection from the exact frozen W9 projection.
Primary/replica differ only through independently seeded minibatch ordering.
Semantic operator identical to P0.

Objective:
equal sum of three binary factor CEs;
fixed temperature 0.07.

Training:
- AdamW;
- lr 3e-4;
- wd .01;
- 8 epochs;
- no scheduler/warmup/AMP;
- grad clip 1.0;
- seed 2519.

### T1 — atomic projection-retune replica

Same as T0.
Independent seed 2539.
Never substituted for primary.

## 5. Probe training

Q0/Q1:
- AdamW;
- lr 1e-3;
- wd 0;
- 20 epochs;
- BCEWithLogits;
- equal factor weights;
- grad clip 1.0;
- Q0 seed 2609;
- Q1 seed 2617.

Diagnostic only.

## 6. DEV freeze

All trainable paths freeze independently on DX.

T0/T1 lexicographic:
1. composed severity;
2. vector accuracy;
3. minimum factor balanced accuracy;
4. lower invalid rate;
5. lower factor CE;
6. earlier epoch.

Q0/Q1:
1. composed severity;
2. vector;
3. minimum factor balanced accuracy;
4. lower BCE;
5. earlier epoch.

Only after T0/T1/Q0/Q1 are frozen may DY/DZ materialize.

## 7. Independent reference

Exact W24 directional DeBERTa:
- `cross-encoder/nli-deberta-v3-base`;
- revision `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`;
- weight SHA256 `d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa`.

Premise=query severity narrative.
Hypothesis=factor-option statement.
Use entailment probability.

Reference never supplies training targets.

Per DY/DZ require:
- F0/F1/F2 >= .92;
- BA >= .90 each;
- vector >= .85;
- composed >= .85;
- invalid <= .05;
- probability mass error <=1e-6.

Failure on either:
`W25_REFERENCE_INADEQUATE`.

## 8. Frozen adequacy gates

Probe adequate/domain:
- every factor >= .90;
- every BA >= .88;
- vector >= .80;
- composed >= .85;
- invalid <= .08.

T0 primary rescue/domain:
- every factor >= .85;
- every BA >= .82;
- vector >= .70;
- composed >= .75;
- invalid <= .12;
- composed gain vs P0 >= +.25.

T1 replica rescue/domain:
- every factor >= .82;
- vector >= .65;
- composed >= .70;
- invalid <= .15;
- gain vs P0 >= +.20.

Both DY and DZ are mandatory.

## 9. Frozen classifications

Precedence:

1. `W25_REFERENCE_INADEQUATE`
2. `TRAINABLE_PROJECTION_RESCUE`
3. `W9_PROJECTION_INFORMATION_LOSS`
4. `SEMANTIC_MATCHING_INTERFACE_LIMIT`
5. `A13_LINEAR_PROBE_LIMIT`
6. `ATOMIC_GEOMETRY_DECOMPOSITION_UNRESOLVED`

### TRAINABLE_PROJECTION_RESCUE

Reference adequate both;
T0 rescue both;
T1 rescue both.

### W9_PROJECTION_INFORMATION_LOSS

Reference adequate both;
Q0 adequate both;
Q1 inadequate both;
Q1 composed <= Q0-.15 both.

### SEMANTIC_MATCHING_INTERFACE_LIMIT

Reference adequate both;
Q1 adequate both;
P0 composed <.65 both;
no full projection rescue.

### A13_LINEAR_PROBE_LIMIT

Reference adequate both;
Q0 inadequate both;
T0 composed <.70 both;
T1 composed <.65 both.

This is explicitly limited to the frozen mean-pooled linear-probe/shared-linear-projection class.

Else unresolved.

## 10. Required reporting

A0/P0/T0/T1:
- per-factor top1/BA/MRR/margin;
- vector;
- invalid rate;
- composed severity;
- MAE;
- probability mass.

Q0/Q1:
- factor top1/BA;
- vector;
- invalid;
- composed;
- BCE.

Training:
- optimizer steps;
- selected DEV epoch;
- seeds;
- parameter counts;
- checkpoint hashes.

Reference:
- exact revision/hash;
- factor/vector/composed metrics.

Integrity:
- exact A13/W9 hashes;
- one A13 encode/case;
- no DP-DS;
- no W23-W18;
- no Banking77;
- no typed final/test;
- campaign cells 0;
- DY/DZ only after all candidate freeze receipts exist.

## 11. Authorization boundary

Full projection rescue:
- later fresh typed-integration/reliability phase may use the frozen rescued projection;
- no production claim yet;
- calibration/OOD mandatory;
- K32/K64 still closed.

Projection information loss:
- redesign projection preservation.

Semantic matching interface limit:
- preserve projection information, redesign matching/scoring.

A13 linear-probe limit:
- next lane may compare bounded encoder adaptation/new frozen encoder.

Unresolved:
- no rescue claim.

## 12. Permanent forbidden evidence

Never use for W25 fitting/selection:
- DP/DQ/DR/DS;
- DL/DM/DN/DO;
- DH/DI/DJ/DK;
- DD/DE/DF/DG;
- CZ/DA/DB/DC;
- CV/CW/CX/CY;
- CR/CS/CT/CU;
- older exposed authorities;
- Banking77 0-799;
- typed final/test;
- campaign cells.

## 13. Current state

Completed:
- W24 authority closed;
- W24 PR merged main `3484dedcdfde1e85c9411023a694ca04c08fdae6`;
- W24 canonical issue #135 closed;
- duplicate W24 issue #136 already closed;
- W25 issue #138 preregistered;
- W25 branch created from exact post-W24 main;
- all data partitions/candidate classes/training/gates frozen;
- this handoff created before exposure.

Exposure:
**NONE.**

Next:
1. implement fresh DT-DZ authority generator;
2. implement raw/projected semantic operators;
3. implement Q0/Q1 probes;
4. implement T0/T1 projection training;
5. implement DX DEV freeze;
6. implement sealed DY/DZ materialization gate;
7. implement directional DeBERTa reference;
8. implement frozen classifier;
9. add tests + pre-data workflow;
10. exact-head unit + repo CI;
11. only then run TRAIN/DEV;
12. freeze candidates;
13. only then expose DY/DZ;
14. freeze W25 verdict and merge.

A future AI must update this file after every meaningful W25 session.


### Pre-exposure amendment — initialization and probe geometry

Frozen before any DT-DZ encoder/reference exposure:
- T0/T1 initialize from the exact W9 projection weights, not random weights;
- T0/T1 use independently seeded TRAIN minibatch order to test reproducibility;
- Q0 uses L2-normalized raw A13 masked-mean severity embeddings;
- Q1 uses tokenwise L2-normalized frozen-W9 projections, then mean pooling and final L2 normalization.

This amendment removes an initialization confound and makes Q0/Q1 feature geometry exact.
