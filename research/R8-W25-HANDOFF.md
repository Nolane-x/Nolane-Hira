# R8-W25 handoff — atomic severity projection-capacity localization

Status: **CLOSED DIAGNOSTIC. Frozen verdict: `W25_REFERENCE_INADEQUATE`. DT-DZ are exposed; T0/T1 show strong replicated projection-rescue signal but cannot be promoted because the sealed DY/DZ reference gate failed on both domains.**

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
**DT-DW TRAIN, DX DEV, DY/DZ CONFIRM permanently exposed by authority run `36231631652`.**

DY/DZ were materialized only after Q0/Q1/T0/T1 DEV-freeze receipts existed; the sealing contract passed in the authoritative workflow.

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
- fresh DT-DZ authority generator implemented;
- raw A13 / frozen-W9 semantic operators implemented;
- Q0/Q1 probes implemented;
- T0/T1 projection retuning implemented;
- DX DEV freeze implemented;
- machine-enforced DY/DZ sealed materialization gate implemented;
- directional DeBERTa CONFIRM reference implemented;
- frozen W25 classifier implemented;
- contract tests + pre-exposure unit workflow implemented;
- exact implementation head `f1b3b0e681961d572e89e03ce2b420c96067984b`;
- W25 unit run `36231134639`: PASS;
- repository CI run `36231137196`: PASS on Python 3.10 and 3.12.

Exposure:
**NONE at the completion of the pre-exposure gate.**

Next:
1. enable the frozen W25 authority workflow;
2. materialize DT-DW TRAIN and DX DEV;
3. freeze Q0/Q1/T0/T1 independently on DX;
4. only after all four freeze receipts exist, materialize sealed DY/DZ;
5. run independent DeBERTa reference + all six W25 candidates on DY/DZ;
6. freeze W25 verdict and merge.

A future AI must update this file after every meaningful W25 session.


### Pre-exposure amendment — initialization and probe geometry

Frozen before any DT-DZ encoder/reference exposure:
- T0/T1 initialize from the exact W9 projection weights, not random weights;
- T0/T1 use independently seeded TRAIN minibatch order to test reproducibility;
- Q0 uses L2-normalized raw A13 masked-mean severity embeddings;
- Q1 uses tokenwise L2-normalized frozen-W9 projections, then mean pooling and final L2 normalization.

This amendment removes an initialization confound and makes Q0/Q1 feature geometry exact.


### Pre-exposure amendment — optimizer minibatch exactness

Frozen before any DT-DZ encoder/reference exposure:
- Q0/Q1/T0/T1 training minibatch size = **32 logical severity cases**;
- TRAIN has 384 cases, therefore exactly 12 optimizer steps/epoch;
- Q0/Q1 run 20 epochs -> exactly 240 optimizer steps;
- T0/T1 run 8 epochs -> exactly 96 optimizer steps;
- each epoch uses a deterministic candidate-seeded permutation;
- no gradient accumulation;
- loss is averaged across cases in the minibatch;
- T0/T1 per-case objective is the equal sum of F0/F1/F2 binary cross-entropies at fixed temperature 0.07.

This amendment closes the last optimizer-step ambiguity before exposure.


---

## 14. Authoritative W25 closure

Exact empirical head:
`386c1e8e898ee7e1f837de6bc79e0133c500b68c`

Authority run:
`36231631652`

All frozen authority stages PASS:
- unit/contracts/freshness;
- exact W9 provenance;
- DT-DW TRAIN + DX DEV cache;
- independent DEV freeze of Q0/Q1/T0/T1;
- machine-enforced sealed DY/DZ materialization only after all four freeze receipts;
- sealed DY/DZ candidate evaluation;
- pinned directional DeBERTa reference;
- frozen classifier/audit verification.

Repository CI on exact empirical head:
`36231635135` — PASS.

## 15. Frozen verdict

Official outcome:

**`W25_REFERENCE_INADEQUATE`**

Reference-adequate CONFIRM domains:
**0 / 2**.

The frozen precedence rule therefore blocks every HIRA-side rescue/localization claim, even though T0/T1 satisfy their own rescue gates on both DY and DZ.

Do not relabel W25 as `TRAINABLE_PROJECTION_RESCUE`.

## 16. Sealed reference result

Pinned directional DeBERTa, pooled DY+DZ:
- F0 top1 **87.500%**, balanced accuracy **91.667%**;
- F1 top1 **97.917%**, balanced accuracy **97.917%**;
- F2 top1 **85.938%**, balanced accuracy **89.931%**;
- full factor-vector **75.000%**;
- deterministic composed severity **75.000%**;
- invalid vector rate **7.813%**;
- probability-mass max error <= `1.1920928955078125e-07`.

DY:
- F0 **86.458%**;
- F1 **97.917%**;
- F2 **86.458%**;
- vector/composed **75.000%**;
- invalid **7.292%**.

DZ:
- F0 **88.542%**;
- F1 **97.917%**;
- F2 **85.417%**;
- vector/composed **75.000%**;
- invalid **8.333%**.

Frozen reference gates required every factor >=92%, every BA >=90%, vector/composed >=85%, invalid <=5%.
Therefore both DY and DZ are reference-inadequate.

## 17. Frozen A0/P0/Q0/Q1 result

Pooled DY+DZ:

A0 raw A13 symmetric semantic MaxSim:
- composed severity **0.000%**;
- factor vector **0.000%**;
- invalid vector **100.000%**.

P0 frozen W9 projection:
- composed severity **1.042%**;
- factor vector **1.042%**;
- invalid vector **98.958%**.

Q0 raw A13 mean-pooled linear probe:
- composed severity **38.021%**;
- vector **38.021%**;
- invalid **0%**.

Q1 W9-projected mean-pooled linear probe:
- composed severity **42.708%**;
- vector **42.708%**;
- invalid **0%**.

Q0 and Q1 both fail the frozen probe-adequacy gate.

Selected DEV epochs/checkpoints:
- Q0 epoch **15**, checkpoint SHA `d376f7b66953c7f3074c8f8aec58eb54e44703f9f66017f10d9e5f3cd9bf8736`;
- Q1 epoch **10**, checkpoint SHA `a98b2b2a8bb55409404c0113cebf0f12cf5f577be14fa2585ac5e4e4bb1eb842`.

## 18. Strong T0/T1 hypothesis-generating signal

T0 primary, pooled DY+DZ:
- F0 **95.313%**;
- F1 **90.625%**;
- F2 **89.583%**;
- factor vector **78.646%**;
- composed severity **78.646%**;
- invalid **1.563%**.

T0 by CONFIRM:
- DY composed **78.125%**;
- DZ composed **79.167%**.

T1 replica, pooled:
- F0 **96.354%**;
- F1 **91.667%**;
- F2 **90.104%**;
- factor vector **80.208%**;
- composed severity **80.208%**;
- invalid **1.563%**.

T1 by CONFIRM:
- DY composed **79.167%**;
- DZ composed **81.250%**.

Frozen P0 composed severity is only **1.042%** on each CONFIRM domain.

Thus T0/T1 each satisfy their preregistered rescue gates on both DY and DZ, with large gains over P0 and independent seed replication.

However this remains **hypothesis-generating only** because the reference gate has precedence and failed.

Selected DEV epochs/checkpoints:
- T0 epoch **7**, checkpoint SHA `97d45d584877e18e6faa3deb07431c9f22707703fb65a1e27b973397f933a4cb`;
- T1 epoch **8**, checkpoint SHA `e087db48832f4c29b79351918a41674b38c3cac8741c8494935e26b73f8f2ee3`.

## 19. Authority artifacts

- frozen W9 bundle: `10902815814`;
  digest `sha256:b1c8bf792a18e68a7d0a7ec491d6f5df7e8c940a127cc4c58f5e89b9c8bb7d7c`;
- TRAIN/DEV cache: `10901974166`;
  digest `sha256:044649f90ce2bed1b6a7852ce844f53c1ee0ac916a68615a0914b865d6403ba5`;
- Q0: `10902647462`;
  digest `sha256:b32b668c616e65c8252d1c1da7437d330e3da2212b2bcafb0da69a4296de0800`;
- Q1: `10903015700`;
  digest `sha256:dc77a50a56bd5c9274811baaba778e55eac840bcdac9c60030050c591a774174`;
- T0: `10903005712`;
  digest `sha256:d31fc7916736404f74251917afa4521142608ba184e8bd75b39cd2dd9a507cd4`;
- T1: `10903255169`;
  digest `sha256:56e86278c483b49a0ea1ece2536cf14b08239771c179d92411d2906f89c57d97`;
- sealed DY/DZ cache: `10902642700`;
  digest `sha256:3a88705183cca3b6ee7b3f33397a4560240c33a1e1ba0e44aaaf989df824e9d2`;
- authoritative audit: `10901974661`;
  digest `sha256:2de65da980b1af703498dda8cbbce12c5523428a2bbe601cb14634eab5af40e1`.

## 20. Scientific interpretation

W25 cannot formally localize the production bottleneck because its sealed independent reference unexpectedly fails on the new authority.

The strongest permitted interpretation is:

> A 32,768-parameter projection retune initialized from W9 produces a large, reproducible atomic-severity improvement on DEV and both sealed CONFIRM domains, while raw/frozen semantic paths and tiny pooled linear probes remain weak. This is a strong projection-rescue signal, but the fresh DY/DZ authority is not independently adequate under the preregistered reference, so the signal cannot be promoted to a causal HIRA rescue claim.

W25 weakens:
- the idea that the frozen W9 projection is already adequate for atomic severity;
- the idea that a simple mean-pooled linear probe is sufficient;
- the idea that the atomic collapse seen in W24 necessarily requires changing A13.

W25 does **not** establish:
- that projection retuning is production-ready;
- that A13 is adequate in general;
- that semantic MaxSim is adequate after projection rescue;
- that T0/T1 should enter HIRA v0;
- that the reference gates may be relaxed post-exposure.

## 21. Permanent exposure after W25

DT/DU/DV/DW TRAIN, DX DEV, DY/DZ CONFIRM are permanently exposed.

Never reuse DY/DZ for:
- training;
- DEV selection;
- epoch/seed selection;
- reference wording/model/gate changes;
- projection objective/lr/batch/epoch tuning;
- architecture selection;
- calibration;
- promotion.

Never reuse DT-DX as fresh CONFIRM evidence.

All prior forbidden evidence remains forbidden.

## 22. Authorized continuation

Because the official verdict is reference inadequate:
- no production integration;
- no typed-integration/reliability promotion;
- no K32/K64 opening;
- T0/T1 remain diagnostic checkpoints only.

A next phase should **validate the projection-rescue hypothesis under a new, independently adequate atomic authority**, not tune T0/T1 on exposed W25 evidence.

A defensible next phase should:
1. keep the W25 T0/T1 training recipe frozen as a hypothesis;
2. use entirely new TRAIN/DEV and sealed dual-CONFIRM data;
3. preregister a stronger independent authority before exposure, preferably multiple independent sentence-pair references or a reference protocol already demonstrated adequate on a matched fresh pilot that is not used for HIRA selection;
4. require reference adequacy before reading rescue status;
5. retain frozen P0 control;
6. require primary + replica rescue on both CONFIRM domains;
7. keep all W25 rows forbidden;
8. only after a full fresh rescue may a later phase integrate the projection into a HIRA v0 semantic core.

A future AI must read this frozen W25 closure before opening the next phase.
