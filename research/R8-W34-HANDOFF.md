# R8-W34 handoff — fresh qualified authority for co-evidence composition

Status: **TRAIN-DEV FROZEN / SEALED CONFIRM PRE-EXPOSURE**

Issue: #157  
Branch: `feat/r8-w34-qualified-coevidence`  
Base main: `81551ab38329959aed25a6cd2704ebb8f4a996cb`

## Why W34 exists

W33 closed `W33_REFERENCE_QUALIFICATION_FAIL`.

Important boundary:
- SA failed;
- SB passed;
- A13 was not loaded;
- HIRA candidate was never evaluated;
- SC–SG were never exposed to candidate training/selection;
- SH/SI remained sealed;
- therefore the co-evidence candidate hypothesis is still untested.

W34 preserves the candidate and replaces only the authority instance with wholly fresh evidence.

## Frozen candidate

Reuse the W33 shared co-evidence scorer unchanged.

Runtime mode:

`coevidence_symmetric_semantic`

Capacity:
- state residual rank-8: 2,048
- schema residual rank-8: 2,048
- shared interaction maps: 2,048
- shared composition maps: 2,048
- total candidate parameters: **8,192**

Composition mechanism:
- project state/schema semantic tokens into a shared rank-8 composition space;
- per semantic view, derive the additional signal from the second-strongest distinct state-token support;
- this rewards support distributed over multiple evidence pieces.

Frozen invariants:
- A13 frozen;
- exact W28 T0 frozen;
- initial candidate logits exactly equal unbridged T0;
- no factor-specific head;
- no primitive-specific head;
- relation refinement OFF;
- state-once/full-K typed runtime.

## W34 authority principle

W34 changes authority wording, not candidate geometry.

Fresh authority uses short declarative evidence clauses:
- F0: ordinary route usable vs ordinary route unusable and alternate required;
- F1: required capability available vs unavailable and objective blocked;
- U: short delay allowed vs no delay allowed / immediate action required;
- C: short delay harmless vs short delay causes serious consequence;
- F2: U AND C.

Qualification state wording and reference hypotheses remain semantically independent but deliberately direct.

No exposed W33 text may be edited/reused.

## Fresh partitions

Reference qualification:
- TA
- TB

TRAIN:
- TC
- TD
- TE
- TF

DEV:
- TG

SEALED CONFIRM:
- TH
- TI

Each domain contains 96 balanced cases.

## Qualification protocol

Frozen external NLI panel only:
- DeBERTa NLI
- RoBERTa NLI

During qualification:
- A13 load forbidden;
- HIRA candidate evaluation forbidden;
- candidate checkpoint load forbidden;
- TRAIN/DEV/CONFIRM exposure forbidden.

Both TA and TB must pass.

Otherwise:

`W34_REFERENCE_QUALIFICATION_FAIL`

and the wave stops.

## Optimizer contract if qualified

- seed: 3417
- epochs: 20
- batch: 32
- AdamW lr: 2e-4
- weight decay: 0.01
- grad clip: 1.0
- semantic temperature: 0.07
- anchor coefficient: 0.35
- frozen-T0 positive anchor threshold: 0.08
- invalid-vector mass coefficient: 0.15
- trainable candidate parameters: exactly 8,192

Primary objective:
- factor-balanced F0/F1/F2 CE.

Anchor:
- preserve frozen T0 correct decisions with signed margin >= 0.08.

Structural validity:
- valid vectors 000, 100, 110, 111;
- invalid vectors 001, 010, 011, 101.

DEV selection:
1. worst-factor balanced accuracy;
2. worst-factor top-1;
3. composed severity;
4. lower invalid-vector rate;
5. earlier epoch.

## Promotion gate

Per sealed domain:
- F0/F1/F2 top-1 >= 0.90
- each factor BA >= 0.88
- factor-vector top-1 >= 0.82
- composed severity top-1 >= 0.82
- invalid-vector rate <= 0.05
- option-order invariance = 1
- state-once = 1
- full-K = 1
- relation delta = 0
- probability mass error <= 1e-6

Pooled transfer versus frozen T0:
- composed severity delta >= +0.15
- worst-factor top-1 delta >= +0.08
- no factor top-1 regression > 0.02

## Frozen outcomes

1. `W34_REFERENCE_QUALIFICATION_FAIL`
2. `W34_COEVIDENCE_COMPOSITION_FAIL`
3. `HIRA_V0_TRANSFER_CORE_READY`

No partial promotion.

## Evidence firewall

Forbidden for fitting/selection/authority repair:
- W29 EW..EZ
- W30 FA..FG
- W31 QH..QP
- W32 RA..RI
- W33 SA..SI
- W28 and older authority rows
- Banking77 final/test
- typed final/test
- Laya/JEV result cells

Only aggregate prior-wave conclusions may motivate W34.

## Reference qualification — frozen

Authoritative run:

`36298394141`

Outcome:

`W34_REFERENCE_QUALIFIED`

Per-domain:
- TA: PASS
- TB: PASS

Isolation:
- case count: 192
- HIRA candidate evaluated: false
- A13 loaded: false
- W33/W32/W31/W30/W29/older authority rows used: false
- exact-text overlap: none

Frozen artifact:
- name: `r8-w34-reference-qualification`
- artifact ID: `10924507541`
- digest: `sha256:13cfd1301f7fa7e03c7c2580c80409fd48c2aecb48b022f7e5fea811b71f5432`

TA/TB are now permanently exposed and forbidden from candidate fitting or DEV selection.

## TRAIN/DEV — frozen

Authoritative run:

`36299087572`

Partitions:
- TRAIN: TC/TD/TE/TF = 384 cases
- DEV: TG = 96 cases
- confirm rows used: 0
- qualification rows used for training: false
- prior-wave rows used: false

Selected checkpoint:
- epoch: **20**
- SHA256: `d69fa11805291e6a06631d5bda941065f209ea96f5c46091187e984ff083834c`
- anchor rate: `0.6050347222222222`

Training artifact:
- name: `r8-w34-coevidence-training`
- artifact ID: `10924803779`
- digest: `sha256:8374f2eb849cf80d1201e4ad05a8c8bf4c9bf1f824da17306f57c87456fd186c`

### TG frozen T0 baseline

- F0 top-1: 0.875
- F0 BA: 0.9166666666666667
- F1 top-1: 0.5
- F1 BA: 0.5
- F2 top-1: 0.7395833333333334
- F2 BA: 0.6736111111111112
- factor-vector / composed severity: 0.23958333333333334
- composed severity MAE: 1.5104166666666667
- invalid-vector rate: 0.375
- probability-mass max error: 1.1920928955078125e-07

### TG frozen W34 candidate

- F0 top-1: 0.8958333333333334
- F0 BA: 0.9305555555555556
- F1 top-1: 0.9895833333333334
- F1 BA: 0.9895833333333333
- F2 top-1: 0.875
- F2 BA: 0.7777777777777778
- factor-vector / composed severity: 0.7708333333333334
- composed severity MAE: 0.23958333333333334
- invalid-vector rate: 0.0
- probability-mass max error: 1.1920928955078125e-07

Scientific DEV observation:
- co-evidence gives a large fresh gain over T0;
- F1 is essentially solved on TG;
- F2 improves strongly but remains the weakest factor;
- vector legality is clean on TG;
- DEV alone does not authorize promotion.

## Sealed-confirm stack — implemented, not exposed

Present:
- `scripts/r8_w34_confirm.py`
- `.github/workflows/r8-w34-sealed-confirm.yml`
- exact frozen quality and transfer gate helpers
- exact frozen checkpoint/artifact provenance checks
- typed primitive agreement audit
- option-order invariance audit
- state-once audit
- full-K audit
- relation-delta-zero audit
- probability-mass audit
- baseline-vs-candidate transfer audit

Frozen TH/TI outcome space:
1. `W34_COEVIDENCE_COMPOSITION_FAIL`
2. `HIRA_V0_TRANSFER_CORE_READY`

The qualification failure outcome is no longer reachable because TA/TB already qualified.

## Current boundary

TH/TI remain sealed and must not be inspected, tuned against, or used for candidate selection.

Next:
1. pass dedicated W34 unit gate and full repository CI on this sealed-confirm implementation;
2. only then create `research/R8-W34-ENABLE-CONFIRM`;
3. expose TH/TI exactly once using the frozen checkpoint `d69fa118...`;
4. freeze authoritative outcome regardless of result;
5. write closure;
6. build FULL source/research/evidence bundle + standalone HANDOFF + integrity manifest;
7. merge PR #158 only after final CI and bundle integrity pass.
