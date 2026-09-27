# R8-W34 handoff — fresh qualified authority for co-evidence composition

Status: **PREREGISTERED / PRE-QUALIFICATION**

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

## Current boundary

No W34 empirical evidence is exposed.

Next:
1. implement TA..TI fresh authority;
2. lock exact-text freshness against W28..W33;
3. build reference-only TA/TB qualification evaluator;
4. pass dedicated/full CI;
5. create one-way qualification marker;
6. only if TA and TB pass, build and expose TC..TG TRAIN/DEV;
7. freeze checkpoint before TH/TI implementation/exposure.
