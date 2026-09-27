# R8-W34 handoff — fresh qualified authority for co-evidence composition

Status: **CLOSED — W34_COEVIDENCE_COMPOSITION_FAIL / MAINLINE ENTRY AUTHORIZED**

Issue: #157  
PR: #158  
Branch: `feat/r8-w34-qualified-coevidence`  
Base main: `81551ab38329959aed25a6cd2704ebb8f4a996cb`

## Purpose

W34 preserved W33's untested 8,192-parameter co-evidence hypothesis and replaced only the failed W33 authority instance with wholly fresh evidence.

The scientific question was whether a shared second-distinct-token co-evidence operator can materially improve fresh compositional transfer without factor-specific or primitive-specific learned heads.

## Candidate

Runtime mode:

`coevidence_symmetric_semantic`

Capacity:

- state residual rank-8: 2,048
- schema residual rank-8: 2,048
- shared interaction maps: 2,048
- shared composition maps: 2,048
- total trainable parameters: **8,192**

Frozen invariants:

- A13 frozen;
- exact W28 T0 frozen;
- exact unbridged-T0 identity at initialization;
- no factor-specific learned head;
- no primitive-specific learned head;
- relation refinement OFF;
- full-K/state-once typed runtime.

## Qualification — frozen

Run:

`36298394141`

Outcome:

`W34_REFERENCE_QUALIFIED`

- TA: PASS
- TB: PASS
- 192 cases
- A13 loaded: false
- HIRA candidate evaluated: false
- prior-wave rows used: false

Artifact:

- `r8-w34-reference-qualification`
- ID: `10924507541`
- digest: `sha256:13cfd1301f7fa7e03c7c2580c80409fd48c2aecb48b022f7e5fea811b71f5432`

## TRAIN/DEV — frozen

Run:

`36299087572`

- TRAIN TC/TD/TE/TF: 384 cases
- DEV TG: 96 cases
- selected epoch: 20
- checkpoint SHA256:
  `d69fa11805291e6a06631d5bda941065f209ea96f5c46091187e984ff083834c`
- anchor rate: 0.6050347222222222
- confirm rows used: 0

Artifact:

- `r8-w34-coevidence-training`
- ID: `10924803779`
- digest: `sha256:8374f2eb849cf80d1201e4ad05a8c8bf4c9bf1f824da17306f57c87456fd186c`

### TG candidate

- F0 top-1 0.8958333333333334 / BA 0.9305555555555556
- F1 top-1 0.9895833333333334 / BA 0.9895833333333333
- F2 top-1 0.875 / BA 0.7777777777777778
- factor-vector / severity 0.7708333333333334
- invalid-vector rate 0.0

## SEALED CONFIRM — frozen

First and only TH/TI run:

`36301624396`

Outcome:

`W34_COEVIDENCE_COMPOSITION_FAIL`

Runtime:

- TH: PASS
- TI: PASS

Absolute quality:

- TH: FAIL
- TI: FAIL

Audit artifact:

- `r8-w34-authoritative-audit`
- ID: `10925661475`
- digest: `sha256:6899d5e332b0d12fc9e4bcedb264533cb7693dafe10bfa35e251af85a58ceb64`

### Pooled TH+TI baseline

- F0 top-1: 0.875
- F1 top-1: 0.5
- F2 top-1: 0.6979166666666666
- F0 BA: 0.9166666666666667
- F1 BA: 0.5
- F2 BA: 0.6597222222222223
- vector/severity: 0.19791666666666666
- invalid-vector rate: 0.375

### Pooled TH+TI W34

- F0 top-1: 0.8854166666666666
- F1 top-1: 1.0
- F2 top-1: 0.8541666666666666
- F0 BA: 0.9236111111111112
- F1 BA: 1.0
- F2 BA: 0.7361111111111112
- vector/severity: 0.7395833333333334
- composed severity MAE: 0.2604166666666667
- invalid-vector rate: 0.0
- probability-mass max error: 1.1920928955078125e-07

## Transfer verdict

Relative transfer gate: **PASS**

- severity delta: +0.5416666666666667
- worst-factor top-1 delta: +0.35416666666666663
- F0 delta: +0.01041666666666663
- F1 delta: +0.5
- F2 delta: +0.15625
- no factor regression

Absolute production promotion: **FAIL**

Remaining pooled gaps:

- F0 top-1 0.8854 < 0.90
- F2 top-1 0.8542 < 0.90
- F2 BA 0.7361 < 0.88
- severity/vector 0.7396 < 0.82

Therefore:

`HIRA_V0_TRANSFER_CORE_READY` is not authorized.

## Scientific meaning

W34 establishes that compact co-evidence composition is not merely a DEV artifact.

It transfers strongly to fresh sealed TH/TI:
- F1 is perfect pooled;
- F2 improves materially over frozen T0;
- composed severity improves by more than 54 percentage points;
- invalid vectors fall to zero;
- state-once/full-K/typed runtime integrity remains intact.

The remaining weakness is absolute semantic discrimination, especially F2 balanced accuracy.

## Evidence firewall

TA/TB and TH/TI are permanently exposed.

No W34 evidence may be reused for:
- future training;
- selection;
- tuning;
- architecture ranking.

The selected W34 checkpoint is immutable with respect to these exposed rows.

## Mainline transition

W34 closes the blocking pre-mainline transfer research phase.

The W34 checkpoint is now designated:

`PROVISIONAL_TRANSFER_CORE`

It is not production-promoted, but it is strong enough to be integrated as a replaceable module while the complete Hira v0 model is built.

There is no blocking W35 before mainline construction.

Next phase:

# HIRA V0 MAINLINE

Initial module status:
- language/semantic front-end: A13 path, provisional/frozen;
- projection: W28 T0;
- transfer/composition: W34 co-evidence checkpoint, provisional;
- state-once memory: available;
- typed decision primitives: available;
- relation-free full-K runtime: available;
- calibration/OOD/null: pending;
- dynamic schemas and high-K: pending;
- multilingual: pending;
- latency/RAM optimization: pending;
- matched Laya/JEV benchmark: pending.

Future transfer research becomes a parallel replaceable-module track and must not block mainline model construction.

See `research/R8-W34-CLOSURE.md` for canonical closure.
