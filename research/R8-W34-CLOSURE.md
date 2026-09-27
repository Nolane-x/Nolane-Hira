# R8-W34 closure — fresh qualified authority for co-evidence composition

Status: **CLOSED — W34_COEVIDENCE_COMPOSITION_FAIL**

Issue: #157  
PR: #158  
Branch: `feat/r8-w34-qualified-coevidence`  
Base main: `81551ab38329959aed25a6cd2704ebb8f4a996cb`

## 1. Purpose

W34 followed W33's `W33_REFERENCE_QUALIFICATION_FAIL`.

W33 never tested the co-evidence candidate because SA failed reference qualification. W34 therefore preserved the exact W33 co-evidence candidate hypothesis and replaced only the authority instance with wholly fresh evidence.

## 2. Frozen candidate

Runtime mode:

`coevidence_symmetric_semantic`

Candidate capacity:

- state residual rank-8 adapter: 2,048 params;
- schema residual rank-8 adapter: 2,048 params;
- shared interaction maps: 2,048 params;
- shared composition maps: 2,048 params;
- total trainable candidate parameters: **8,192**.

Frozen invariants:

- A13 frozen;
- exact W28 T0 frozen;
- exact T0 identity at initialization;
- no factor-specific learned head;
- no primitive-specific learned head;
- relation refinement OFF;
- state-once/full-K typed runtime.

## 3. Reference qualification

Authoritative run:

`36298394141`

Outcome:

`W34_REFERENCE_QUALIFIED`

Domains:

- TA: PASS
- TB: PASS

Isolation:

- 192 cases;
- HIRA candidate evaluated: false;
- A13 loaded: false;
- prior-wave authority rows used: false;
- exact-text overlap: none.

Qualification artifact:

- name: `r8-w34-reference-qualification`
- artifact ID: `10924507541`
- digest: `sha256:13cfd1301f7fa7e03c7c2580c80409fd48c2aecb48b022f7e5fea811b71f5432`

## 4. TRAIN / DEV

Authoritative run:

`36299087572`

Partitions:

- TRAIN TC/TD/TE/TF: 384 cases;
- DEV TG: 96 cases.

Selected epoch:

`20`

Selected candidate checkpoint SHA256:

`d69fa11805291e6a06631d5bda941065f209ea96f5c46091187e984ff083834c`

Anchor rate:

`0.6050347222222222`

Training artifact:

- name: `r8-w34-coevidence-training`
- artifact ID: `10924803779`
- digest: `sha256:8374f2eb849cf80d1201e4ad05a8c8bf4c9bf1f824da17306f57c87456fd186c`

### TG frozen T0 baseline

- F0 top-1: 0.875
- F1 top-1: 0.5
- F2 top-1: 0.7395833333333334
- F0 BA: 0.9166666666666667
- F1 BA: 0.5
- F2 BA: 0.6736111111111112
- factor-vector / composed severity: 0.23958333333333334
- composed severity MAE: 1.5104166666666667
- invalid-vector rate: 0.375

### TG W34 candidate

- F0 top-1: 0.8958333333333334
- F1 top-1: 0.9895833333333334
- F2 top-1: 0.875
- F0 BA: 0.9305555555555556
- F1 BA: 0.9895833333333333
- F2 BA: 0.7777777777777778
- factor-vector / composed severity: 0.7708333333333334
- composed severity MAE: 0.23958333333333334
- invalid-vector rate: 0.0

## 5. SEALED CONFIRM

First and only TH/TI sealed run:

`36301624396`

Frozen outcome:

`W34_COEVIDENCE_COMPOSITION_FAIL`

Runtime gates:

- TH: PASS
- TI: PASS

Absolute quality gates:

- TH: FAIL
- TI: FAIL

This is a scientific quality failure, not an infrastructure/runtime failure.

Audit artifact:

- name: `r8-w34-authoritative-audit`
- artifact ID: `10925661475`
- digest: `sha256:6899d5e332b0d12fc9e4bcedb264533cb7693dafe10bfa35e251af85a58ceb64`

## 6. Frozen pooled TH+TI metrics

### Unbridged T0 baseline

- F0 top-1: 0.875
- F1 top-1: 0.5
- F2 top-1: 0.6979166666666666
- F0 balanced accuracy: 0.9166666666666667
- F1 balanced accuracy: 0.5
- F2 balanced accuracy: 0.6597222222222223
- factor-vector top-1: 0.19791666666666666
- composed severity top-1: 0.19791666666666666
- composed severity MAE: 1.5520833333333333
- invalid-vector rate: 0.375
- probability-mass max error: 1.1920928955078125e-07

### W34 co-evidence candidate

- F0 top-1: 0.8854166666666666
- F1 top-1: 1.0
- F2 top-1: 0.8541666666666666
- F0 balanced accuracy: 0.9236111111111112
- F1 balanced accuracy: 1.0
- F2 balanced accuracy: 0.7361111111111112
- factor-vector top-1: 0.7395833333333334
- composed severity top-1: 0.7395833333333334
- composed severity MAE: 0.2604166666666667
- invalid-vector rate: 0.0
- probability-mass max error: 1.1920928955078125e-07

## 7. Transfer gate

The preregistered pooled transfer gate **PASSED**.

Observed:

- composed severity delta: **+0.5416666666666667**
- worst-factor top-1 delta: **+0.35416666666666663**
- F0 top-1 delta: +0.01041666666666663
- F1 top-1 delta: +0.5
- F2 top-1 delta: +0.15625
- maximum factor regression: -0.01041666666666663
- no factor regressed

All three frozen relative-transfer requirements passed:

- severity delta >= +0.15: PASS
- worst-factor top-1 delta >= +0.08: PASS
- no factor regression > 0.02: PASS

This is the strongest fresh relative-transfer result in the W29-W34 line.

## 8. Why absolute promotion failed

Per sealed domain the frozen production gate required:

- every factor top-1 >= 0.90;
- every factor BA >= 0.88;
- factor-vector >= 0.82;
- composed severity >= 0.82;
- invalid-vector <= 0.05;
- runtime invariants PASS.

The pooled candidate shows the remaining gap:

- F0 top-1: 0.8854166666666666
- F2 top-1: 0.8541666666666666
- F2 BA: 0.7361111111111112
- composed severity: 0.7395833333333334

Therefore `HIRA_V0_TRANSFER_CORE_READY` is not authorized.

## 9. Scientific conclusion

W34 establishes a materially stronger result than W32:

- shared co-evidence composition generalizes strongly to fresh sealed data;
- F1 reaches 1.0 top-1 / 1.0 BA pooled;
- F2 improves +0.15625 absolute over frozen T0;
- severity/vector accuracy improves +0.5416666666666667 absolute;
- invalid vectors fall from 0.375 to 0.0;
- runtime integrity passes on both sealed domains;
- the relative transfer gate passes decisively.

The remaining problem is no longer whether the compact transfer mechanism works at all. It does.

The remaining problem is **absolute semantic quality**, particularly F2 balanced discrimination and final composed severity quality.

## 10. Evidence boundary

TH/TI are permanently exposed.

They are forbidden for all future:

- fitting;
- DEV selection;
- hyperparameter tuning;
- candidate ranking;
- architecture selection.

The W34 checkpoint must never be retuned against TH/TI.

## 11. Mainline transition decision

W34 is the final blocking pre-mainline transfer research wave.

No W35 is required before beginning the main Hira model.

The frozen W34 checkpoint is authorized only as:

`PROVISIONAL_TRANSFER_CORE`

It is **not** authorized as `HIRA_V0_TRANSFER_CORE_READY`.

Hira v0 mainline may now begin using W34's co-evidence core as a replaceable provisional module while the complete model architecture, typed decision system, calibration/OOD/null handling, dynamic schema, high-K, multilingual, latency/RAM and matched benchmark stack are built around it.

Future transfer-core research must proceed as a parallel replaceable-module track and must not block construction of the main Hira model.

## 12. Mainline entry

The next project phase is not R8-W35.

The next project phase is:

**HIRA V0 MAINLINE**

with this architectural status:

- semantic front-end: frozen/provisional A13 path;
- projection: exact W28 T0;
- transfer/composition: W34 co-evidence checkpoint, provisional;
- state-once runtime: available;
- typed primitives: available;
- full-K binary authority runtime: available;
- production calibration/OOD/null: pending;
- dynamic schema/high-K: pending;
- multilingual: pending;
- matched Laya/JEV benchmark: pending.

This closes the pre-mainline transfer-core research phase.
