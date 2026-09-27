# HIRA V0 MAINLINE M1 closure — reliability / OOD / abstention

Status: **CLOSED — HIRA_V0_M1_RELIABILITY_FAIL / FAIL-CLOSED MECHANISM RETAINED**

Issue: #161  
PR: #162  
Branch: `feat/hira-v0-mainline-m1-reliability`  
M0 base: `17ad32104ac610908230c13d4b936a1b1ac3ecac`

## 1. Purpose

M1 added first-class reliability behavior to the actual Hira v0 mainline.

The product requirement was not merely to emit a confidence number. M1 had to keep three authorities separate:

1. probability calibration;
2. out-of-distribution detection;
3. selective correctness / abstention.

Automatic ACCEPT was required to fail closed unless every required authority was qualified.

## 2. Mechanism delivered

M1 implements:

- structured reliability receipts;
- `accept / abstain / escalate`;
- max-probability, normalized entropy and top-margin diagnostics;
- independent OOD authority;
- calibration authority;
- selective correctness authority;
- `HiraV0Session.decide_reliable()`;
- state-once execution;
- full-K preservation;
- typed `noul` kept separate from reliability abstention;
- high raw confidence forbidden from acting as OOD authority;
- missing or provisional authority -> ESCALATE.

These mechanism contracts remain valid even though empirical production promotion failed.

## 3. Frozen decision core

Throughout M1:

- semantic front-end: frozen;
- W28 T0: frozen;
- W34 provisional transfer core: frozen;
- relation core: frozen;
- decision-core trainable parameters: **0**.

Exact provenance:

W28 T0 checkpoint:

`1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f`

W34 provisional transfer checkpoint:

`d69fa11805291e6a06631d5bda941065f209ea96f5c46091187e984ff083834c`

## 4. M1 R1 — frozen DEV failure

Run:

`36305969587`

Artifact:

- `hira-v0-mainline-m1-train-dev`
- ID `10927625673`
- digest `sha256:79ff7eb67c3ce82ed82562787d813b75d35e7eca1bed0211f01ddf1b09c31dca`

Fresh exposure:

- calibration TRAIN UA/UB/UC: 108 cases;
- calibration DEV UD: 36 cases;
- OOD TRAIN UF/UG: 72 cases;
- OOD DEV UH: 36 cases;
- UE/UI remained sealed.

R1 DEV outcome:

`dev_qualification.pass = false`

The R1 OOD component was strong:

- candidate: `semantic-linear`;
- parameters: 6;
- threshold: 0.52;
- AUROC: 0.9969135522842407;
- balanced accuracy: 0.9722222222222222;
- OOD recall: 0.9722222222222222;
- OOD false-accept: 0.027777777777777776;
- ID accept: 0.9722222222222222.

But R1 failed because:

- calibration candidate selection allowed hard-accuracy regression;
- max-probability selective policy reached only 0.8 accepted accuracy at 0.2777777777777778 coverage.

UE/UI were not exposed.

## 5. M1 R2 — fresh DEV rescue

R2 preserved the frozen R1 semantic OOD head and changed only the calibration-selection guardrail plus selective correctness authority.

Fresh R2 authority:

- TRAIN UJ/UK/UL/UM: 144 cases;
- DEV UN/UO: 72 cases.

Authoritative run:

`36307009727`

Artifact:

- `hira-v0-mainline-m1-r2-train-dev`
- ID `10928285228`
- digest `sha256:2e4bed910adc75ac2a3579765d712ea881a30e9c0888e693f0de293ff14e9178`

### R2 calibration

Selected:

`primitive-temperature`

- trainable parameters: 3;
- selected epoch: 3;
- checkpoint SHA256:
  `dccee8d11c42e56e2ea6ef030c7044adb963ac8a48ae40daec082f17fe52ce3a`
- DEV hard accuracy: 0.5277777777777778;
- control hard accuracy: 0.5277777777777778;
- soft ECE: 0.054004438428415194;
- soft NLL: 0.9380818158388138;
- soft Brier: 0.2184828238354789.

Safe calibration selection therefore preserved hard accuracy exactly while improving calibration.

### R2 selective correctness

Selected:

`semantic-confidence-risk-linear`

- trainable parameters: 9;
- selected epoch: 78;
- threshold: 0.52;
- checkpoint SHA256:
  `08e4afcb3db9f4be3e7553b40d955aa2d304ffad7f544b78f01cb18f4ed3f39e`
- correctness AUROC: 0.8351393342018127;
- coverage: 0.375;
- accepted accuracy: 0.9259259104728699;
- selective risk: 0.07407408952713013.

### Frozen R1 OOD reused in R2

- candidate: `semantic-linear`;
- trainable parameters: 6;
- threshold: 0.52;
- checkpoint SHA256:
  `2fda61a53fd2db05be5a703c6640a87e52588091fd22ed320cc0ab3f4cc80a3e`
- retrained: false;
- retuned: false.

R2 DEV qualification:

`pass = true`

This authorized one UE/UI sealed confirm.

## 6. M1 R2 sealed confirm

Pre-confirm code head:

`a70afd4ac84c65252a382b223967dbeb61da3a47`

Marker commit:

`ff53cf4991dfbe4f1e120472d545201fa9fa58c0`

First and only sealed run:

`36308933278`

Artifact:

- `hira-v0-mainline-m1-r2-sealed-confirm`
- ID `10928705130`
- digest `sha256:48caf250ab9273a046b6070b3c23319c313367de6079f3961136afda18ef4e10`

Frozen outcome:

`HIRA_V0_M1_RELIABILITY_FAIL`

UE case count: 36  
UI case count: 36  
State encode count: 72  
State encodes/case: 1.0

No sealed row was used for fitting, threshold selection, candidate ranking or architecture selection.

## 7. Sealed calibration

Control UE:

- hard accuracy: 0.5;
- soft ECE: 0.058990628665520066;
- soft Brier: 0.2374046551477578;
- soft NLL: 0.9600778768459955.

Selected 3-param calibration:

- hard accuracy: **0.5**;
- soft ECE: **0.04066400757680334**;
- soft Brier: **0.22894535659222776**;
- soft NLL: **0.9512354135513306**;
- probability-mass max error: 1.1920928955078125e-07.

All frozen calibration sealed gates PASS.

Conclusion:

**calibration mechanism transferred successfully.**

## 8. Sealed selective correctness

Frozen threshold:

0.52

UE result:

- correctness AUROC: 0.6728395223617554;
- accepted count: 10 / 36;
- coverage: 0.2777777777777778;
- accepted accuracy: **0.800000011920929**;
- selective risk: **0.19999998807907104**.

Frozen target required:

- coverage >= 0.25;
- accepted accuracy >= 0.90;
- selective risk <= 0.10.

Result:

- coverage: PASS;
- accepted accuracy: FAIL;
- selective risk: FAIL;
- meets target: FAIL.

The correctness-risk head did not transfer adequately from UN/UO to UE.

## 9. Sealed OOD

Frozen R1 OOD threshold:

0.52

UE + UI result:

- AUROC: **0.9984567761421204**;
- balanced accuracy: **0.8611111111111112**;
- ID accept: **1.0**;
- OOD recall: **0.7222222222222222**;
- OOD false-accept: **0.2777777777777778**.

Interpretation:

Ranking quality remained excellent, but the frozen absolute threshold did not transfer.

The sealed OOD gate therefore failed on:

- OOD recall;
- OOD false-accept.

This is a threshold-transfer failure, not a failure of semantic OOD ranking.

## 10. Final combined policy

Frozen combined ACCEPT required both:

- OOD score below 0.52;
- correctness-risk score at or above 0.52.

Sealed result:

- ID coverage: 0.2777777777777778;
- ID accepted accuracy: **0.8**;
- ID selective risk: **0.2**;
- OOD final accept count: 1 / 36;
- OOD final accept rate: **0.027777777777777776**.

The final OOD accept-rate gate passed.

The final selective quality gates failed.

## 11. Scientific conclusion

M1 establishes several useful results even though production reliability is not promoted.

### Established

1. Calibration can be improved with only 3 parameters without changing the hard decision.
2. Semantic geometry can rank fresh OOD extremely well:
   - DEV AUROC 0.9969;
   - sealed AUROC 0.9985.
3. Confidence should not be used as OOD authority.
4. A correctness-risk head can improve DEV selective behavior but did not transfer strongly enough to sealed UE.
5. Fixed OOD thresholds are materially less stable than OOD ranking quality.
6. Fail-closed separation between calibration, OOD and selective correctness prevented an invalid reliability promotion.

### Not established

M1 does **not** establish a production-qualified automatic ACCEPT policy.

Therefore:

`reliability_ood_abstention = provisional`

must remain unchanged.

## 12. Product decision

M1 mechanism code is retained and may merge because it is fail-closed:

- without a qualified reliability policy, it escalates rather than overclaims;
- high confidence cannot silently become OOD authority;
- reliability abstention remains distinct from semantic `noul`.

No M1 empirical checkpoint is promoted to production-qualified reliability.

The mainline must **not** be blocked by an R3 reliability experiment.

Future reliability research becomes a parallel replaceable-control track.

## 13. Evidence firewall

The following M1 rows are now exposed and forbidden from future fitting/selection:

- R1 UA/UB/UC/UD/UF/UG/UH;
- R2 UJ/UK/UL/UM/UN/UO;
- sealed UE/UI.

They may only be used as historical evidence.

No future reliability candidate may tune:
- calibration parameters;
- OOD thresholds;
- correctness-risk thresholds;
- architecture;
against UE/UI.

## 14. Mainline transition

M1 closes with:

- reliability mechanism: implemented;
- calibration mechanism: empirically promising;
- OOD semantic ranking: empirically strong;
- automatic reliability promotion: failed;
- default product behavior: fail-closed / provisional.

The next blocking mainline phase is:

# HIRA V0 MAINLINE M2 — DYNAMIC HIGH-K

M2 targets:
- dynamic schemas beyond tiny K;
- K = 4 / 8 / 16 / 32 / 64 / 128 / 255;
- state-once preserved;
- full-K preserved;
- memory and latency measured;
- typed `choice / score / noul` remain valid;
- no reliability promotion claim is required for M2.

Reliability research may continue later in parallel without blocking M2.
