# R8-W32 closure — dual residual adapter + shared interaction metric

Status: **CLOSED — W32_INTERACTION_ADAPTER_FAIL**

Issue: #153  
PR: #154  
Branch: `feat/r8-w32-interaction-semantic-adapter`  
Base main: `762289911da2b7cde860383268c5d35d6ebd2f98`

## 1. Purpose

W32 followed W31's `W31_DUAL_ADAPTER_FAIL`.

W31 established that compact state/schema-asymmetric adaptation can improve semantic transfer substantially, but absolute F1/F2 and composed-vector quality remained below the production threshold.

W32 preserved the successful dual residual adapters and added:

- a shared low-rank state↔schema interaction metric;
- a structural invalid-vector probability-mass loss.

The frozen candidate contains exactly 6,144 trainable parameters:

- state residual adapter: 2,048;
- schema residual adapter: 2,048;
- state interaction map: 1,024;
- schema interaction map: 1,024.

A13 and exact W28 T0 remained frozen.

## 2. Reference qualification

Authoritative qualification run:

`36292606367`

Outcome:

`W32_REFERENCE_QUALIFIED`

Domains:

- RA: PASS
- RB: PASS

Qualification properties:

- 192 cases total;
- no HIRA candidate evaluated;
- A13 not loaded;
- no W29/W30/W31/older authority rows used;
- no exact-text overlap with exposed prior authorities.

Frozen artifact:

- name: `r8-w32-reference-qualification`
- artifact ID: `10922738458`
- digest: `sha256:02cafde04b10c075df848d3c0d45b9f9c139b3ae1516a82c2a428e2163131751`

## 3. TRAIN/DEV

Authoritative TRAIN/DEV run:

`36293558503`

Partitions:

- TRAIN: RC/RD/RE/RF = 384 cases
- DEV: RG = 96 cases

Frozen optimizer:

- seed: 3217
- epochs: 18
- batch: 32
- AdamW lr: 2e-4
- weight decay: 0.01
- grad clip: 1.0
- temperature: 0.07
- anchor coefficient: 0.35
- positive frozen-T0 anchor margin: 0.08
- invalid-vector probability-mass coefficient: 0.15

Selected DEV epoch:

`17`

Selected checkpoint SHA256:

`212caf6d5cdb06743979da5d7b64fff5887fc5c2b11e1be91006bc6463e63a5f`

Anchor rate:

`0.7100694444444444`

Frozen training artifact:

- name: `r8-w32-interaction-training`
- artifact ID: `10923630412`
- digest: `sha256:4b8289345f92f7fb45b5af1f4c2e6d2765ca56b4e1e52dffc62968f26b868f92`

### RG unbridged T0 baseline

- F0 top-1: 0.7916666666666666
- F1 top-1: 0.875
- F2 top-1: 0.7604166666666666
- factor-vector top-1: 0.5104166666666666
- composed severity top-1: 0.5104166666666666
- invalid-vector rate: 0.08333333333333333

### RG W32 candidate

- F0 top-1: 0.9166666666666666
- F1 top-1: 0.96875
- F2 top-1: 0.8020833333333334
- factor-vector top-1: 0.6875
- composed severity top-1: 0.6875
- invalid-vector rate: 0.0
- probability-mass max error: 1.1920928955078125e-07

W32 therefore improved RG strongly, especially F0/F1 and structural vector validity, while F2 remained the weakest factor.

## 4. SEALED CONFIRM

First sealed RH/RI run:

`36294714414`

Outcome:

`W32_INTERACTION_ADAPTER_FAIL`

Runtime gates:

- RH: PASS
- RI: PASS

Absolute quality gates:

- RH: FAIL
- RI: FAIL

The failure is scientific, not infrastructure-related.

Frozen audit artifact:

- name: `r8-w32-authoritative-audit`
- artifact ID: `10923651937`
- digest: `sha256:b7b9f078a36af924e2a3a409756df5a2b66a7d80e038890b792434244c50340a`

## 5. Frozen pooled RH+RI metrics

### Unbridged T0 baseline

- F0 top-1: 0.796875
- F1 top-1: 0.8697916666666666
- F2 top-1: 0.765625
- F0 balanced accuracy: 0.8645833333333333
- F1 balanced accuracy: 0.8697916666666667
- F2 balanced accuracy: 0.5868055555555556
- factor-vector top-1: 0.515625
- composed severity top-1: 0.515625
- composed severity MAE: 0.6510416666666666
- invalid-vector rate: 0.08333333333333333
- probability-mass max error: 1.1920928955078125e-07

### W32 interaction candidate

- F0 top-1: 0.9427083333333334
- F1 top-1: 0.953125
- F2 top-1: 0.8125
- F0 balanced accuracy: 0.9409722222222222
- F1 balanced accuracy: 0.953125
- F2 balanced accuracy: 0.7916666666666667
- factor-vector top-1: 0.7135416666666666
- composed severity top-1: 0.7135416666666666
- composed severity MAE: 0.2916666666666667
- invalid-vector rate: 0.0
- probability-mass max error: 1.1920928955078125e-07

## 6. Transfer gate

The preregistered pooled transfer gate **FAILED**.

Observed:

- composed severity delta: +0.19791666666666663
- worst-factor top-1 delta: +0.046875
- F0 top-1 improvement: +0.14583333333333337
- F1 top-1 improvement: +0.08333333333333337
- F2 top-1 improvement: +0.046875
- no factor regressed

The severity-improvement requirement passed.

The worst-factor improvement requirement failed:

- required: >= +0.08
- observed: +0.046875

The dominant worst factor was F2.

## 7. Why promotion failed

The frozen absolute sealed gate required, per domain:

- every factor top-1 >= 0.90;
- every factor balanced accuracy >= 0.88;
- factor-vector top-1 >= 0.82;
- composed severity top-1 >= 0.82;
- invalid-vector rate <= 0.05.

The pooled result already localizes the remaining bottleneck:

- F0 top-1: 0.9427083333333334
- F1 top-1: 0.953125
- F2 top-1: 0.8125
- F2 BA: 0.7916666666666667
- composed severity: 0.7135416666666666
- invalid-vector rate: 0.0

W32 therefore solved neither the F2 absolute threshold nor the composed-severity threshold, despite excellent F0/F1 and perfect observed monotone-vector validity.

`HIRA_V0_TRANSFER_CORE_READY` is not authorized.

## 8. What W32 establishes

W32 positively establishes:

- shared low-rank state↔schema interaction can coexist with state/schema residual asymmetry;
- only 6,144 trainable parameters can push fresh sealed F0/F1 above 0.94/0.95 pooled;
- invalid-vector probability-mass regularization can reduce observed invalid-vector rate to zero on DEV and sealed confirmation;
- composed severity improved by nearly +0.20 absolute over frozen T0;
- full-K, state-once, typed primitive consistency, option-order invariance and relation-delta-zero remain intact;
- runtime integrity is no longer a blocking issue;
- simple monotone-vector validity is not enough to solve F2 semantics.

The dominant remaining bottleneck is now sharply localized to **F2 semantic/compositional discrimination**, not F0/F1, vector legality, runtime integrity, or generic state↔schema transfer.

## 9. Evidence boundary

RH/RI are permanently exposed.

They must never be used for:

- future training;
- DEV selection;
- hyperparameter tuning;
- candidate choice;
- architecture ranking between future candidates.

Only aggregate W32 observations may motivate a fresh next-wave hypothesis.

The W32 checkpoint must never be retuned against RH/RI.

## 10. Next-wave direction

A future W33 must use wholly fresh qualification/TRAIN/DEV/CONFIRM domains.

W33 should preserve:

- frozen A13 + exact W28 T0;
- state/schema asymmetric residual adaptation;
- shared low-rank interaction;
- anchor protection;
- structural monotone-vector validity;
- state-once/full-K/typed semantics.

But it must attack F2 specifically at the **composition mechanism**, without introducing a factor-specific classifier or using exposed RH/RI.

A defensible next hypothesis is a compact shared compositional latent operator that learns a generic conjunction/interaction relation between semantic evidence components before candidate scoring, while remaining shared across factors/domains and initialized to exact T0 identity.

This is only a fresh hypothesis. It requires preregistration and wholly fresh evidence before empirical use.

## 11. Promotion boundary

W32 authorizes no HIRA-v0 transfer-core promotion.

Calibration/OOD/null, high-K, latency/RAM and matched Laya/JEV claims remain blocked until a future fresh transfer wave earns:

`HIRA_V0_TRANSFER_CORE_READY`
