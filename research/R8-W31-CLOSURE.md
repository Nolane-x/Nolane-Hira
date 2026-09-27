# R8-W31 closure — qualified dual semantic adapter

Status: **CLOSED — W31_DUAL_ADAPTER_FAIL**

Issue: #151  
PR: #152  
Branch: `feat/r8-w31-qualified-dual-semantic-adapter`  
Base main: `fc906391fd55ffc6edcce60e383364801260701a`

## 1. Purpose

W31 followed W30's `W30_REFERENCE_INADEQUATE` result.

It introduced two changes:

1. a reference-only qualification stage before HIRA training;
2. separate rank-8 residual adapters for state tokens and schema tokens.

The frozen candidate contains exactly 4,096 trainable adapter parameters:
- state adapter: 2,048;
- schema adapter: 2,048.

A13 and the exact W28 T0 projection remained frozen.

## 2. Reference qualification

Authoritative qualification run:
`36281931836`

Outcome:
`W31_REFERENCE_QUALIFIED`

Domains:
- QH: PASS
- QI: PASS

Qualification properties:
- 192 cases total;
- no HIRA candidate evaluated;
- A13 not loaded;
- no W29/W30/older authority rows used;
- no exact-text overlap with exposed prior authorities.

Frozen artifact:
- name: `r8-w31-reference-qualification`
- artifact ID: `10918919102`
- digest: `sha256:badc9d6e91864519d7a3472b6b7cde313e42908611be38faa978a95b8f622d5e`

## 3. TRAIN/DEV

Authoritative TRAIN/DEV run:
`36290117569`

Partitions:
- TRAIN: QJ/QK/QL/QM = 384 cases
- DEV: QN = 96 cases

Frozen optimizer:
- seed: 3117
- epochs: 16
- batch: 32
- AdamW lr: 2e-4
- weight decay: 0.01
- grad clip: 1.0
- temperature: 0.07
- anchor coefficient: 0.35
- positive baseline anchor margin threshold: 0.08

Selected DEV epoch:
`12`

Selected adapter checkpoint SHA256:
`2a14479cc45b4e92c9ac76ce1535500b11778c8aa5b194b4ff53669f7f11054f`

Anchor rate:
`0.6102430555555556`

Frozen training artifact:
- name: `r8-w31-dual-adapter-training`
- artifact ID: `10922330271`
- digest: `sha256:2e055d2050b3fc27c85bdb47aa0044947b5b33b27b76d5c4f96989286aef358c`

### QN unbridged T0 baseline

- F0 top-1: 0.625
- F1 top-1: 0.7291666666666666
- F2 top-1: 0.7916666666666666
- factor-vector top-1: 0.3958333333333333
- composed severity top-1: 0.3958333333333333
- invalid factor-vector rate: 0.3958333333333333

### QN selected dual adapter

- F0 top-1: 0.8645833333333334
- F1 top-1: 0.875
- F2 top-1: 0.8645833333333334
- factor-vector top-1: 0.6354166666666666
- composed severity top-1: 0.6354166666666666
- invalid factor-vector rate: 0.010416666666666666
- probability mass max error: 1.1920928955078125e-07

W31 therefore improved DEV transfer substantially before sealed evaluation.

## 4. SEALED CONFIRM

First sealed QO/QP run:
`36290775556`

Outcome:
`W31_DUAL_ADAPTER_FAIL`

Runtime gates:
- QO: PASS
- QP: PASS

Absolute quality gates:
- QO: FAIL
- QP: FAIL

The failure is scientific, not infrastructure-related.

Frozen audit artifact:
- name: `r8-w31-authoritative-audit`
- artifact ID: `10922082803`
- digest: `sha256:2ac2825617223df4bf8d68701618e938f60117d2c1a2db1370781e51ae06bc9d`

## 5. Frozen pooled QO+QP metrics

### Unbridged T0 baseline

- F0 top-1: 0.609375
- F1 top-1: 0.6666666666666666
- F2 top-1: 0.7708333333333334
- F0 balanced accuracy: 0.7395833333333334
- F1 balanced accuracy: 0.6666666666666666
- F2 balanced accuracy: 0.5416666666666666
- factor-vector top-1: 0.2864583333333333
- composed severity top-1: 0.2864583333333333
- composed severity MAE: 1.6614583333333333
- invalid factor-vector rate: 0.4739583333333333
- probability mass max error: 1.1920928955078125e-07

### W31 dual adapter

- F0 top-1: 0.828125
- F1 top-1: 0.8541666666666666
- F2 top-1: 0.8541666666666666
- F0 balanced accuracy: 0.8854166666666667
- F1 balanced accuracy: 0.8541666666666666
- F2 balanced accuracy: 0.75
- factor-vector top-1: 0.6197916666666666
- composed severity top-1: 0.6197916666666666
- composed severity MAE: 0.5364583333333334
- invalid factor-vector rate: 0.057291666666666664
- probability mass max error: 1.1920928955078125e-07

## 6. Transfer gate

The preregistered pooled transfer gate PASSED.

Observed:
- composed severity delta: +0.3333333333333333
- worst-factor top-1 delta: +0.21875
- F0 top-1 change: +0.21875
- F1 top-1 change: +0.1875
- F2 top-1 change: +0.08333333333333326
- no factor regressed

This is the key W31 result:

**the dual asymmetric adapter materially improves semantic transfer, but absolute quality remains below the frozen production threshold.**

## 7. Why promotion failed

The absolute sealed gate required, per domain:
- every factor top-1 >= 0.90;
- every factor BA >= 0.88;
- factor-vector top-1 >= 0.82;
- composed severity top-1 >= 0.82;
- invalid-vector rate <= 0.05.

The pooled result already shows the remaining bottleneck:
- F0 top-1: 0.828125
- F1 top-1: 0.8541666666666666
- F2 top-1: 0.8541666666666666
- F1 BA: 0.8541666666666666
- F2 BA: 0.75
- composed severity: 0.6197916666666666
- invalid-vector rate: 0.057291666666666664

Therefore `HIRA_V0_TRANSFER_CORE_READY` is not authorized.

## 8. What W31 establishes

W31 positively establishes:

- reference qualification can be separated from candidate training;
- qualification can run without loading A13 or evaluating HIRA;
- state/schema semantic roles benefit from independent compact adaptation;
- 4,096 trainable parameters can substantially improve held-out transfer;
- anchor preservation avoids the strong regression pattern observed in W30;
- full-K, state-once, typed primitive consistency, option-order invariance and relation-delta-zero survive;
- transfer is no longer the primary failure mode.

The new dominant bottleneck is **absolute F1/F2 semantic discrimination and composed-vector accuracy**, not runtime integrity.

## 9. Evidence boundary

QO/QP are permanently exposed.

They must never be used for:
- future training;
- DEV selection;
- hyperparameter tuning;
- candidate choice;
- architecture ranking between future candidates.

Only aggregate W31 observations may motivate a fresh next-wave hypothesis.

The W31 checkpoint must never be retuned against QO/QP.

## 10. Next-wave direction

A future W32 must use wholly fresh qualification/TRAIN/DEV/CONFIRM domains.

The next candidate should preserve the successful state/schema asymmetry and anchor protection while attacking the remaining F1/F2 bottleneck more directly.

A reasonable research direction is a compact **interaction adapter** that improves cross-token relational composition after the separate state/schema residual maps, without introducing factor-specific heads or abandoning state-once/full-K semantics.

This is a new hypothesis only. It must be preregistered on fresh evidence before empirical use.

## 11. Promotion boundary

W31 authorizes no HIRA-v0 transfer-core promotion.

Calibration/OOD/null, high-K, latency/RAM and matched Laya/JEV claims remain blocked until a future fresh transfer wave earns:

`HIRA_V0_TRANSFER_CORE_READY`
