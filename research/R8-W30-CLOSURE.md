# R8-W30 closure — low-rank semantic-transfer bridge

Status: **CLOSED — W30_REFERENCE_INADEQUATE**

Issue: #149  
PR: #150  
Authority branch: `feat/r8-w30-semantic-transfer-bridge`  
Authority head: `290825ee698158eb33c4193018731d3892924563`  
Base main: `bc8827d3a86a7ade940e6d66ea72287aa6d624f9`

## 1. Authoritative execution

GitHub Actions run: `36280493847`

The preregistered execution order completed:

- unit: PASS
- exact W28 T0 provenance: PASS
- TRAIN/DEV bridge fitting and selection: PASS
- first sealed FF/FG CONFIRM exposure: PASS as an audit execution
- original bundle packaging: content creation succeeded; verification step failed only because `sha256sum` was invoked from the wrong working directory

The packaging failure occurred after the scientific audit and does not alter the W30 outcome.

## 2. Frozen artifacts

### Exact W28 T0
- artifact: `r8-w30-frozen-t0`
- artifact ID: `10918563713`
- artifact digest: `sha256:5dc21d2443fba08214e762fe167788dbfc77076c3ba72327289bde5fc02881b3`
- internal candidate checkpoint SHA256:
  `1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f`

### W30 bridge training
- artifact: `r8-w30-bridge-training`
- artifact ID: `10918956343`
- artifact digest: `sha256:0d542a9778e3049812af417474a523a6e0e6eadb1ed2476372f22a1c5a100508`
- selected bridge checkpoint SHA256:
  `58387533c061a1306cf84c7910663924df2f7f0a1a2903cdc3f6db034e859d31`
- selected DEV epoch: `12`

### W30 sealed audit
- artifact: `r8-w30-authoritative-audit`
- artifact ID: `10919026510`
- artifact digest:
  `sha256:2d24b7b2081bbb238b0b50b9f01233db531b1cbcdbfd80d42509aa80be34a92c`

## 3. TRAIN/DEV result

DEV contained 96 cases and was the only selection authority.

Frozen unbridged T0 baseline on DEV:
- F0 top-1: 0.78125
- F1 top-1: 0.78125
- F2 top-1: 0.78125
- factor-vector top-1: 0.53125
- composed severity top-1: 0.53125
- invalid-vector rate: 0.1875

Selected rank-8 bridge on DEV:
- F0 top-1: 0.96875
- F1 top-1: 0.7916666666666666
- F2 top-1: 0.8125
- factor-vector top-1: 0.5833333333333334
- composed severity top-1: 0.5833333333333334
- invalid-vector rate: 0.010416666666666666

Interpretation: the bridge improved DEV geometry, especially F0 and vector validity, but F1/F2 remained far below the intended production gate.

## 4. First sealed FF/FG result

Authoritative outcome:

`W30_REFERENCE_INADEQUATE`

Reference qualification:
- FF: PASS
- FG: FAIL

Because reference adequacy has absolute precedence, W30 cannot use FF/FG as a clean semantic promotion authority.

The HIRA/runtime observations are still frozen historical measurements and may not be tuned against.

### Frozen unbridged pooled FF+FG

- F0 top-1: 0.8541666666666666
- F1 top-1: 0.8229166666666666
- F2 top-1: 0.75
- factor-vector top-1: 0.46875
- composed severity top-1: 0.46875
- invalid-vector rate: 0.010416666666666666
- probability-mass max error: 1.1920928955078125e-07

### Frozen bridged pooled FF+FG

- F0 top-1: 0.9010416666666666
- F1 top-1: 0.8125
- F2 top-1: 0.6197916666666666
- factor-vector top-1: 0.3958333333333333
- composed severity top-1: 0.3958333333333333
- invalid-vector rate: 0.13541666666666666
- probability-mass max error: 1.1920928955078125e-07

Observed pooled composed-severity delta versus unbridged:
`-0.07291666666666669`

Transfer gate: **FAIL**

Quality gates:
- FF: FAIL
- FG: FAIL

Runtime gates:
- FF: PASS
- FG: PASS

## 5. What W30 establishes

W30 successfully demonstrates that:

- the W29 runtime can host a distinct bridged semantic scorer without breaking backward compatibility;
- a shared rank-8 residual bridge can be trained while keeping A13 and W28 T0 frozen;
- exactly 2,048 bridge parameters can be isolated, checkpointed, frozen and replayed;
- TRAIN/DEV/CONFIRM evidence separation works;
- typed state-once runtime, full-K behavior, option-order invariance and relation-refinement-off behavior survive the bridge path;
- W8 semantic-transfer modules can coexist untouched with a separate W30 authority namespace.

W30 does **not** establish semantic-transfer readiness.

## 6. Interpretation boundary

The scientific outcome is `W30_REFERENCE_INADEQUATE`, not
`HIRA_V0_TRANSFER_CORE_READY`.

Because FG reference adequacy failed, no clean claim about model quality on the full FF/FG authority is permitted.

Separately, the observed rank-8 bridge did not show a promising sealed pooled signal:
- F1 regressed slightly;
- F2 regressed materially;
- composed severity regressed;
- invalid-vector rate increased.

These are frozen observations, not authorization to tune on FF/FG.

FF and FG are now permanently exposed and forbidden for all future fitting, selection or candidate choice.

## 7. Required next direction

A future W31 must not retry or tune this rank-8 bridge on W30 evidence.

W31 should first introduce a **fresh semantic-reference qualification stage** before any candidate is trained or promoted, so weak authority domains are rejected before they consume a sealed model evaluation.

The next architectural candidate must attack F1/F2 compositional transfer more directly than a single shared linear residual bridge while remaining compact and state-once.

All W31 TRAIN/DEV/CONFIRM/qualification text must be wholly fresh.

## 8. Promotion boundary

W30 authorizes **no** HIRA-v0 transfer-core promotion.

Calibration, OOD/null, high-K and Laya/JEV matched claims remain blocked until a future fresh transfer wave passes its own qualified sealed authority.
