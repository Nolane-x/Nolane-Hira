# R8-W32 handoff — dual residual adapter + shared interaction metric

Status: **REFERENCE QUALIFIED / TRAIN-DEV PRE-EXPOSURE**

Issue: #153  
Branch: `feat/r8-w32-interaction-semantic-adapter`  
Base main: `762289911da2b7cde860383268c5d35d6ebd2f98`

## Why W32 exists

W31 closed `W31_DUAL_ADAPTER_FAIL`, but it produced a strong positive result:
- reference qualification passed;
- runtime integrity passed on both sealed domains;
- pooled transfer gate passed strongly;
- state/schema asymmetry plus anchor preservation improved held-out transfer;
- absolute F1/F2 discrimination and composed-vector quality remained below production thresholds.

W32 preserves the successful W31 principles but adds a shared low-rank state↔schema interaction metric.

## Frozen base

- A13 frozen;
- exact W28 T0 frozen;
- W29 unbridged T0 scorer is the baseline;
- W31 checkpoint is not reused as initialization;
- relation refinement OFF;
- no factor-specific learned head;
- no primitive-specific learned head.

## Candidate

Residual semantic geometry:

```
t_s = T0(state)
t_o = T0(schema)

z_s = normalize(t_s + U_s(D_s(t_s)))
z_o = normalize(t_o + U_o(D_o(t_o)))
```

Shared interaction metric:

```
r_s = M_s(z_s)
r_o = M_o(z_o)
I(o,s) = <r_o, r_s> / sqrt(8)
S(o,s) = <z_o, z_s> + I(o,s)
```

Capacity:
- state residual rank-8 adapter: 2,048;
- schema residual rank-8 adapter: 2,048;
- state interaction 128→8: 1,024;
- schema interaction 128→8: 1,024;
- total trainable: **6,144**.

Initialization:
- both residual up projections zero;
- state interaction map zero;
- schema interaction map seeded normal/default basis;
- interaction exactly zero initially;
- initial logits exactly equal frozen unbridged T0.

Runtime mode:
`interaction_symmetric_semantic`

## Training contract

- seed: 3217
- epochs: 18
- batch: 32
- AdamW lr: 2e-4
- weight decay: 0.01
- grad clip: 1.0
- temperature: 0.07
- anchor coefficient: 0.35
- frozen T0 anchor margin: 0.08
- structural invalid-vector coefficient: 0.15
- trainable candidate parameters: exactly 6,144

Primary loss is factor-balanced F0/F1/F2 CE.

Anchor loss preserves frozen-T0 correct decisions with signed margin >= .08.

Structural validity loss penalizes probability mass assigned to the four invalid binary vectors:
001, 010, 011, 101.
Allowed vectors remain 000, 100, 110, 111.

DEV selection:
1. worst-factor balanced accuracy;
2. worst-factor top-1;
3. composed severity;
4. lower invalid-vector rate;
5. earlier epoch.

## Fresh evidence

### Reference qualification
RA, RB — 96 cases/domain.

Qualification uses only the frozen DeBERTa/RoBERTa NLI panel. No HIRA candidate and no A13 load.

### TRAIN
RC, RD, RE, RF.

### DEV
RG.

### SEALED CONFIRM
RH, RI.

All partition contexts, style banks and generated texts are fresh and disjoint from W29/W30/W31 and older authority evidence.

## Promotion gate

Per sealed domain:
- F0/F1/F2 top-1 >= .90
- each factor BA >= .88
- factor-vector top-1 >= .82
- composed severity >= .82
- invalid-vector <= .05
- option-order invariance = 1
- state-once = 1
- full-K = 1
- relation delta = 0
- probability mass error <= 1e-6

Pooled transfer versus frozen T0:
- composed severity delta >= +.15
- worst-factor top-1 delta >= +.08
- no factor top-1 regression > .02

## Frozen outcomes

1. `W32_REFERENCE_QUALIFICATION_FAIL`
2. `W32_INTERACTION_ADAPTER_FAIL`
3. `HIRA_V0_TRANSFER_CORE_READY`

No partial promotion.

## Evidence firewall

Forbidden for W32 fitting/selection/candidate choice:
- W29 EW/EX/EY/EZ;
- W30 FA..FG;
- W31 QH..QP;
- all W28 and older authority rows;
- Banking77 final/test;
- typed final/test;
- Laya/JEV result cells.

Prior aggregate metrics may motivate architecture only.

## Current boundary

Reference qualification is complete and frozen:
- authority run: `36292606367`
- outcome: `W32_REFERENCE_QUALIFIED`
- RA: PASS
- RB: PASS
- 192 reference-only cases
- no HIRA candidate evaluated
- A13 not loaded
- no W29/W30/W31/older authority rows used
- qualification artifact: `r8-w32-reference-qualification`
- artifact ID: `10922738458`
- artifact digest: `sha256:02cafde04b10c075df848d3c0d45b9f9c139b3ae1516a82c2a428e2163131751`

Implemented before qualification:
- `src/nmd/w32_interaction_semantic_adapter.py`
- runtime mode `interaction_symmetric_semantic`
- exact 6,144-parameter capacity and T0-identity tests
- typed/full-K/state-once/relation-delta-zero runtime tests
- fresh `src/nmd/w32_transfer_authority.py` with RA..RI
- exact-text freshness tests against W28/W29/W30/W31
- `scripts/r8_w32_qualify.py`
- marker-gated reference qualification workflow
- dedicated W32 unit/backward-compat gate

Next frozen boundary:
1. implement RC/RD/RE/RF TRAIN cache and RG DEV cache;
2. implement factor-balanced CE + T0 anchor preservation + invalid-vector probability-mass penalty;
3. implement 6,144-parameter checkpoint/core and DEV selector;
4. pass dedicated + full repository CI;
5. only then create `research/R8-W32-ENABLE-TRAIN`;
6. freeze selected RG checkpoint and receipt;
7. RH/RI remain sealed until that freeze is complete;
8. sealed confirm, closure and FULL bundle.

RA/RB are now exposed and permanently forbidden from candidate loss or DEV selection.

TRAIN/DEV implementation is now present but remains pre-exposure:
- `src/nmd/w32_transfer_cache.py`
- `src/nmd/w32_transfer_eval.py`
- `src/nmd/w32_transfer_core.py`
- `scripts/r8_w32_train.py`
- `tests/test_w32_transfer_training.py`
- marker-gated `r8-w32-train-dev` workflow

Frozen training objective:
- factor-balanced F0/F1/F2 CE;
- frozen-T0 positive-anchor hinge, threshold 0.08, coefficient 0.35;
- invalid-vector probability mass over 001/010/011/101, coefficient 0.15;
- exactly 6,144 trainable candidate parameters;
- RC/RD/RE/RF TRAIN = 384 cases;
- RG DEV = 96 cases;
- RH/RI confirm usage = 0 during training/selection.

Do not create `research/R8-W32-ENABLE-TRAIN` until the extended W32 unit gate and full repository CI are green.

