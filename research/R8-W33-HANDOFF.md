# R8-W33 handoff — shared co-evidence compositional metric

Status: **PREREGISTERED / PRE-QUALIFICATION**

Issue: #155  
Branch: `feat/r8-w33-coevidence-composition`  
Base main: `8be5d068225d364211987d0b73071030e39401c4`

## Why W33 exists

W32 closed `W32_INTERACTION_ADAPTER_FAIL`.

The important W32 result was not a generic transfer collapse:

- sealed F0 top-1: 0.9427083333333334;
- sealed F1 top-1: 0.953125;
- sealed F2 top-1: 0.8125;
- sealed F2 BA: 0.7916666666666667;
- composed severity: 0.7135416666666666;
- invalid-vector rate: 0.0;
- runtime integrity passed on RH and RI.

The remaining dominant bottleneck is therefore F2 compositional discrimination.

W33 tests whether generic **co-evidence** — support from more than one distinct semantic state token — can recover conjunction-like semantics without introducing a factor-specific head.

## Frozen base

- A13 frozen;
- exact W28 T0 frozen;
- W29 unbridged symmetric scorer remains the baseline;
- W32 checkpoint is not reused as initialization;
- relation refinement OFF;
- no factor-specific learned head;
- no primitive-specific learned head.

## Candidate

W33 preserves from-scratch W32 geometry:

```
z_s = normalize(T0(state)  + U_s(D_s(T0(state))))
z_o = normalize(T0(schema) + U_o(D_o(T0(schema))))

r_s = M_s(z_s)
r_o = M_o(z_o)
I(o,s) = <r_o, r_s> / sqrt(8)
```

Then adds a separate shared rank-8 composition space:

```
c_s = C_s(z_s)
c_o = C_o(z_o)
P(o,s) = <c_o, c_s> / sqrt(8)
```

For each option semantic view, the co-evidence term is derived from the **second strongest distinct state-token match** in the composition matrix. It therefore rewards semantic support that is distributed across multiple state tokens instead of being explained entirely by a single strongest token.

The operator is shared across every factor, schema, primitive and domain.

## Capacity

- state residual rank-8: 2,048
- schema residual rank-8: 2,048
- state interaction 128→8: 1,024
- schema interaction 128→8: 1,024
- state composition 128→8: 1,024
- schema composition 128→8: 1,024
- total trainable candidate parameters: **8,192**

Initialization:

- residual up projections zero;
- state interaction map zero;
- state composition map zero;
- schema interaction/composition maps retain seeded basis initialization;
- interaction term = 0 at initialization;
- co-evidence term = 0 at initialization;
- initial W33 logits must exactly equal frozen T0.

Runtime mode:

`coevidence_symmetric_semantic`

## Training contract

- seed: 3317
- epochs: 20
- batch: 32
- AdamW lr: 2e-4
- weight decay: 0.01
- grad clip: 1.0
- semantic temperature: 0.07
- anchor coefficient: 0.35
- frozen-T0 positive anchor margin: 0.08
- invalid-vector mass coefficient: 0.15
- exactly 8,192 candidate parameters may receive gradients

Primary loss:

- factor-balanced F0/F1/F2 CE.

Anchor loss:

- preserve frozen T0 correct decisions with margin >= 0.08.

Structural validity:

- valid vectors: 000, 100, 110, 111;
- invalid vectors: 001, 010, 011, 101.

No F2-specific loss weight or factor-ID feature is permitted.

## DEV selection

1. worst-factor balanced accuracy;
2. worst-factor top-1;
3. composed severity top-1;
4. lower invalid-vector rate;
5. earlier epoch.

## Fresh evidence

### Reference qualification
- SA
- SB

### TRAIN
- SC
- SD
- SE
- SF

### DEV
- SG

### SEALED CONFIRM
- SH
- SI

Each domain has 96 balanced cases. All contexts, style banks and exact texts must be fresh against W28/W29/W30/W31/W32.

## Promotion gate

Per sealed domain:

- every factor top-1 >= 0.90
- every factor BA >= 0.88
- factor-vector top-1 >= 0.82
- composed severity >= 0.82
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

1. `W33_REFERENCE_QUALIFICATION_FAIL`
2. `W33_COEVIDENCE_COMPOSITION_FAIL`
3. `HIRA_V0_TRANSFER_CORE_READY`

No partial promotion.

## Evidence firewall

Forbidden for W33 fitting/selection/candidate choice:

- W29 EW/EX/EY/EZ
- W30 FA..FG
- W31 QH..QP
- W32 RA..RI
- W28 and older authority rows
- Banking77 final/test
- typed final/test
- Laya/JEV benchmark cells

Only aggregate prior-wave metrics may motivate architecture.

## Current boundary

No W33 empirical evidence has been exposed.

Next:

1. implement co-evidence scorer/runtime;
2. lock capacity/identity/runtime contracts;
3. create wholly fresh SA..SI authority;
4. pass unit/full CI;
5. run reference-only SA/SB qualification;
6. only if qualified, build TRAIN/DEV;
7. freeze selected checkpoint before SH/SI implementation/exposure;
8. sealed confirm, closure and full evidence bundle.
