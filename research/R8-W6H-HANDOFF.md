# R8-W6h handoff — fresh field-semantic rescue authority

Status: **PRE-AUTHORITY IMPLEMENTATION ACTIVE. No W6h empirical verdict is valid yet.**

Issue: #91

## Frozen premise

W6g closed as `STABLE_SECOND_ORDER_LOCALIZATION`.

Across fresh Q/R/S diagnostics, all four semantic roles converged on:
`FIELD_SEMANTIC_COLLAPSE`.

This authorizes one rescue target only: improve field-semantic separability before high-K competition.

Forbidden:
- W6e CONFIRM L/M;
- W6f N/O/P;
- W6g Q/R/S;
- all earlier W6 CONFIRM rows;
- typed final/test rows;
- public campaign cells;
- A13 scaling;
- domain-specific parser features.

## Frozen upstream

A13:
- model `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`.

W6e joint-primary initialization:
- HIRA SHA `d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588`;
- scorer SHA `6d5a7f2d3ed63ecd756181b1cb54e4704f68e5f74983a897f0a68fb4d1d63d2e`;
- artifact `10805567861`.

## Primary rescue mechanism

`SemanticResidualAdapter`:
- operates in the frozen scorer's 128-d projected token space;
- shared for state, question and option tokens;
- 128 -> 32 -> 128 bias-free residual MLP;
- GELU bottleneck;
- zero-initialized up projection;
- exact trainable budget **8,192 parameters**;
- no domain IDs, role IDs, gold field indices or candidate metadata are model inputs.

The initial adapter scorer is functionally identical to the frozen base scorer.

## Fresh authority domains

TRAIN:
- T aerospace ground support;
- U food cold-chain;
- V coastal flood pumping;
- W data-center thermal operations.

DEV:
- X railway signalling.

Sealed CONFIRM:
- Y pharmaceutical packaging;
- Z solar-storage microgrid.

Seeds:
- T 221227;
- U 221231;
- V 221237;
- W 221239;
- X 222347;
- Y 223451;
- Z 224557;
- optimizer/global 1601.

Counts:
- TRAIN 384 states / 1,920 decisions;
- DEV-X 192 / 960;
- CONFIRM-Y 192 / 960;
- CONFIRM-Z 192 / 960.

All domains use the same 48 K/severity/confidence strata.

A pre-data generator audit found that the shared W6 domain generator did not guarantee every K8 state contains at least one one-field negative for each semantic role. W6h therefore deterministically replaces only non-one-field distractors as needed so every state exposes all four one-field semantic pair labels while preserving:
- case count;
- K;
- gold option;
- gold probabilities;
- state text;
- optimizer budget.

This repair occurred before any W6h A13 cache, training or CONFIRM exposure.

## Preregistered paths

1. `frozen-joint-control`
   - exact W6e joint-primary;
   - 0 trainable params.

2. `projection-retune-control`
   - HIRA frozen;
   - original competitive scorer trainable;
   - 32,769 trainable params.

3. `semantic-residual-adapter`
   - HIRA frozen;
   - original W6e scorer frozen;
   - residual adapter only;
   - 8,192 trainable params.

No cross-candidate winner selection is allowed. Each trainable path freezes independently on DEV-X.

## Objective and selection

Six epochs, AdamW lr 3e-4, weight decay .01.

Loss:
- existing typed W6 objective, weight 1.0;
- one-field semantic pair logistic loss, weight 0.5.

DEV-X ordering:
1. diagnosis K64;
2. mean one-field pair accuracy;
3. overall typed accuracy;
4. choice;
5. score;
6. noul;
7. lower hard Brier;
8. lower score MAE;
9. earlier epoch.

CONFIRM-Y/Z remain sealed until all three paths freeze.

## Current implementation

Implemented:
- `src/nmd/field_semantic_rescue.py`;
- `src/nmd/field_semantic_rescue_authority.py`;
- `scripts/r8_w6h_build_cache.py`;
- `scripts/r8_w6h_train_candidate.py`;
- `scripts/r8_w6h_freeze_candidates.py`;
- `scripts/r8_w6h_confirm.py`;
- `tests/test_field_semantic_rescue.py`;
- `tests/test_field_semantic_rescue_authority.py`;
- pre-authority unit workflow.

Before empirical authority is enabled, unit CI must prove:
- exact 8,192 adapter params;
- zero-init functional identity;
- HIRA/base-scorer freezing for adapter path;
- exact 32,769 projection control budget;
- fresh T-Z value/template/role separation;
- four one-field pairs per state;
- Y/Z seal;
- no prior authority leakage;
- frozen gate/verdict semantics;
- production regression compatibility.

No W6h empirical result is valid until these contracts pass.
