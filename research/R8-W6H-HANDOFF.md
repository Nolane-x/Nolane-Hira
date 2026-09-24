# R8-W6h handoff — fresh field-semantic rescue authority

Status: **CLOSED. Frozen authoritative verdict: `FIELD_SEMANTIC_FAIL`.**

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


## Authoritative closure

Exact empirical head:
- `8e1fa435318f39780ce0cc4f0acc4f0c7be5770e`.

Authority run:
- `36016490423`;
- every job PASS: unit -> upstream -> fresh cache -> three paths -> DEV-X freeze -> untouched CONFIRM-Y/Z.

Artifacts:
- fresh TRAIN/DEV cache `10815905048`, digest `sha256:d9147cd26771d9466b0f66f2599cfb4e86251e957b6af765a1a6f0f3efce49b6`;
- frozen control `10815531951`, digest `sha256:1119bb3bd51d82ed785dce5a31fdb6a8966b282edc0d2ded41fb0242d8158903`;
- projection-retune control `10816055074`, digest `sha256:54a673b057e8b509a6086e1b46c60263d195150ccfd4e7f4aca8f1667b9d74c1`;
- semantic residual adapter `10815347443`, digest `sha256:d9c7e2089140247a33822f5acdb6810dbc88ffa078390d1b56dacedd5f486da1`;
- DEV freeze `10816010756`, digest `sha256:f8e1b4bbf4280fb27b2227e053affc91cf3ad0cf740ad748c406d3cd9c612e81`;
- untouched CONFIRM `10815792599`, digest `sha256:b600b6cdf45fbea6f5d554d1286a46772610541913e370ac8c2052cce0c3bf19`.

Fresh cache provenance:
- TRAIN SHA `912a12299ba92c81ee41066d0469f9155f678754495d2f7565c6767d5fb684a6`;
- DEV-X SHA `75b27c76a9a6b7e46ed9efa84e886f3dcbbfbbd59eb521ac189eb49513df9e4a`;
- state encode 1/case;
- prior-authority value overlap 0;
- every state exposes all four one-field semantic pairs;
- no W6e CONFIRM / W6f / W6g rows used.

DEV-X freezes:
- frozen control epoch 0: overall 87.083%, K64 39.583%, field-pair 92.324%;
- projection-retune epoch 6: overall 89.271%, K64 68.750%, field-pair 97.561%;
- semantic adapter epoch 6: overall 86.771%, K64 43.750%, field-pair 94.261%.

### Untouched CONFIRM-Y

Frozen control:
- overall 90.3125%;
- choice 83.594%;
- noul 97.396%;
- score 93.490%;
- K64 47.917%;
- mean field-pair 94.820%.

Projection-retune control:
- overall 89.688%;
- choice 86.719%;
- noul 98.438%;
- score 88.281%;
- K64 56.250%;
- mean field-pair 96.141%.

Semantic residual adapter:
- overall 90.417%;
- choice 84.375%;
- noul 98.438%;
- score 92.448%;
- K64 45.833%;
- mean field-pair 95.391%;
- every role pair gate PASS;
- probability integrity PASS;
- state once PASS.

Adapter vs frozen:
- K64 **-2.083 pp**;
- field-pair **+0.572 pp**;
- overall +0.104 pp.

Adapter vs projection:
- K64 -10.417 pp;
- field-pair -0.750 pp.

Y absolute K64 gate fails. Causal K64-gain, field-pair-gain and K64-comparable-to-projection gates fail.

### Untouched CONFIRM-Z

Frozen control:
- overall 89.479%;
- choice 78.125%;
- noul 99.479%;
- score 95.833%;
- K64 37.500%;
- mean field-pair 95.129%.

Projection-retune control:
- overall 90.625%;
- choice 85.417%;
- noul 98.958%;
- score 91.667%;
- K64 64.583%;
- mean field-pair 96.454%.

Semantic residual adapter:
- overall 91.146%;
- choice 82.031%;
- noul 100%;
- score 95.833%;
- K64 62.500%;
- mean field-pair 96.132%;
- all absolute gates PASS;
- probability integrity PASS;
- state once PASS.

Adapter vs frozen:
- K64 **+25.000 pp**;
- field-pair **+1.003 pp**;
- overall +1.667 pp.

Adapter vs projection:
- K64 -2.083 pp;
- field-pair -0.322 pp.

Z passes the absolute K64 and projection-comparability gates and shows a large K64 gain, but the preregistered field-pair-gain causal gate requires +10 pp and observes only +1.003 pp.

### Frozen verdict

**`FIELD_SEMANTIC_FAIL`**

- full pass Y: false;
- full pass Z: false;
- material signal Y: false;
- material signal Z: false.

No gate or threshold was changed after CONFIRM exposure.

## Scientific interpretation

W6h falsifies the proposed universal rescue mechanism.

The 8,192-param residual adapter is not a stable cross-domain field-semantic rescue:
- it produces a large K64 improvement on Z;
- it fails to improve K64 on Y;
- its full-option one-field pair accuracy is already high before adaptation and improves only marginally;
- the larger 32,769-param projection retune is materially stronger on DEV-X and on both held-out K64 domains.

This exposes an important mismatch between the W6g diagnosis and the first rescue operationalization. W6g localized weak **isolated changed-value retrieval**. W6h trained and gated **full-option one-field pair margins**. Those full-option pairs are already near saturation (~95% on held-out controls), so they are not a sensitive surrogate for the isolated-value failure that W6g identified.

Per the preregistered boundary, do not enlarge A13 and do not stack another local synthetic mechanism onto this adapter. The next lane should reassess the production representation/interface itself, specifically whether free-form textual criterion strings are the right carrier for high-cardinality structured decisions, and should establish a fresh representation-level diagnostic before authorizing another rescue.


## Authoritative closure

Status: **CLOSED. Frozen verdict: `FIELD_SEMANTIC_FAIL`.**

Exact empirical head:
- `8e1fa435318f39780ce0cc4f0acc4f0c7be5770e`.

Authority run:
- `36016490423`;
- all jobs PASS through untouched CONFIRM-Y/Z.

CONFIRM artifact:
- ID `10815792599`;
- digest `sha256:b600b6cdf45fbea6f5d554d1286a46772610541913e370ac8c2052cce0c3bf19`.

### Untouched CONFIRM-Y

Frozen joint control:
- overall 90.3125%;
- diagnosis K64 47.917%;
- mean one-field pair accuracy 94.820%.

Projection-retune control:
- overall 89.6875%;
- diagnosis K64 56.250%;
- mean one-field pair accuracy 96.141%.

Semantic residual adapter:
- overall 90.417%;
- diagnosis K64 45.833%;
- mean one-field pair accuracy 95.391%;
- choice 84.375%;
- noul 98.438%;
- score 92.448%;
- probability-mass max error 1.48e-7;
- state encodes/case 1.0.

Adapter vs frozen control:
- overall +0.104 pp;
- K64 -2.083 pp;
- field-pair +0.572 pp.

The adapter fails the absolute K64 >=55% gate and fails the preregistered K64/pair causal-gain gates on Y.

### Untouched CONFIRM-Z

Frozen joint control:
- overall 89.479%;
- diagnosis K64 37.500%;
- mean one-field pair accuracy 95.129%.

Projection-retune control:
- overall 90.625%;
- diagnosis K64 64.583%;
- mean one-field pair accuracy 96.454%.

Semantic residual adapter:
- overall 91.146%;
- diagnosis K64 62.500%;
- mean one-field pair accuracy 96.132%;
- choice 82.031%;
- noul 100%;
- score 95.833%;
- probability-mass max error 1.36e-7;
- state encodes/case 1.0.

Adapter vs frozen control:
- overall +1.667 pp;
- K64 +25.000 pp;
- field-pair +1.003 pp.

The adapter passes all absolute Z competence gates and is K64/pair-comparable to projection retuning, but it fails the preregistered >=+10 pp field-pair causal-gain gate. Therefore Z is not a full rescue either.

### Frozen verdict

`FIELD_SEMANTIC_FAIL`.

No threshold is changed after CONFIRM exposure. Y/Z are permanently exposed and forbidden for future training, selection, threshold tuning or mechanism search.

### Scientific interpretation

W6h exposes an important bridge mismatch between W6g localization and the W6h rescue proxy.

W6g localized weak **isolated changed-value retrieval** on fresh domains. W6h trained/evaluated **full structured one-field option pairs**. On untouched Y/Z, the frozen control already scores about 94.8-95.1% on that structured pair metric before adaptation. The proxy therefore has little headroom and is not equivalent to the weak isolated-value measurement that authorized the lane.

The 8,192-param adapter does not reproducibly rescue K64: it regresses on Y while improving strongly on Z. The 32,769-param projection-retune control is also inconsistent across domains but reaches K64 56.25%/64.58%, showing that frozen projection geometry can move without establishing the adapter's localized causal claim.

Per the preregistered boundary, do not enlarge A13 and do not stack another local semantic adapter. The next lane must reassess the representation/interface bridge itself: whether free-form structured criterion strings and the current pair proxy preserve the field-semantic distinctions that the production decision actually requires.
