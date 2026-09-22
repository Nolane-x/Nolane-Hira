# R8-W3 handoff — typed-decisions specialist lane

Status: **W3 COMPLETE. W3b one-shot final executed exactly once: 1 Laya typed WIN, 0 TIE, 7 LOSS; Jev typed remains protocol-separated/MISSING.**

## Frozen phase boundary

W3a is model selection and may load only the pinned typed-decisions TRAIN split.

W3b is final evaluation and is forbidden until one W3a candidate config and head SHA are frozen.

No typed-decisions test row may participate in W3a.

## Frozen W3a authority

Dataset:
- LocalLLaMA/typed-decisions;
- revision c76749ec58bd8c3d2ea706b31c333a9059c38f90.

A13:
- microsoft/xtremedistil-l6-h256-uncased;
- revision 4226d9e4d2c08703e5cb0491b479bfc6a1607181;
- model.safetensors SHA-256 5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880;
- max_length 256;
- state segments 32 valid tokens;
- encoder frozen.

Train/dev:
- deterministic SHA256("R8-W3-DEV:" + case_id) ordering inside each workflow;
- 240 train + 60 dev per workflow;
- total 960 train / 240 dev.

## Candidate grid

Exactly eight:
- init: exact R15 or fresh seed 13;
- loss: balanced or soft;
- lr: 3e-4 or 1e-3.

All candidates:
- head-only;
- 422,159 parameters;
- AdamW, weight decay 0.01;
- eight epochs;
- global RNG seed 71;
- deterministic case shuffle seed 71 + epoch;
- full-K only;
- no adaptive budget.

## Selection

Within each candidate and across candidates:
1. higher dev accuracy;
2. lower hard Brier;
3. higher soft accuracy;
4. lower score MAE;
5. lower ECE;
6. earlier epoch.

Cross-candidate tie after those metrics:
- R15 init before fresh13;
- lower LR;
- balanced before soft.

## W3a artifacts

Expected:
- frozen A13 feature cache + receipt;
- eight candidate heads + receipts;
- one selection receipt;
- one selected train/dev head.

The selection receipt must state final_test_exposed=false.

## W3b

Only after W3a selection is merged/frozen:
- add explicit final-test adapter path;
- evaluate all 400 pinned test cases exactly once;
- do not train/select/calibrate on test;
- populate only metric-compatible Laya typed cells;
- label Jev comparison protocol-separated.


## W3a authoritative completion

Merged to `main` as `327679b87988f375e97b2a8a96456d35a8c54dde`.

Authoritative workflow:
- run `35748778854`: PASS;
- cache: PASS;
- all eight candidates: PASS;
- selector: PASS.

Frozen cache:
- 1,200 TRAIN cases / 6,000 decisions;
- 960 TRAIN / 240 DEV;
- exactly 1,200 A13 state encodes = 1 per case;
- cache SHA-256 `7e1e7fcc46cf107be138285efd320b37529adc141d0219e40696842ebd451b6f`.

Selected candidate:
- `r15-balanced-lr1e3`;
- selected epoch 7;
- lr 0.001;
- balanced frozen loss;
- selected head SHA-256 `2505e2cf99d741e590ff26ff7c70a587065e5713a4c6a3e53b6783a03b20446c`;
- selected artifact `10706135649`;
- artifact digest `sha256:8a2efd31a6ccea3b987957d328c4b73267ab00ff9024c469bcb7f3cd578d5d92`.

Frozen DEV selection metrics:
- accuracy 0.5491666667;
- choice accuracy 0.4777777778;
- noul accuracy 0.7222222222;
- score accuracy 0.4729166667;
- soft accuracy 0.3859879950;
- hard Brier 0.5646017494;
- raw ECE 0.0855358069;
- score MAE 0.5405605146.

These are **DEV selection metrics only**, not final benchmark results.

The selector receipt records `final_test_exposed=false`.

Machine-readable authority:
`artifacts/r8-w3a-selection/summary.json`.

## W3b one-shot rule

W3b must not retrain, recalibrate, choose thresholds, choose epochs, or change the head.

Final authority is frozen to:
- dataset `LocalLLaMA/typed-decisions@c76749ec58bd8c3d2ea706b31c333a9059c38f90`;
- config `all`;
- split `test`;
- all 400 cases / 2,000 decisions;
- selected head SHA-256 `2505e2cf99d741e590ff26ff7c70a587065e5713a4c6a3e53b6783a03b20446c`.

W3b code must be unit-tested without loading final test. The actual final evaluation is triggered once by a dedicated execution-marker commit after evaluator code is merged.


## W3b authoritative one-shot final — COMPLETE

Evaluator merged before exposure:
- `d35716b01e4962b9ea561904267b453b982797ac`.

Execution authority:
- branch `exec/r8-w3b-final`;
- single marker commit `df226492d0387aa8ee91ea513d6aa8557759e770`;
- workflow run `35795982760`;
- job `106975249753`;
- workflow result: **SUCCESS**;
- exposure boundary: `R8_W3B_FINAL_TEST_EXPOSURE_BEGIN`;
- final artifact `10723399141`;
- artifact digest `sha256:2dde91ffe7b79b831fdac5d853093122b99684e0595d4f5e8c3cfdac120023d7`.

Frozen model:
- candidate `r15-balanced-lr1e3`;
- selected epoch 7;
- selected head SHA-256 `2505e2cf99d741e590ff26ff7c70a587065e5713a4c6a3e53b6783a03b20446c`;
- HIRA head parameters: 422,159;
- full-K forced budget 255;
- adaptive budget disabled.

Final authority:
- `LocalLLaMA/typed-decisions@c76749ec58bd8c3d2ea706b31c333a9059c38f90`;
- config `all`;
- split `test`;
- 400 cases / 2,000 decisions;
- exactly 400 state encodes = 1 per case;
- exactly 5 decisions per state encode.

### Exact Laya typed result

| Cell | HIRA | Laya target | Status |
|---|---:|---:|---|
| accuracy | 0.5245 | 0.766 | LOSS |
| soft accuracy | 0.3831433658 | 0.471 | LOSS |
| hard Brier ↓ | 0.5749368837 | 0.061 | LOSS |
| raw ECE ↓ | 0.0655248929 | 0.213 | **WIN** |
| score MAE ↓ | 0.5103628148 | 0.242 | LOSS |
| noul accuracy | 0.7016666667 | 0.857 | LOSS |
| choice accuracy | 0.4183333333 | 0.733 | LOSS |
| score accuracy | 0.47125 | 0.723 | LOSS |

Typed specialist summary:
- **WIN 1**
- **TIE 0**
- **LOSS 7**

53-cell headline board after this lane:
- WIN 1;
- TIE 0;
- LOSS 7;
- MISSING 45.

All five Jev typed published cells remain **MISSING** because the Jev result is a protocol-separated generalist authority. The W3 specialist numbers must not be used to declare a Jev win or loss.

### Scientific interpretation

This W3 result is a **scientific negative** for current HIRA typed specialist quality versus the frozen Laya typed target.

The low raw ECE is real under the frozen metric, but it does not imply high decision competence. Accuracy, primitive accuracies, score MAE and especially hard Brier remain substantially behind Laya.

DEV accuracy was 0.5491666667 and final accuracy is 0.5245, so the primary failure is not a dramatic final-only collapse. The current specialist is simply not competitive enough on this benchmark.

Do **not** open a posthoc W3c that tunes against this public final. Any future typed-quality improvement must use a new preregistered development/confirmatory authority and preserve this final result unchanged.

Machine-readable final:
`artifacts/r8-w3b-final/summary.json`.
