# R8-W3 handoff — typed-decisions specialist lane

Status: **W3a TRAIN-only implementation in progress under issue #50.**

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
