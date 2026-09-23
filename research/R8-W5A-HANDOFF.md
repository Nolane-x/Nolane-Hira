# R8-W5a handoff — fresh semantic routing competence rebuild

Status: **implementation active under issue #59; no W5a empirical result yet.**

## Frozen rationale

W4a:
- Banking77 semantic generalization: 0/400.

W4b:
- MECHANICS_PASS at K=128 and K=255;
- semantic-key routing near random.

Therefore W5a trains semantic relation/routing using only fresh generated data.

## Forbidden

No Banking77, typed final, W4b cases/vocab/templates, MASSIVE/XNLI eval or Laya/Jev final examples.

## Authorities

TRAIN:
- 640 cases;
- K 8/16/32/64/128/255;
- seed 51001;
- four training templates.

DEV:
- 160 cases;
- seed 52002;
- held-out templates;
- only authority for candidate/epoch selection.

CONFIRM:
- 256 cases;
- K=32/64/128/255, 64 each;
- seed 53003;
- disjoint vocabulary and templates;
- generated only after selected head/config is frozen.

## Model

Frozen A13 encoder.

Train only 422,159-parameter HIRA head.

Candidate grid:
- init R15 / W3;
- lr 3e-4 / 1e-3;
- four candidates;
- six epochs;
- AdamW wd 0.01;
- hard CE 1.0 + hard Brier 0.1;
- full-K forced budget 255;
- no adaptive budget.

## Confirm gate

SEMANTIC_COMPETENCE_PASS requires all:
- overall accuracy >= 0.50;
- K128 accuracy >= 0.30;
- K255 accuracy >= 0.20;
- K255 top5 >= 0.50;
- probability mass max error <= 1e-6;
- candidate budget exactly K;
- state-once 1 encode/case.

W5a populates zero R8 campaign cells.
