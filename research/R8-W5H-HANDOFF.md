# R8-W5h handoff — balanced anti-collapse token binding

Status: **implementation active under issue #73; no empirical W5h result yet.**

## Frozen premise

W5g reached 50.52% fresh untouched accuracy, MRR 0.6402 and K255 accuracy 41.67%, but missed:
- overall accuracy >= 0.60;
- K255 top-5 >= 0.70.

The remaining hypothesis is many-to-one MaxSim collapse: several option tokens can reuse one context token as their strongest evidence.

## Controlled candidates

All candidates have exactly 32,769 trainable parameters:
- bias-free 256->128 projection;
- scalar logit scale.

Only binding differs:
1. `idf-maxsim-proj128` — fresh W5g-style control;
2. `idf-competitive-proj128` — subtract within-option sibling-token common mode per context token before MaxSim;
3. `idf-greedy-unique-proj128` — deterministic unique context assignment by salience before fallback.

IDF salience and final weighted-mean + minimum-coverage aggregation are frozen.

## Fresh authority

TRAIN 512 / DEV 176 / post-selection CONFIRM 192.

Seeds:
- train 131001;
- dev 132002;
- confirm 133003;
- initialization/training 503.

W5h vocab/templates/case IDs must be disjoint from W5a-W5g.

## Selection

DEV-only:
accuracy -> K255 top-5 -> K255 accuracy -> K128 accuracy -> overall top-5 -> MRR -> Brier -> earlier epoch.

Selector freezes one candidate and carries the fresh `idf-maxsim-proj128` checkpoint as the same-authority mechanism control.

## Frozen rescue gates

Competence:
- overall >= 0.60;
- K128 >= 0.40;
- K255 >= 0.30;
- K255 top-5 >= 0.70;
- probability error <= 1e-6.

Mechanism:
- overall gain vs control >= +0.05;
- K255 top-5 gain vs control >= +0.05.

No public campaign cells may be populated by W5h.
