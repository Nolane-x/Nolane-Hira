# R8-W5h handoff — balanced anti-collapse token binding

Status: **implementation active under issue #73; pre-authority protocol repair applied. No W5h empirical verdict is valid until the repaired fresh authority completes.**

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

Repaired fresh-authority seeds:
- train 141109;
- dev 142211;
- confirm 143313;
- initialization/training 503 (unchanged).

Protocol repair boundary:
- branch head `8ebdcac...` still reused W5g textual template scaffolds even though its template IDs and domain vocabulary were new;
- issue #73 requires templates themselves to be disjoint, so any empirical run from that pre-repair head is non-authoritative and must not supply a W5h verdict;
- the repair changes only textual templates and authority seeds; candidates, A13, objectives, optimizer budget, selector order and rescue gates remain frozen;
- repaired template scaffolds are explicitly guarded against the W5g phrases in unit tests;
- W5h vocabulary/template text/case IDs/seeds must be disjoint from W5a-W5g.

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
