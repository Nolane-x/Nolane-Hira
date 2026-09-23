# R8-W5i handoff — cross-candidate evidence intersection

Status: **implementation branch active under issue #75; no W5i empirical result is valid yet.**

## Frozen premise

W5h closed as `BALANCED_BINDING_PARTIAL`:
- accuracy 47.40%;
- MRR 0.6500;
- top-5 86.98%;
- K128 accuracy 43.75%;
- K255 accuracy 27.08%;
- K255 top-5 70.83%.

The W5h mechanism control for W5i is soft within-option competitive binding.

W5i asks whether remaining top-1 ambiguity is caused by scoring each option independently after within-option competition.

## Frozen candidates

All candidates have exactly 32,769 trainable parameters and share:
- frozen A13;
- IDF salience;
- W5h competitive adjusted similarity;
- W5h forward aggregate;
- CE+Brier objective;
- six-epoch budget.

Candidates:
1. `idf-competitive-forward-proj128` — fresh W5h control.
2. `idf-competitive-bidir-mean-proj128` — add context->candidate residual against arithmetic candidate mean.
3. `idf-competitive-bidir-logmeanexp-proj128` — add context->candidate residual against log-mean-exp candidate baseline.

No extra learned coefficient or temperature.

## Fresh authority

TRAIN 512 / DEV 176 / post-selection CONFIRM 192.

Seeds:
- train 151117;
- dev 152219;
- confirm 153321;
- init/training 607.

Vocabulary, rendered templates, IDs and seeds must be disjoint from W5a-W5h.

CONFIRM must remain sealed until DEV selection/checkpoint hashes freeze.

## Frozen gates

Absolute competence:
- overall >= 0.60;
- K128 >= 0.40;
- K255 >= 0.30;
- K255 top-5 >= 0.70;
- probability error <= 1e-6.

Mechanism:
- overall gain vs fresh forward control >= +0.05;
- K255 accuracy gain vs fresh forward control >= +0.03.

Valid verdicts:
- `CROSS_CANDIDATE_BINDING_RESCUE`;
- `CROSS_CANDIDATE_CONTROL_ALREADY_RESCUES`;
- `CROSS_CANDIDATE_BINDING_PARTIAL`;
- `CROSS_CANDIDATE_BINDING_FAIL`.

Campaign cells populated = 0.

## Current implementation boundary

Before adding a push-only empirical workflow:
1. compile the W5i module/scripts;
2. pass W5i unit contracts;
3. prove vocabulary/template freshness against W5a-W5h;
4. audit CONFIRM sealing;
5. then create one clean push-authority lineage.

No empirical authority is accepted from a branch state before these checks pass.
