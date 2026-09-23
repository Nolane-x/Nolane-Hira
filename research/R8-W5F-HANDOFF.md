# R8-W5f handoff — state-once late-interaction semantic binding

Status: **implementation active under issue #69; no empirical result yet.**

## Why W5f exists

W5a–W5e failed semantic competence despite:
- correct high-K mechanics;
- option/state token plumbing;
- learned probes;
- A13 top-layer adaptation;
- larger A22 capacity.

W5b did not construct direct state-token × option-token interaction. Its option tokens were independently pooled before final scoring.

W5f tests that missing interaction directly.

## Frozen mechanism

A13 remains exact/frozen.

Per case:
- state encoded once;
- question encoded once;
- options encoded on the schema side;
- BERT boundary tokens excluded from coverage;
- option-token × context-token cosine matrix;
- each option token takes MaxSim over context;
- score = mean coverage + 0.5 * minimum coverage;
- all K options scored.

Candidates:
- raw-maxsim;
- proj64-maxsim;
- proj128-maxsim.

Only projection + scalar scale are trainable for projected candidates.

## Fresh authority

TRAIN = 512 cases.
DEV = 176 cases.
CONFIRM = 192 cases, generated only after selection freeze.

K reaches 255.

All vocabulary/templates/seeds are fresh and must remain disjoint from W5a–W5e.

No public benchmark row is allowed.

## Decision rule

Freeze one candidate on DEV.

Then compare selected candidate against same-set frozen pooled-cosine A13 on untouched CONFIRM.

Possible verdicts:
- LATE_INTERACTION_RESCUE;
- LATE_INTERACTION_PARTIAL;
- LATE_INTERACTION_FAIL.

Zero public campaign cells are populated by W5f.
