# R8-W5f handoff — state-once late-interaction semantic binding

Status: **COMPLETE / LATE_INTERACTION_PARTIAL; merged to main in `6504e4ebd93efe78c93caaed0f9d18843131402b`.**

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


## Authoritative result

Run `35858733250`, exact head `5c5fbf829f13b6094a94f6a54a9bebfdaace0046`.

Selected on DEV:
- `proj128-maxsim`;
- epoch 6;
- matcher SHA-256 `a2c862feff387e4fd5af04543ba152016a48aee14ab1977812eb2ec861666e5a`;
- selector recorded `confirm_exposed=false`.

Untouched CONFIRM:
- 192 cases;
- generated only after selection freeze;
- state encode/case = 1.0;
- accuracy **0.296875**;
- MRR **0.44158**;
- top-5 **0.609375**;
- K128 accuracy **0.25**;
- K255 accuracy **0.16667**;
- K255 top-5 **0.41667**;
- probability mass max error **2.38e-7**.

Same-set frozen pooled baseline:
- accuracy **0.005208**;
- MRR **0.04804**;
- top-5 **0.03125**.

Verdict:
`LATE_INTERACTION_PARTIAL`.

This is the strongest positive semantic-mechanism result in R8 so far, but it misses the frozen rescue thresholds. Do not rerun W5f CONFIRM or alter its fixed coverage weight post hoc.

Machine-readable authority:
`artifacts/r8-w5f-late-interaction/summary.json`.
