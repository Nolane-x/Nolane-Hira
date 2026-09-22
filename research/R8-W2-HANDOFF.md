# R8-W2 handoff — typed-decisions train adapter + state-once execution

Status: **COMPLETE; merged to `main` in `425b89ede2de251efa990f5586a92924edb71535`; final typed-decisions test split was not exposed.**

## Frozen source

- dataset: `LocalLLaMA/typed-decisions`;
- revision: `c76749ec58bd8c3d2ea706b31c333a9059c38f90`;
- W2 authority: TRAIN split only;
- 4 workflows × 300 train cases = 1,200 cases;
- exactly 5 typed decisions per case = 6,000 train decisions.

## Source contract

Each row contains:
- `state`: JSON model input;
- `questions`: JSON map of typed questions;
- `gold`: full soft distributions;
- `factors`: latent generation factors, forbidden as model input;
- `label_agreement`: teacher diagnostics, forbidden as model input.

Question entries contain:
- `type`: choice / score / noul;
- `instructions`;
- `criteria`: label/rubric -> semantic description.

The semantic criteria text is part of the task. Bare labels are not a valid substitute.

## W2 implementation

Library primitives:
- `src/nmd/typed_decisions.py`;
- `src/nmd/typed_eval.py`.

Runtime/training changes:
- `NolaneHira.state_encode_calls` instruments state-once execution;
- case inference compiles one StateMemory then answers all five schemas;
- case training compiles one differentiable StateMemory, aggregates five typed losses, and performs one backward/optimizer step;
- training-mode schema caching remains disabled;
- ordinal loss accepts explicit numeric score support instead of assuming option index.

Adapter safety:
- hard-coded immutable dataset revision;
- loader requests only `split="train"`;
- parser explicitly rejects non-train rows;
- probability labels must exactly match criteria labels;
- probability mass must be finite, non-negative and approximately one;
- noul must be exactly false/true with values 0/1;
- score criterion IDs must be finite numeric rubric values;
- question/gold IDs must match;
- exactly five questions required;
- factors/label_agreement are never read.

Train-only evaluator reports:
- total and primitive accuracy;
- soft accuracy;
- hard and soft Brier;
- NLL;
- KL(gold || prediction);
- 15-bin ECE compatible with the frozen benchmark semantics;
- score MAE from explicit support;
- case/decision counts;
- state encode calls per case;
- decisions per state encode;
- probability mass integrity.

## Selection discipline

W2 is infrastructure correctness only.

Do not:
- evaluate typed-decisions test;
- populate R8 final scorecard;
- tune against Laya/Jev final values;
- infer a benchmark win from train metrics.

W3 must be preregistered separately before specialized training/final evaluation.


## Completion receipt

PR #49 merged as `425b89ede2de251efa990f5586a92924edb71535`.

Verification:
- W2-specific compile PASS;
- W2-specific contract suite: **18/18 PASS**;
- repository CI Python 3.10 PASS;
- repository CI Python 3.12 PASS;
- repository preflight PASS on both matrix jobs.

W2 produced no final benchmark metric and populated no scorecard cell.
