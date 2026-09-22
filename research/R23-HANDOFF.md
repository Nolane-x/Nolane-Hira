# R23 handoff — finite teacher-KL trust region

Status: **completed empirical negative result; merged to `main` in `fb5017bcd60240722dbf2b81ab7eb881cbc7f634`.**

## Authoritative execution

Workflow run:
- `35725111435`;
- empirical head: `669d167f7bf6d8af8300ac75b5fe205b1ee63206`.

Artifacts:
- evidence ID `10693494535`;
- evidence digest `sha256:0b9cd9ebe61757cdb66ea169ea84868359f62ea6bf160c87c007fe422a403141`;
- full bundle ID `10693499464`;
- full bundle digest `sha256:a765467dc694f427982a07791203c5316b3307916f4867cfef903d8a5b6a4d33`.

Execution correctness:
- CI Python 3.10 PASS;
- CI Python 3.12 PASS;
- unit tests PASS;
- preflight PASS;
- exact R15/R16 SHA verification PASS;
- empirical workflow execution PASS.

Scientific result: **PRIMARY FAIL**.

## R23 mechanism

R23 used:
- exact R15 as competence anchor;
- exact R16 only to place the inherited fixed beta=0.25 functional probe;
- exact 40,000-example R19/R20/R21/R22 scored-train exclusion union;
- new R23 structural proposal train of 4,000;
- new R23 functional trust train of 6,000, seed 31;
- exact rank-32 functional teacher-KL damage subspace;
- native, unnormalized structural null tangent;
- fixed eta grid {1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4};
- train-only finite trust selection;
- final validation only after train selection was frozen.

## Functional geometry

The functional-damage machinery remains numerically valid:
- probe mean teacher KL: **0.0009998879433**;
- functional subspace rank: **63**;
- basis orthonormal error: **5.2747e-11**;
- structural gradient L2: **4.9198733**;
- native null tangent L2: **0.2637185**;
- retained structural-gradient energy: **0.0028732494 = 0.2873%**;
- null leakage: **3.3993e-23**;
- raw null first-order structural predicted benefit: **0.0695474**.

Direction validity: **PASS**.

R23 therefore reproduces the R22 geometry: almost all structural-gradient energy lies inside the measured functional-damage span.

## Finite trust tournament

No candidate is trust-safe.

Smallest preregistered step:
- eta: **1/64 = 0.015625**;
- applied update L2: **0.0041206014**;
- mean teacher KL: **6.0325e-7**;
- teacher argmax agreement: **5993 / 6000**;
- changed teacher decisions: **7**;
- trust-safe: **FAIL**.

The KL number is extremely small, but exact decisions already change.

This is important evidence:
- mean KL alone is not a sufficient safety authority near teacher decision boundaries;
- the discrete argmax boundary is much tighter than the average distribution-drift budget;
- R23's exact 6000/6000 gate correctly prevents those changes from being hidden by a tiny mean KL.

Every larger eta changes more teacher decisions and eventually also violates the KL ceiling.

Therefore:
- train-selected candidate: **none**;
- persisted head mode: **exact R15 fallback**;
- persisted head SHA-256:
  `007e24fff0e0e48a096de59276f7ab8d0e25bdb826a3f81a7838c5bd6151723f`.

## Independent final-baseline invalidity

The frozen final matched slice also exposes a separate protocol failure:

Exact R15:
- matched accuracy: **0.5159999728**;
- exact correct count: **774 / 1500**;
- label counts: entailment 522, neutral 480, contradiction 498.

Frozen R23 absolute baseline floor:
- **0.53**.

So the exact R15 anchor itself fails the floor on this fresh slice.

Baseline validity: **FAIL**.

This failure is preserved without choosing a friendlier slice post hoc.

The R23 result therefore fails for two independent reasons:
1. no train candidate satisfies exact finite trust;
2. the preregistered absolute baseline-validity rule rejects exact R15 on the final slice.

## Why this changes the successor protocol

The scientific question is retention **relative to exact R15**.

A fixed absolute floor such as 0.53 is not necessary to test that question and can invalidate the experiment for reasons unrelated to whether a candidate preserves R15.

R24 should therefore:
- keep exact zero-drop retention relative to exact R15;
- remove the unrelated absolute R15-accuracy floor;
- preserve label-support/data-integrity validity checks;
- never lower the relative zero-drop gate.

This is not permission to accept weaker candidates. The final candidate must still equal or exceed exact R15 correct count on the same authority.

## Successor mechanism constraint

R24 should not:
- add a smaller one-shot eta after seeing R23;
- relax exact teacher argmax agreement;
- increase the KL ceiling;
- use final validation to choose a step.

The principled successor is a **sequential relinearized trust-region walk**:

1. start at exact R15;
2. compute a local structural direction;
3. use a preregistered microstep backtracking ladder;
4. accept only a cumulative state that remains finite-trust safe against the frozen R15 teacher;
5. recompute the structural direction after every accepted step;
6. repeat for a fixed number of rounds or until no safe step exists;
7. freeze the final train-selected state;
8. evaluate exactly once on a fresh competence authority and fresh structural validation;
9. require zero-drop correct count relative to exact R15.

This tests whether the model can follow a curved safe manifold rather than forcing one global straight-line update.

## Scope discipline

No HANS, Breaking NLI, XNLI, MASSIVE, Banking77, Laya, or Jev/JEV final benchmark cell was used to select R23.

R23 does not establish Laya/Jev parity or broad general reasoning.
