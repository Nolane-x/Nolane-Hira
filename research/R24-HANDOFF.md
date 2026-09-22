# R24 handoff — sequential relinearized exact-trust walk

Status: **protocol frozen before empirical evaluation in issue #44; implementation in progress on `feat/r24-sequential-exact-trust-walk`.**

## Evidence entering R24

R23 is a completed scientific negative result.

Two independent findings matter:

1. No one-shot native-null eta in the frozen R23 grid preserves all 6,000 teacher decisions.
   - eta 1/64 has mean KL only 6.03e-7;
   - yet teacher argmax agreement is only 5993/6000.

2. Exact R15 itself scores only 0.516 on the frozen R23 matched slice and therefore fails the unrelated 0.53 absolute baseline floor.

R24 therefore tests a different mechanism and a cleaner retention authority.

## Frozen mechanism

Issue #44 is authoritative.

R24:
- reconstructs exactly 50,000 prior scored-train indices from R19–R23;
- selects 16 deterministic balanced structural shards of 250 examples each;
- uses one new 6,000-example train-only exact trust authority;
- builds one fixed rank-32 functional-damage basis;
- starts from exact R15;
- recomputes the structural gradient after every round;
- projects each round's structural descent into the fixed functional nullspace;
- tries exactly the microstep ladder:
  1/64, 1/128, 1/256, 1/512, 1/1024, 1/2048, 1/4096;
- tests every cumulative candidate directly against frozen exact R15 teacher outputs;
- accepts the largest safe eta per round;
- continues all 16 rounds even if a particular round has no safe candidate.

A cumulative state is train-safe only if:
- mean teacher KL <= 1e-4;
- exact teacher argmax agreement is 6000/6000;
- outputs are finite;
- non-relation tensors remain bit-identical.

Validation cannot select any microstep.

## Final competence

R24 removes the slice-sensitive absolute 0.53 floor.

It does **not** weaken retention.

The final candidate must preserve exact R15 correct count on both:
- C1: fresh 3,000-example MultiNLI train-heldout competence set, seed 43;
- C2: untouched official validation_matched tail from shuffled position 9000 to end.

Zero-drop must pass independently on both authorities.

## Structural final

Fresh validation_mismatched structural ranks 1500:1750 per non-entailment label, exactly 500 examples.

Primary structural gain remains >= +0.020 absolute versus exact R15 on the same slice.

No HANS, Breaking NLI, XNLI, MASSIVE, Banking77, Laya, or Jev/JEV final cell may select R24.
