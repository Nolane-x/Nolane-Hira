# R24 handoff — sequential relinearized exact-trust walk

Status: **completed empirical negative result; merged to `main` in `4ef6f355c5d419ec4d1b0c1c4047d43092e24875`.**

## Why R24 existed

R23 showed that even a native functional-null step with tiny teacher KL could flip teacher decisions, and that an absolute matched-accuracy floor was slice-sensitive.

R24 therefore tested:
- sequential microsteps rather than one global step;
- relinearized structural gradients;
- cumulative exact trust versus frozen R15;
- relative final competence preservation rather than an unrelated absolute accuracy floor.

Issue #44 is the frozen protocol authority.

## Technical provenance

Initial empirical run:
- run `35730839711`;
- head `302f4d80015b2d155fd9517c57c5d8bee45222b9`;
- failed technically before scientific completion because `collect_structural_gradient()` was called with unsupported keyword `layout`.

Fix:
- commit `5baaad1cb276722d60181b759def999bce13af1d`;
- removes only the unsupported keyword;
- no scientific protocol/data/threshold/gate changed.

Authoritative rerun:
- workflow `35734857791`;
- head `5baaad1cb276722d60181b759def999bce13af1d`.

Artifacts:
- evidence ID `10697218852`;
- evidence digest `sha256:38e13b75f47325aaba25d56a1cef9d92e62e029e14ca6a408360956ff70534df`;
- full bundle ID `10697273832`;
- full bundle digest `sha256:9a06c69e47cb0fce7aa5250020173b1a897dee8fa489d7af5daa06f7bcc01bec`.

Execution correctness:
- Python 3.10 CI PASS;
- Python 3.12 CI PASS;
- unit tests PASS;
- preflight PASS;
- exact R15/R16 SHA checks PASS;
- workflow execution PASS;
- artifact upload PASS.

Scientific primary result: **FAIL**.

## Frozen R24 authorities

Prior scored-train exclusion:
- exact R19–R23 union = **50,000** distinct MultiNLI train indices.

R24 proposal:
- 4,000 structural examples;
- split into 16 balanced shards of 250.

Trust:
- 6,000 train-only examples, seed 37;
- frozen exact R15 teacher probabilities and argmax labels.

Final competence:
- C1 fresh train-heldout = 3,000 examples;
- C2 untouched validation_matched shuffled tail = 815 examples.

Fresh structural validation:
- 500 mismatched examples;
- non-entailment per-label ranks 1500:1750.

## Functional basis

The inherited fixed rank-32 functional teacher-KL basis remained numerically valid:
- probe mean teacher KL: **0.0009912024**;
- numerical rank: **63**;
- orthonormal error: **1.041e-10**.

## Sequential walk result

Accepted rounds: **2 / 16**.

Round 1:
- eta = **1/256 = 0.00390625**;
- update L2 = **0.00108584**;
- teacher KL = **3.87e-8**;
- teacher agreement = **6000/6000**.

Round 2:
- eta = **1/1024 = 0.0009765625**;
- update L2 = **0.00030341**;
- teacher KL = **6.42e-8**;
- teacher agreement = **6000/6000**.

Cumulative update L2:
- **0.00138591**.

Rounds 3–16:
- every direction remained finite and first-order useful;
- null leakage remained near machine zero;
- no eta in the frozen ladder was exact-trust-safe.

At the smallest eta, **1/4096**, every round 3–16 still produced:
- agreement **5999/6000**;
- exactly one changed teacher decision;
- KL only around **6.95e-8 to 7.27e-8**.

The repeatedly changed example has frozen teacher margin:
- **0.0004096627**.

This is a persistent discrete decision-boundary wall, not a large-KL failure.

## Fresh final competence

### C1 — 3,000 fresh heldout examples

Exact R15:
- **1694 / 3000 correct**;
- accuracy **0.5646667**.

R24 experimental cumulative state:
- **1692 / 3000 correct**;
- accuracy **0.5640000**.

Zero-drop: **FAIL by 2 correct predictions**.

### C2 — untouched official matched tail

n = **815**.

Exact R15:
- **430 / 815 correct**;
- accuracy **0.5276074**.

R24 experimental state:
- **430 / 815 correct**;
- accuracy **0.5276074**.

Zero-drop: **PASS exactly**.

Final eligibility: **FAIL** because C1 zero-drop fails.

## Fresh structural validation

Exact R15:
- non-entailment accuracy **0.656**.

R24 experimental state:
- non-entailment accuracy **0.658**.

Gain:
- **+0.002 absolute**.

Frozen primary requirement:
- **>= +0.020**.

Structural gain gate: **FAIL**.

## Artifact output

Because final eligibility fails, the persisted R24 head is exact R15 fallback.

Persisted SHA-256:
`007e24fff0e0e48a096de59276f7ab8d0e25bdb826a3f81a7838c5bd6151723f`.

No experimental R24 state becomes the default model.

## R24 conclusion

R24 demonstrates that:
1. sequential relinearization can find a very small exact-trust-safe path;
2. that path saturates after two microsteps against a persistent discrete boundary;
3. exact preservation on a 6,000-example train trust authority does not guarantee zero-drop on fresh competence data;
4. the safe structural improvement is only +0.002.

Therefore extending the same NLI repair mechanism with smaller eta values or more rounds is low-value and risks turning the project into benchmark-specific tuning.

## Project direction after R24

R24 closes this NLI repair campaign **for now**.

Do not open an R25 whose main purpose is to:
- extend the eta ladder;
- relax exact trust;
- replace the final authority;
- continue tuning MultiNLI structural accuracy.

Return to the original HIRA program:
- typed decisions;
- dynamic schemas;
- full-K probability fidelity;
- Banking77 / K=128 / K=255;
- state-once Q=1/5/10/50 systems;
- multilingual held-out evaluation;
- calibration and OOD authority;
- option-order robustness;
- frozen Laya/Jev campaign.

No HANS, Breaking NLI, XNLI, MASSIVE, Banking77, Laya, or Jev/JEV final cell selected R24.
