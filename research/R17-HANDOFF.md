# R17 handoff — endpoint weight interpolation

Status: **protocol frozen; empirical workflow active**.

## Starting point

R15 remains the competence anchor:
- selected HIRA head SHA-256: `007e24fff0e0e48a096de59276f7ab8d0e25bdb826a3f81a7838c5bd6151723f`;
- matched MultiNLI accuracy: 0.5613;
- HIRA parameter count: exactly 422,159.

R16 is preserved as a negative result:
- strict structural validation collapsed to only two balanced examples;
- all trained R16 candidates failed the matched-retention eligibility floor;
- no HANS/Breaking diagnostic was authorized after that primary failure;
- failure-analysis head SHA-256: `dbe0ddd8bf3811c98f5062bbf482f5d5b4f1f5991fa6f734f7ca88c24e88cc75`.

A13 stays frozen:
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- safetensors SHA-256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`.

## R17 hypothesis

R16 may contain a useful structural anti-entailment update direction even though its full move leaves the R15 competence basin.

R17 therefore does **no additional training**. It evaluates the convex endpoint path:

`theta(alpha) = (1-alpha) * theta_R15 + alpha * theta_R16`

The deployed result is still one HIRA head with exactly 422,159 parameters. There is no runtime ensemble, expert routing, or hidden parameter increase.

## Frozen alpha grid

Exactly:

- 0.00
- 0.025
- 0.05
- 0.10
- 0.15
- 0.20
- 0.30

No alpha may be added or removed after looking at results.

## Model-selection authority

Only pinned MultiNLI is permitted:

`nyu-mll/multi_nli@da70db2af9d09693783c3320c4249840212ee221`

Forbidden for R17 selection:
- HANS
- Breaking NLI
- XNLI
- MASSIVE
- Banking77

## Frozen near-structural validation predicate

Tokenization:
- lowercase alphanumeric tokens only.

A mismatched-validation example is eligible when:
- gold label is neutral or contradiction;
- hypothesis length is at least 3 tokens;
- multiset hypothesis-token recall in premise >= 0.80 **OR**
- ordered LCS recall of hypothesis tokens in premise >= 0.80.

Ranking within each label:
1. descending max(multiset recall, ordered LCS recall);
2. descending min(multiset recall, ordered LCS recall);
3. ascending source index.

Selection:
- balance neutral and contradiction;
- select up to 500 per label.

Support validity gate:
- at least 100 available neutral examples;
- at least 100 available contradiction examples.

If this gate fails, R17 model selection is invalid.

## Matched retention slice

Exact R12/R15 convention:
- `validation_matched`;
- filter valid labels;
- shuffle seed 14;
- first 1,500 examples.

Eligibility:
- matched accuracy >= 0.5606666612625122.

## Selection rule

Among eligible alpha values:

1. maximize balanced near-structural non-entailment accuracy;
2. tie-break by matched accuracy;
3. then prefer smaller alpha.

## Preregistered primary gates

All must pass:

- support validity pass;
- matched accuracy >= 0.5606666612625122;
- near-structural non-entailment accuracy >= 0.60;
- near-structural improvement >= +0.05 absolute versus alpha=0/R15 on the same frozen slice;
- selected alpha is eligible;
- HIRA parameter count remains exactly 422,159.

No gate or threshold may be changed after the empirical run.

## Evidence implementation

R17 adds:
- `src/nmd/interpolation.py` — near-structural mining, ordered-LCS recall, convex state interpolation;
- `tests/test_interpolation.py` — deterministic support/ranking and endpoint invariants;
- `scripts/r17_weight_interpolation.py` — frozen empirical tournament and receipt;
- `.github/workflows/r17-weight-interpolation.yml` — exact artifact retrieval, hash verification, empirical run, and packaging.

The workflow retrieves:
- exact R15 artifact from run `35688830869`;
- exact R16 failure-analysis artifact from run `35690994565`.

Both endpoint hashes are checked before evaluation.

## Post-selection diagnostics

Only if every primary R17 gate passes may adapted HANS and adapted Breaking NLI be run.

Those suites can never return to held-out status because their prior failures are already known.

XNLI remains unopened for model selection.

## Interpretation discipline

A successful R17 result would establish only that a small interpolation along the observed R16 update direction improves the preregistered MultiNLI near-structural slice while retaining matched competence.

It would **not** establish:
- Laya parity;
- Jev/JEV parity;
- broad OOD robustness;
- multilingual generalization;
- general reasoning competence.

A failed R17 result must be preserved as a negative receipt and must not trigger post-hoc alpha/grid/threshold edits under the R17 label.
