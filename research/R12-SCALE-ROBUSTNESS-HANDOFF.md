# R12 handoff — scale frozen-A13 relation learning + permutation robustness

Status: **protocol frozen; empirical run pending**.

## Purpose

R11 proved that frozen A13 + 422,159 trainable HIRA parameters can learn semantic NLI signal:
34.9% -> 54.4% on a 5k MultiNLI pilot.

R12 tests whether that gain scales with more non-evaluation data **without learning option-position shortcuts**.

## Frozen inputs

A13:
- model: `microsoft/xtremedistil-l6-h256-uncased`
- revision: `4226d9e4d2c08703e5cb0491b479bfc6a1607181`
- safetensors SHA-256: `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`
- frozen.

MultiNLI:
- dataset: `nyu-mll/multi_nli`
- revision: `da70db2af9d09693783c3320c4249840212ee221`

Frozen direct-comparison corpora remain unused:
- XNLI: NO
- MASSIVE: NO
- Banking77: NO

## Scaling ladder

A13 encodes at most 15,000 train examples + 1,500 validation examples once.

Independent HIRA heads are trained from the same seed at:
- 5,000 examples
- 10,000 examples
- 15,000 examples

Each rung:
- 8 epochs
- batch size 96
- LR 0.002
- CE + 0.1 multiclass Brier
- K=3 all-option relation refinement
- same validation set

The winning rung is chosen by validation accuracy.

## Robustness requirement

Every trained rung is evaluated under all six permutations of the three NLI logical options.

For a permutation, option embeddings are permuted and labels are remapped. Probabilities are then restored to canonical semantic order.

The architecture should be equivariant:
- accuracy delta ~ 0
- prediction flip rate ~ 0
- restored probability error ~ numerical noise

Preregistered robustness gate:
- max |accuracy delta| <= 1e-7
- max flip rate <= 1e-7
- max probability equivariance error <= 1e-5

## Scale gate

Preregistered R12 scale gate:
- best validation accuracy >= 0.56
- best accuracy >= 5k rung accuracy

A failure is preserved. Do not relax either threshold after seeing the run.

## Interpretation

Passing R12 means:
- semantic relation quality increased beyond the R11 pilot floor, and
- the gain did not come from option-position shortcuts.

It still does **not** establish:
- XNLI parity;
- multilingual quality;
- Laya/Jev parity;
- OOD capability;
- production readiness.

## Next decision

If scale passes:
1. freeze the R12 head-selection protocol;
2. run a held-out relation stress suite (negation/contradiction/paraphrase) from non-XNLI sources;
3. only then open frozen XNLI English as evaluation;
4. unfreeze A13 top layers only if the frozen-head capacity curve plateaus.

If scale fails:
- inspect rung history/representation bottleneck before increasing model capacity.
