# R9 handoff — trainable HIRA

Status: **implementation active; semantic quality not yet claimed**.

## What R9 changes

R7 contained the 422,159-parameter HIRA modules but the relation path was intentionally inactive.
R9 activates those existing parameters without increasing the HIRA parameter ledger:

1. full-K coarse logits;
2. candidate selection with all-K fail-safe by default;
3. selected logical options query state segments through micro cross-attention;
4. relation deltas are scattered only into selected option logits;
5. unselected logits are preserved exactly;
6. full-K probability normalization happens after refinement;
7. relation gradients reach cross-attention, cross-score, options and state.

## End-to-end path

```text
state text
  -> semantic encoder ONCE
  -> StateMemory segments
  -> compiled/cached schema
       question + descriptions + aliases + exemplars
  -> logical option embeddings
  -> all-K HIRA coarse scores
  -> selected relation refinement
  -> full-K probabilities
  -> Choice / Score / Noul
```

## Semantic encoders

- `HFAutoSemanticEncoder`: lazy optional Transformers adapter for pinned A13/A22 weights.
- `TrainableSemanticEncoder`: self-contained hashed-token Transformer used only to prove the training/runtime software path without external weights.

Do not report the self-contained encoder as A13 quality.

## Cache safety

Schema tensors are cached only when the encoder is in eval mode. Training bypasses the cache so no stale autograd graph or stale semantic embedding can survive an optimizer update.

Option IDs are included in cache identity because they are routing keys, but they are never passed into semantic encoding. Opaque-ID permutations must leave embeddings/probabilities unchanged when semantic text/order are unchanged.

## Training

`nmd.training` now supports an end-to-end DecisionExample and optimizer step.
`nmd.losses` provides hard CE, KL distribution distillation, hard/soft Brier, ordinal expected-score MAE and pairwise margin loss.

Synthetic/local tests only prove gradient flow and contracts, not model intelligence.

## Checkpoints

`nmd.checkpoint` writes:
- model.pt
- config.json
- manifest.json

The manifest pins code revision, encoder revision, config hash and SHA-256 of the weights. Load refuses corrupted weights.

## Safety/reliability rules retained

- adaptive budget is OFF by default; before calibration HIRA uses all-K;
- candidate misses cannot be hidden;
- OOD remains separate from classifier confidence;
- MASSIVE/XNLI/Banking77 training allowances follow the R8 ledger;
- no Laya/Jev win can be recorded before immutable benchmark receipts exist.

## Tests required before merge

- exact HIRA params remain 422,159;
- relation delta changes selected candidates only;
- unselected logits are bit-identical to coarse logits;
- relation gradients reach cross-attention/cross-score;
- all-K default has zero tail mass;
- registered schema cache reuses schema but never re-encodes an existing state;
- training gradients reach both semantic encoder and HIRA;
- opaque ID renames preserve semantic embeddings/probabilities;
- checkpoint roundtrip/hash corruption guard;
- existing R7/R8 benchmark and reproducibility tests remain green.

## Next empirical step

1. pin an immutable A13 revision + weight hash;
2. install the optional HF dependency;
3. run frozen A13 inference through this exact runtime;
4. create the first immutable A13 run receipt;
5. only then begin relation/NLI training and the R8 scorecard.

No semantic benchmark result is claimed by R9 itself.
