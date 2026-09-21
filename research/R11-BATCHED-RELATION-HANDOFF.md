# R11 handoff — frozen-A13 cached relation learning

Status: **pilot implementation ready; empirical run pending**.

## Why R11 exists

R9 made HIRA trainable. R10 proved the real A13 checkpoint can be materialized and connected. R11 removes the largest training-efficiency problem: repeated A13 forward passes every epoch.

When A13 is frozen:

```text
MultiNLI examples
   -> A13 encode once
   -> cached state segments + hypothesis embeddings + option prototypes
   -> train 422,159-param HIRA for many epochs
```

The semantic cache is keyed in the receipt by:
- A13 revision and verified weight SHA;
- dataset revision;
- deterministic seed;
- segment geometry.

## Pilot

A13:
- `microsoft/xtremedistil-l6-h256-uncased`
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`
- safetensors SHA-256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`
- frozen.

MultiNLI:
- `nyu-mll/multi_nli`
- revision `da70db2af9d09693783c3320c4249840212ee221`
- deterministic shuffled train subset;
- `validation_matched` pilot validation.

Direct Laya benchmark corpora:
- XNLI: **not used**
- MASSIVE: **not used**
- Banking77: **not used**

## Pilot default

- train: 5,000 examples
- validation: 1,000
- A13 max length: 128
- 8 state segments
- 8 HIRA epochs
- HIRA train batch: 64
- AdamW LR: 2e-3
- objective: CE + 0.1 multiclass Brier
- all 3 NLI options retained; no candidate pruning.

## Preregistered learning gate

Random is approximately 33.3%.

Pilot semantic-learning gate:
`validation accuracy >= 0.50`.

The workflow must preserve a run even if the gate fails. Failure means the frozen-A13/HIRA geometry needs diagnosis; it is not permission to hide the result or jump directly to A22.

## Evidence boundary

Passing 50% means relation learning is real enough to proceed. It does **not** mean:
- Laya parity;
- Jev parity;
- XNLI parity;
- multilingual parity;
- production readiness.

After a pass, scale the cached relation stage and run adversarial/held-out relation suites before touching the R8 scorecard.
