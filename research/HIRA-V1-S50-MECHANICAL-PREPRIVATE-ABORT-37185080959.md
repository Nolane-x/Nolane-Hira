# HIRA V1 S50 mechanical pre-private abort receipt

Status: **NATIVE COMPLETE / PRIVATE DEV UNEXPOSED / ONE HASH-LOCKED MECHANICAL REPLAY AUTHORIZED**

Failed run: `37185080959`  
Job: `111385220575`  
Scientific head: `68df39904b620f1bc9aa5ce74072457a165eae20`

## Completed before failure

- install PASS
- S50 contracts PASS
- canonical S50-A0 PASS
- M4 integrity PASS
- shared native TRAIN-only authority completed **24/24 epochs**
- native fixed authority reached epoch **24**
- final native runtime hash:
  `29c29fc34624f9096c16d54115e2db1047ccd477939780738c66bcd1f9f23913`
- TRAIN cache materialized
- DEV cache materialized according to the preregistered phase-2 protocol.

All 24 native runtime hashes are frozen in:
`research/HIRA-V1-S50-RECOVERED-NATIVE.json`.

## What was NOT exposed

Reference private training emitted:
`HIRA_V1_S50_PRIVATE_REFERENCE_TRAIN_BEGIN`

but failed on the **first backward of epoch 1**.

Therefore:
- reference private epoch records: **0**
- treatment private TRAIN_BEGIN: **not reached**
- private DEV scoring: **0**
- private checkpoint selection: **0**
- treatment/reference DEV comparison: **0**
- scientific A/B/C/D/E verdict: **not exposed**.

DEV had been encoded into the immutable cache as preregistered, but no private branch scored it.

## Mechanical failure

Failure:
`RuntimeError: Inference tensors cannot be saved for backward`

Cause:
native evidence generation/cache materialization runs under `torch.inference_mode()`. The cache helper used a normal `detach().clone()` while inference mode remained active, so the supposedly frozen cache tensors retained PyTorch's inference-tensor flag. They had `requires_grad=False`, but autograd could not save them for backward when training the private correction branch.

## Mechanical repair

Only cache ownership transfer changes:

- enter `torch.inference_mode(False)`
- clone/detach/contiguous
- keep `requires_grad=False`.

No changes to:
- S50 rows
- seed 71001
- native equations
- native optimizer
- native 24-epoch authority
- cache fields
- reference private signature
- treatment query-free identity
- correction initialization
- loss coefficients
- private optimizer
- selector
- capacity
- DEV gates.

A regression test reproduces cache creation inside `torch.inference_mode()` and requires that all resulting cache tensors are normal non-inference tensors and support private backward.

## Replay governance

One mechanical replay is authorized only if:

1. shared native TRAIN is reproduced from the same seed/data/mechanics;
2. **all 24 replay runtime hashes exactly equal** the frozen run `37185080959` hashes before private results are accepted;
3. no semantic variable changes;
4. no private DEV metric from the failed run is used, because none exists;
5. replayed cache is generated only after the exact native replay is verified;
6. after private TRAIN begins in the replay, no further mechanical/scientific retry is authorized for weakness.

If any native hash differs, S50 is invalidated rather than continued.
