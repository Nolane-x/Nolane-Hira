# HIRA V1 S50 mechanical replay staging receipt

Status: **STAGED / REPLAY NOT AUTHORIZED**

Issue: #283  
PR: #284

## Failed authority

Fresh S50 run:
`37185080959`

Scientific head:
`68df39904b620f1bc9aa5ce74072457a165eae20`

Completed:
- exact stack install
- S50 contracts
- canonical S50-A0
- M4 integrity
- one shared native TRAIN-only trajectory
- native epochs **24/24**
- fixed native authority epoch **24**
- TRAIN cache materialized
- DEV cache materialized per the preregistered phase-2 protocol.

Not exposed:
- reference private completed epochs: **0**
- treatment private TRAIN_BEGIN: **not reached**
- private DEV scoring: **0**
- private model selection: **0**
- A/B/C/D/E verdict: **none**.

## Frozen native authority

Recovered file:
`research/HIRA-V1-S50-RECOVERED-NATIVE.json`

Frozen final native hash:
`29c29fc34624f9096c16d54115e2db1047ccd477939780738c66bcd1f9f23913`

All 24 epoch hashes are frozen.

## Mechanical bug

Production cache materialization executes native evidence generation under
`torch.inference_mode()`.

The original cache clone:
`tensor.detach().clone().contiguous()`

was itself executed while inference mode was active. Therefore the cache carried PyTorch's inference-tensor flag even though `requires_grad=False`.

Reference private training then failed on the first backward of epoch 1:

`RuntimeError: Inference tensors cannot be saved for backward`

## Mechanical repair only

`_clone_frozen` now temporarily exits inference mode solely for the ownership transfer:

`with torch.inference_mode(False): clone/detach/contiguous`

The resulting cache remains:
- detached
- `requires_grad=False`
- immutable
- digestable
- normal autograd-compatible input for private parameter training.

Regression:
`test_s50_cache_created_inside_inference_mode_is_normal_autograd_compatible_tensor`

This reproduces the production cache path and requires a successful private backward.

## Hash-locked replay

Workflow:
`.github/workflows/hira-v1-s50-mechanical-replay.yml`

Marker:
`research/HIRA-V1-S50-ENABLE-MECHANICAL-REPLAY`

The replay:
1. reuses exact seed **71001** and exact S50 fresh rows;
2. reruns the same shared-native TRAIN-only phase;
3. requires all **24/24** runtime hashes to equal run `37185080959`;
4. requires final epoch-24 native hash exact;
5. only then rematerializes cache with corrected tensor ownership;
6. trains the already-frozen reference and treatment private branches;
7. makes no scientific-variable change.

## Authorization rule

The replay marker MUST remain absent until the exact final replay-staging head passes generic CI on Python 3.10 and 3.12.

After replay private TRAIN begins:
- no further mechanical repair is authorized;
- no scientific retry is authorized;
- no cache/operator/identity tuning;
- no second S50 private DEV.

If any replay native hash differs, S50 stops as invalid.
