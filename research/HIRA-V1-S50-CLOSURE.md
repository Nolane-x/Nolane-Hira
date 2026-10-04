# HIRA V1 S50 closure — Shared-Native Forked Private Readouts

Status: **SCIENTIFICALLY CLOSED / INVALID REPRODUCIBILITY COURT / INCONCLUSIVE**

Issue: #283  
PR: #284

## Qualified A0

S50-A0:
- run `37183981097`
- artifact `11296132080`
- digest `sha256:364545c53977aba290faacb4a064b59b104a3d9f67e1ea0ad8a6b0cfe543ab1`
- outcome `HIRA_V1_S50_A0_SHARED_NATIVE_FORKED_PRIVATE_READOUTS_READY`

A0 established:
- immutable shared-evidence cache mechanics
- source-mutation isolation
- branch-order replay identity
- bit-identical correction initialization
- native optimizer parameter count **0** in private phase
- correction parameters **114,688** per branch
- identity trainable parameters **0**
- K=3/7/255
- full-K probability mass.

## Fresh S50 authority — mechanical pre-private abort

Fresh run:
`37185080959`

Scientific head:
`68df39904b620f1bc9aa5ce74072457a165eae20`

Completed:
- shared native TRAIN-only phase **24/24 epochs**
- fixed authority epoch **24**
- final native hash:
  `29c29fc34624f9096c16d54115e2db1047ccd477939780738c66bcd1f9f23913`
- TRAIN/DEV shared cache materialization.

Private scientific exposure:
- reference completed private epochs: **0**
- treatment TRAIN_BEGIN: **not reached**
- private DEV scoring: **0**
- private model selection: **0**.

Failure:
`RuntimeError: Inference tensors cannot be saved for backward`

Cause was mechanical cache tensor ownership under `torch.inference_mode()`.

The cache fix was isolated and regression-tested.

## Hash-locked mechanical replay

Replay staging head:
`6bee77820d18d280a03354f150ec1cf02bd1da42`

Exact-head CI:
`37188412137`
- Python 3.10 PASS
- Python 3.12 PASS.

Replay authorization:
`aa2672da4355123538f5bc23eeb3f1655f96724e`

Replay run:
`37188685173`

The replay completed the same shared-native TRAIN-only phase for **24/24 epochs**.

However, the replay runtime hash differed from the frozen native authority at **every epoch 1–24**.

Source epoch-1 hash:
`fae0fe2fdfd390b01e1928015260b2958784cbb21aa2fb8598514dfe200e59bf`

Replay epoch-1 hash:
`f8661c600fa54e828872cd05f180503cc68350923de3206b13c24b8e599d06d8`

Source epoch-24 hash:
`29c29fc34624f9096c16d54115e2db1047ccd477939780738c66bcd1f9f23913`

Replay epoch-24 hash:
`1d5f86447f756f54a863a2e57353699fd4de224dbd19e76fa4e6f09f71853bf8`

The hash guard stopped the replay before any private branch was allowed to run.

## Scientific consequence

S50's controlled-variable experiment was **never executed**.

No reference-vs-treatment private DEV metrics exist.

The query-free identity hypothesis therefore cannot be classified as A, B, C, D, or E.

S50 instead establishes a court-design fact:

**retraining the native authority in a later workflow run is not a valid mechanism for recovering bit-identical native evidence.**

The exact root cause of cross-run numerical divergence is not claimed here. Seed/data/code equality was insufficient to reproduce the same native bytes under the current execution stack.

## Frozen conclusion

**S50 scientific question remains unresolved.**

The next court must persist the native authority itself as an artifact and consume that exact artifact downstream, rather than retraining native and checking equality afterward.

## Stop-rule compliance

No:
- private identity tuning
- cache regeneration after a valid private DEV
- second private DEV
- branch-specific native retraining
- alternate selector
- scientific retry
- Laya/Jev external evaluation.

S50 is closed.
