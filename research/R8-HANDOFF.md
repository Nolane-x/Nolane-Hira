# R8 handoff — Laya/Jev benchmark campaign

Status: **OPEN; W1 + W2 + W3 + W4a complete. Campaign now 1 WIN / 0 TIE / 8 LOSS / 44 MISSING. Current task: W4b fresh synthetic K=128/K=255 high-cardinality stress.**

## Read first

1. `benchmarks/BENCHMARK_CAMPAIGN.md`
2. `benchmarks/laya_jev_targets.json`
3. `benchmarks/laya_fullboard.py`
4. `benchmarks/laya_protocol_manifest.json`
5. `benchmarks/jev_protocol_manifest.json`
6. `benchmarks/training_allowance.json`
7. `research/R8-CORRECTIONS.md`
8. R7 architecture/build-order docs.

## Frozen reference

Laya:
- commit `42626c348753fbb17572a813127df2278a1ec527`
- BENCHMARKS blob `2d448beccd1f8f0b87ad9fe0b090addf3cafdbae`
- T4 results `ddc400a430834abc30c4bd028b6238153a968eee`
- CPU51 results `1cb7d5ed5c498a6919797181d0f9ecb205e05b3d`

Jev zero-shot pilot:
- `AbdelStark/jev-benchmarks@0d610cc53e79bcbec691312b0c4adb4a0e371642`
- BTZSC revision `fef2a2ac62b69c58670047dddf045c53d7c3cb5e`

Jev Banking77 retrieved-24 challenge:
- `simonmesmith/jev-banking77-experiment@5cac4ff7783a4cfc0badba4124a309dd2a2b9862`
- Banking77 source commit `57ec275d8078af65b7731c2a98be812d844a6d6b`

## Scoreboards

- **53-cell headline board:** fixed public targets, including Laya and protocol-separated Jev targets.
- **706-cell Laya quality board:** generated from the pinned T4 + CPU51 artifacts. It compares HIRA to the best Laya checkpoint independently per metric cell.
- Application/systems extras remain separately labeled because their source artifacts/protocols differ.

A missing HIRA result is `MISSING`, never zero and never silently removed.

## Critical correction from R6

For the direct Laya generalization lane, do **not** train HIRA on MASSIVE, XNLI, SST-5, DAIR Emotion, prompt-injections or Banking77. Laya's frozen benchmark source marks these as held out (and MASSIVE/XNLI not used in base checkpoint training).

Earlier Banking77 sparse/prototype scores used labeled task data and therefore remain architecture probes only.

Typed-decisions has two lanes:
- base/zero-shot: no typed-decisions task training;
- fine-tuned: training allowed because Laya's 0.766 checkpoint is benchmark-fine-tuned.

Adapted/specialized lanes are allowed, but cannot replace direct generalization cells.

## Execution order

### E0 — reproduce manifests
Materialize exact raw bytes/revisions and regenerate Laya/Jev evaluation manifests. Store hashes and immutable run receipts. Do not train a neural candidate before the direct-test exclusions are machine-enforced.

### E1 — frozen A13 proof
Acquire/pin A13. Run held-out/direct suites without task adaptation. Establish which capabilities exist before HIRA-specific training.

### E2 — semantic relation training
Use non-evaluation corpora + teacher/synthetic relation data. Test pointwise/listwise/pairwise objectives. Public held-out benchmark labels remain unavailable for selection.

### E3 — typed-decisions specialized lane
Fine-tune on its training split. Optimize full distribution quality as well as argmax:
- accuracy target >0.766,
- soft accuracy target >0.580 if attempting to beat Jev too,
- Brier <0.061 to beat Laya,
- raw ECE <0.144 to beat both published Jev and Laya raw,
- score MAE <0.242.

### E4 — high cardinality
Direct held-out Banking77 first; K=128/K=255 synthetic/fresh schema stress next.
Candidate recall must remain explicit. If recall gate fails, force all-K.
Then run a separate equal-information retrieved-24 Banking77 challenge against Jev 92.40%.

### E5 — reliability
Option permutations, opaque IDs, paraphrase, negation, contradiction, calibration, OOD and selective risk. Calibration and OOD remain separate authorities.

### E6 — systems
Matched Tesla T4 protocol for direct comparison with Laya Q=1/5/10/50. Record p50/p95, throughput, peak RAM/VRAM, model bytes and semantic encoder calls. Never compare a local CPU time directly to Laya T4 as a winner claim.

### E7 — multilingual
Train using corpora other than the frozen MASSIVE/XNLI direct tests. Evaluate EN/VI first, then the Laya 14/15-language lanes and 51-language MASSIVE board. An adapted MASSIVE/XNLI lane, if run, must be labeled separately.

### E8 — compression
Only after A13 quality is understood: A6 rival and A7-FE tournament. A22 only if the capacity trigger fires.

## Attack priorities

1. **Banking77 / K=255:** HIRA was designed specifically to avoid Laya's shared option-token budget. This is the most architecture-revealing opportunity.
2. **typed probability quality:** optimize soft accuracy, Brier/NLL/ECE, not just top-1.
3. **option-order invariance:** HIRA logical options should make ordering close to irrelevant; enforce and test.
4. **state-once Q=10/Q=50:** strongest systems hypothesis.
5. **OOD/selective risk:** do not repeat confidence-as-OOD failures.
6. **held-out generalization:** public target wins are secondary to fresh confirmatory success.

## Kill / downgrade conditions

- If A13 cannot beat strong sparse/rival controls on semantic held-out families after non-leaky relation training, do not continue claiming neural value.
- If candidate recall misses its target, disable pruning for that regime.
- If OOD gate fails, autonomous mode stays disabled.
- If A6 matches A13 across the full typed/reliability envelope, 13M is not a justified production target.
- If public benchmark gains disappear on frozen confirmatory data, label them benchmark specialization.
- Never alter a target after seeing HIRA final-test results.

## W1 reproducibility authority — COMPLETE

Merged in `7a1b163afeea1f7198ae13784e04323225c96b56`.

Machine-readable contract preflight now enforces:
- exactly 53 unique headline cells;
- WIN/TIE/LOSS/MISSING semantics with missing never treated as zero;
- frozen direct-lane training allowances;
- matched Tesla T4 Q=1/5/10/50 systems authority;
- immutable final benchmark sources.

Source registry authority:
- **26 / 26 sources pinned**;
- **0 blocked final sources**;
- **0 mutable revisions**.

Previously blocked sources are now pinned:
- `LocalLLaMA/typed-decisions@c76749ec58bd8c3d2ea706b31c333a9059c38f90`;
- `mteb/amazon_massive_intent@940fd47a81eaa7f2cc7b129674d945d618ac38c2`;
- `mteb/amazon_massive_scenario@58871793b91addb7c5f7afff26ccf08737fb6697`.

Strict benchmark-contract preflight and repo CI pass on the merged W1 head.

This closes source/manifests reproducibility only. It does **not** populate any HIRA scorecard cell.

## NLI repair boundary after R24

R24 closes the deep NLI repair loop for now.

The authoritative R24 result is preserved separately in `research/R24-HANDOFF.md` and `artifacts/r24-sequential-exact-trust-walk/summary.json`.

Do not open another round merely to tune MultiNLI microsteps. The project returns to the original typed-decision/high-cardinality/state-once program.

## W2 typed/state-once infrastructure — COMPLETE

Merged in `425b89ede2de251efa990f5586a92924edb71535`.

W2 established:
- pinned typed-decisions TRAIN-only adapter;
- hard guard rejecting non-train rows;
- one state encode shared across five typed decisions;
- one differentiable state graph shared across five typed losses;
- full soft-distribution evaluator;
- explicit score support;
- factors/label_agreement excluded from model input;
- 18/18 W2 contract tests PASS;
- repository CI 3.10/3.12 PASS.

No test row was used and no scorecard cell was populated.

## W3a typed specialist selection — COMPLETE

Merged in `327679b87988f375e97b2a8a96456d35a8c54dde`.

Selected TRAIN/DEV-only head:
- candidate `r15-balanced-lr1e3`;
- epoch 7;
- head SHA-256 `2505e2cf99d741e590ff26ff7c70a587065e5713a4c6a3e53b6783a03b20446c`;
- selector explicitly records `final_test_exposed=false`.

No final scorecard cell was populated by W3a.

## W3b typed specialist one-shot final — COMPLETE

One-shot run `35795982760` evaluated the frozen head exactly once on typed-decisions `all/test`.

Result versus the eight metric-compatible Laya typed cells:
- **1 WIN / 0 TIE / 7 LOSS**;
- only raw ECE wins;
- accuracy, soft accuracy, Brier, score MAE and all three primitive accuracies lose;
- Jev typed cells remain MISSING because W3 is a specialist protocol.

This is a negative result and must remain unchanged. Do not posthoc tune against this final test.

## W4a held-out Banking77 direct — COMPLETE

Authoritative run `35822128041` executed the frozen first-400 Banking77 application protocol exactly once after a pure pre-exposure technical retry.

Result:
- HIRA accuracy **0.000 = 0/400**;
- Laya frozen target **0.492**;
- headline `laya.app.banking77_full` = **LOSS**;
- full-K K=77 mechanics PASS;
- state-once PASS;
- probability mass PASS;
- no Banking77 task training/retrieval/calibration.

Campaign after W4a:
- **1 WIN / 0 TIE / 8 LOSS / 44 MISSING**.

Machine-readable authority:
- `artifacts/r8-w4a-banking77/summary.json`.

Do not tune or rerun against the exposed first-400 authority.

## Current next task — W4b fresh K=128/K=255 stress

W4b must isolate high-cardinality mechanics from semantic zero-shot generalization.

Use fresh synthetic schemas/data that are generated independently of Banking77 and frozen before execution.

Required questions:
1. can HIRA preserve full probability mass at K=128 and K=255?
2. can option-order permutation be inverted exactly back to canonical probabilities/predictions?
3. can the correct option remain identifiable when semantic labels are fresh and opaque routing IDs carry no meaning?
4. how do latency and memory scale from K=77 -> 128 -> 255?
5. if candidate pruning is tested, what is candidate recall and does the frozen recall gate force all-K fallback?

Do not use W4a Banking77 rows, labels, errors or predictions to design W4b examples.

After W4b, decide whether high-cardinality mechanics themselves remain viable before any equal-information Jev retrieved-24 challenge.

The remaining 44 headline cells stay `MISSING` until their own authorized lanes execute.