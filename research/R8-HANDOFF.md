# R8 handoff — Laya/Jev benchmark campaign

Status: **OPEN; W1–W5g complete. Campaign remains 1 WIN / 0 TIE / 8 LOSS / 44 MISSING. W5g raises fresh semantic-routing accuracy to 50.52% with contrastive salience but remains PARTIAL; next priority is fresh balanced/anti-collapse token binding, not backbone scaling or exposed-CONFIRM retuning.**

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

## W4b fresh K=128/K=255 stress — COMPLETE

Authoritative run `35824879417`.

Track A:
- **MECHANICS_PASS** at K=128 and K=255;
- full-K budgets exactly K;
- tail mass zero;
- probability mass preserved;
- permutation equivariance PASS 64/64 at both K;
- repeatability exact in eval mode.

Track B:
- K=128 semantic-key accuracy **2/128 = 1.5625%**;
- K=255 semantic-key accuracy **1/128 = 0.78125%**;
- state-once remained exact.

Interpretation:
- high-cardinality implementation is viable;
- semantic ranking/generalization is the dominant blocker;
- candidate pruning is not the next priority.

Machine-readable authority:
- `artifacts/r8-w4b-high-cardinality/summary.json`.

## W5a fresh semantic competence rebuild — COMPLETE / NEGATIVE

Merged in `ebbba0d29574386624f53b9b0f1538aa75189295`.

Authoritative run `35826397280`:
- 4/4 candidate training PASS;
- DEV-only selector PASS;
- selected `w3-lr3e4`, epoch 1;
- untouched CONFIRM generated after selection;
- CONFIRM **0/256 top-1**;
- K=128 and K=255 top-1 both 0;
- K=255 top-5 0;
- mechanics, probability mass and state-once all PASS.

Verdict:
`SEMANTIC_COMPETENCE_FAIL`.

Machine-readable authority:
`artifacts/r8-w5a-semantic-competence/summary.json`.

## Current next task — W5b token-aware semantic matcher

The failure anatomy reveals an untested path already present in HIRA:
- A13 exposes token embeddings;
- HIRACore accepts option-token tensors;
- W5a supplied only pooled option embeddings;
- relation context also used segment-pooled state memory.

W5b must:
1. wire option token embeddings through schema/cache/runtime without changing opaque-ID semantics;
2. test state-token relation context as a separate mechanism;
3. use entirely fresh TRAIN/DEV authorities and an untouched post-selection CONFIRM set;
4. compare pooled baseline vs token-aware mechanisms under the same data and fixed optimization budget;
5. freeze one candidate on DEV before CONFIRM generation;
6. populate zero public campaign cells;
7. proceed to public held-out lanes only if fresh CONFIRM passes.

Do not rerun W5a or tune against its exposed CONFIRM.

## W5b token-aware semantic comparison — COMPLETE / NEGATIVE

Authoritative run `35829561505`:
- pooled remained DEV-selected;
- token-aware alternatives did not improve;
- untouched CONFIRM 0/192 top-1;
- K128/K255 accuracy 0;
- mechanics remained healthy.

Verdict: `TOKEN_SEMANTIC_FAIL`.

Machine-readable authority:
`artifacts/r8-w5b-token-semantic/summary.json`.

## W5c alignment/representation probes — COMPLETE / NEGATIVE

Merged in `a691ef7b0fad2ce55cb13d09e0047faa9bd5b0e8`.

Authoritative run `35834712173`.

Fresh CONFIRM:
- pooled cosine 0/256;
- token-max 1/256;
- learned bilinear 0/256;
- learned pair-MLP 0/256;
- all K128/K255 accuracy gates fail;
- probability mass and state-once PASS.

Verdict: `A13_PROBE_FAIL`.

Machine-readable authority:
`artifacts/r8-w5c-semantic-alignment/summary.json`.

## Current next task — W5d controlled A13 top-layer adaptation

The repository capacity trigger does not justify jumping directly to A22 before controlled A13 top-layer unfreezing.

W5d must:
1. use fresh non-public TRAIN/DEV/CONFIRM semantic authorities, disjoint from W5a–W5c;
2. fine-tune only the upper A13 transformer layers under a frozen lower encoder;
3. train a direct semantic alignment objective rather than HIRA scorer parameters;
4. freeze the adapted encoder checkpoint on DEV before untouched CONFIRM;
5. compare adapted A13 to exact frozen A13 on the same fresh authority;
6. populate zero public campaign cells.

If adapted A13 still misses fresh competence by the frozen gate, the A22 capacity trigger is considered satisfied and the next lane should run the pinned A22 capacity control.


## W5d controlled A13 top-layer adaptation — COMPLETE / NEGATIVE

Merged in `c44a16452b9a013fdccdd9486d5fdbca9dab0433`.

Authoritative run `35844159119`:
- 4/4 top-layer adaptation candidates PASS execution;
- DEV-only selector chose `top2-lr1e5`, epoch 2;
- untouched CONFIRM generated after selection;
- frozen A13 top-1 **0/192**;
- adapted A13 top-1 **0/192**;
- adapted MRR **0.04472** vs frozen **0.04022**;
- K128/K255 top-1 remain **0**;
- every competence/gain gate FAIL;
- probability mass PASS;
- campaign cells populated = 0.

Verdict:
`A13_ADAPTATION_FAIL_CAPACITY_TRIGGER`.

This satisfies the preregistered capacity trigger. The next authorized lane is **W5e A22 diagnostic capacity control** using `microsoft/xtremedistil-l6-h384-uncased`. Do not continue A13 scorer or adaptation sweeps before that control.


## W5e paired A13/A22 capacity control — COMPLETE / NEGATIVE

Merged in `c7dd96f0e854ea3ebf14c466ff3574945fa8ebee`.

Authoritative run `35850323054`:
- A13 adapted CONFIRM accuracy **0/192**;
- A22 adapted CONFIRM accuracy **1/192**;
- A22 adapted MRR **0.07065** vs A13 adapted **0.06068**;
- A22 adapted K128 top-1/top-5 **0 / 0**;
- A22 adapted K255 top-1 **0**, top-5 **0.0625**;
- all capacity-rescue gates FAIL except probability-mass integrity.

Verdict:
`A22_CAPACITY_NO_RESCUE`.

Machine-readable authority:
`artifacts/r8-w5e-capacity-control/summary.json`.

This falsifies the simple capacity hypothesis. Do not continue by merely scaling A22 to a larger encoder. The next lane must change the semantic representation/learning mechanism while keeping fresh-data and untouched-CONFIRM discipline.


## W5f state-once late interaction — COMPLETE / PARTIAL

Merged in `6504e4ebd93efe78c93caaed0f9d18843131402b`.

Authoritative run `35858733250` selected `proj128-maxsim`, epoch 6.

Untouched CONFIRM:
- selected accuracy **29.6875%** vs pooled **0.5208%**;
- selected MRR **0.44158** vs pooled **0.04804**;
- selected top-5 **60.9375%** vs pooled **3.125%**;
- K128 accuracy **25%** vs pooled 0%;
- K255 accuracy **16.667%** vs pooled 0%;
- K255 top-5 **41.667%**;
- probability integrity PASS;
- campaign cells populated = 0.

Verdict: `LATE_INTERACTION_PARTIAL`.

Interpretation:
- independent pooling was a real blocker;
- direct token binding recovers substantial competence;
- remaining error is concentrated at high K and near-neighbor distractors;
- W5f gives every option token uniform aggregation weight, so tokens shared by nearly every option can dilute the discriminative field-value tokens.

## W5g contrastive-salience late interaction — COMPLETE / PARTIAL

Merged in `1846f2a401e55e35228ef9f14437baa5088af928`.

Authoritative push run `35862655705` at exact empirical head `e36a57b849db6fec1bb2e570d9e4003a533338b0`.

Selected:
- `idf-proj128`, epoch 6;
- matcher SHA-256 `f416a2786ed907608017ffffc8cf61442b4cfc68d75bb9f5102841659afd8788`.

Untouched CONFIRM:
- overall accuracy **50.5208%** vs fresh uniform **38.5417%** and pooled A13 **0%**;
- MRR **0.64019** vs uniform **0.54627**;
- top-5 **79.6875%** vs uniform **75%**;
- K128 accuracy **47.9167%**, top-5 **72.9167%**;
- K255 accuracy **41.6667%**, top-5 **64.5833%**;
- probability-mass max error **2.384e-7**;
- state text encodes per case **1.0**;
- forbidden benchmark data used = false;
- campaign cells populated = 0.

Frozen rescue gates PASS:
- K128 accuracy;
- K255 accuracy;
- overall gain vs uniform;
- K128 gain vs uniform;
- K255 gain vs uniform;
- probability integrity.

Frozen rescue gates FAIL:
- overall accuracy >= 0.60;
- K255 top-5 >= 0.70.

Verdict:
`CONTRASTIVE_SALIENCE_PARTIAL`.

Authoritative closeout:
`research/R8-W5G-HANDOFF.md`.

Interpretation:
- candidate-relative salience is a second real semantic-binding mechanism on top of direct late interaction;
- the remaining blocker is no longer near-random semantic ordering;
- high-K errors now look like hard-negative binding/ranking ambiguity;
- W5g still lets every option token independently take MaxSim over the same context tokens, so several semantic fields may collapse onto the same context evidence.

## Current next task — W5h balanced anti-collapse token binding

W5h must be a new fresh authority. Do not reuse W5g TRAIN/DEV/CONFIRM.

Core hypothesis:
**independent MaxSim creates many-to-one token-binding collapse; a constrained/balanced matching operator can improve hard-negative separation without increasing encoder capacity.**

Keep fixed:
- exact frozen A13;
- bias-free 256->128 projection;
- one scalar logit scale;
- state-once execution;
- full-K scoring through K=255;
- same training objective/optimizer/epoch budget across controlled candidates;
- opaque option semantics;
- zero public campaign cells.

The experiment should compare a fresh W5g-style `idf-maxsim` control against anti-collapse alternatives with the same trainable parameter count. Candidate mechanisms should alter only the token-to-context assignment/coverage rule, not backbone size or data access.

The decisive target is not merely another small average gain. W5h should attack the two W5g failures:
1. overall accuracy < 0.60;
2. K255 top-5 < 0.70.

Selection must be DEV-only and CONFIRM must be generated only after candidate/checkpoint freeze.

Do not:
- tune W5g IDF/min-coverage thresholds using exposed CONFIRM;
- reuse W5g cases/templates/vocabulary/seeds;
- scale to a larger encoder;
- reopen Banking77/typed final/MASSIVE/XNLI authorities;
- populate any public campaign cell before fresh rescue is established.
