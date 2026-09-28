# HIRA V0 MAINLINE M5 closure — matched external benchmark and research release

Status: **CLOSED — RESEARCH EVIDENCE RELEASE READY; LAYA/JEV PARITY NOT ESTABLISHED**

Issue: #169  
PR: #170  
Branch: `feat/hira-v0-mainline-m5-matched-benchmark`  
Base main: `97840f88b592ddfa10ae66e156fab781a4f18af7`

## 1. Purpose

M5 is the final Hira v0 mainline benchmark/release phase.

It evaluates the exact frozen M4 runtime against pinned Laya/Jev authorities under preregistered matched contracts, retains unsupported or cross-protocol cells as such, runs a fresh confirmatory suite, and produces a reproducible research evidence release.

M5 does not authorize post-exposure model tuning.

## 2. Frozen candidate

Hira runtime:
- M4 merge: `97840f88b592ddfa10ae66e156fab781a4f18af7`
- resident parameters: **13,213,199**
- trainable parameters: **0**
- state-once: required
- full-K: required
- relation refinement: disabled
- adaptive budget: disabled
- production_ready: **false**

Exact model identities:
- W28 T0 SHA256: `1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f`
- W34 SHA256: `d69fa11805291e6a06631d5bda941065f209ea96f5c46091187e984ff083834c`
- A13 revision: `4226d9e4d2c08703e5cb0491b479bfc6a1607181`
- A13 weight SHA256: `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`

## 3. M5-A — benchmark contract

Outcome:

`HIRA_V0_M5_CONTRACT_READY`

Run: `36358456335`  
Artifact: `10943744649`  
Digest: `sha256:856031dec357cb0a0d0f1b0fff82a5d7661832a5c67aebc7c2b45d309ec45230`

Frozen:
- 53 preregistered target cells;
- 27/27 source authorities;
- exact M4 runtime binding;
- direct/adapted/protocol-separated lane taxonomy;
- MISSING != zero;
- UNSUPPORTED != LOSS;
- NOT_COMPARABLE retained explicitly;
- no cross-protocol/global winner claim;
- final labels forbidden for model selection/training;
- fresh confirmatory evidence required.

Pinned external authorities:
- Laya: `NandhaKishorM/laya@42626c348753fbb17572a813127df2278a1ec527`
- Jev BTZSC: `AbdelStark/jev-benchmarks@0d610cc53e79bcbec691312b0c4adb4a0e371642`
- Jev model: `jev-1.13.0`

## 4. M5-B — matched quality evidence

### B1 — Banking77 direct

Outcome: `HIRA_V0_M5_BANKING77_DIRECT_READY`  
Run: `36358746320`  
Artifact: `10944752687`  
Digest: `sha256:7291569fb47f21b22b24b1eb9a8ec6690dc7a6d3f2e5658185e4c991e78f59e8`

- Hira: **0.0600**
- Laya matched target: **0.4920**
- result: **LOSS**
- K=77 full-K
- 400/400 states encoded exactly once
- Jev retrieved-24 score 0.924: **NOT_COMPARABLE** because the information budget differs.

### B2 — English held-out

Outcome: `HIRA_V0_M5_ENGLISH_HELDOUT_READY`  
Run: `36359178166`  
Artifact: `10944159765`  
Digest: `sha256:865f29d48ad9e7354e756453e22cf170b0927b0110dc5c86f34fa80ddab9dd9e`

- SST-5: Hira **0.263333** vs Laya **0.371667** — LOSS
- DAIR Emotion: Hira **0.206667** vs Laya **0.573333** — LOSS
- Prompt-Injections: Hira **0.431034** vs Laya **0.698276** — LOSS

The first attempted run failed in a Python 3.12 test-module harness before final exposure. The qualified retry changed only the harness, not dataset, prompt, target, runtime or weights.

### B3 — XNLI multilingual

Outcome: `HIRA_V0_M5_XNLI_READY`  
Run: `36359330862`  
Artifact: `10944709225`  
Digest: `sha256:dba2b9ca1cc61b5bceb62171f6da9d2fa2e8854838433317607f2c0462a1639d`

Exact 15-language protocol, 300 cases/language:
- English: Hira **0.283333** vs Laya **0.860000** — LOSS
- Vietnamese: Hira **0.343333** vs Laya **0.723333** — LOSS
- other-14 macro: Hira **0.334048** vs Laya **0.731000** — LOSS

### B4 — exact Jev BTZSC zero-shot pilot

Outcome: `HIRA_V0_M5_JEV_BTZSC_READY`  
Run: `36359563201`  
Artifact: `10945541326`  
Digest: `sha256:b0cd43fb6b9c580dc4360eb6a03365387d2521d09651c344d191cf555b970b5a`

Matched result cells:
- AG News accuracy: 0.24 vs 0.91 — LOSS
- AG News Brier: 0.749670 vs 0.145914 — LOSS
- AG News coverage@5% error: 0.00 vs 0.83 — LOSS
- Banking77 accuracy: 0.06 vs 0.87 — LOSS
- Banking77 Brier: 0.985177 vs 0.179125 — LOSS
- Banking77 coverage@5% error: 0.00 vs 0.86 — LOSS
- Emotion accuracy: 0.25 vs 0.48 — LOSS
- Emotion Brier: **0.826398 vs 0.846289 — WIN**

This Emotion Brier cell is the single matched WIN in the final scorecard.

### B5 — MASSIVE intent dynamic schema

Outcome: `HIRA_V0_M5_MASSIVE_INTENT_READY`  
Run: `36359747332`  
Artifact: `10945381496`  
Digest: `sha256:530ccadc71ad31a96fa03cea306c153d3e425f2799b72eda846d80694fe2770c`

- English: Hira **0.076667** vs Laya **0.783333** — LOSS
- other-13 macro: Hira **0.024872** vs Laya **0.451000** — LOSS

All 4,200 cases retained state-once/full-K invariants.

## 5. M5-C — systems evidence

Outcome: `HIRA_V0_M5_SYSTEMS_CPU_READY`  
Run: `36359707957`  
Artifact: `10945511699`  
Digest: `sha256:4341c4d6e452c5db2d0031ee7d3ebcf8bac5516738a7b52fb75167b6fc27bbbf`

GitHub-hosted CPU authority:
- Q=1 p50: **18.9965 ms**
- Q=5 p50: **23.4679 ms** total / 4.6936 ms per question
- Q=10 p50: **28.8772 ms** total / 2.8877 ms per question
- Q=50 p50: **74.1656 ms** total / 1.4833 ms per question
- state encoder calls: **48/48 expected**
- local load excluding snapshot download: **~2838.66 ms**
- wheel: **670,913 bytes**
- runtime bundle: **862,376 bytes**

Laya T4 latency cells remain **NOT_COMPARABLE** because the hardware/runtime boundary differs.

## 6. M5-D — fresh confirmatory authority

Outcome: `HIRA_V0_M5_CONFIRMATORY_READY`  
Run: `36380411753`  
Artifact: `10952835400`  
Digest: `sha256:2a4d2eb7a1d523b54158fc3993571e675fbfaac8fd47eba620646906fafacae3`

Suite SHA256:

`7c0126acb42ff1b361aabe5467a5a04513a26ce9ee3b9d94ca7d32b239d040ca`

Fresh suite:
- 24 base cases;
- 12 English / 12 Vietnamese;
- 48 total queries;
- opaque option IDs;
- original + reversed option order per state;
- not sourced from public Laya/Jev rows;
- not used for model selection.

Observed:
- original accuracy: **0.2500**
- reversed accuracy: **0.2500**
- paired both-correct: **0.2500**
- option-order flip rate: **0.0000**
- English accuracy: **0.333333**
- Vietnamese accuracy: **0.166667**
- composition accuracy: **0.0000**
- unseen-schema accuracy: **0.0000**
- max probability mass error: **1.043e-7**

This confirms strong order invariance/mechanical integrity but weak semantic reasoning quality in the frozen v0 candidate.

No model weights changed after public or confirmatory exposure.

## 7. Final 53-cell scorecard and research release

Final authority:

Run: `36380679813`  
Artifact: `10952164776`  
Artifact digest: `sha256:21334052dee015be8196eb89c6be3c787b0bdf70844e06481ca61d1eabffee06`

Scorecard outcome:

`HIRA_V0_M5_FINAL_SCORECARD_READY`

Counts across all **53 preregistered cells**:
- WIN: **1**
- TIE: **0**
- LOSS: **16**
- NOT_COMPARABLE: **5**
- UNSUPPORTED: **0**
- MISSING: **31**

Matched scored cells: **17**.

Research-release outcome:

`HIRA_V0_FINAL_RESEARCH_RELEASE_READY`

Archive:
- name: `NOLANE-HIRA-V0-FINAL-RESEARCH-RELEASE.zip`
- bytes: **3,518,599**
- SHA256: `2b524645c831b12521cd7edb506e708afad81336f1f23ebf13a271cfde514e5d`

The uploaded GitHub Actions artifact contains:
- frozen M4 runtime;
- all M5 authority evidence;
- raw per-example/per-case outputs;
- final scorecard;
- benchmark contracts/source manifests;
- fresh confirmatory evidence;
- integrity manifests;
- research-release archive.

## 8. Scientific conclusion

Hira v0 successfully closes its engineering/runtime mainline:
- compact 13.2M resident parameter system;
- zero trainable packaged parameters;
- state-once execution;
- dynamic schema;
- full-K mechanics through K=255;
- bounded LRU cache;
- deterministic package;
- local CPU load;
- reproducible benchmark/evidence chain;
- strong option-order invariance in the fresh confirmatory suite.

However, the matched M5 evidence does **not** establish Laya or Jev semantic parity.

The frozen candidate loses most executed matched semantic-quality cells, including Banking77, English held-out tasks, XNLI and MASSIVE. Reliability/OOD, multilingual quality and semantic reasoning therefore remain provisional. Production readiness remains false.

The correct final classification is:

**Hira v0 = completed reproducible research runtime/evidence release, not a production-ready or Laya/Jev-parity model.**

## 9. Closure decision

M5-A: **CLOSED**  
M5-B: **CLOSED for the executable matched lanes selected by the frozen contract**  
M5-C: **CLOSED**  
M5-D: **CLOSED**  
Final scorecard: **CLOSED**  
Research evidence release: **READY**

No post-exposure Hira v0 model rescue is authorized inside M5.

Future semantic improvements must start a new explicitly versioned research/mainline track and must not rewrite v0 evidence.
