# R8 Laya/Jev benchmark campaign

Status: **OPEN / preregister before model selection**.

This campaign turns the public Laya and Jev results into a fixed evaluation contract for Nolane HIRA. The goal is not one headline number: it is to test whether a 6–13M state-once decision engine can match or exceed large decision models across discrimination, probability quality, dynamic schemas, multilingual transfer, high-cardinality choices, reliability and systems efficiency.

## Pinned Laya reference

- repository commit: `42626c348753fbb17572a813127df2278a1ec527`
- `BENCHMARKS.md` blob: `2d448beccd1f8f0b87ad9fe0b090addf3cafdbae`
- T4 result blob: `ddc400a430834abc30c4bd028b6238153a968eee`
- 51-language CPU result blob: `1cb7d5ed5c498a6919797181d0f9ecb205e05b3d`
- T4 run: seed 13, Tesla T4, torch 2.11.0+cu128, transformers 5.17.0.

Exact numeric targets live in `laya_jev_targets.json` and are scored by `scorecard.py`.

## Laya lanes

### L1 — typed-decisions
Run the same 400 cases / 2,000 decisions. Preserve teacher probability vectors. Report accuracy, soft accuracy, Brier, raw ECE, calibrated ECE, NLL, score MAE, primitive breakdown and all workflows. Beating argmax accuracy alone is not parity if probability quality collapses.

### L2 — multilingual
Reproduce the T4 MASSIVE intent/scenario and XNLI lanes, then the 51-language MASSIVE sweep. Preserve seed 13 and the Laya 20-option sampling rule for many-label MASSIVE. Report Vietnamese separately from macro averages.

### L3 — English public tasks
Run AG News, BoolQ, DAIR Emotion, prompt-injections and SST-5 with the same prompt/sample rules. Label AG News/BoolQ as retention-style comparisons because Laya trained on them; Emotion, prompt-injections and SST-5 are the stronger held-out checks.

### L4 — application workflows
Reproduce email spam, phishing, jailbreak guardrails, toxicity moderation, RAG relevance, support triage and model routing from Laya's application script. Do not average in-training and held-out workflows into one unqualified intelligence score.

### L5 — high cardinality
Run full 77-label Banking77, then K=128 and K=255. Candidate recall is a first-class metric. If a candidate gate misses the preregistered recall target, report the miss and fall back to all-K; never hide a missing gold option behind reranker accuracy.

### L6 — reliability/robustness
Measure option-order flip rate, opaque-ID invariance, paraphrase, negation, contradiction, ECE, Brier, NLL, AURC/selective risk and OOD separately. Calibration is not OOD. OOD is not `1-max_softmax`.

### L7 — systems
On matched hardware/runtime report warm p50/p95 for Q=1/5/10/50, throughput, peak RAM/VRAM and model storage. For state-once HIRA report semantic-encoder calls per state. Laya's T4 figures are only direct speed targets on an equivalent T4 setup.

## Jev lanes

Jev does not publish one permanent standard benchmark table, so protocols stay separate.

### J1 — preregistered zero-shot pilot
Reproduce the `AbdelStark/jev-benchmarks` BTZSC pilot before direct Jev claims: 100 held-out examples per condition, no task-specific demonstrations, pinned data/model revisions, Jev 1.13.0. Report accuracy, macro-F1, Brier, NLL, ECE, true-label-zero rate, coverage at <=5% empirical error, failures and latency.

### J2 — typed-decisions published lane
Track the Jev figures surfaced by the Laya comparison table: 0.727 accuracy, 0.580 soft accuracy, 0.148 Brier, 0.144 ECE, 0.391 score MAE. Treat these as secondary-source targets until an independently runnable Jev manifest is available.

### J3 — full Banking77 challenge
A separate September 2026 Jev experiment reports 92.40% on all 3,080 official Banking77 test examples while supplying 24 relevant labeled examples per prediction. This is **not zero-shot**. HIRA may attempt it only with the same information budget and must not merge it into J1.

### J4 — K=255
TypeSafe states Jev supports cardinality up to 255 and uses a two-stage score-then-choice strategy at higher cardinality. HIRA must therefore report K=255 correctness, probability integrity, candidate recall and latency.

## Anti-benchmaxxing confirmatory lane

Public scores may guide engineering but cannot close the scientific claim. Before any parity/release claim, freeze a fresh confirmatory suite unused for architecture/loss/threshold/model selection. Include unseen schemas, opaque labels, order permutations, near-duplicate choices, negation/contradiction, OOD, English and Vietnamese.

If a public-target gain disappears there, record it as benchmark specialization, not closure.

## Optimization map

- typed accuracy: relation semantics + dynamic-schema training.
- soft accuracy/Brier/NLL: distribution distillation + proper scoring objectives.
- score MAE: ordinal/cumulative objective.
- Banking77/K=255: logical-option pooling + schema descriptions + candidate-recall authority + all-K fallback.
- option order: permutation training + consistency loss/check.
- raw ECE: proper-scoring/calibration-aware training; raw and post-temperature must be reported separately.
- OOD: energy/distance/OOD-head tournament; fail closed.
- multilingual: paired MASSIVE/XNLI EN–VI stage before expansion.
- Q=10/50 speed: state-once encoding + registered schema cache; count encoder calls.

## Claim language

The scorecard can mark an individual cell `WIN`. A release may say "beats Laya on X under protocol Y" only after protocol Y was run. "Laya parity" or "Jev parity" requires the preregistered multi-dimensional gate, not a majority of selected cells.
