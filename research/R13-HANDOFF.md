# R13 handoff — held-out NLI stress falsification

Status: **HANS protocol frozen; empirical run pending**.

## Frozen model

R13 must not train, calibrate, select thresholds, or alter architecture.

The evaluated model is exactly the R12-selected head:
- HIRA head SHA-256: `0ca95572399d15717e1069545439165e46083c58afea8707cbbadeca67ad3b86`
- source workflow run: `35665673683`
- source artifact: `r12-scale-evidence`
- A13 revision: `4226d9e4d2c08703e5cb0491b479bfc6a1607181`
- A13 safetensors SHA-256: `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`

Any SHA mismatch is a hard failure.

## HANS source

- repo: `tommccoy1/hans`
- revision: `7299f6f657089ce06a0f98e7e81f8d0f5b7741ce`
- evaluation blob SHA: `15a8339b4f20fd21536a3f631682f7f1f52e5a2f`
- license: MIT
- evaluation set size: 30,000

HANS official binary view maps:
- 3-way HIRA entailment -> entailment
- 3-way HIRA neutral or contradiction -> non-entailment

The original 3-way probabilities/counts are also preserved.

## Pre-registered gate

Before seeing the result:

- overall HANS accuracy >= 0.55
- entailment accuracy >= 0.50
- non-entailment accuracy >= 0.50
- each of the three heuristic aggregate accuracies >= 0.50

All four conditions must pass.

This is intentionally stricter than a degenerate one-label classifier. Failure remains evidence and does not authorize threshold tuning on HANS.

## Required reporting

Report:
- overall accuracy;
- entailment vs non-entailment accuracy;
- lexical_overlap / subsequence / constituent accuracy;
- every HANS subcase;
- 3-way prediction counts under each binary gold label;
- mean 3-way probabilities by binary gold label.

## Other R13 preregistered sources

Issue #19 freezes:
- Breaking NLI, CC BY-SA 4.0;
- ANLI, CC BY-NC 4.0 research-only;
- NLI StressTest source, currently BLOCKED because license/generated-set terms are unresolved.

Do not substitute another stress set if HANS is weak.

## Next state

After HANS receipt is immutable, the same frozen R12 head may be evaluated on Breaking NLI. No model update is permitted between the two held-out evaluations.
