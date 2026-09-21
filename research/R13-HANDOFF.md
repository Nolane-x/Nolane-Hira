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


## Held-out empirical results

### HANS full 30k — gate FAIL

Workflow run: `35666437449`

Overall:
- accuracy: **0.5109**
- entailment accuracy: **0.9381**
- non-entailment accuracy: **0.0837**

Heuristic aggregates:
- constituent: 0.5266
- lexical_overlap: 0.4947
- subsequence: 0.5114

Non-entailment accuracy by heuristic is the critical failure:
- constituent: 0.1538
- lexical_overlap: 0.0394
- subsequence: 0.0580

The preregistered overall/label-balance/heuristic gates all failed.

Interpretation: the R12 head is strongly biased toward entailment when lexical/syntactic overlap is high. This is not a threshold problem; mean entailment probability is almost the same for HANS entailment and non-entailment examples.

Evidence artifact:
- id: `10669502726`
- digest: `sha256:ed56455d152d4852315a2fcedb72dddbceb54335f23c880e58cffebf9ff47f00`

### Breaking NLI 8,193 — gate FAIL

Workflow run: `35666600462`

- accuracy: **0.2234**
- macro recall: **0.3564**
- macro F1: **0.1515**
- majority baseline accuracy: 0.8744 (dataset is heavily contradiction-skewed)
- entailment recall: **0.9430**
- neutral recall: **0.0000**
- contradiction recall: **0.1262**

The preregistered macro-recall/overall/contradiction gates failed.

Strongest category:
- synonyms: 0.9418

Examples of weak categories:
- countries: 0.0326
- nationalities: 0.0344
- drinks: 0.0438
- materials: 0.0504
- colors: 0.0672
- cardinals: 0.0698

Evidence artifact:
- id: `10668819251`
- digest: `sha256:7aa53b6949af7194674d2ceabcb19a852697671c8cca4e6641319f37a300b856`

## R13 conclusion

R12's 58.07% MultiNLI result is real semantic learning, but it does **not** generalize robustly to syntactic counterexamples or lexical/world-knowledge minimal pairs.

Two separate deficits are now evidenced:

1. **anti-entailment / structural reasoning deficit**
   - HANS non-entailment collapse;
   - model overweights lexical/subsequence overlap.

2. **lexical/world-knowledge deficit**
   - Breaking NLI contradiction collapse across many semantic categories;
   - synonyms are easy while many substitution categories fail.

Do not open XNLI as a victory benchmark yet. R14 must repair these failure modes using training data that does not include the HANS evaluation set or Breaking NLI evaluation set.

The untouched R13 failures must remain permanently visible even if an adapted R14 model later improves on the same suites.
