# R7 implementation-research baseline

**Status:** OPEN. **Purpose:** convert the R6 construction blueprint into a reproducible repository without weakening scientific gates.

## Frozen question

Can a 6-13M state-once typed decision system match a large decision encoder on a preregistered envelope of fixed-schema quality, dynamic/unseen-schema semantics, K<=255 candidate coverage, calibration, OOD/selective risk, EN/VI transfer, and multi-question systems efficiency?

## R7 workstreams

### W1 — Reproducibility authority
Pin dataset revisions, checkpoint revisions, tokenizer hashes, seeds, environment, and immutable run receipts. Recompute R3-R5 sparse controls from materialized bytes before neural claims.

### W2 — A13 proof
Run frozen A13 first. A22 is forbidden unless A13 vs A22 diagnostic trigger is explicitly reached. A7 compression is forbidden before A13 passes core gates.

### W3 — Relation semantics
Evaluate gold NLI warm-up and pointwise/listwise/pairwise teacher distillation without assuming one objective is superior. Negation, contradiction, paraphrase, opaque IDs and option-order perturbations are mandatory.

### W4 — Dynamic high-cardinality schemas
Descriptions, aliases, multiple exemplars and counterexamples compile into logical options. Prototype rows must pool before K accounting. Candidate miss is a first-class failure.

### W5 — Reliability
Calibration and OOD are separate. Missing/stale OOD authority fails closed. Registered-fast mode requires fresh domain/schema calibration; dynamic mode cannot sparse-exit.

### W6 — EN/VI and compression
MASSIVE en-US/vi-VN and XNLI EN/VI are release gates. A7-FE24 is a post-proof compression candidate, not a guaranteed winner.

## Mandatory rivals

- strong sparse word+char prototype control;
- ~6M dynamic-router challenger;
- A13 reference;
- A22 control only on trigger;
- pinned Laya version on matched hardware/runtime.

## Kill conditions

- If neural models do not beat sparse controls on semantic/dynamic gates, do not claim neural value.
- If candidate recall misses its preregistered target, fall back to all-K; never hide misses behind reranking accuracy.
- If OOD/selective-risk gates fail, autonomous mode is disabled.
- If A7 loses >2 absolute points on multiple EN/VI semantic families under matched training, retain A13 or redesign compression.
- If the ~6M rival matches A13 on the full typed/reliability envelope, 13M is not a justified production target.

## Evidence boundary

Repository tests prove software contracts only. They do not prove semantic quality, Laya/JEV parity, multilingual parity, or production latency.
