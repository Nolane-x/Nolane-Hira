# R8-W6g handoff — field-role vs candidate-set interference localization

Status: **PRE-DIAGNOSTIC IMPLEMENTATION ACTIVE. No W6g localization result is valid yet.**

Issue: #89

## Frozen premise

W6f closed with pooled classification `UNRESOLVED_HIGH_K_FAILURE` for:
- W6e multi-source scorer-only;
- W6e joint-primary;
- W6e joint-replica.

W6f also established:
- uniform salience is dramatically worse than native candidate-relative salience;
- relation reranking is net helpful, not the dominant high-K failure;
- K64 errors frequently keep gold inside coarse top5;
- one-field hard negatives dominate many winning wrong candidates;
- gold margins and ranks degrade as candidate-set cardinality grows.

Therefore W6g is a second diagnostic localization lane. It performs no training and cannot promote a production mechanism.

## Frozen checkpoints

Use exact W6e checkpoints only:
- scorer-only artifact `10805971002`;
- joint-primary artifact `10805567861`;
- joint-replica artifact `10805871471`.

A13 remains exact and frozen.

Forbidden:
- all W6b/W6c/W6d/W6e CONFIRM rows;
- W6f N/O/P diagnostic rows;
- typed final/test rows;
- public campaign cells;
- any parameter update or threshold tuning.

## Fresh domains

Q:
- orbital greenhouse logistics;
- seed 211193.

R:
- biomedical cold-chain instrumentation;
- seed 211199.

S:
- geothermal generation control;
- seed 211213.

Each domain:
- 64 base states;
- four semantic roles;
- 10 controlled views/base.

Total:
- 192 base states;
- 1,920 diagnostic views.

## Fixed semantic roles

Every diagnosis signature has exactly four roles:
1. entity/equipment;
2. location/zone;
3. anomaly/event;
4. channel/source.

Every base state has:
- one gold signature;
- one exact one-field distractor per role;
- three fixed two-field negatives;
- controlled far and same-role-dense spectators.

## Ten views per base

Pair views:
- pair-entity K2;
- pair-location K2;
- pair-anomaly K2;
- pair-channel K2.

Reference:
- core-k8.

Candidate-set controls:
- far64;
- dense-entity64;
- dense-location64;
- dense-anomaly64;
- dense-channel64.

The gold and target one-field hard-negative option IDs/texts must remain byte-identical across pair/core/far/dense contexts.

## Primary diagnostic question

For a fixed gold vs one-field hard-negative pair, measure whether the pair margin changes when only the surrounding candidate set changes.

This separates:
- semantic value separability;
- role binding;
- generic cardinality;
- same-role density;
- candidate-relative IDF set dependence;
- common-mode set dependence.

## Frozen mechanism probes

No probe trains or mutates weights.

Required attribution probes for the fixed pair:
- native production scorer;
- K8-reference IDF only in final aggregation/salient-mask, with current-view native-IDF common mode;
- K8-reference IDF only in common-mode subtraction, with current-view native-IDF aggregation;
- K8-reference IDF in both pathways.

Important preregistration correction:
- production common-mode has no independent candidate-set input;
- its set dependence is downstream of candidate-relative IDF;
- W6g therefore decomposes **two pathways of IDF drift**, not two independent set-relative mechanisms.

Field-role decomposition uses tokenizer-robust counterfactual probes:
- isolated changed-value separability;
- role+value phrase separability;
- full structured K2 pair separability;
- value-only -> role+value -> full-pair accuracy/margin degradation.

A pre-data unit audit rejected the earlier token-span attribution idea because punctuation/tokenization made separately encoded substrings fail exact alignment inside full option text. No W6g A13 empirical cache existed. The protocol was amended before diagnostic exposure rather than weakening token matching.

## Scientific boundary

W6g may only localize.

A rescue lane is authorized only if the same causal classification:
- appears for joint-primary and joint-replica;
- holds on at least two of Q/R/S;
- is not contradicted by scorer-only.

Otherwise W6g must close unresolved/mixed and the project must reassess the textual structured interface before adding more model complexity.

## Current implementation

Implemented:
- `src/nmd/second_order_localization_authority.py`;
- `tests/test_second_order_localization_authority.py`.

Frozen generator invariants already encoded:
- 64 bases/domain;
- 10 views/base;
- stable state/gold across all views of a base;
- exact one-field pair construction;
- exact K8 core preservation inside every K64 view;
- far spectators differ in at least three fields;
- dense spectators differ in exactly the target role;
- deterministic option IDs across processes;
- Q/R/S value/template/role disjointness;
- freshness against W5-W6f values;
- no prior rows used.

No W6g cache/evaluator/result is valid until unit contracts pass.


## Pre-data tokenizer-robust attribution amendment

Before any W6g A13 cache or diagnostic result existed, unit tests showed that matching separately encoded role/value token-ID subsequences inside punctuated full option text is not tokenizer invariant.

The failed pre-data heads are not diagnostic authorities. They performed only local unit tests using the self-contained trainable test encoder.

W6g now uses three nested frozen counterfactual contexts for every one-field pair:
1. value-only;
2. role+value phrase;
3. full structured K2 production option.

All use the same frozen scorer projection and perform no training. This preserves the field-binding question while removing dependence on punctuation-specific token span recovery.


## Pre-data contextual-probe refinement

A second pre-data audit found that using the gold value itself as the probe reference would make value-only retrieval too close to a self-match.

Before any W6g A13 cache existed, the counterfactual probes were therefore frozen to score:
- gold value vs negative value against the full cached state-token sequence;
- gold role+value phrase vs negative role+value phrase against the same full state-token sequence.

This keeps the probe tokenizer-robust while making it contextual and non-tautological.


## Pre-data gate-alignment audit

Before any W6g A13 cache or diagnostic output existed, the first gated workflow
attempts stopped in the unit stage.

The audit found three implementation/contract defects and repaired them before
empirical exposure:

1. the synthetic margin fixture expected the wrong
   `phrase_to_full_margin_delta`; the implementation correctly defines this
   as full structured K2 margin minus role+value phrase margin;
2. cross-checkpoint stability logic lived in the CLI script, making the unit
   contract depend on importing `scripts/`; the logic is now centralized in
   `nmd.second_order_localization.diagnostic_stability` and consumed by both
   tests and the evaluator;
3. the first localization implementation incorrectly applied the far64
   stability guard to the IDF-interference classes. The preregistered issue
   only requires far64 stability for `DENSITY_NEAR_NEIGHBOR_LIMIT`.
   IDF aggregation/common-mode/mixed classes now use the frozen K2 -> dense64
   fixed-pair drop, while the density-near-neighbor class alone requires
   far64 to stay within 0.05 of K2 and dense64 to drop at least 0.10 from
   far64.

The evaluator also gives `MIXED_IDF_PATHWAY_INTERFERENCE` precedence whenever
both IDF pathways independently cross their recovery thresholds, matching the
frozen rule.

These repairs do not change Q/R/S generators, seeds, checkpoints, model
parameters, diagnostic thresholds or any empirical result. No eligible W6g
cache/evaluator result existed before the repairs.
