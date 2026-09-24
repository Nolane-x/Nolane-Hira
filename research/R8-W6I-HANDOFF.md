# R8-W6i handoff — representation-bridge audit

Status: **CLOSED DIAGNOSTIC. Authoritative outcome: `REPRESENTATION_BRIDGE_UNRESOLVED`. No rescue lane is authorized.**

Issue: #93

## Frozen premise

W6h closed with authoritative verdict `FIELD_SEMANTIC_FAIL`.

The key negative finding is not merely that the 8,192-param adapter failed. W6h exposed a measurement bridge mismatch:

- W6g localized weak isolated changed-value retrieval on fresh Q/R/S;
- W6h optimized full structured one-field option pairs;
- untouched W6h Y/Z frozen control already reached about 94.8-95.1% on that structured pair metric;
- adapter field-pair gain was therefore only about +0.6/+1.0 pp while K64 changed inconsistently.

W6i is diagnostic only. It must resolve whether this discrepancy is caused by the proxy bridge, domain difficulty, free-form conjunction loss, or a global ranking failure that representation changes do not fix.

## Frozen checkpoints

No training or parameter updates.

Evaluate only:
- exact W6e joint-primary;
- exact W6h DEV-X projection-retune checkpoint;
- exact W6h DEV-X semantic-residual-adapter checkpoint.

No W6i row may select, tune or modify a checkpoint.

## Forbidden rows

Do not reuse:
- W6e CONFIRM L/M;
- W6f N/O/P;
- W6g Q/R/S;
- W6h CONFIRM Y/Z;
- W6h TRAIN T/U/V/W or DEV-X as diagnostic rows;
- typed final/test rows;
- public campaign cells.

## Fresh domains

- AA satellite battery servicing, seed 231269;
- AB sterile water distribution, seed 231271;
- AC automated warehouse safety, seed 231277.

Each domain has 64 base states.

Every base has:
- one frozen state text;
- one four-field gold signature;
- one exact one-field negative per role;
- one K8 core;
- one K64 semantic master set.

The same gold and same one-field negative identities are reused across pair/K8/K64 views.

Total:
- 192 base states;
- 1,152 generated production-text views.

## Same-case representation ladder

For every role/base:

- P0: isolated changed value vs matched negative against full frozen state;
- P1: role+value phrase vs matched negative against same state;
- P2: full production structured K2 pair;
- P3: exact same P2 pair embedded into K8 and K64.

Counterfactual renderings of the same K8/K64 semantic signatures:
- canonical tagged text: `[F0] value0 ... [F3] value3`;
- factorized role+value fields with frozen arithmetic-mean and minimum operators.

No learned parameter or post-result operator selection is allowed.

## Current implementation

Implemented:
- `src/nmd/representation_bridge_authority.py`;
- `tests/test_representation_bridge_authority.py`.

Generator contracts include:
- 64 bases/domain;
- six views/base;
- exact pair identities;
- K8 nested byte-identically inside K64;
- deterministic K64 controlled cross-combinations;
- pairwise domain value/template/role separation;
- freshness against W5-W6h;
- deterministic IDs;
- canonical and factorized representations derived only from the same four semantic values.

Before any A13 cache/evaluation:
1. generator unit contracts must pass;
2. frozen checkpoints/provenance must be verified;
3. state encoding and all additional probe encoder calls must be explicitly accounted;
4. no prior authority/diagnostic row may be materialized as W6i input.

No W6i localization result is valid before these gates pass.


## Authoritative diagnostic closure

Exact diagnostic head before this closure-doc commit:
- `6993efdefa962ee6fca50f44f6d3dc6c39532b0e`.

Authority run:
- `36022251442`;
- all jobs PASS: unit -> exact upstream provenance + fresh AA/AB/AC cache -> fixed-checkpoint representation audit.

Artifacts:
- cache `10818615287`, digest `sha256:052d375d270457dfcf699139dff32839398fe46ce30848ff8f4bb96e0f117ef9`;
- representation audit `10818157410`, digest `sha256:595a668b91e3f89e88473ffa798328d527351f7be16ce3202dcdba6c17739c8d`;
- frozen checkpoints `10816673873`, digest `sha256:1927a020b433985b0eb9377e3f6b51671a163dc5948548054c97fc057cf938e6`.

Integrity:
- 192 fresh base states / 1,152 production-text views;
- AA/AB/AC only;
- state encoding exactly 1/base;
- probability mass max error `2.384185791015625e-07`;
- no training;
- no W6b-W6h exposed CONFIRM/diagnostic rows;
- no typed final/test rows;
- campaign cells 0.

### Frozen stability result

Checkpoint stability:
- `w6e-joint-primary` -> `REPRESENTATION_BRIDGE_UNRESOLVED`; not stable;
- `w6h-projection-retune` -> `STRUCTURED_PAIR_PROXY_MISMATCH` on AA/AB; checkpoint-local stability only;
- `w6h-semantic-adapter` -> `REPRESENTATION_BRIDGE_UNRESOLVED`; not stable.

Cross-checkpoint outcome:
- **`REPRESENTATION_BRIDGE_UNRESOLVED`**;
- stable cross-checkpoint classification: none;
- `rescue_lane_authorized = false`.

### Key empirical anatomy

W6e joint-primary pooled:
- P0 isolated-value accuracy 59.505%;
- P2 structured-K2 accuracy 78.125%;
- fixed pair inside K64 81.641%;
- production K64 top1 41.146%.

W6h projection-retune pooled:
- P0 isolated-value accuracy 63.802%;
- P2 structured-K2 accuracy 90.885%;
- fixed pair inside K64 92.578%;
- production K64 top1 66.146%;
- this checkpoint alone satisfies `STRUCTURED_PAIR_PROXY_MISMATCH`, but the result does not replicate across the frozen checkpoints.

W6h semantic-adapter pooled:
- P0 isolated-value accuracy 63.932%;
- P2 structured-K2 accuracy 85.417%;
- fixed pair inside K64 86.589%;
- production K64 top1 52.083%.

Representation-only counterfactuals do not rescue global ranking. Across the frozen checkpoints, canonical tagged and both factorized operators are materially worse than production free-form scoring at K64. Therefore W6i does **not** support promoting the tested canonical/factorized interfaces.

### Scientific conclusion

W6i confirms that isolated-value retrieval and full structured-pair competence are not interchangeable measurements: projection retuning can make the full structured pair strong while isolated field retrieval remains weak.

However that proxy-mismatch story is not stable across W6e joint-primary and the semantic-adapter checkpoint. The tested canonical/factorized renderings also fail to improve high-K ranking.

Per the preregistered boundary, do not open another local adapter, IDF, relation-head, or representation rescue from W6i. The next research work must move one level up and reassess HIRA's high-cardinality decision decomposition and supervision interface before another rescue authority is authorized.
