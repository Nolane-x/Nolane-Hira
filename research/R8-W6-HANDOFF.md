# R8-W6 handoff — production competitive coarse integration

Status: **Phase A MERGED as `bd250d37726bcbc975c5244ccb8dd0cb597b1351`. Phase B fresh typed/reliability authority is next; no W6 empirical typed/reliability verdict exists yet.**

## Frozen evidence entering W6

W5i authoritative verdict:
`CROSS_CANDIDATE_CONTROL_ALREADY_RESCUES`.

Fresh forward competitive control on untouched W5i CONFIRM:
- accuracy 82.29%;
- K128 accuracy 85.42%;
- K255 accuracy 79.17%;
- K255 top-5 91.67%;
- probability max error 2.384e-7;
- state encode/case 1.0.

The production mechanism promoted into W6 is only:
`idf-competitive-forward-proj128`.

W5i reverse/listwise evidence is not promoted.

## Integration shape

Legacy `HIRACore` remains the same 422,159-parameter module.

W6 adds an optional `CompetitiveCoarseScorer` owned by `NolaneHira`:
- d_model 256 -> d_rel 128 bias-free projection;
- scalar logit scale;
- exactly 32,769 trainable parameters;
- candidate-relative IDF;
- within-option competitive common-mode subtraction;
- forward MaxSim;
- weighted mean + 0.5 * salient minimum coverage.

`HIRACore.forward` accepts an optional `coarse_override`.
When absent, the legacy pooled coarse path is unchanged.

Runtime modes:
- `coarse_mode="legacy"` — default and backward compatible;
- `coarse_mode="competitive"` — production integration under evaluation.

## Artifact extension

Optional additions only:
- TextBatch token IDs + special-token mask;
- StateMemory content-token embeddings;
- CompiledSchema question token artifacts;
- CompiledSchema option token IDs + content mask.

Existing fields remain unchanged.

## Phase A gates

Before any fresh typed/reliability authority:
1. all modified modules compile;
2. dedicated W6 contracts pass;
3. legacy runtime tests pass;
4. token relation mode tests pass;
5. typed state-once tests pass;
6. default HIRACore parameter count remains 422,159;
7. strict legacy HIRACore state_dict roundtrip passes;
8. competitive scorer count is 32,769;
9. competitive mode is state-once;
10. competitive mode fails closed without scorer/artifacts;
11. opaque IDs are semantically inert;
12. option permutation only permutes outputs;
13. probability mass error <= 1e-6;
14. gradients reach scorer, encoder and relation head.

## Phase B boundary

Do not use previously exposed typed-decisions final rows for tuning.

After Phase A is clean, freeze a fresh train/dev/confirm typed/reliability authority from train-only or newly generated data, with CONFIRM sealed until selection freeze.

No public campaign cells are populated by Phase A.


## Phase A authoritative integration closure

Merged PR:
- PR #78;
- squash merge `bd250d37726bcbc975c5244ccb8dd0cb597b1351`;
- exact Phase A head `0b125657109723047bfc4d1476c7e5d68d6c8836`.

Exact-head evidence:
- dedicated push run `35931861897`: compile PASS, **44/44 PASS**;
- dedicated PR run `35931867744`: compile PASS, **44/44 PASS**;
- legacy W5b token-relation PR run `35931867662`: PASS;
- legacy W2 typed state-once PR run `35931867709`: PASS.

Repo-wide CI on immediate predecessor `fad4a8b...` passed Python 3.10, Python 3.12 and preflight. The only subsequent source change before the frozen Phase A head was vectorized IDF mean normalization inside the new optional competitive scorer; exact-port and W6 integration tests passed on the frozen head.

Phase A proves:
- production scorer parameter count = 32,769;
- legacy HIRACore parameter count = 422,159;
- strict legacy HIRACore state_dict compatibility;
- default `legacy` coarse mode unchanged;
- explicit `competitive` full-K coarse injection;
- exact scorer-port equivalence to W5h `idf-competitive-proj128`;
- state-once execution;
- opaque-ID semantic invariance;
- option-order equivariance;
- fail-closed artifact/scorer requirements;
- valid probability normalization;
- gradients through scorer, trainable encoder and relation head.

No Phase A benchmark/final authority was exposed.
Campaign cells populated = 0.

## Phase B preregistration boundary

Phase B must branch from merged Phase A and freeze its authority before any empirical run.

It must not:
- load `typed-decisions` final/test for tuning or selection;
- reinterpret W3b final;
- reuse W5 semantic CONFIRM rows;
- promote W5i reverse/listwise evidence.

Phase B should test the integrated forward competitive scorer on a new typed/reliability authority with:
- multiple decisions per state;
- choice, noul and score primitives;
- soft probability targets;
- high-cardinality choice slices;
- one state encode per case;
- TRAIN/DEV/CONFIRM split with CONFIRM sealed until DEV selection freeze;
- explicit legacy same-authority control.
