# R8-W6 handoff — production competitive coarse integration

Status: **Phase A implementation active under issue #77. No W6 empirical typed/reliability verdict exists yet.**

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
