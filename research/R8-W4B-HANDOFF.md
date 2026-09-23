# R8-W4b handoff — fresh K=128/K=255 stress

Status: **COMPLETE; authoritative W4b empirical result merged and preserved.**

## Why W4b exists

W4a held-out Banking77 completed with 0/400 accuracy while full-K K=77/state-once/probability-mass mechanics executed correctly.

W4b does not retry or modify Banking77.

It isolates two questions:
1. are the frozen HIRA head's large-K mechanics numerically/permutation stable at K=128 and K=255?
2. can the frozen A13 + HIRA stack route an independently generated exact semantic key when K is large?

## Frozen model

Selected W3a head:
- SHA-256 `2505e2cf99d741e590ff26ff7c70a587065e5713a4c6a3e53b6783a03b20446c`;
- 422,159 parameters.

A13:
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA-256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- frozen.

## Track A

Exact K values:
- 128;
- 255.

64 deterministic embedding-level cases per K.

Primary authority is mechanical invariants only:
- finite outputs;
- candidate budget exactly K;
- zero tail mass;
- probability mass error <= 1e-6;
- permutation probability error <= 2e-6;
- 64/64 inverse-permutation argmax invariance;
- repeatability error <= 1e-7.

Track A alone determines MECHANICS_PASS/FAIL.

## Track B

128 fresh semantic-key cases per K.

Each case is generated only from frozen independent color/object/action/location vocabularies.

No external benchmark data is loaded.

Report:
- accuracy;
- top-5 recall;
- MRR;
- correct-option probability;
- confidence;
- probability mass;
- state-once ratio.

Track B is diagnostic and cannot alter Track A verdict.

## Anti-leak

W4b contains no Banking77 loader, rows, labels, predictions, prompts or error analysis.

It also does not populate any 53-cell campaign metric.

Issue #57 is the frozen protocol authority.


## Authoritative W4b result

Workflow:
- run `35824879417`;
- head `b24df386613f66cf5647f83f216601b966262df9`;
- artifact `10735251248`;
- digest `sha256:5337fa194ba629bbddeec2190c6b06d94605618c263dffaa5f90dd394d31bbb2`.

### Track A — mechanics

Verdict: **MECHANICS_PASS**.

K=128:
- 64/64 inverse-permutation argmax invariance;
- candidate budget 128/128;
- tail mass 0;
- probability-mass max error 2.3842e-7;
- permutation max error 5.9605e-8;
- repeatability max error 0;
- p50 1.8216 ms CPU;
- p95 1.8943 ms CPU.

K=255:
- 64/64 inverse-permutation argmax invariance;
- candidate budget 255/255;
- tail mass 0;
- probability-mass max error 2.3842e-7;
- permutation max error 2.9802e-8;
- repeatability max error 0;
- p50 2.4213 ms CPU;
- p95 2.4777 ms CPU.

### Track B — fresh semantic-key diagnostic

K=128:
- accuracy 0.015625 = 2/128;
- top-5 recall 0.046875;
- MRR 0.0437583.

K=255:
- accuracy 0.0078125 = 1/128;
- top-5 recall 0.0078125;
- MRR 0.0263185.

This diagnostic is near-random and is not a campaign cell.

## W4b conclusion

The K=128/K=255 implementation is not the dominant failure.

Large-K probability mass, permutation equivariance, determinism and full-K execution are healthy.

The primary blocker is semantic routing/generalization of the frozen A13 + HIRA head.

Do not proceed to candidate-pruning optimization as the next priority. Pruning cannot rescue near-random semantic ranking.

Next: fresh non-benchmark semantic competence training with independent confirmatory data.

Machine-readable authority:
`artifacts/r8-w4b-high-cardinality/summary.json`.
