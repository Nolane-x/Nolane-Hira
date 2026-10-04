# HIRA V1 S51 contract — Persisted Native Authority + Artifact-Pinned Private Court

Status: **PREREGISTERED / NO S51-A0 EXPOSURE**

Issue: #285

Parent:
- S50 fresh run `37185080959`: native 24/24 completed, private DEV unexposed, then mechanical cache failure.
- S50 governed replay `37188685173`: same seed/data/code but native hashes differed at all 24 epochs.
- S50 closed as **INVALID REPRODUCIBILITY COURT / INCONCLUSIVE**.

## Scientific question

> With one artifact-pinned native authority consumed byte-for-byte by a later private court, does query-free state↔option identity improve cross-view stability without sacrificing useful correctness versus question-conditioned native relation signatures?

## Structural rule

S51 has exactly two workflows.

### Phase A — Native Authority

Phase A:
- uses wholly fresh S51 TRAIN rows only;
- trains native exactly once;
- never imports/generates S51 DEV rows;
- never creates a private correction module;
- fixed seed **72001**;
- fixed native epoch **24**;
- uploads one sealed native-authority artifact.

Artifact contents:
- `native-authority.pt`
- `train-manifest.json`
- `authority-receipt.json`
- `INTEGRITY.sha256`

Checkpoint payload must contain:
- LoRA state tensors;
- projection state tensor;
- native runtime state SHA-256;
- semantic revision;
- initialization T0 SHA-256;
- native trainable parameter surface;
- fixed epoch;
- seed.

Phase A DEV exposure is structurally forbidden.

### Phase B — Artifact-Pinned Private Court

Phase B:
- downloads the exact Phase-A artifact by frozen run ID and exact artifact name;
- verifies artifact integrity before loading;
- builds the frozen S17 runtime shell;
- loads LoRA/projection state from the artifact;
- verifies loaded runtime-state hash exactly equals authority receipt;
- freezes all native parameters;
- never creates a native optimizer;
- contains no native training loop or native TRAIN update call;
- only then generates fresh S51 TRAIN/DEV authority rows and encodes shared evidence.

Reference private signature:
`question_conditioned_native_relation_signature`

Treatment private signature:
`query_free_state_option_identity`

Both:
- exact same shared evidence cache bytes;
- correction parameters **114,688**;
- bit-identical correction initialization;
- identity added trainable parameters **0**;
- same raw query readout;
- same private optimizer/loss/selector;
- private epochs **24**;
- batch **16**.

## Persisted authority payload

Logical native tensor digest is independent of serialization container metadata.

Digest order:
1. sorted LoRA tensor keys;
2. projection weight;
3. tensor name;
4. shape;
5. dtype;
6. contiguous CPU bytes.

Runtime-state digest after loading MUST equal payload runtime-state digest.

File SHA-256 is separately frozen in `INTEGRITY.sha256`.

## Required S51-A0

Persist/load mechanics:
- checkpoint save/load runtime hash equality;
- logical tensor digest stable across roundtrip;
- file tamper/integrity mismatch rejected;
- wrong semantic revision rejected;
- wrong T0 authority rejected;
- malformed/missing LoRA key rejected;
- wrong projection shape rejected;
- non-finite checkpoint tensor rejected;
- loaded native params all frozen;
- loaded native optimizer parameter count 0.

Phase isolation:
- Phase-B implementation contains no call to native training helper;
- Phase-A implementation contains no S51 DEV generation/import;
- Phase-A implementation contains no private correction construction;
- authority artifact is immutable input to Phase B.

Cache:
- cache generated under native inference mode but transferred to normal non-inference tensors;
- detached;
- contiguous;
- requires_grad false;
- source mutation isolation;
- stable digest;
- branch-order replay exact.

Private mechanics:
- reference/treatment A/B/W initialization bit-identical;
- 114,688 correction params each;
- query-free identity params 0;
- identity API accepts no question tensors;
- K=3/7/255;
- full-K probability mass <=1e-6.

A0 is mechanical only and cannot select a model.

## Fresh S51 authority

Intended:
- seed **72001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S51 domains
- K=4
- native fixed epoch **24**
- private epochs **24**
- private batch **16**
- exactly one Phase-A authority artifact
- exactly one Phase-B private DEV.

No exact S50 A0/TRAIN/DEV text may be reused.

## Prohibited

No:
- Phase-B native retraining;
- second Phase-A authority;
- authority artifact regeneration after Phase-B exposure;
- identity variant;
- temperature/mixing sweep;
- raw/native identity blend;
- learned projector;
- query leak into identity;
- private loss/capacity/selector change;
- seed/LR/epoch retry;
- gate weakening;
- second S51 DEV;
- external Laya/Jev evaluation before separate confirmation.

Scientific failure is valid.
