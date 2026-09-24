# R8-W6e handoff — replicated joint-adaptation held-out generalization

Status: **PRE-AUTHORITY IMPLEMENTATION ACTIVE. No W6e empirical verdict is valid yet.**

Issue: #85

## Scientific premise

W6d closed with frozen verdict `GENERALIZATION_FAIL`.

Untouched domain-F evidence:
- frozen W6b control overall 40.208%, K64 6.25%;
- single-source scorer-only overall 84.792%, K64 25.00%;
- multi-source scorer-only overall 84.271%, K64 27.083%;
- multi-source joint overall 89.167%, K64 54.167%.

Equal-budget scorer-only multi-source training did not beat single-source training on the preregistered diversity gates.

The joint HIRA+scorer path gained +27.083 pp K64 over multi-source scorer-only but missed the frozen K64 >=55% competence threshold by one of 48 held-out K64 cases.

W6e does not relax that gate and does not reuse W6d domain F.

## Frozen upstream

A13:
- model `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA-256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- fully frozen.

W6b production initialization:
- HIRA SHA `925f74094ac4ae583c015ea2a0be32ec692d885ec64dcf9cf3d94b64be0ccf42`;
- scorer SHA `50abb2e8136599bcaf5c41d61036e3c335a7589c6244536cc0292dcea15b1ef0`;
- artifact `10792488690`.

Forbidden:
- W6b CONFIRM;
- W6c CONFIRM;
- W6d domain F;
- typed final/test rows;
- public campaign cells.

## Fresh seven-domain authority

Source TRAIN:
- G aviation;
- H geothermal;
- I semiconductor;
- J food processing.

Held-out DEV:
- K electrical grid.

Untouched CONFIRM replicas:
- L mining;
- M datacenter infrastructure.

All domain values/templates/role wording/case IDs are pairwise disjoint.

Counts:
- TRAIN G/H/I/J: 96 states/domain, 384 total, 1,920 decisions;
- DEV-K: 192 states / 960 decisions;
- CONFIRM-L: 192 / 960;
- CONFIRM-M: 192 / 960.

Joint strata:
`K {8,16,32,64} x severity {0,1,2,3} x confidence {verified, provisional, uncertain}`.

TRAIN has 2 cases per stratum/domain.
DEV/CONFIRM have 4 cases per stratum.

Seeds:
- G 191161;
- H 191167;
- I 191173;
- J 191179;
- DEV-K 192283;
- CONFIRM-L 193387;
- CONFIRM-M 194489;
- primary training 1409;
- replica training 2411.

## Preregistered paths

1. `frozen-w6b-control`: 0 trainable params.
2. `multi-source-scorer-only`: 32,769 trainable params.
3. `multi-source-joint-primary`: 454,928 trainable params, seed 1409.
4. `multi-source-joint-replica`: 454,928 trainable params, seed 2411.

No cross-candidate winner selection is allowed.

Each path freezes independently on DEV-K.

The primary joint path is the hypothesis test. The replica cannot rescue a failing primary by itself.

## Frozen gates

Semantic competence on each CONFIRM domain:
- overall >=0.65;
- choice >=0.65;
- score >=0.60;
- noul >=0.65;
- diagnosis K64 >=0.55;
- probability error <=1e-6;
- state encodes/case =1.0.

Primary joint causal gates vs scorer-only on each CONFIRM domain:
- overall gain >=+0.03;
- K64 gain >=+0.05;
- choice regression <=0.02;
- score regression <=0.02.

Verdicts:
- `JOINT_GENERALIZATION_REPLICATION_RESCUE`;
- `JOINT_GENERALIZATION_REPLICATION_PARTIAL`;
- `JOINT_GENERALIZATION_REPLICATION_FAIL`.

## Current implementation

Implemented:
- `src/nmd/typed_joint_replication_authority.py`;
- `tests/test_typed_joint_replication_authority.py`.

The W6e authority module reuses only the already-tested generic W6d domain-generation primitives. It defines wholly new G-M domain specs, seeds and identities, then relabels shared-generator case IDs from the internal W6d prefix to explicit W6e IDs.

Pre-cache contracts must prove:
- source/domain counts and exact stratum balance;
- TRAIN/DEV semantic disjointness;
- pairwise value/template/role disjointness across G-M;
- no exact authority-value overlap with W5/W6b/W6c/W6d;
- W6e-only case IDs/workflows;
- CONFIRM-L/M remain sealed;
- unit tests never request `allow_confirm=True`.

No W6e cache, model training, DEV freeze or CONFIRM materialization is valid until these contracts pass.
