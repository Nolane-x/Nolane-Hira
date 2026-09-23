# R8-W5g handoff — contrastive-salience late interaction

Status: **COMPLETE / PARTIAL. Merged to main in `1846f2a401e55e35228ef9f14437baa5088af928`.**

## Authority

Issue: #71  
PR: #72  
Exact empirical head: `e36a57b849db6fec1bb2e570d9e4003a533338b0`  
Authoritative push run: `35862655705`

Exact-head verification:
- W5g push authority: PASS;
- PR W5g unit workflow `35862661515`: PASS;
- repository CI `35862661504`: PASS;
- cache: PASS;
- 3/3 candidates: PASS;
- DEV selector: PASS;
- untouched post-selection CONFIRM: PASS;
- forbidden benchmark data used: false;
- campaign cells populated: 0.

Artifacts:
- cache `10750344798`, digest `sha256:c32ceb78f2483215fa4815599a022e379f447b56656a723af265a91e0d16f056`;
- uniform candidate `10751164369`, digest `sha256:905b43533852324f1778995d6c4e300f66f8dc49b6c0813eda8f58f7acb7e3ba`;
- idf candidate `10751333656`, digest `sha256:8475d5df15d2c20e5022b7328153b24510ad831b457d0dcdc2d61311d7f5849c`;
- idf-centered candidate `10751931241`, digest `sha256:06d288c25d1d0a4041ef5fa9753ecc48f25acbedf8beb385948f6eae1393fbdc`;
- selected `10752261174`, digest `sha256:d8767ec2ca90673a1c8b9c9d958f43e385d3fab8ec7a3a284a7a9d727845b720`;
- confirm `10751452082`, digest `sha256:b7bc62f04fe01859d2309ae4043ec56465ad6f412fa2b60b8931cd38cce4a0f7`.

## Controlled experiment

Frozen encoder:
- `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA-256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`.

Fresh authority:
- TRAIN 512;
- DEV 176;
- untouched CONFIRM 192;
- CONFIRM K=32/64/128/255, 48 cases each;
- seeds 121001 / 122002 / 123003;
- all W5g vocabulary/templates/case IDs disjoint from W5a-W5f.

All three candidates had the same trainable surface:
- bias-free 256->128 projection;
- one scalar logit scale;
- same objective, optimizer, epochs and seeds.

Candidates:
1. `uniform-proj128`;
2. `idf-proj128`;
3. `idf-centered-proj128`.

Only candidate-relative aggregation differed.

## DEV selection

Selected:
`idf-proj128`, epoch 6.

Selected matcher SHA-256:
`f416a2786ed907608017ffffc8cf61442b4cfc68d75bb9f5102841659afd8788`.

Fresh uniform matcher SHA-256:
`fa9642be4331e6847d87993755946bace03d62736717ff0529ad55be4e1be857`.

Selection was frozen before CONFIRM generation.

## Untouched CONFIRM

Selected IDF late interaction:
- accuracy **0.505208 = 50.52%**;
- MRR **0.640186**;
- top-5 recall **0.796875 = 79.69%**;
- K32 accuracy **0.645833**, top-5 **0.958333**;
- K64 accuracy **0.479167**, top-5 **0.854167**;
- K128 accuracy **0.479167**, top-5 **0.729167**;
- K255 accuracy **0.416667**, top-5 **0.645833**;
- hard Brier **0.692422**;
- probability-mass max error **2.384e-7**;
- state text encodes per case **1.0**.

Fresh uniform baseline:
- accuracy **0.385417**;
- MRR **0.546269**;
- top-5 **0.75**;
- K128 accuracy **0.395833**;
- K255 accuracy **0.3125**.

Frozen pooled A13 baseline:
- accuracy **0.0**;
- MRR **0.039988**;
- top-5 **0.020833**.

W5f -> W5g top-1 movement on separate fresh authorities:
**29.69% -> 50.52%**.

## Frozen rescue gates

PASS:
- K128 accuracy >= 0.40;
- K255 accuracy >= 0.30;
- overall accuracy gain vs fresh uniform >= +0.10;
- K128 accuracy gain vs fresh uniform;
- K255 accuracy gain vs fresh uniform;
- probability-mass integrity.

FAIL:
- overall accuracy >= 0.60;
- K255 top-5 recall >= 0.70.

Verdict:
**`CONTRASTIVE_SALIENCE_PARTIAL`**

Do not relabel this as RESCUE.

## Failure anatomy and next hypothesis

W5g has moved the problem from semantic collapse to hard-negative binding ambiguity.

Evidence:
- K255 top-1 is **41.67%** but top-5 is **64.58%**;
- overall MRR is **0.6402**;
- the correct option is therefore often near the top even when not ranked first.

The W5g matcher still computes independent per-option-token MaxSim:

`coverage = similarity.max(dim=-1).values`

This permits multiple semantic fields/tokens in one option to reuse the same context token as their best match. IDF suppresses common option boilerplate, but does not enforce distinct or balanced semantic coverage.

The next fresh lane should test **anti-collapse / balanced token binding** while preserving:
- exact frozen A13;
- the 256->128 projection budget;
- state-once execution;
- full-K scoring through K=255;
- fresh pre-registered TRAIN/DEV/CONFIRM;
- zero public campaign cells.

Do not:
- tune on W5g CONFIRM;
- sweep IDF thresholds/weights on exposed data;
- scale to A22/A30/A50;
- rerun Banking77 or other exposed final authorities.
