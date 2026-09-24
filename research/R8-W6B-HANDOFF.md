# R8-W6b handoff — fresh typed/reliability production authority

Status: **CLOSED — authoritative verdict `PRODUCTION_COMPETITIVE_PARTIAL`.**

Issue: #79
PR: #80

## Scientific purpose

W6 Phase A integrated the promoted forward competitive binder into HIRA's real state-once coarse path without changing legacy defaults.

W6b is the first fresh typed/reliability authority that tests whether the integrated production competitive path improves over a same-authority legacy HIRA control.

This lane is no longer a synthetic semantic-scorer search.

## Frozen upstream identities

A13:
- model `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA-256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`.

W3 HIRA head:
- run `35748778854`;
- artifact `10706135649`;
- artifact name `r8-w3-selected-head`;
- artifact digest `sha256:8a2efd31a6ccea3b987957d328c4b73267ab00ff9024c469bcb7f3cd578d5d92`;
- head SHA-256 `2505e2cf99d741e590ff26ff7c70a587065e5713a4c6a3e53b6783a03b20446c`;
- HIRACore parameters 422,159.

W5i promoted forward scorer:
- run `35884130186`;
- artifact `10764240783`;
- artifact name `r8-w5i-candidate-idf-competitive-forward-proj128`;
- artifact digest `sha256:f3c0ff451c976caff10b8098b010abc7b8758d4c4797110c4cce18943edb4f74`;
- candidate `idf-competitive-forward-proj128`;
- matcher SHA-256 `5ce9cdceb5a87c8b18b8d50e68396f23184e1dfb9acad338f25f24277d438a0d`;
- scorer parameters 32,769.

The W5i reverse/listwise scorer is not promoted.

## Fresh authority

Every case has exactly five decisions:
1. diagnosis — choice;
2. response — choice;
3. needs_review — noul;
4. risk — score;
5. urgency — score.

Counts:
- TRAIN: 320 cases / 1,600 decisions;
- DEV: 128 cases / 640 decisions;
- CONFIRM: 160 cases / 800 decisions.

Seeds:
- TRAIN 161127;
- DEV 162229;
- CONFIRM 163331;
- training/global 809.

CONFIRM remains sealed until DEV selection and all selected checkpoint hashes are frozen.

Forbidden:
- typed-decisions final/test rows;
- prior W5 semantic CONFIRM rows;
- public campaign cells.

## Exactly three candidates

1. `legacy-w3-joint`
   - legacy coarse mode;
   - W3 init;
   - 422,159 trainable/total parameters.

2. `competitive-w5i-joint`
   - W3 + W5i forward scorer init;
   - competitive coarse mode;
   - 454,928 trainable/total parameters.

3. `competitive-w5i-scorer-only`
   - frozen W3;
   - train competitive scorer only;
   - 32,769 trainable / 454,928 total parameters.

All use the same frozen A13 cached features, six epochs and frozen loss/optimizer contract from issue #79.

## DEV freeze

Each candidate freezes its own epoch by:
1. higher overall accuracy;
2. lower hard Brier;
3. higher soft accuracy;
4. higher diagnosis K64 accuracy;
5. lower score MAE;
6. lower soft-target ECE;
7. earlier epoch.

Select one competitive candidate between joint and scorer-only.
Freeze legacy independently.
Only then may CONFIRM be generated.

## Frozen production gates

Absolute selected gates:
- overall accuracy >= 0.65;
- choice accuracy >= 0.65;
- noul accuracy >= 0.70;
- score accuracy >= 0.60;
- diagnosis K64 accuracy >= 0.55;
- hard Brier <= 0.50;
- soft-target ECE <= 0.15;
- score MAE <= 0.55;
- probability error <= 1e-6;
- source state encodes/case = 1.0.

Mechanism gates vs legacy:
- overall accuracy gain >= +0.03;
- diagnosis K64 gain >= +0.05;
- hard Brier no worse by > 0.02;
- soft-target ECE no worse by > 0.02;
- score MAE no worse by > 0.05.

Valid verdicts:
- `PRODUCTION_COMPETITIVE_RESCUE`;
- `PRODUCTION_LEGACY_ALREADY_STRONG`;
- `PRODUCTION_COMPETITIVE_PARTIAL`;
- `PRODUCTION_COMPETITIVE_FAIL`.

## Pre-authority evidence

Head `29ebe5a1971ed8214bc1593007aa7934fa4a1ee0`:
- W6b push unit workflow PASS;
- W6b PR unit workflow PASS;
- **14/14 pre-authority contracts PASS**;
- repo CI Python 3.10 PASS;
- repo CI Python 3.12 PASS;
- preflight PASS.

No W6b CONFIRM has been generated or exposed by this boundary.

## Authority chain

The first eligible authority workflow must gate:

`unit -> upstream provenance -> fresh A13 TRAIN/DEV cache -> 3 candidates -> DEV selection freeze -> untouched CONFIRM`

No branch mutation is allowed while an eligible exact-head authority run is active.


## Pre-data calibration amendment

Before any W6b TRAIN/DEV cache or CONFIRM was generated, audit found that hard ECE conflicts with the declared soft reliability targets.

A model matching a target confidence of 0.90/0.75/0.60 should be considered reliability-calibrated. Hard ECE compares those confidences with binary correctness and can penalize an otherwise perfect soft-target match for not being overconfident.

Therefore:
- raw hard ECE remains recorded as a diagnostic;
- soft-target ECE compares model confidence with the frozen teacher probability assigned to the predicted class;
- DEV tie-breaking uses soft-target ECE;
- the absolute 0.15 calibration gate uses soft-target ECE;
- the +0.02 non-regression mechanism gate uses soft-target ECE.

Regression coverage explicitly proves that [0.90, 0.75, 0.60] against the same soft targets yields approximately zero soft-target ECE while hard ECE exceeds 0.15.

This amendment occurred before empirical authority execution and does not change authority rows, model architecture, optimizer, loss weights, checkpoint identities, accuracy gates or verdict taxonomy.

## Pre-data integrity strengthening

Also before empirical authority execution:
- cache receipts were extended with TRAIN/DEV/CONFIRM template IDs and template hash;
- TRAIN/DEV and reserved-CONFIRM value lexicons receive separate hashes;
- candidate receipts bind to exact TRAIN/DEV cache SHA-256;
- selector rejects candidates from mismatched caches;
- confirm receipt carries cache provenance and explicit parameter counts;
- a regression contract proves TRAIN and DEV gold diagnosis semantic signatures are internally unique and cross-split disjoint.

Earlier queued authority heads are superseded; only the final post-amendment exact head may be authoritative.


## Authoritative closure

Status: **CLOSED — authoritative verdict `PRODUCTION_COMPETITIVE_PARTIAL`.**

Final exact authority head:
- `883e15bb37bf887401865d73f37eaf88e97af2c2`.

Sole authoritative end-to-end run:
- `35959578866`;
- unit -> upstream provenance -> fresh TRAIN/DEV cache -> 3 candidates -> DEV-only selector -> untouched CONFIRM;
- every job PASS.

Exact-head validation:
- frozen W6b contracts: **17/17 PASS**;
- PR unit run `35959582383`: PASS;
- push unit run `35959578839`: PASS;
- repo CI run `35959581533`: PASS.

### Fresh cache receipt

TRAIN:
- 320 cases / 1,600 decisions;
- seed 161127;
- cache SHA-256 `1a99f9068c1aa527fc00d13024e47e1714bd17a5e84404ac71fff493fdb31866`.

DEV:
- 128 cases / 640 decisions;
- seed 162229;
- cache SHA-256 `48282d09e170964d6634d18789647728ee6c347254118a17b541cc1c3d58d0ce`.

Reserved CONFIRM seed:
- 163331.

Integrity:
- TRAIN state encodes/case = 1.0;
- DEV state encodes/case = 1.0;
- prior-W5 value overlap = empty;
- typed-decisions final/test used = false;
- campaign cells populated = 0.

### DEV selection

Competitive joint:
- epoch 6;
- accuracy 90.94%;
- K64 diagnosis accuracy 84.38%;
- hard Brier 0.2356;
- score MAE 0.2018;
- soft ECE 0.1609;
- HIRA SHA `fdad44c5d30cc93ceb2efa80ff38ab5800e2493b8087886a9db9b4c923719f1b`;
- scorer SHA `32379f0dfcec0d7d98107fd7ec26693ffd9504c1dafc4ecb1492caa5a35cb244`.

Competitive scorer-only:
- epoch 6;
- **accuracy 93.13%**;
- K64 diagnosis accuracy 84.38%;
- hard Brier 0.2160;
- score MAE 0.1513;
- soft ECE 0.1833;
- HIRA SHA `925f74094ac4ae583c015ea2a0be32ec692d885ec64dcf9cf3d94b64be0ccf42`;
- scorer SHA `50abb2e8136599bcaf5c41d61036e3c335a7589c6244536cc0292dcea15b1ef0`.

Legacy:
- epoch 6;
- accuracy 44.38%;
- K64 diagnosis accuracy 0%;
- hard Brier 0.6254;
- score MAE 0.5767;
- soft ECE 0.0341;
- HIRA SHA `a83da6c92aaa515518600bf379c897d90f27d12b5a6ef9f31b76c68297139e7e`.

DEV selector froze:
- `competitive-w5i-scorer-only`;
- epoch 6;
- trainable parameters **32,769**;
- total production path parameters 454,928.

### Untouched CONFIRM

CONFIRM:
- 160 cases / 800 decisions;
- generated only after DEV freeze;
- cache SHA-256 `3c0ebd9c9f516ea51778da8557eb13893459da34e195037dc36c03eec3714292`;
- source state encodes/case = 1.0.

Selected scorer-only production path:
- overall accuracy **81.50%**;
- choice accuracy **80.31%**;
- noul accuracy **69.375%**;
- score accuracy **88.75%**;
- diagnosis K8 **57.50%**;
- diagnosis K16 **67.50%**;
- diagnosis K32 **65.00%**;
- diagnosis K64 **60.00%**;
- hard Brier **0.3283**;
- raw hard ECE **0.1333**;
- soft-target ECE **0.1637**;
- score MAE **0.2367**;
- probability mass max error **1.4063e-7**.

Legacy same-authority control:
- overall accuracy **40.125%**;
- choice accuracy **21.25%**;
- noul accuracy **66.25%**;
- score accuracy **45.94%**;
- diagnosis K8 **12.50%**;
- diagnosis K16/K32/K64 **0%**;
- hard Brier **0.6519**;
- raw hard ECE **0.0839**;
- soft-target ECE **0.0607**;
- score MAE **0.6398**.

Same-authority deltas:
- overall accuracy **+41.375 percentage points**;
- diagnosis K64 **+60.0 points**;
- hard Brier improvement **0.3236**;
- score MAE improvement **0.4031**;
- soft-target ECE worsened by about **0.1030**.

Frozen absolute gates:
- PASS overall accuracy;
- PASS choice accuracy;
- **FAIL noul accuracy**: 0.69375 < 0.70;
- PASS score accuracy;
- PASS diagnosis K64;
- PASS hard Brier;
- **FAIL soft-target ECE**: 0.16373 > 0.15;
- PASS score MAE;
- PASS probability integrity;
- PASS state-once.

Frozen mechanism gates:
- PASS overall accuracy gain;
- PASS diagnosis K64 gain;
- PASS hard-Brier non-regression;
- **FAIL soft-ECE non-regression**;
- PASS score-MAE non-regression.

Therefore the frozen verdict is:

**`PRODUCTION_COMPETITIVE_PARTIAL`**

It is not RESCUE because two absolute production gates fail and the calibration mechanism gate fails.

### Evidence artifacts

- frozen upstream: `10792845362`, digest `sha256:12cc959893474bef1a6ef387f5ab34ce3b10f4fb55da4ea5696e3e8ddaa7ba66`;
- fresh cache: `10792084200`, digest `sha256:ec8dc1ef205bb4f773fca12f4136d097f1bab6b87d9478241689612dd041d509`;
- competitive joint: `10792562782`, digest `sha256:f9040d77709e71fc0313a21b8887b618868a7088f7e23efa71e116d55577cb8a`;
- legacy: `10792905891`, digest `sha256:a38dee06a9abb395f734aaef8a7c7491254a4381f57f31c895fb9b92deb5a454`;
- competitive scorer-only: `10792970760`, digest `sha256:abc6856162d0320074eec0ac9532a3cd73d54164b3d902e9318e3ac2b86b462a`;
- selected: `10792488690`, digest `sha256:3f0fe10a60701a0a654e5c71170432d55a96d1345b42fa2e53a38e9cb1778711`;
- CONFIRM: `10792554860`, digest `sha256:bc3d62e644d0e0124467d45c14dd73319a219384b973885bd8f13b5cba7ca170`.

### Scientific interpretation

The semantic/coarse-routing question is no longer the dominant blocker:
- a 32,769-trainable-parameter scorer-only adaptation on frozen HIRA more than doubles same-authority legacy overall accuracy;
- high-cardinality diagnosis transfers to fresh reserved vocabulary, including 60% at K64 while legacy is 0%;
- choice, score, Brier and score-MAE production gates pass.

The remaining blocker is concentrated in **reliability/review calibration**:
1. `needs_review` misses by only 0.625 percentage points;
2. soft-target ECE exceeds the absolute limit by about 0.0137;
3. competitive soft-ECE is materially worse than legacy despite far higher correctness.

Do not reopen W5 semantic scorer search and do not retune exposed W6b CONFIRM.

The next lane should preserve the scorer-only winner and target reliability/review calibration on a wholly fresh authority.
