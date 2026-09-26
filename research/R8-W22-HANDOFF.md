# R8-W22 handoff — independent reference-panel ceiling

Status: **PRE-DIAGNOSTIC. No DH/DI/DJ/DK HIRA/reference cache or W22 empirical result exists yet.**

Issue: #131

Branch:
`feat/r8-w22-reference-panel-ceiling`

Base main:
`18d594998718b88f10ba71b59733aa97becf6eeb`

Read first:
1. this file;
2. `research/R8-W21-HANDOFF.md`;
3. `research/R8-W20-HANDOFF.md`;
4. issue #131.

## 0. HIRA identity

Nolane HIRA is a compact non-autoregressive typed decision engine.

W22 remains diagnostic:
- zero HIRA training;
- no typed-kernel integration;
- no production promotion;
- no K32/K64 opening;
- fresh authority only.

## 1. Why W22 exists

W18-W21 were four consecutive latent-interface phases whose frozen interpretation was blocked because the sole MiniLM reference did not meet adequacy gates.

Latest W21:
- `PROTOTYPE_LATENT_UNRESOLVED`;
- 4/4 `W21_REFERENCE_INADEQUATE`;
- HIRA prototype: severity 48.44%, confidence 73.96%, joint 35.94%;
- MiniLM prototype: severity 53.91%, confidence 71.88%, joint 38.54%.

Therefore W22 stops changing only the HIRA-side interface and directly diagnoses the reference ceiling.

## 2. Frozen HIRA base

A13:
- repo `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- model weight SHA256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length 256;
- frozen.

W9 scorer:
- 256->128 projection;
- scorer SHA256 `8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102`;
- source run `36122220588`;
- freeze artifact `10858424139`.

W22 HIRA trainable parameters:
**0**.

## 3. Fresh W22 authority

Domains:
- DH — municipal document-delivery incident assessment — seed 411101;
- DI — community food-distribution service triage — seed 411107;
- DJ — university media-equipment service assessment — seed 411119;
- DK — regional bicycle-share support triage — seed 411131.

Each:
- 4 severity classes;
- 3 confidence classes;
- 8 query variants per S×C;
- 96 cases/domain.

Total:
- 384 query cases.

Each HIRA query:
- one severity sequence;
- one confidence sequence;
- one batched A13 invocation;
- exactly two encoded query sequences.

Exposure:
**NONE.**

## 4. Frozen semantic interface

Fresh prototype-grounded interface:
- severity: exactly 3 concrete prototypes/class;
- confidence: exactly 3 concrete prototypes/class;
- prototype bank shared within domain;
- arithmetic mean of all 3 prototype scores/class;
- no top-k;
- no selection;
- no learned weights;
- query/prototype exact sentence overlap = 0;
- no W18-W21 exact text reuse.

Fresh abstract D0/D1/D2 categorical control remains descriptive.

## 5. Frozen reference panel

All panel members are selected and pinned before exposure.

### R0 — legacy MiniLM
- `sentence-transformers/all-MiniLM-L6-v2`;
- revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- model.safetensors SHA256 `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`;
- mean pooling over attention-masked last hidden states;
- L2 normalize.

### R1 — MPNet
- `sentence-transformers/all-mpnet-base-v2`;
- revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`;
- model.safetensors SHA256 `78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658`;
- mean pooling;
- L2 normalize.

### R2 — E5-small-v2
- `intfloat/e5-small-v2`;
- revision `93da57dece4e396a19773b2658fe3ffdd358e5eb`;
- model.safetensors SHA256 `45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1`;
- average pooling;
- query fields receive prefix `query: `;
- prototype/abstract option text receives prefix `passage: `;
- L2 normalize.

### R3 — BGE-small-en-v1.5
- `BAAI/bge-small-en-v1.5`;
- revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`;
- model.safetensors SHA256 `3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad`;
- CLS-token pooling;
- no retrieval instruction prefix in W22;
- L2 normalize.

No panel member, revision, pooling rule or prefix rule may change after exposure.

## 6. Per-reference scoring

For prototype interface:
- cosine query-to-each-prototype;
- class score = arithmetic mean of exactly 3 prototype cosine scores;
- softmax over class scores only for probability-mass integrity.

For abstract control:
- D0/D1/D2 option cosine scores;
- arithmetic mean across views.

Per domain/reference report:
- severity top1/MRR/MAE;
- confidence top1/MRR/MAE;
- joint S+C;
- probability mass.

## 7. Frozen individual reference pass

Per domain a reference passes iff:
- prototype severity >= .80;
- prototype confidence >= .85;
- prototype joint >= .68;
- probability mass error <=1e-6.

W22 gates are new diagnostic gates only.
They do not relabel W18-W21.

## 8. Frozen panel adequacy

Per domain require all:
- >=3/4 references individually pass;
- >=2/3 non-legacy references {MPNet,E5,BGE} pass;
- majority-vote severity >= .85;
- majority-vote confidence >= .88;
- majority-vote joint >= .75;
- mean pairwise prediction agreement across {MPNet,E5,BGE} >= .82 severity;
- same agreement >= .85 confidence.

Otherwise:
`W22_REFERENCE_PANEL_INADEQUATE`.

Ties in 4-reference majority are resolved only by the 3 non-legacy reference majority. No metric or HIRA output may break ties.

## 9. Legacy-reference diagnosis

If panel is adequate and MiniLM individually fails:
`LEGACY_MINILM_REFERENCE_LIMIT`.

This is a reference-authority diagnosis, not a HIRA promotion.

## 10. Frozen HIRA interpretation

HIRA is interpreted only on panel-adequate domains.

HIRA adequacy:
- prototype severity >= .75;
- prototype confidence >= .80;
- prototype joint >= .62;
- probability mass <=1e-6.

If panel adequate and HIRA passes:
`HIRA_LATENT_AUTHORITY_ADEQUATE`.

If panel adequate and HIRA fails:
`HIRA_LATENT_GEOMETRY_LIMIT`.

If panel inadequate:
`W22_REFERENCE_PANEL_INADEQUATE`.

The legacy-limit flag is reported orthogonally.

## 11. Cross-domain outcome

- same interpretable HIRA class on >=3/4 domains ->
  `STABLE_REFERENCE_PANEL_LOCALIZATION`;
- panel adequate on >=3/4 and legacy MiniLM limit on >=3/4 ->
  `STABLE_LEGACY_REFERENCE_LIMIT` if no stable HIRA class is available;
- panel inadequate on >=3/4 ->
  `AUTHORITY_REFERENCE_UNRESOLVED`;
- otherwise ->
  `REFERENCE_PANEL_MIXED`.

## 12. Integrity

Must prove:
- 384 cases;
- 96/domain;
- balanced latent cells;
- 3 prototypes/class exactly;
- exact query/prototype overlap = 0;
- exact prior-authority overlap = 0;
- one HIRA state compile/query;
- one HIRA A13 invocation/query;
- two HIRA sequences/query;
- exact HIRA/W9 hashes;
- exact 4 reference repo revisions and model weight hashes;
- panel frozen before exposure;
- zero HIRA training/selection;
- no W21/W20/W19/W18 rows;
- no Banking77;
- no typed final/test;
- campaign cells = 0.

## 13. Authorization boundary

W22 may change future **reference methodology** only if the frozen panel proves the legacy single-reference limit on fresh data.

W22 alone cannot:
- promote prototype grounding to production;
- integrate deterministic typed composition;
- tune HIRA;
- alter W18-W21 verdicts;
- open K32/K64.

## 14. Current state

Completed:
- W21 authority frozen and closure merged;
- W21 issue closed;
- W22 issue #131 preregistered;
- W22 branch created from exact post-W21 main;
- fresh domains/seeds frozen;
- panel members/revisions/hashes/pooling frozen;
- all gates frozen;
- this handoff created before exposure.

Exposure:
**NONE.**

Next:
1. implement fresh W22 authority/prototype bank;
2. implement freshness contracts;
3. implement HIRA two-field/prototype cache;
4. implement HIRA scoring;
5. implement four-reference evaluator;
6. implement panel majority/agreement;
7. implement classifier/outcome;
8. tests + pre-data workflow;
9. exact-head unit + repo CI;
10. only then enable empirical authority;
11. freeze result before merge.

A future AI must update this file after every meaningful W22 session.
