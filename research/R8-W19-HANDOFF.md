# R8-W19 handoff — latent ordinal-axis recoverability decomposition

Status: **PRE-DIAGNOSTIC. No CV/CW/CX/CY A13/reference cache or W19 result exists yet.**

Issue: #125

Branch:
`feat/r8-w19-latent-ordinal-axis`

Base main:
`5a3b158404cb444121dc1aace9ac3e34ae452668`

Read first when restoring:
1. this file;
2. `research/R8-W18-HANDOFF.md`;
3. `research/R8-W17-HANDOFF.md`;
4. `research/R8-W16-HANDOFF.md`;
5. issue #125.

## 0. HIRA identity

Nolane HIRA is a compact non-autoregressive typed decision engine.

Target:
- state logically compiled once;
- isolated semantic fields where needed;
- dynamic candidate/schema semantics;
- typed choice/score/noul;
- transferable semantic extraction;
- explicit deterministic composition where rules are known;
- calibrated probabilities;
- small local-friendly trainable footprint;
- sealed fresh authorities;
- external validation only after internal replication.

Scientific discipline:
- freeze gates before exposure;
- exposed rows are permanently forbidden;
- PARTIAL/FAIL remains PARTIAL/FAIL;
- no post-exposure threshold/wording changes;
- pinned reference models are diagnostic only;
- do not promote descriptive gains into production rescue.

## 1. Frozen semantic base

A13:
- `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length 256;
- frozen.

W9 semantic projection:
- 256->128 bias-free;
- scorer SHA `8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102`;
- source run `36122220588`;
- freeze artifact `10858424139`.

W19 trainable parameter count:
**0**.

## 2. Decisive history entering W19

W14:
- 4/4 `CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE`;
- multiview semantic E 94.92/90.23/84.77% K4/K8/K16.

W15:
- `ANCHOR_PRESERVING_FAIL`;
- W14 anchor did not transfer to flattened typed state.

W16:
- 4/4 `CONTEXTUAL_STATE_CONTAMINATION`;
- adding orthogonal typed metadata to one A13 sequence destroyed intent token geometry before scoring.

W17:
- zero-parameter field isolation massively repaired diagnosis;
- but whole typed system remained weak;
- frozen verdict `FIELD_ISOLATED_TYPED_FAIL`.

W18:
- exact authority head `37bc9befa2e13a6adf08f45ccabe547e1609d5be`;
- authority run `36209370261`;
- merged main `5a3b158404cb444121dc1aace9ac3e34ae452668`;
- frozen outcome **`TYPED_COMPOSITION_UNRESOLVED`**;
- CR/CS/CT/CU all `W18_REFERENCE_INADEQUATE`.

W18 pooled HIRA:
- FIELD_DIRECT overall 45.31%;
- HARD composition 69.11%;
- SOFT composition 69.38%;
- conditioned on correct severity, response/risk = 100%;
- conditioned on correct severity+confidence, review/urgency = 100%.

W18 reference:
- MiniLM intent K4/K8/K16 95.31/74.22/49.22%;
- severity 64.58%;
- confidence 87.76%.

W18 HIRA atomic:
- intent 92.19/73.44/58.59%;
- severity 66.41%;
- confidence 90.89%;
- joint S+C 60.68%.

Therefore W18 does not authorize typed-kernel integration.

## 3. W19 scientific question

Are severity/confidence difficult because W18 treated intrinsically ordered latent variables as flat categorical semantic classes?

W19 compares:
1. matched flat categorical extraction;
2. cumulative ordinal-threshold extraction.

No training.

## 4. Fresh W19 authority

Fresh domains:
- CV municipal noise-impact reporting — seed 381101;
- CW appliance service-impact triage — 381107;
- CX campus facility incident assessment — 381119;
- CY neighborhood mobility disruption reporting — 381131.

Each:
- 4 severity classes;
- 3 confidence classes;
- 6 wording variants per S×C cell;
- 72 cases/domain;
- 288 total.

Each logical case has exactly:
- severity semantic sequence;
- confidence semantic sequence.

One batched A13 invocation/case.
Exactly two encoded sequences/case.

No W18 exact field text/definition reuse.

Exposure currently:
**NONE.**

## 5. Flat categorical baseline

Severity:
- four natural D0/D1/D2 definitions;
- exact W9 symmetric bidirectional MaxSim;
- mean across three views;
- frozen scale;
- softmax over 4.

Confidence:
- three corresponding classes;
- same operator.

## 6. Cumulative ordinal thresholds

Severity thresholds:
- T_S1: exceeds minimal/slight impact;
- T_S2: at least substantial/high;
- T_S3: at least extreme/critical.

Each:
- FALSE/TRUE natural option definitions;
- D0/D1/D2;
- exact W9 semantic operator;
- no learned threshold.

Hard severity:
`S_hat = sum_j 1[T_Sj=true]`.

Valid monotone patterns:
- 000;
- 100;
- 110;
- 111.

Confidence certainty order:
- 0 uncertain;
- 1 provisional;
- 2 verified.

Thresholds:
- T_C1: at least preliminary credible support;
- T_C2: fully confirmed/corroborated.

Hard confidence:
`C_hat = 1[T_C1=true] + 1[T_C2=true]`.

Valid patterns:
- 00;
- 10;
- 11.

No post-hoc monotonic repair.

## 7. Soft ordinal distribution

Severity:
- p0=1-q1
- p1=q1(1-q2)
- p2=q1 q2 (1-q3)
- p3=q1 q2 q3

Confidence:
- p0=1-q1
- p1=q1(1-q2)
- p2=q1 q2

Descriptive only.
No learned calibration.

## 8. Pinned reference

MiniLM:
- `sentence-transformers/all-MiniLM-L6-v2`;
- revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- weight SHA `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.

Reference evaluates flat and ordinal threshold interfaces independently.
Reference never enters HIRA.

## 9. Frozen gates

Reference adequacy/domain:
- severity ordinal hard >=.90;
- confidence ordinal hard >=.90;
- severity monotonicity >=.95;
- confidence monotonicity >=.97.

HIRA ordinal adequacy:
- severity >=.82;
- confidence >=.85;
- joint S+C >=.72;
- severity monotonicity >=.92;
- confidence monotonicity >=.95.

Causal gain:
- ordinal severity >= flat severity +.12;
- severity mean absolute ordinal error improves >=.20;
- ordinal confidence >= max(.85, flat-.03);
- ordinal joint >= flat joint +.10.

## 10. Frozen classes

`ORDINAL_LATENT_INTERFACE_LIMIT`:
reference + HIRA ordinal + causal gains all pass.

`LATENT_AXIS_EXTRACTION_LIMIT`:
reference passes but HIRA ordinal adequacy fails.

`FLAT_LATENT_INTERFACE_ADEQUATE`:
reference passes and flat HIRA already strong, with ordinal joint gain <.08.

Reference failure:
`W19_REFERENCE_INADEQUATE`.

Otherwise:
`LATENT_AXIS_UNRESOLVED`.

Cross-domain stable:
same non-unresolved/non-reference-inadequate class >=3/4 ->
`STABLE_LATENT_AXIS_LOCALIZATION`.

## 11. Required metrics

Flat:
- severity/confidence top1/MRR/margin;
- joint S+C.

Ordinal:
- every threshold binary top1/MRR/margin;
- hard decoded accuracy;
- soft top1;
- monotonic pattern rate;
- joint S+C;
- probability mass error.

Error anatomy:
- adjacent/non-adjacent error rate;
- MAE in ordinal levels;
- flat->ordinal wrong-to-right;
- flat->ordinal right-to-wrong.

Integrity:
- one batched A13 invocation/case;
- exactly two sequences;
- exact A13/W9/reference hashes;
- zero training;
- no W18 rows;
- no Banking77;
- no typed final/test;
- campaign cells zero.

## 12. Authorization boundary

Stable ORDINAL_LATENT_INTERFACE_LIMIT:
- W20 may test ordinal isolated latent extraction + deterministic typed kernel on wholly fresh TRAIN/DEV/dual-CONFIRM;
- retain flat latent control;
- no K32/K64.

Stable LATENT_AXIS_EXTRACTION_LIMIT:
- no symbolic production integration;
- diagnose latent encoder/projection geometry.

Flat adequate:
- do not replace categorical interface just because ordinal is cleaner.

Mixed/unresolved/reference inadequate:
- no rescue integration.

## 13. Forbidden evidence

Never reuse:
- CR/CS/CT/CU;
- CK-CQ;
- CG-CJ;
- all older exposed authorities;
- Banking77 0-799;
- typed final/test;
- public campaign cells.

## 14. Current execution state

Completed:
- W18 authority frozen and closure merged;
- issue #125 preregistered before W19 exposure;
- branch created from exact W18 main;
- all W19 equations/gates/fresh seeds frozen in issue #125;
- this handoff created before exposure.

Exposure:
**NONE**.

No CV/CW/CX/CY A13 cache.
No W19 MiniLM scores.
No W19 classification.

Next:
1. implement fresh balanced latent-axis authority;
2. implement isolated two-sequence cache;
3. implement flat and threshold schemas;
4. implement HIRA flat/ordinal evaluator;
5. implement MiniLM matched reference;
6. implement frozen classifiers/outcome;
7. add contracts + pre-data workflow;
8. exact-head unit + repo CI;
9. only then enable authority;
10. freeze authoritative result here;
11. merge only clean closure.

A future AI must update this file after every meaningful W19 session.
