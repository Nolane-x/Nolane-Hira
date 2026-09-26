# R8-W19 handoff — latent ordinal-axis recoverability decomposition

Status: **CLOSED DIAGNOSTIC. Frozen outcome: `LATENT_AXIS_UNRESOLVED`; CV/CW/CX/CY are all `W19_REFERENCE_INADEQUATE`.**

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


---

## 15. Authoritative W19 closure

Exact verdict-bearing head:
`28d5a004348e62e5ee6d8e87bef9a698286a18f2`

Authority run:
`36214486773`

All authority jobs PASS:
- frozen unit/contracts/freshness;
- exact W9 upstream provenance;
- fresh CV/CW/CX/CY cache;
- pinned MiniLM matched multiview reference;
- frozen HIRA flat/ordinal evaluation;
- frozen per-domain classifier/outcome.

Artifacts:
- frozen W9 bundle: `10897515042`;
  digest `sha256:70674e6475e496416620c2cd5913bf484bebb8d1b0fd42fc53dbd5a7672c01df`;
- fresh W19 cache: `10896868113`;
  digest `sha256:a60f5aa62282cd08bf6db2f8b889949a07a18eff9b88a1257f12e09c1cf29806`;
- authoritative W19 audit: `10896788580`;
  digest `sha256:84c1b420c2055a4994ddabedbc13d58b9abc3cc624b6624e786340d1fe2f1c19`.

Integrity:
- 288 cases;
- 72/domain;
- severity balance 18/class/domain;
- confidence balance 24/class/domain;
- one logical state compile/case;
- one batched A13 state invocation/case;
- exactly two encoded isolated sequences/case;
- trainable parameters = 0;
- training = false;
- selection = false;
- probability-mass max error <= `2.220446049250313e-16`;
- no W18 rows;
- no Banking77;
- no typed final/test;
- campaign cells = 0.

## 16. Frozen verdict

Overall:

**`LATENT_AXIS_UNRESOLVED`**

Stable classification:
`null`.

Per-domain:
- CV -> `W19_REFERENCE_INADEQUATE`;
- CW -> `W19_REFERENCE_INADEQUATE`;
- CX -> `W19_REFERENCE_INADEQUATE`;
- CY -> `W19_REFERENCE_INADEQUATE`.

Classification counts:
- `W19_REFERENCE_INADEQUATE`: 4/4.

No production W20 typed-kernel integration is authorized by W19.

## 17. Pooled HIRA result

Flat categorical:
- severity top1 **50.000%**;
- severity MAE **0.7396**;
- confidence top1 **48.611%**;
- confidence MAE **0.7500**;
- joint severity+confidence **24.306%**.

Cumulative ordinal:
- severity hard **23.958%**;
- severity soft top1 **33.333%**;
- severity MAE **1.2188**;
- severity monotonicity **81.250%**;
- confidence hard **40.278%**;
- confidence soft top1 **41.667%**;
- confidence MAE **0.8333**;
- confidence monotonicity **98.611%**;
- joint severity+confidence **9.722%**.

Threshold anatomy:
- severity T1 **79.167%**;
- severity T2 **52.083%**;
- severity T3 **38.542%**;
- confidence T1 **75.000%**;
- confidence T2 **41.667%**.

Transitions from flat -> ordinal:
- severity wrong->right **12.500%**;
- severity right->wrong **38.542%**;
- confidence wrong->right **6.944%**;
- confidence right->wrong **15.278%**.

Thus the frozen ordinal candidate is not a HIRA rescue on this authority.

## 18. Pinned MiniLM reference result

Flat:
- severity **39.583%**;
- confidence **79.167%**;
- joint **31.944%**.

Ordinal:
- severity hard **31.250%**;
- severity soft top1 **25.000%**;
- severity monotonicity **76.042%**;
- confidence hard **50.000%**;
- confidence soft top1 **33.333%**;
- confidence monotonicity **91.667%**;
- joint **16.667%**.

Reference threshold anatomy:
- severity T1 **78.125%**;
- severity T2 **50.000%**;
- severity T3 **66.667%**;
- confidence T1 **90.278%**;
- confidence T2 **51.389%**.

Frozen reference gates required:
- ordinal severity >=90%;
- ordinal confidence >=90%;
- severity monotonicity >=95%;
- confidence monotonicity >=97%.

The reference misses every adequacy gate by a large margin. Therefore W19 cannot causally classify flat-vs-ordinal interface choice for HIRA.

## 19. Scientific interpretation

The strongest defensible W19 conclusion is:

> Cumulative threshold decomposition does not rescue the fresh latent fields under the frozen W19 interface. The failure is not HIRA-specific: the pinned MiniLM reference also fails ordinal decoding and monotonicity badly, especially at the upper severity/certainty boundaries. Therefore the experiment does not establish an ordinal-interface limitation or a latent-axis extraction limitation in HIRA.

A descriptive pattern is still useful:
- lower thresholds are materially easier than upper thresholds;
- confidence T1 is much easier than confidence T2;
- severity T1 is much easier than severity T2/T3;
- HIRA confidence threshold predictions are nearly monotone despite low decoded accuracy;
- flat extraction is stronger than the frozen ordinal candidate on HIRA;
- both HIRA and reference show that the current cumulative boundary semantics are not independently adequate.

These observations are hypothesis-generating only.

## 20. Permanent exposure after W19

CV/CW/CX/CY are permanently exposed.

Never reuse them for:
- threshold wording selection;
- flat/ordinal schema redesign;
- gate tuning;
- A13/W9 tuning;
- reference selection;
- calibration;
- production promotion.

All older forbidden evidence remains forbidden.

## 21. Authorized continuation

Because W19 is reference-inadequate/unresolved:
- do **not** integrate ordinal latent extraction into production;
- do **not** integrate the deterministic typed kernel on the basis of W19;
- do **not** lower the frozen reference gates;
- do **not** rewrite CV/CW/CX/CY thresholds and rerun;
- do **not** open K32/K64 from this result.

A next phase must remain diagnostic on wholly fresh domains and first establish an independently adequate latent-boundary interface.

A defensible next question is narrower than W19:

> Can ordered latent values be recovered through direct pairwise boundary comparisons whose positive/negative alternatives are locally contrastive and independently reference-adequate, without using cumulative-threshold count decoding?

Any such phase must:
1. preregister fresh domains and wording before exposure;
2. retain flat W19-style control on fresh data;
3. retain zero training for the primary diagnostic;
4. include a pinned independent reference adequacy gate;
5. separate boundary-identifiability from global ordinal decoding;
6. keep CV/CW/CX/CY permanently forbidden;
7. authorize production integration only after stable fresh-domain localization.

A future AI should read this frozen W19 closure before opening the next diagnostic.
