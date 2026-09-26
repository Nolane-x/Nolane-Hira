# R8-W18 handoff — latent-field extraction vs deterministic typed composition

Status: **CLOSED DIAGNOSTIC. Frozen outcome: `TYPED_COMPOSITION_UNRESOLVED`; all four domains are `W18_REFERENCE_INADEQUATE`.**

Issue: #123

Branch:
`feat/r8-w18-latent-typed-composition`

Base main:
`ad3de4cdbebdada013d12f201a433011af912c18`

Read first when restoring:
1. this file;
2. `research/R8-W17-HANDOFF.md`;
3. `research/R8-W16-HANDOFF.md`;
4. `research/R8-W15-HANDOFF.md`;
5. issue #123.

## 0. Project identity

Nolane HIRA is a compact non-autoregressive typed decision engine, not a next-token LLM.

Long-term target:
- state encoded once/logically compiled once;
- dynamic semantic schemas;
- changing/high-cardinality candidate sets;
- typed choice / score / noul primitives;
- strong semantic transfer;
- explicit reliability/calibration;
- small local-friendly trainable footprint;
- sealed fresh authorities and immutable negative results;
- eventual external/public validation only after internal replication.

Scientific rules:
- gates freeze before exposure;
- exposed rows are permanently forbidden;
- PARTIAL/FAIL is never promoted because it is close;
- no post-hoc candidate additions;
- diagnostic references never become HIRA candidates;
- internal evidence is not broad external superiority.

## 1. Frozen semantic base

A13:
- model `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length 256;
- fully frozen.

W9 semantic projection:
- 256 -> 128 bias-free projection;
- scorer SHA `8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102`;
- source run `36122220588`;
- source freeze artifact `10858424139`.

W18 trainable parameter count:
**0**.

HIRACore is not required for the primary W18 diagnostic.

## 2. Decisive history entering W18

W5:
late interaction + candidate-relative salience/competition rescued semantic binding after simple pooled/capacity probes failed.

W6-W7:
typed state-once execution, reliability calibration and high-K decomposition were built, but fresh-domain high-K generalization remained unstable.

W7c:
public Banking77 transfer was essentially absent.

W8:
stable low-K `GENERAL_SEMANTIC_TRANSFER_LIMIT`.

W9:
small post-projection semantic bridges failed; full projection semantic retuning was strongest but not a rescue.

W10-W12:
candidate-independent semantic geometry was often much stronger than the production path; no stable micro-stage or scalar-confidence router explained all domains.

W13:
three-view semantic ensemble was strong, but strict 3/3 agreement was too sparse at K16.

W14:
major stable result:
- outcome `STABLE_CONTINUOUS_RELIABILITY_LOCALIZATION`;
- target `CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE`;
- E 94.92 / 90.23 / 84.77% K4/K8/K16;
- production 66.80 / 53.91 / 40.23%.

W15:
six-scalar anchor-preserving residual failed because W14 anchor strength did not transfer to the W15 typed regime:
`ANCHOR_PRESERVING_FAIL`.

W16:
matched renderings localized the transfer gap 4/4:
`CONTEXTUAL_STATE_CONTAMINATION`.
Full flattened typed state contaminates intent token representations inside A13 before scoring.

W17:
field-isolated A13 representation tested on fresh typed domains.
Exact authority head:
`3c570b8b9faa16c6677b2a4a09d3ac33bbc37674`.

Authority run:
`36206591909`.

Frozen verdict:
**`FIELD_ISOLATED_TYPED_FAIL`**.

But W17 has a strong causal positive:
CP diagnosis:
- FULL K4/K8/K16 53.13/34.38/9.38%;
- isolated 90.63/75.00/56.25%.

CQ:
- FULL 43.75/28.13/15.63%;
- isolated 90.63/50.00/53.13%.

Thus pre-context field separation repairs much of direct intent semantics.

Whole typed failure:
- response ~43–48%;
- review ~52–55%;
- urgency ~18%;
- overall ~51%.

W17 interpretation:
field-isolated representation is strongly supported for direct field-semantic decisions, but direct semantic similarity is weak for deterministic/compositional typed primitives.

## 3. W18 scientific question

Can we separate:
1. atomic semantic extraction of latent fields;
2. deterministic typed composition over those extracted values?

W18 tests whether the weak response/review/urgency paths come from asking semantic similarity to implement a known deterministic relation.

No training occurs.

## 4. Fresh W18 domains

Wholly fresh from W5-W17:

- CR condominium package-room access services — seed 371101;
- CS regional farm-share membership services — seed 371107;
- CT youth sports league registration services — seed 371119;
- CU household smart-meter support services — seed 371131.

Each:
- 4 subjects;
- 4 actions;
- 16 intents;
- 6 state variants/intent;
- 96 cases;
- diagnosis K4/K8/K16 exactly 32/32/32;
- 5 typed decisions/case.

Total:
- 384 cases;
- 1,920 typed decisions.

All CR/CS/CT/CU become permanently exposed after first eligible empirical cache.

## 5. Isolated fields

Each case has exactly three semantic fields:
- intent;
- severity;
- confidence.

They are encoded in one batched A13 invocation as three separate sequences.

No full flattened semantic representation enters the primary W18 diagnostic.

Fresh W18 severity/confidence wording must not literally expose the canonical latent class as a bare structured value.

## 6. Atomic latent schemas

### Intent

16 domain-specific intents.

Each option has D0/D1/D2 natural definitions.

### Severity

Four latent categories:
- S0 minimal;
- S1 moderate;
- S2 high;
- S3 critical.

Each has D0/D1/D2 frozen natural definitions distinct from final response/risk wording.

### Confidence

Three latent categories:
- C0 verified;
- C1 provisional;
- C2 uncertain.

Each has D0/D1/D2 frozen natural definitions.

## 7. Atomic semantic operator

For each field and each D0/D1/D2 schema view:
- exact W9 projection;
- L2 normalize;
- bidirectional symmetric token MaxSim;
- arithmetic directional mean;
- arithmetic mean across D0/D1/D2;
- multiply by frozen scorer scale.

Softmax gives:
- p_I;
- p_S;
- p_C.

No labels enter inference.

## 8. FIELD_DIRECT baseline

Exact W17 style:
- diagnosis: intent field vs diagnosis options;
- response: severity field vs response options;
- needs_review: severity+confidence token union vs review options;
- risk: severity field vs risk options;
- urgency: severity+confidence token union vs urgency options.

No typed symbolic composition.

## 9. LATENT_COMPOSED_HARD

Use argmax latent predictions:
- diagnosis = argmax p_I;
- response = argmax p_S;
- risk = argmax p_S;
- review = true iff severity=critical OR confidence=uncertain;
- urgency = min(3, severity + 1[confidence=uncertain]).

No learned rule.

## 10. LATENT_COMPOSED_SOFT

Diagnosis:
p_diag = p_I.

Response/risk:
p_response = p_risk = p_S.

Review:
p_true = 1 - (1-p_S[critical])*(1-p_C[uncertain]).
p_false = 1-p_true.

Urgency:
for all s,c:
u=min(3,s+1[c=uncertain]);
p_urgency[u] += p_S[s]*p_C[c].

Independence is frozen before exposure.
No learned weight/threshold.

## 11. ORACLE sanity

Oracle uses synthetic known latent I/S/C identities only to verify deterministic typed equations.

Oracle hard-label accuracy must be exactly 1.0.

Oracle:
- is never a HIRA candidate;
- cannot choose a mechanism;
- cannot affect thresholds.

## 12. Pinned reference

MiniLM:
- `sentence-transformers/all-MiniLM-L6-v2`;
- revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- weight SHA `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.

Reference measures D0:
- intent extraction;
- severity extraction;
- confidence extraction.

Never enters HIRA or composition.

## 13. Frozen adequacy/gates

Reference adequacy/domain:
- oracle typed hard accuracy = 1.0;
- MiniLM intent K4 >=.80;
- MiniLM intent K16 >=.60;
- MiniLM severity >=.90;
- MiniLM confidence >=.90.

HIRA atomic extraction:
- intent K4 >=.82;
- intent K16 >=.50;
- severity >=.80;
- confidence >=.80;
- joint S+C correct >=.68.

One-field composition rescue:
- atomic extraction adequate;
- HARD response >=.80;
- HARD risk >=.80;
- response >= FIELD_DIRECT +.20;
- risk >= FIELD_DIRECT +.10.

Multi-field composition rescue:
- atomic adequate;
- HARD review >=.80;
- HARD urgency >=.75;
- review >= FIELD_DIRECT+.15;
- urgency >= FIELD_DIRECT+.30;
- conditioned on S+C argmax both correct:
  review=1.0;
  urgency=1.0.

Soft viability:
- SOFT overall >= HARD-.03;
- SOFT non-diagnosis >= HARD-.03;
- probability mass error <=1e-6.

## 14. Frozen classifications

`TYPED_COMPOSITION_LIMIT`:
reference + atomic + one-field + multi-field all pass.

`ONE_FIELD_MAPPING_LIMIT`:
reference+atomic pass; one-field passes; multi-field fails.

`MULTI_FIELD_COMPOSITION_LIMIT`:
reference+atomic pass; multi-field passes; one-field fails.

`LATENT_FIELD_EXTRACTION_LIMIT`:
reference passes; HIRA atomic extraction fails.

Reference failure:
`W18_REFERENCE_INADEQUATE`.

Otherwise:
`TYPED_COMPOSITION_UNRESOLVED`.

Cross-domain:
same non-unresolved/non-reference-inadequate classification >=3/4 ->
`STABLE_TYPED_COMPOSITION_LOCALIZATION`.

Two incompatible classes >=2 each ->
`MIXED_TYPED_COMPOSITION_LOCALIZATION`.

Else:
`TYPED_COMPOSITION_UNRESOLVED`.

## 15. Required metrics

Atomic:
- intent per-K top1/top5/MRR/margin;
- severity top1/MRR/margin;
- confidence top1/MRR/margin;
- joint S+C top1 correctness.

Typed paths:
- overall;
- primitive;
- question;
- non-diagnosis;
- Brier where probability exists;
- score MAE;
- probability mass.

Composition:
- hard-vs-soft prediction identity;
- direct wrong→composed right;
- direct right→composed wrong;
- correctness conditioned on S correct;
- correctness conditioned on S+C correct;
- field error propagation.

Integrity:
- one batched A13 invocation / logical case;
- exactly three isolated sequences;
- exact model/scorer/reference hashes;
- no training;
- no W17 data;
- no Banking77;
- no typed final/test;
- campaign cells zero.

## 16. Authorization boundary

W18 is diagnostic only.

Stable TYPED_COMPOSITION_LIMIT:
- W19 may integrate isolated latent extraction + deterministic typed kernel on wholly fresh TRAIN/DEV/dual-CONFIRM.

Stable ONE_FIELD_MAPPING_LIMIT:
- integrate only deterministic one-field mapping first.

Stable MULTI_FIELD_COMPOSITION_LIMIT:
- integrate only explicit multi-field composition.

Stable LATENT_FIELD_EXTRACTION_LIMIT:
- no symbolic production integration;
- diagnose latent schema representation/A13 geometry.

Mixed/unresolved/reference inadequate:
- no rescue training.

Never open K32/K64 from W18 alone.

## 17. Permanent forbidden evidence

Includes all previous exposed rows through:
- W16 CG/CH/CI/CJ;
- W17 CK/CL/CM/CN/CO/CP/CQ;
- Banking77 0-799;
- typed final/test;
- public campaign cells.

Never use them for W18 tuning.

## 18. Prior-art boundary

See `research/R8-W18-PRIOR-ART.md`.

Prior art motivates architectural separation only and cannot change frozen gates after exposure.

## 19. Current state

Completed:
- W17 merged main `ad3de4cdbebdada013d12f201a433011af912c18`;
- issue #123 preregistered before exposure;
- branch created from exact post-W17 main;
- this handoff created before W18 exposure.

Exposure:
**NONE.**

No CR/CS/CT/CU A13 cache exists.
No W18 reference score exists.
No W18 metric/classification exists.

Next:
1. freeze prior-art note;
2. implement fresh authority generator;
3. implement isolated field + latent-schema cache;
4. implement atomic extractor;
5. implement FIELD_DIRECT/HARD/SOFT/ORACLE evaluator;
6. implement frozen classifiers/outcome;
7. implement pinned MiniLM reference;
8. add tests + unit workflow;
9. run exact-head unit + repo CI;
10. only then enable authority;
11. freeze result here before merge.

A future AI must update this file after every meaningful session.


---

## 20. Authoritative W18 closure

Exact verdict-bearing head:
`37bc9befa2e13a6adf08f45ccabe547e1609d5be`

Authority run:
`36209370261`

All authority jobs PASS:
- unit;
- exact W9 upstream provenance;
- fresh CR/CS/CT/CU isolated-field cache;
- frozen HIRA atomic extractor;
- FIELD_DIRECT / HARD / SOFT / ORACLE evaluator;
- pinned MiniLM reference;
- frozen per-domain classifiers/outcome.

### Artifacts

Frozen W9 checkpoint bundle:
- artifact `10895325285`;
- digest `sha256:6ffecd42667a917be4660d7b709ac3d21e8e15be89e8006af798f201ad5ffa3a`.

Fresh W18 cache:
- artifact `10894844785`;
- digest `sha256:929a248e2ecf3891080fdf7cfbf5bd891d4c299b0cbfa23483159f875051ea30`.

Authoritative W18 audit:
- artifact `10894249286`;
- digest `sha256:b9ad1c8ebb6bad81d69cbbba1e233cc8bda0931f2cdc82095d30d66f5532d370`.

Integrity:
- 384 cases;
- 1,920 typed decisions;
- one logical state compile/case;
- one batched A13 invocation/case;
- exactly three encoded isolated sequences/case: intent, severity, confidence;
- trainable parameters = 0;
- training = false;
- selection = false;
- ORACLE hard typed accuracy = 1.0 on CR/CS/CT/CU;
- probability mass max error <= `4.440892098500626e-16`;
- no W17 rows;
- no Banking77;
- no typed final/test;
- campaign cells = 0.

## 21. Frozen verdict

Overall:

**`TYPED_COMPOSITION_UNRESOLVED`**

Stable classification:
`null`.

Per-domain:
- CR -> `W18_REFERENCE_INADEQUATE`;
- CS -> `W18_REFERENCE_INADEQUATE`;
- CT -> `W18_REFERENCE_INADEQUATE`;
- CU -> `W18_REFERENCE_INADEQUATE`.

Classification counts:
- `W18_REFERENCE_INADEQUATE`: 4/4.

The frozen protocol intentionally does **not** convert four reference-inadequate domains into a composition localization.

No W19 production integration is authorized by W18.

## 22. Why the reference adequacy gate failed

Pinned MiniLM pooled D0 reference:

Intent:
- K4 **95.313%**;
- K8 **74.219%**;
- K16 **49.219%**.

Severity:
- **64.583%**.

Confidence:
- **87.760%**.

Frozen reference requirements were:
- intent K4 >=80%;
- intent K16 >=60%;
- severity >=90%;
- confidence >=90%.

Thus:
- K4 intent passes strongly;
- K16 intent misses by 10.78 pp;
- severity misses by 25.42 pp;
- confidence misses by 2.24 pp.

Every fresh domain independently fails reference adequacy.

Per-domain MiniLM:
- CR intent K4/K16 96.88/53.13%, severity 62.50%, confidence 86.46%;
- CS 90.63/53.13%, severity 66.67%, confidence 87.50%;
- CT 96.88/37.50%, severity 64.58%, confidence 86.46%;
- CU 96.88/53.13%, severity 64.58%, confidence 90.63%.

Therefore the authority is not permitted to decide whether deterministic composition itself is the stable bottleneck.

## 23. HIRA atomic extraction anatomy

Pooled HIRA atomic extraction:

Intent:
- K4 **92.188%**;
- K8 **73.438%**;
- K16 **58.594%**.

Severity:
- **66.406%**.

Confidence:
- **90.885%**.

Joint severity+confidence top1:
- **60.677%**.

Frozen HIRA atomic requirements:
- intent K4 >=82%;
- intent K16 >=50%;
- severity >=80%;
- confidence >=80%;
- joint S+C >=68%.

So HIRA:
- passes intent K4;
- passes intent K16;
- fails severity;
- passes confidence;
- fails joint S+C.

The atomic failure pattern is highly aligned with the reference weakness in the severity/latent schema rather than being a uniquely HIRA-only collapse.

## 24. Deterministic-composition signal

Although W18 cannot promote a composition conclusion, the descriptive causal signal is large.

Pooled FIELD_DIRECT:
- overall **45.313%**;
- non-diagnosis **37.956%**;
- diagnosis **74.740%**;
- response **40.104%**;
- needs_review **36.719%**;
- risk **46.875%**;
- urgency **28.125%**.

Pooled LATENT_COMPOSED_HARD:
- overall **69.115%**;
- non-diagnosis **67.708%**;
- diagnosis **74.740%**;
- response **66.406%**;
- needs_review **74.740%**;
- risk **66.406%**;
- urgency **63.281%**.

Pooled LATENT_COMPOSED_SOFT:
- overall **69.375%**;
- non-diagnosis **68.034%**;
- diagnosis **74.740%**;
- response **66.406%**;
- needs_review **75.260%**;
- risk **66.406%**;
- urgency **64.063%**.

Soft composition:
- hard-vs-soft top1 identity **99.271%**;
- hard Brier improves from HARD `0.6177` to SOFT `0.4204`;
- soft Brier improves from HARD `0.4989` to SOFT `0.2868`;
- score MAE improves from HARD `0.4926` to SOFT `0.4643`;
- probability mass max error <= `4.44e-16`.

Composition transitions from FIELD_DIRECT -> HARD:
- response wrong->right: 101/384 = **26.30%**; right->wrong = 0%;
- risk wrong->right: 88/384 = **22.92%**; right->wrong = 3.39%;
- needs_review wrong->right: 195/384 = **50.78%**; right->wrong = 12.76%;
- urgency wrong->right: 160/384 = **41.67%**; right->wrong = 6.51%.

Critically, conditioned on correct extracted latent values:
- when severity argmax is correct, HARD response accuracy = **100%**;
- when severity argmax is correct, HARD risk accuracy = **100%**;
- when both severity+confidence argmax are correct, HARD needs_review accuracy = **100%**;
- when both severity+confidence argmax are correct, HARD urgency accuracy = **100%**.

This proves the deterministic equations are internally correct and that remaining composed errors propagate from latent extraction errors.

However, because reference adequacy fails, this signal is descriptive and hypothesis-generating only.

## 25. Scientific interpretation

W18 rules out a premature conclusion that "typed composition is now solved".

The strongest defensible interpretation is:

> On the fresh W18 authority, deterministic typed composition behaves exactly as intended once the latent variables are correct, and it substantially outperforms direct semantic matching. But the newly introduced severity/confidence latent schemas are themselves insufficiently recoverable even for the pinned reference, so W18 cannot distinguish a genuine composition bottleneck from a latent-schema/wording difficulty artifact under its frozen gates.

This creates a new, narrower question:

> Can atomic severity/confidence semantics be represented in a fresh authority whose latent classes are independently recoverable, without trivial exact-label leakage?

That question must be answered before symbolic composition can be promoted into production.

## 26. Permanent exposed evidence after W18

CR/CS/CT/CU are permanently exposed.

Never reuse them for:
- latent schema wording selection;
- severity/confidence definition design;
- composition-rule selection;
- threshold/gate tuning;
- A13/projection tuning;
- checkpoint/seed selection;
- calibration;
- production promotion.

All prior exposed evidence remains forbidden.

## 27. Authorized continuation

W18 does **not** authorize W19 production integration.

A next phase must remain diagnostic and focus on **latent schema recoverability** on wholly fresh data.

A valid continuation should:
1. keep field isolation;
2. keep zero training in the primary diagnostic;
3. use fresh intent/severity/confidence wording;
4. separate lexical paraphrase difficulty from class-boundary difficulty;
5. include at least one non-HIRA pinned reference adequacy ceiling;
6. test whether severity/confidence classes are semantically identifiable before any typed composition gate;
7. keep composition equations frozen only as a descriptive downstream probe;
8. use wholly fresh domains and permanently forbid them after exposure.

It must not:
- rewrite CR/CS/CT/CU latent definitions;
- lower W18 reference gates post hoc;
- call the 69% composed result a rescue;
- integrate the deterministic kernel into production yet;
- open K32/K64;
- use MiniLM as HIRA supervision.

A future AI should read W18 first, then W17/W16/W15.
