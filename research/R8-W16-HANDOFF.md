# R8-W16 handoff — matched semantic-regime transfer decomposition

Status: **CLOSED DIAGNOSTIC. Frozen outcome: `STABLE_REGIME_TRANSFER_LOCALIZATION`; stable target: `CONTEXTUAL_STATE_CONTAMINATION`.**

Issue: #119

Branch:
`feat/r8-w16-regime-transfer-decomposition`

Base main:
`d23006f8074195ed6eba404427fb82f09aac7ea7`

Read first when restoring:
1. this file;
2. `research/R8-W15-HANDOFF.md`;
3. `research/R8-W14-HANDOFF.md`;
4. `research/R8-W13-HANDOFF.md`;
5. issue #119.

## 0. Project identity

Nolane HIRA is a compact non-autoregressive typed decision engine, not a next-token LLM.

Long-term target:
- state encoded once;
- dynamic semantic schemas;
- changing/high-cardinality candidate sets;
- typed choice / score / noul;
- small local-friendly trainable footprint;
- transferable semantic binding;
- robust high-K decisions;
- calibrated/reliable probabilities;
- sealed fresh authorities and immutable negative results.

Research discipline:
- no gate moves after exposure;
- exposed rows are permanently forbidden;
- partial/fail is never upgraded because it is close;
- internal fresh authorities are not broad external superiority.

## 1. Frozen base

A13:
- `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length 256;
- frozen.

W9 semantic projection:
- scorer SHA `8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102`;
- freeze run `36122220588`;
- artifact `10858424139`.

HIRACore:
- remains frozen;
- not required for the primary W16 diagnostic.

No W16 trainable parameter exists.

## 2. Decisive history entering W16

W5 established late-interaction competitive semantic binding after simple pooled/capacity probes failed.

W6-W7 established typed execution, reliability, and several internal high-K mechanisms, but fresh-domain high-K rescue remained unstable.

W7c showed Banking77 external transfer was essentially absent.

W8 localized a low-K general semantic transfer limit.

W9 found full projection semantic retuning was stronger than tiny post-projection bridges but still failed rescue.

W10 showed W9 semantic geometry could be strong before production transforms.

W11 could not localize one stable damaging production micro-stage.

W12 showed candidate-independent semantic anchor often exceeded production, but scalar margin did not define a stable router.

W13 showed exact 3/3 paraphrase agreement was high-quality but insufficiently supported at K16.

W14 produced the first stable 4/4 architecture-level semantic result:
- outcome `STABLE_CONTINUOUS_RELIABILITY_LOCALIZATION`;
- target `CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE`;
- E 94.92/90.23/84.77% at K4/K8/K16;
- production 66.80/53.91/40.23%.

W15 tested a six-scalar anchor-preserving residual production redesign and failed:
`ANCHOR_PRESERVING_FAIL`.

The deeper W15 observation:
- CE anchor 46.88/34.38/9.38%;
- CF anchor 40.63/31.25/12.50%.

Thus W14's strong E did not transfer into W15's typed regime.

Exact W15 empirical head:
`1bee96afc789641b56edf1b8061951042d233de1`

Authority:
`36150235281`

Authoritative CONFIRM artifact:
`10872390453`
digest `sha256:f5baa513c6d9c972321b95d9e255ac9177256e349e2c87b03d2eb52b08b56324`.

## 3. W16 scientific question

Why does the exact frozen W9 multiview semantic geometry remain extremely strong under W14-style fresh semantic classification yet collapse under W15-style typed cases?

Primary matched hypothesis:
W15 diagnosis state text appends orthogonal metadata:

`Reported impact is <severity>; evidence confidence is <confidence>.`

Diagnosis gold intent is independent of this metadata.

The semantic anchor is bidirectional:
- definition -> state MaxSim mean;
- state -> definition MaxSim mean;
- symmetric average.

Therefore irrelevant state-side tokens can directly dilute the S2D half.

W16 tests this hypothesis without training.

## 4. Fresh domains

Fresh and disjoint from W5-W15:
- CG municipal tree-permit administration — seed 351101;
- CH home-appliance warranty services — 351107;
- CI community-college equipment lending — 351121;
- CJ bicycle-share account support — 351133.

Each:
- 16 intents = 4 subjects × 4 actions;
- actions open / revise / close / check;
- 4 independent base-state variants per intent;
- D0/D1/D2 natural definitions;
- 64 bases/domain.

Total:
- 256 bases.

After first eligible authority, CG/CH/CI/CJ are permanently exposed.

## 5. Paired renderings

Candidate identity/order/gold/definitions are identical across renderings.

R0 BARE:
- intent-bearing state only.

R1 DECORATED_FULL:
- exact R0 text;
- append exact W15-style severity/confidence suffix;
- score all state content tokens.

R2 DECORATED_PREFIX_MASK:
- encode exact same R1 full string;
- score only state content-token positions belonging to original R0 prefix.

R2 is diagnostic only.

## 6. Frozen semantic operators

For each R and D0/D1/D2, exact W9 projection + L2 normalization.

D2S:
- definition token -> state token MaxSim;
- mean over definition tokens.

S2D:
- state token -> definition token MaxSim;
- mean over active state tokens.

SYM:
`0.5 * (D2S + S2D)`.

Multiview:
`E = mean(D0,D1,D2)`.

No question tokens.
No IDF.
No common-mode subtraction.
No salient-min.
No HIRACore.

## 7. Reference ceiling

Pinned MiniLM:
- `sentence-transformers/all-MiniLM-L6-v2`;
- revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- weight SHA `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.

R0/R1 D0 cosine only.
Reference never enters HIRA or classification selection.

## 8. Frozen adequacy

Primary-evaluable domain:
- MiniLM R0 K4 >= .75;
- R0 SYM E K4 >= .70;
- R0 SYM E K16 >= .50.

Otherwise:
`BASE_SEMANTIC_REGIME_DIFFICULTY`.

## 9. Frozen classifications

`EXTRA_STATE_TOKEN_DILUTION`:
- adequacy;
- R1 SYM <= R0-.12 at K4 and K16;
- R2 >= R1+.10 at K4 and K16;
- R2 >= R0-.05 at K4 and K16;
- D2S regression <=.05 at K4/K16;
- S2D regression >=.10 at K4/K16.

`CONTEXTUAL_STATE_CONTAMINATION`:
- adequacy;
- R1 <= R0-.12 at K4/K16;
- R2 recovery <.06 at K4 or K16;
- R2 remains <=R0-.08 at K4/K16.

`DECORATION_EFFECT_MIXED`:
- adequacy;
- R1 <=R0-.08 at K4/K16;
- neither stronger class passes.

`DECORATION_NOT_PRIMARY`:
- adequacy;
- abs(R1-R0)<.08 at K4/K16.

Otherwise:
`REGIME_TRANSFER_UNRESOLVED`.

Cross-domain stable:
same non-unresolved/non-inadequate class on >=3/4.

## 10. Required diagnostics

Per domain/rendering/operator/K:
- top1/top5/MRR;
- gold margin.

Matched transitions:
- R0 c->R1 w;
- R0 w->R1 c;
- R1 c->R2 w;
- R1 w->R2 c;
- R0 c->R2 w;
- rank deltas.

Directional attribution:
- D2S, S2D, SYM R1-R0;
- S2D share of SYM margin loss.

Token integrity:
- active token counts;
- prefix-token ID identity;
- state encoding counts.

Leakage:
- exact text freshness against W5-W15;
- no W15 CE/CF;
- no Banking77;
- no typed final/test;
- campaign cells 0.

## 11. Authorization boundary

W16 is diagnostic only.

Stable EXTRA_STATE_TOKEN_DILUTION:
- W17 may test primitive-relevant state views while preserving shared/state-once encoding.

Stable CONTEXTUAL_STATE_CONTAMINATION:
- W17 must test representation separation before scoring, not token masking only.

DECORATION_NOT_PRIMARY / base difficulty:
- investigate semantic granularity/task-family transfer.

Mixed/unresolved:
- no rescue training.

## 12. Permanent forbidden evidence

Never reuse:
- BN/BO/BP/BQ;
- BR/BS/BT/BU;
- BV/BW/BX/BY;
- BZ/CA/CB/CC/CD/CE/CF;
- Banking77 0-799;
- typed final/test;
- campaign cells.

## 13. Prior-art note

See `research/R8-W16-PRIOR-ART.md`.

Prior art is motivation only. It must not be used to alter frozen gates after exposure.

## 14. Current execution state

Completed:
- W15 merged main `d23006f8074195ed6eba404427fb82f09aac7ea7`;
- issue #119 created before W16 exposure;
- branch created from exact post-W15 main;
- W16 protocol/gates/renderings/operators frozen in issue #119;
- this handoff created before empirical exposure.

Exposure:
**NONE.**

No CG/CH/CI/CJ A13 cache exists.
No MiniLM W16 score exists.
No W16 classification exists.

Next:
1. freeze prior-art note;
2. implement fresh matched authority generator;
3. implement R0/R1/R2 state-once-per-rendering cache;
4. implement prefix-token identity contract;
5. implement directional D2S/S2D/SYM evaluator;
6. implement MiniLM reference;
7. implement classifiers/outcome;
8. add tests + pre-data unit workflow;
9. run exact-head unit + repo CI;
10. only then enable authority;
11. freeze result here before merge.

A future AI must update this file after every meaningful session.


## 15. Pre-data implementation update

Scientific exposure remains:

**NONE.**

No CG/CH/CI/CJ A13 cache has been materialized.
No W16 MiniLM score exists.
No W16 classification exists.

Implemented on `feat/r8-w16-regime-transfer-decomposition`:

### Fresh matched authority
- `src/nmd/regime_transfer_authority.py`;
- CG/CH/CI/CJ exactly as preregistered;
- 16 intents/domain = 4 subjects × 4 actions;
- 4 state variants/intent;
- nested K4/K8/K16;
- D0/D1/D2;
- identical candidate IDs/order/gold across all state renderings;
- deterministic W15-style severity/confidence suffix independent of diagnosis gold.

### Matched state cache
- `src/nmd/regime_transfer_cache.py`;
- R0 bare and R1 decorated are encoded together exactly once per base;
- exactly two encoded state texts/base;
- R2 performs **zero additional encoder calls**;
- R2 reuses contextual embeddings from R1 but truncates semantic scoring to the exact R0 prefix token IDs;
- R0 token IDs must equal the leading R1 content-token IDs exactly or cache construction fails;
- `prefix_token_identity_rate` must equal 1.0;
- schema D0/D1/D2 token artifacts are cached once and shared across R0/R1/R2;
- nested candidate identity is validated.

This construction separates:
1. extra-token aggregation dilution;
2. contextual representation contamination.

### Directional evaluator
- `src/nmd/regime_transfer_eval.py`;
- exact W9 projection;
- D2S definition->state MaxSim mean;
- S2D state->definition MaxSim mean;
- SYM = .5*(D2S+S2D);
- D0/D1/D2 arithmetic multiview ensemble;
- per-domain/rendering/operator/K top1/top5/MRR/margin;
- matched R0->R1, R1->R2 and R0->R2 transitions;
- directional margin/top1 attribution;
- active-token accounting;
- frozen classifier precedence and >=3/4 cross-domain outcome.

### Frozen reference evaluator
- `scripts/r8_w16_evaluate.py`;
- exact W9 freeze provenance;
- exact W9 semantic scorer SHA check;
- pinned MiniLM reference revision/SHA;
- reference only scores R0/R1 against D0 as adequacy ceiling;
- reference cannot enter HIRA.

### Fresh cache builder
- `scripts/r8_w16_build_cache.py`;
- exact A13 revision/SHA verification;
- exact-text freshness firewall through W15;
- no W14/W15 rows;
- no CE/CF;
- no Banking77;
- no typed final/test;
- campaign cells zero.

### Pre-data contracts
- `tests/test_regime_transfer.py`;
- fresh authority shape;
- paired D0/D1/D2 identity;
- exact R0/R1/R2 prefix-token contract;
- R2 zero-extra-encode contract;
- directional D2S/S2D/SYM equation;
- frozen `EXTRA_STATE_TOKEN_DILUTION` classifier fixture;
- >=3/4 stability rule.

### CI
- `.github/workflows/r8-w16-unit.yml`;
- compiles all W16 core/scripts;
- download-free W16 unit tests;
- no A13 materialization;
- no CG/CH/CI/CJ authority exposure.

Current implementation head:
`f078c94fbbc82729ba622283a348466fdab0e2bc`.

Validation currently in progress:
- W16 unit run `36202258480`;
- repository CI run `36202258442`.

Do not enable authority until both exact-head gates are PASS.

## 16. Remaining pre-exposure sequence

1. obtain exact-head W16 unit PASS;
2. obtain exact-head repository CI PASS on Python 3.10 and 3.12;
3. record run IDs and exact green head here and in issue #119;
4. add authority workflow only after the green gate;
5. authority chain:
   `unit -> exact W9 projection provenance + fresh CG-CJ cache -> frozen directional evaluator + MiniLM reference -> frozen classifications/outcome`;
6. after first CG-CJ A13 materialization, no scientific mutation is allowed;
7. freeze exact artifacts/metrics/verdict here;
8. merge only a clean frozen result.

A future AI must not treat current unit/CI execution as empirical evidence.


---

## 17. Authoritative W16 closure

Exact empirical authority head:
`af83c92e7f25c70f044abf38f9a64ff5503746d2`

Authority run:
`36202648911`

All jobs PASS:
- unit;
- exact W9 upstream provenance;
- fresh CG/CH/CI/CJ A13 cache;
- frozen D2S/S2D/SYM evaluator;
- pinned MiniLM reference;
- frozen classifier/outcome.

### Artifacts

Frozen W9 checkpoint bundle:
- artifact `10892338589`;
- digest `sha256:1b9ee5ab5ee442e00e5cec664dc24c08cea74aa67bbbdb2dc667180d8007c8db`.

Fresh W16 cache:
- artifact `10892383907`;
- digest `sha256:92056e45db26463cfc113496e6befd929beb1bd22723a18cea8cb1eeb3fc8bfd`;
- internal cache SHA `3607bc02528f87159ad3f63ab9796f6e2830a2c7a0c6c987af3034b3b11194af`.

Authoritative W16 audit:
- artifact `10893240677`;
- digest `sha256:3306a48b5c9081ab400a88d89ea74134ef9f162c1197470789255dccb07d174b`.

### Integrity

- 256 fresh bases;
- 2,304 schema views;
- 256 state-encoder batches;
- exactly 512 encoded state texts = R0 + R1 only;
- exactly 2 encoded state texts/base;
- R2 additional encoder calls = 0;
- prefix-token identity rate = 1.0;
- exact-text overlap with W5-W15 = [];
- no training;
- no W14 rows;
- no W15 rows;
- no W15 CE/CF rows;
- no Banking77 rows;
- no typed final/test rows;
- campaign cells = 0.

Cache identity:
- base ID SHA `ccbc4114be7a9444cb42108649df2c795ee1ed533f701b836b8d26300396c2a0`;
- case ID SHA `30c5a150f3c1b1fb52d277c4c4278b5644e6b9655bf2f636954fe8f626d6991c`;
- text-atom SHA `0b2b595e0d64c5f74201acf1bcf77df5e71c3e60ffda58d110d4ce3ccd9b94bc`.

## 18. Frozen outcome

**`STABLE_REGIME_TRANSFER_LOCALIZATION`**

Stable classification:

**`CONTEXTUAL_STATE_CONTAMINATION`**

Classification count:
- `CONTEXTUAL_STATE_CONTAMINATION`: **4/4** domains.

This exceeds the preregistered >=3/4 stability boundary without reinterpretation.

## 19. Pooled matched result

### R0 — bare intent state

SYM multiview:
- K4 **90.625%**;
- K8 **86.328%**;
- K16 **77.734%**.

D2S:
- 87.891 / 82.031 / 71.094%.

S2D:
- 90.625 / 85.547 / 76.563%.

This confirms that the fresh W16 base semantic regime itself is strong under the exact frozen W9 projection.

### R1 — same state + W15-style reliability suffix

SYM:
- K4 **41.797%**;
- K8 **25.391%**;
- K16 **17.578%**.

Thus R0 -> R1 loses:
- **48.828 pp** K4;
- **60.938 pp** K8;
- **60.156 pp** K16.

D2S also collapses:
- 46.094 / 28.125 / 18.750%.

S2D collapses:
- 38.672 / 23.438 / 16.797%.

Therefore the primary W16 hypothesis that the damage would be isolated mainly to S2D aggregation is falsified.

### R2 — decorated sequence, exact R0 prefix scored only

SYM:
- K4 **47.266%**;
- K8 **28.125%**;
- K16 **18.750%**.

Recovery over R1:
- +5.469 pp K4;
- +2.734 pp K8;
- +1.172 pp K16.

But R2 remains below R0 by:
- 43.359 pp K4;
- 58.203 pp K8;
- 58.984 pp K16.

Because:
- R2 uses no suffix tokens in scoring;
- R2 token IDs are exactly the R0 prefix token IDs;
- R2 performs no additional encoder call;
- R2 still uses the contextual embeddings produced when the suffix was present;

the loss is localized upstream of token aggregation.

## 20. Fresh-domain classifications

### CG

`CONTEXTUAL_STATE_CONTAMINATION`

SYM:
- R0 K4/K16: 92.188 / 85.938%;
- R1: 43.750 / 14.063%;
- R2: 51.563 / 14.063%.

D2S R1 regression:
- K4 40.625 pp;
- K16 65.625 pp.

S2D R1 regression:
- K4 50.000 pp;
- K16 67.188 pp.

### CH

`CONTEXTUAL_STATE_CONTAMINATION`

SYM:
- R0: 90.625 / 71.875%;
- R1: 35.938 / 18.750%;
- R2: 35.938 / 21.875%.

### CI

`CONTEXTUAL_STATE_CONTAMINATION`

SYM:
- R0: 90.625 / 81.250%;
- R1: 37.500 / 21.875%;
- R2: 48.438 / 23.438%.

### CJ

`CONTEXTUAL_STATE_CONTAMINATION`

SYM:
- R0: 89.063 / 71.875%;
- R1: 50.000 / 15.625%;
- R2: 53.125 / 15.625%.

The same class holds independently on every domain.

## 21. Reference adequacy and model specificity

Pinned MiniLM reference pooled:

R0:
- K4 92.969%;
- K8 85.156%;
- K16 81.641%.

R1:
- K4 93.359%;
- K8 82.422%;
- K16 69.922%.

The suffix causes only:
- +0.391 pp at K4;
- -2.734 pp at K8;
- -11.719 pp at K16

for the reference, versus catastrophic W9/A13 semantic-anchor drops of roughly 49-61 pp.

Therefore:
- the fresh tasks remain semantically recoverable;
- the effect is not merely that decorated queries become intrinsically impossible;
- the dominant failure is specific to the current A13/W9 token representation geometry.

MiniLM remains diagnostic only and is not a HIRA candidate.

## 22. What W16 proves and falsifies

### Supported

The stable evidence supports:

> Adding orthogonal typed metadata to the same A13 sequence causes severe contextual contamination of the intent-bearing token representations used by the W9 semantic projection.

The contamination happens before the final symmetric token aggregation.

This explains a major part of the W14 -> W15 regime gap:
- W14-like bare states preserve strong semantic geometry;
- W15-like decorated states can destroy it even when diagnosis semantics, candidates and definitions are unchanged.

### Falsified as the primary explanation

**Pure extra-token/S2D mean dilution is not sufficient.**

If dilution alone were dominant, R2 would recover close to R0 after suffix tokens were excluded from scoring.

It does not.

D2S also collapses heavily, which is only possible because the prefix state token embeddings themselves have changed under full-sequence contextualization.

### Not yet proven

W16 does not prove:
- a production fix;
- that every typed state will suffer the same contamination;
- that A13 must be replaced;
- that a particular representation-separation mechanism will work;
- high-K rescue;
- external superiority.

## 23. Permanent exposed evidence

CG/CH/CI/CJ are permanently exposed.

Never use them for:
- representation architecture selection;
- field-boundary tuning;
- attention-mask design;
- segmentation policy selection;
- projection tuning;
- threshold selection;
- checkpoint/seed selection;
- calibration.

All earlier forbidden evidence remains forbidden.

## 24. Authorized continuation

W16 authorizes a **fresh representation-separation mechanism authority**.

The next phase must address contamination **before semantic scoring**, not merely remove suffix tokens afterward.

A valid W17 should test one frozen/pre-registered representation separation strategy on wholly fresh TRAIN/DEV/dual-CONFIRM typed domains.

Required conceptual controls:
1. current full-state contextual encoding;
2. candidate-independent multiview anchor on that full state;
3. representation-separated state encoding for primitive-relevant fields;
4. an equal-information control proving gains are from representation separation rather than deleting necessary state information.

The most direct lane is a structured field-isolated state compiler:
- one logical state compilation per case;
- primitive-relevant fields encoded in isolated sequences within a single batched encoder invocation;
- diagnosis receives intent-field semantic memory;
- risk/response/urgency/review retain the reliability fields they require;
- A13/projection remain frozen for the primary causal comparison.

W17 must explicitly measure compute/state-once accounting because isolation must not silently become unconstrained multiple state passes.

Before any W17 empirical exposure:
- freeze exact state field schema;
- freeze how many encoder sequences/batches are allowed;
- freeze controls;
- freeze trainable parameter budget;
- freeze fresh domains/seeds;
- freeze DEV selection and dual-CONFIRM gates.

Do not:
- tune on CG/CH/CI/CJ;
- reopen CE/CF;
- merely mask final-layer tokens;
- open K32/K64 yet;
- replace A13 without a separate architecture question;
- claim W15 is now solved.

A future AI should read this W16 closure first, then W15/W14.
