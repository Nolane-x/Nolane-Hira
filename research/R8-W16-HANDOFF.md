# R8-W16 handoff — matched semantic-regime transfer decomposition

Status: **PRE-DIAGNOSTIC. No CG/CH/CI/CJ A13/reference cache or W16 classification exists yet.**

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
