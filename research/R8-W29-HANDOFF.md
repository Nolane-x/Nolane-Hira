# R8-W29 handoff — HIRA v0 semantic-core integration

Status: **PRE-INTEGRATION / PRE-EXPOSURE. No W29 empirical authority exists.**

Issue: #147

Branch:
`feat/r8-w29-hira-v0-semantic-core`

Base main:
`953c3324a43bd0f14f285dc13a6720ef9d996f87`

Read first:
1. this file;
2. `research/R8-W28-HANDOFF.md`;
3. `src/nmd/runtime.py`;
4. `src/nmd/schema.py`;
5. `src/nmd/hira.py`.

## 0. Goal

W29 begins the integrated HIRA-v0 model.

This is not another projection-search wave.

W28 froze:
`REPLICATED_COMPOSITIONAL_PROJECTION_RESCUE`.

W29 asks:

> Can the rescued semantic projection be production-ported into the real HIRA runtime, preserve the exact winning token-level operator and multi-view semantics, compile state once, and drive typed choice/score/noul decisions on wholly fresh states without relation-refinement distortion?

Passing W29 authorizes the name:

**HIRA v0 semantic core**

It does not authorize final production.

## 1. Frozen semantic core

Encoder:
- `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- frozen.

Projection:
- use exact W28 **T0** authoritative checkpoint;
- artifact `10905153620`;
- checkpoint SHA256 `1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f`;
- 256->128;
- bias-free;
- 32,768 parameters;
- frozen in W29.

T1 remains replication evidence, not an ensemble member.

No W29 training of the semantic projection is allowed.

## 2. Production scorer

W29 must production-port the exact W28 winning operator.

For one semantic view:

1. project state content tokens with the rescued projection;
2. project option content tokens with the same projection;
3. L2 normalize both;
4. cosine similarity matrix;
5. option->state coverage = per option token max over state tokens, then valid-token mean;
6. state->option coverage = per state token max over valid option tokens, then state-token mean;
7. view score = arithmetic mean of those two directions.

For a logical option with multiple semantic views:

- compute the exact symmetric score independently for each valid view;
- final option logit = arithmetic mean across valid views.

No IDF.
No anti-collapse subtraction.
No question-token concatenation.
No learned view weight.
No top-k.
No post-hoc scale.
No relation delta.

This exactness is important:
the existing W5h `CompetitiveCoarseScorer` is not the same operator and must not silently substitute for W28.

## 3. Multi-view schema semantics

Current `LogicalOption` already contains:
- `criterion_text`;
- `aliases`;
- `exemplars`.

W29 production schema compilation will interpret:
- criterion_text = view 0;
- each alias = another positive semantic view;
- each exemplar = another positive semantic view.

Counterexamples remain metadata and are not positive scoring views.

The compiler must preserve token-level artifacts for all positive views.

Required optional compiled tensors:
- option-view token embeddings;
- option-view content-token masks;
- option-view validity mask.

Existing single-view token artifacts must remain backward-compatible.

## 4. Runtime integration

Add a new coarse mode:

`symmetric_semantic`

It must:
- require the production rescued semantic scorer;
- require multi-view schema token artifacts;
- use `StateMemory.content_token_embeddings`;
- produce full-K logits.

Add a relation-refinement bypass.

For W29:
- refinement = OFF;
- relation delta exactly zero;
- no random/untrained HIRACore parameter may alter rescued semantic logits.

The normal HIRACore path remains available for old modes.

## 5. Typed behavior

The same compiled state must be reused across typed queries.

W29 validates:
- `choice`;
- `score`;
- `noul`.

For each binary semantic factor:
- choice returns semantic option identity;
- score uses values 0.0 / 1.0;
- noul uses values 0.0 / 1.0 and returns true probability.

All three primitive wrappers must preserve the same top-1 semantic decision for the same factor.

## 6. Structured severity composition

Fresh states carry atomic factors:

- F0 meaningful disruption;
- F1 major functional loss;
- F2 immediate criticality.

Valid factor vectors remain:

- S0 = 000;
- S1 = 100;
- S2 = 110;
- S3 = 111.

W29 may derive a structured severity class from the three factor top-1 predictions.

W29 does not claim calibrated class probabilities yet.

## 7. Fresh W29 authority

Fresh domains:

EW — municipal library-access restoration review — seed 481101  
EX — nonprofit meal-delivery routing incident review — seed 481107  
EY — university equipment-loan service review — seed 481119  
EZ — regional transit-pass support review — seed 481131

Per domain:
- 96 cases;
- 24 cases/severity;
- fresh wording not reused from W28 or older authorities;
- deterministic F0/F1/U/C facts;
- F2 = U AND C;
- severity mapping from F0/F1/F2 as above.

Total:
384 fresh cases.

No W29 row may train or tune the frozen projection.

## 8. External semantic authority

Pinned W28 reference panel remains external validation only:

DeBERTa:
- `cross-encoder/nli-deberta-v3-base`;
- revision `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`;
- weight SHA256 `d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa`.

RoBERTa:
- `cross-encoder/nli-roberta-base`;
- revision `1be0567456f0543475805e758725f151f283705a`;
- weight SHA256 `efc90996d2ed80123c26c9091c91385ffddc6d2fd0b2bacf3187fbd6c5b87953`.

Primary reference:
- F0 direct;
- F1 direct;
- U direct;
- C direct;
- F2 = U AND C.

Direct F2 stays diagnostic only.

No reference output may become a model input.

## 9. Fresh reference gate

Every EW/EX/EY/EZ domain must pass:

Panel:
- F0 BA >= .90;
- F1 BA >= .90;
- U BA >= .92;
- C BA >= .92;
- composed F2 BA >= .92;
- composed positive recall >= .88;
- composed negative recall >= .94;
- probability mass error <= 1e-6.

Each reference model:
- F0/F1 BA >= .80;
- U/C BA >= .82;
- composed F2 BA >= .82.

Failure:
`W29_REFERENCE_INADEQUATE`.

Reference precedence is absolute.

## 10. HIRA-v0 semantic-core gate

Every EW/EX/EY/EZ domain must pass.

For each factor under each primitive:
- top1 >= .90.

Cross-primitive semantic agreement:
- choice vs score >= .98;
- choice vs noul >= .98;
- score vs noul >= .98.

Structured factor vector:
- >= .85.

Composed severity:
- >= .85.

Invalid factor-vector rate:
- <= .05.

Probability mass:
- max error <= 1e-6.

State-once:
- exactly one A13 state encode per case;
- nine typed queries may reuse that one `StateMemory`;
- zero state re-encode during typed query fan-out.

Schema reuse:
- repeated compile of identical schema must cache-hit after the first compile.

Option-order invariance:
- semantic output identity must be invariant to binary option order.

## 11. Runtime integrity gate

Must prove:
- frozen T0 checkpoint SHA exact;
- projection trainable parameters in W29 = 0;
- relation refinement disabled;
- relation delta exactly zero;
- scorer has no trainable parameter other than the loaded frozen projection;
- full-K scores emitted;
- no candidate truncation;
- no W28 or older authority rows;
- no Banking77;
- no typed final/test;
- campaign cells = 0.

## 12. Frozen outcomes

Precedence:

1. `W29_REFERENCE_INADEQUATE`
2. `W29_RUNTIME_INTEGRATION_FAIL`
3. `HIRA_V0_SEMANTIC_CORE_READY`

No PARTIAL promotion.

### HIRA_V0_SEMANTIC_CORE_READY

Requires:
- reference adequate on 4/4;
- HIRA semantic gate pass on 4/4;
- state-once/runtime integrity pass;
- option-order invariance pass;
- exact frozen T0 provenance.

## 13. Authorization after pass

Only `HIRA_V0_SEMANTIC_CORE_READY` authorizes the repository to treat the integrated runtime as **HIRA v0 semantic core**.

The next required gates remain:
1. calibration;
2. OOD/null abstention;
3. reliability;
4. high-K / K32-K64;
5. latency/RAM;
6. broader semantic transfer.

No Laya/JEV superiority claim yet.

## 14. Permanent forbidden evidence

All W28 EN-EV and every older exposed authority remain forbidden.

After first W29 empirical execution:
EW/EX/EY/EZ become permanently exposed and may never be used for later tuning.

## 15. Current state

Completed:
- W28 frozen and merged into main `953c3324a43bd0f14f285dc13a6720ef9d996f87`;
- W28 issue #145 closed;
- W29 issue #147 created;
- W29 branch created from exact post-W28 main;
- architecture, authority, gates and promotion boundary frozen here.

Exposure:
**NONE.**

Next:
1. implement production symmetric semantic scorer;
2. add multi-view schema token artifacts;
3. add symmetric runtime coarse mode;
4. add relation-refinement bypass;
5. add unit/backward-compatibility tests;
6. implement fresh EW-EZ authority;
7. implement T0 artifact loader + external reference evaluator;
8. exact-head unit + repo CI;
9. only then enable W29 empirical integration authority;
10. freeze outcome before merge.
