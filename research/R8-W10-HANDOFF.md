# R8-W10 handoff — representation-ceiling decomposition audit

Status: **CLOSED DIAGNOSTIC. Authoritative outcome: `REPRESENTATION_CEILING_UNRESOLVED`. No rescue mechanism is authorized from W10.**

Issue: #107

Branch:
`feat/r8-w10-representation-ceiling`

Base main:
`c09c108783c1c7f120c5b13c5d4b28747e3ef011`

Read first when restoring:
1. this file;
2. `research/R8-W9-HANDOFF.md`;
3. `research/R8-W8-HANDOFF.md`;
4. `research/R8-W7C-HANDOFF.md`;
5. issue #107.

## 0. What HIRA is

Nolane HIRA is a compact non-autoregressive typed decision system.

Long-term goals:
- encode state once/case;
- dynamic semantic schemas;
- choice / score / noul primitives;
- small local-friendly parameter budget;
- transferable semantic binding;
- robust high-cardinality decisions;
- explicit reliability integrity;
- sealed authorities and immutable negative results.

## 1. Frozen production stack

A13:
- `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length 256;
- frozen.

W6e production scorer:
- 256->128 bias-free projection;
- competitive late interaction;
- 32,769 params total;
- scorer SHA `6d5a7f2d3ed63ecd756181b1cb54e4704f68e5f74983a897f0a68fb4d1d63d2e`.

HIRACore:
- frozen W6e SHA `d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588`.

W9 projection-semantic checkpoint:
- selected epoch 5;
- scorer SHA `8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102`;
- source W9 freeze artifact `10858424139`.

## 2. Decisive history through W9

W5 established late-interaction + competitive salience as the strongest tiny semantic matcher.

W6-W6e showed strong internal typed competence but unstable held-out high-K behavior.

W6f-W6j progressively ruled out salience removal, relation damage, simple field adapters and naive representation formats, then localized the internal synthetic high-K issue to coarse conjunction.

W7 explicit smooth-AND conjunction failed.

W7b same-data free-form retuning produced strong synthetic K64 gains but missed a preregistered replication gate.

W7c showed those gains did not transfer to Banking77: primary/replica 0/400 with very high confidence.

W8 stably localized `GENERAL_SEMANTIC_TRANSFER_LIMIT`: low-K unseen semantics were already weak before K64.

W9 tested direct semantic alignment:
- projection retune;
- shared rank-16 bridge;
- asymmetric state/schema bridge.

W9 exact head:
`26203f8f5e3f751de331a0f2aafbb5d447efc4fb`

Authority:
`36122220588`

Frozen verdict:
**`SEMANTIC_ALIGNMENT_FAIL`**.

The strongest W9 path was ordinary full projection retuning:
- BD definition K4/K16 51.56% / 34.38%;
- BE 62.50% / 34.38%;
- alignment ~51-53%.

Small post-projection bridges were weaker.

## 3. Why W10 exists

W9 leaves the key unresolved question:

**Is transferable semantic information already missing in raw frozen A13, or is it damaged by HIRA's projection / production scoring interface?**

W10 is diagnostic only. No optimization occurs.

## 4. Fresh W10 domains

- BF residential energy account support — seed 291701;
- BG outpatient laboratory administration — seed 291707;
- BH freight-booking customer support — seed 291719;
- BI connected-device subscription support — seed 291731.

Each:
- 16 latent intents;
- 4 independently written state utterances/intent;
- one natural definition/intent;
- one terse label/intent;
- 64 bases/domain.

Total 256 bases.

Nested K:
4 / 8 / 16.

Natural definitions are primary.
Terse labels are descriptive only.

BF/BG/BH/BI must not overlap W5-W9 exact text atoms.

## 5. Frozen representation operators

A0 raw A13 mean cosine:
- content-token masked mean;
- 256-d;
- L2 normalize;
- cosine.

A1 raw A13 symmetric token MaxSim:
- 256-d content tokens;
- bidirectional MaxSim means;
- no learned projection.

P0 W6e projected symmetric MaxSim:
- exact frozen W6e projection;
- 128-d;
- same candidate-independent symmetric MaxSim.

P1 W9 semantic projected symmetric MaxSim:
- exact W9 semantic projection;
- same score.

S0 frozen W6e production:
- competitive scorer + frozen pooled relation/HIRACore.

S1 W9 semantic projection through production:
- W9 projection checkpoint;
- same production competitive path/HIRACore.

R0 frozen semantic reference:
- `sentence-transformers/all-MiniLM-L6-v2`;
- pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- frozen sentence embedding cosine;
- diagnostic ceiling only;
- never enters HIRA.

Authority must record exact downloaded R0 weight SHA before accepting results.

## 6. Frozen classifiers

`A13_REPRESENTATION_CEILING` on >=3/4 domains:
- A1 K4 <.60;
- A0 K4 <.60;
- R0 K4 >=.75;
- R0 - max(A0,A1) >=+.20;
- P1 does not beat max raw by >=+.15.

`PROJECTION_DEGRADATION_LIMIT` on >=3/4:
- A1 K4 >=.70;
- P0 <= A1-.15;
- R0 >=.75;
- A1 K16 >=.45.

`PROJECTION_RECOVERABLE_GEOMETRY` on >=3/4:
- P1 K4 >=.70;
- P1 >= P0+.15;
- P1 K16 >=.45;
- R0 >=.75.

`SCORING_INTERFACE_LIMIT` on >=3/4:
- P1 K4 >=.70;
- S1 final K4 <= P1-.15;
- P1 K16 >=.45;
- S1 final K16 <= P1-.10.

`REFERENCE_TASK_AMBIGUOUS` on >=3/4:
- R0 K4 <.70.

Otherwise:
- mixed or unresolved per issue #107.

Overall:
- `STABLE_REPRESENTATION_CEILING_LOCALIZATION`;
- `MIXED_REPRESENTATION_CEILING_LOCALIZATION`;
- `REPRESENTATION_CEILING_UNRESOLVED`.

No post-exposure gate changes.

## 7. External diagnostic reference boundary

R0 is a ceiling probe, not a HIRA candidate.

Never:
- use R0 embeddings inside HIRA;
- train HIRA against R0 outputs in W10;
- select HIRA checkpoints using R0;
- compare parameter efficiency as if R0 were production;
- change R0 revision after exposure.

The current Hub pin is:
`sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

## 8. Permanent forbidden evidence

Never reuse for W10:
- prior W6b-W9 CONFIRM/diagnostic rows;
- W8 AU/AV/AW/AX;
- W9 AY/AZ/BA/BB/BC/BD/BE;
- Banking77 0-799;
- typed final/test;
- public campaign cells.

After authority, BF/BG/BH/BI become exposed.

## 9. Mandatory integrity

- no training;
- one A13 state encode/base;
- exact candidate identity across all operators;
- same K memberships/order;
- exact W6e/W9 checkpoint SHA validation;
- exact A13 pin/weight SHA validation;
- exact R0 revision and local downloaded weight SHA recorded;
- no prior exact text overlap;
- no campaign population.

## 10. Current state

Completed:
- W9 closed and merged main `c09c108783c1c7f120c5b13c5d4b28747e3ef011`;
- W10 issue #107 preregistered;
- branch created from exact post-W9 main;
- this handoff created before any W10 empirical exposure.

No BF/BG/BH/BI A13 or R0 embeddings exist yet.

Remaining:
1. fresh authority generator;
2. state-once raw-A13 cache;
3. representation operators/evaluator;
4. R0 frozen reference encoder;
5. classifier library/tests;
6. exact checkpoint provenance loader;
7. pre-data unit workflow;
8. gated authority workflow;
9. run authority only after pre-data contracts pass;
10. freeze exact result into this file before merge.

## 11. Continuation boundary

If W10 stably localizes:
- A13 ceiling -> next lane may compare a new frozen semantic encoder or tightly bounded encoder adaptation;
- projection degradation -> preserve raw A13 geometry while redesigning projection;
- projection recoverable -> projection-centric semantic training/regularization;
- scoring interface -> fix candidate-relative scorer/HIRACore interface.

If mixed/unresolved:
do not train another local mechanism; perform a representation/prior-art reassessment.

A future AI must be able to continue from this file without chat memory.


---

## 12. Authoritative W10 closure

Exact empirical head:
`c5e8846851d4f520771d402be132f03a25622807`

Authority run:
`36126611084`

All jobs PASS:
- unit `108044069062`;
- upstream provenance `108044454245`;
- fresh cache `108044454240`;
- evaluate `108045394558`.

Pre-data exact-head gates:
- W10 unit `36126239741`: PASS;
- repository CI `36126239687`: PASS;
- exact pre-exposure head `a4ab787cfbae88ec09251ada5dec6db0c63edcd6`.

Authority artifacts:
- W10 fresh representation cache:
  - ID `10860406331`;
  - digest `sha256:297054bd1a5d9987a70de150d2d2dfbbeb494003224d303e0d2c8c51334bc1d0`;
- repackaged frozen W9 checkpoint bundle:
  - ID `10860395891`;
  - digest `sha256:94db4b9aba69f2211cb5cf394f9252700e6890efd4b235f0fa0359e0f8855e65`;
- authoritative representation-ceiling audit:
  - ID `10860241732`;
  - digest `sha256:cc8470a1b841af89de01680e3b54276e565f37f7c9a03ba79fb6c232d6dad12e`.

### Integrity

- 256 fresh base states;
- 1,536 paired K/view cases;
- BF/BG/BH/BI each 64 bases;
- one A13 state encode/base;
- training performed: false;
- probability mass max error: `1.771841198205948e-07`;
- no W8 diagnostic rows;
- no W9 CONFIRM rows;
- no Banking77 rows;
- no typed final/test rows;
- campaign cells populated: 0.

Frozen reference:
- `sentence-transformers/all-MiniLM-L6-v2`;
- revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- downloaded `model.safetensors` SHA256:
  `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`;
- diagnostic only;
- never used as a HIRA candidate, training target, checkpoint selector or calibration source.

### Frozen outcome

**`REPRESENTATION_CEILING_UNRESOLVED`**

Stable classification:
- none.

Frozen classification counts:
- `SCORING_INTERFACE_LIMIT`: 2 domains;
- every other primary class: 0 stable-domain count reaching the required threshold.

The preregistered rule requires >=3/4 domains for a stable target. That gate is not changed after exposure.

## 13. Per-domain frozen anatomy

### BF
Classification:
**`SCORING_INTERFACE_LIMIT`**

Natural-definition top1:
- A0 raw A13 mean K4: 57.81%;
- A1 raw A13 symmetric MaxSim K4: 70.31%;
- A1 K16: 42.19%;
- P0 frozen W6e projection K4: 73.44%;
- P1 W9 semantic projection K4: **87.50%**;
- P1 K16: **48.44%**;
- S1 W9 projection through production final K4: **56.25%**;
- S1 K16: **20.31%**;
- R0 frozen reference K4: **98.44%**.

The P1 -> S1 loss is -31.25 pp at K4 and -28.13 pp at K16.

### BG
Classification:
**`SCORING_INTERFACE_LIMIT`**

- A0 K4: 37.50%;
- A1 K4: 48.44%;
- A1 K16: 20.31%;
- P0 K4: 60.94%;
- P1 K4: **71.88%**;
- P1 K16: **48.44%**;
- S1 final K4: **46.88%**;
- S1 K16: **29.69%**;
- R0 K4: **96.88%**.

The P1 -> S1 loss is -25.00 pp at K4 and -18.75 pp at K16.

### BH
Classification:
**`REPRESENTATION_CEILING_UNRESOLVED`**

- A0 K4: 46.88%;
- A1 K4: 60.94%;
- A1 K16: 34.38%;
- P0 K4: 76.56%;
- P1 K4: 75.00%;
- P1 K16: 50.00%;
- S1 final K4: 67.19%;
- S1 K16: 26.56%;
- R0 K4: 98.44%.

There is still a large K16 P1 -> S1 drop (-23.44 pp), but K4 drops only -7.81 pp, so the frozen scoring-interface classifier does not activate.

### BI
Classification:
**`REPRESENTATION_CEILING_UNRESOLVED`**

- A0 K4: 70.31%;
- A1 K4: 76.56%;
- A1 K16: 42.19%;
- P0 K4: 78.13%;
- P1 K4: 82.81%;
- P1 K16: 70.31%;
- S1 final K4: 71.88%;
- S1 K16: 51.56%;
- R0 K4: 96.88%.

The P1 -> S1 loss is -10.94 pp at K4 and -18.75 pp at K16. Again, this is meaningful but does not satisfy the exact frozen K4 damage gate.

## 14. Pooled representation ladder

Natural-definition pooled top1:

| Operator | K4 | K8 | K16 |
| --- | ---: | ---: | ---: |
| A0 raw A13 mean | 53.13% | 35.94% | 23.44% |
| A1 raw A13 symmetric MaxSim | 64.06% | 46.48% | 34.77% |
| P0 W6e projected symmetric MaxSim | 72.27% | 60.16% | 48.44% |
| P1 W9 semantic projected symmetric MaxSim | **79.30%** | **67.58%** | **54.30%** |
| S0 frozen W6e production final | 46.88% | 34.38% | 22.27% |
| S1 W9 projection through production final | 60.55% | 44.92% | 32.03% |
| R0 frozen MiniLM reference | **97.66%** | **94.92%** | **90.63%** |

The pooled P1 -> S1 production-interface drop is:
- K4: **-18.75 pp**;
- K8: **-22.66 pp**;
- K16: **-22.27 pp**.

The frozen reference remains >90% even at K16, so the authority itself has strong recoverable semantic structure.

## 15. What W10 establishes and what it does not

### Strong evidence

1. **The W10 task is not intrinsically ambiguous.**
   R0 reaches 97.66% K4 and 90.63% K16 pooled.

2. **A pure raw-A13 ceiling is not stably supported.**
   Raw A13 performance varies substantially by domain, and in BI A1 reaches 76.56% K4.

3. **The frozen W6e projection is not generally destroying raw A13 semantics.**
   P0 is stronger than A1 pooled:
   - K4 72.27% vs 64.06%;
   - K16 48.44% vs 34.77%.

4. **Projection geometry is materially recoverable under W9 semantic supervision.**
   P1 is the strongest HIRA-side semantic operator:
   - K4 79.30%;
   - K16 54.30%.

5. **The production path can lose a large amount of that semantic signal.**
   P1 -> S1 loses ~19-23 pp pooled.
   BF/BG satisfy the frozen `SCORING_INTERFACE_LIMIT` rule.

### What W10 does NOT establish

W10 does not establish a stable `SCORING_INTERFACE_LIMIT`, because only BF/BG meet the exact rule and preregistration requires >=3/4.

It also does not establish:
- that A13 must be replaced;
- that W9 projection is production-ready;
- that candidate-relative IDF is the culprit;
- that sibling common-mode subtraction is the culprit;
- that salient-min coverage is the culprit;
- that HIRACore relation is the culprit.

Those production transformations remain confounded inside S0/S1.

Therefore the official result remains:

**`REPRESENTATION_CEILING_UNRESOLVED`**

Do not lower 3/4 to 2/4 after seeing the result.

## 16. Permanent exposure boundary after W10

BF/BG/BH/BI are now exposed.

Never reuse them for:
- training;
- DEV selection;
- projection/scorer selection;
- ablation selection;
- threshold tuning;
- weighting tuning;
- relation tuning;
- candidate-set tuning;
- seed selection;
- calibration;
- mechanism choice.

The R0 reference outputs on BF/BG/BH/BI are also exposed diagnostics and cannot be used as supervision.

All W6-W9 forbidden evidence remains forbidden.

## 17. Authorized next research direction

Because W10 is unresolved, **do not open a rescue/training lane**.

The next phase should be a second fresh diagnostic focused on the production interface, because W10 exposed a large but not yet stable P1 -> S1 degradation.

A fresh W11 should decompose, on wholly new domains and with the exact frozen W9 P1 projection:

1. candidate-independent symmetric semantic pair score;
2. one-direction option->state MaxSim;
3. addition of question/context tokens;
4. candidate-relative IDF weighting;
5. sibling common-mode subtraction;
6. weighted-mean + salient-min coverage aggregation;
7. full competitive coarse;
8. frozen HIRACore relation/final output.

The purpose is to localize **which transformation first turns a correct strong semantic ranking into the wrong production ranking**.

Important historical constraints:
- W6f already showed that removing candidate-relative salience wholesale is much worse;
- W6f also showed relation reranking is net helpful in that authority;
- therefore W11 must use paired stage transitions, not crude “remove everything” ablations;
- no training;
- no BF/BG/BH/BI reuse;
- create a complete W11 handoff before fresh exposure.

Only if W11 finds one stable interface stage may a later W12 mechanism authority target that stage.

## 18. Continuation state

W10 empirical work is complete.

Before continuing:
1. merge PR #108 after this closure-only commit has green unit/CI;
2. close issue #107;
3. create W11 from resulting main;
4. preregister fresh domains/operators/stage-localization rules;
5. create `research/R8-W11-HANDOFF.md` before exposure;
6. keep every W10 threshold and classification frozen.

A future AI should read this file first, then W9 and W8 handoffs.
