# R8-W10 handoff — representation-ceiling decomposition audit

Status: **PRE-DIAGNOSTIC. No W10 BF/BG/BH/BI empirical cache or result exists yet.**

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
