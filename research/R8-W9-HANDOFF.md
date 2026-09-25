# R8-W9 handoff — contrastive semantic-alignment bridge

Status: **CLOSED EMPIRICAL MECHANISM AUTHORITY. Frozen verdict: `SEMANTIC_ALIGNMENT_FAIL`. No external/public phase is authorized.**

Issue: #105

Branch:
`feat/r8-w9-semantic-alignment`

Base main:
`e92e1bd7fad5dea3c552ab2fc2008a46367f3afe`

Read first when restoring:
1. this file;
2. `research/R8-W8-HANDOFF.md`;
3. `research/R8-W7C-HANDOFF.md`;
4. `research/R8-W7B-HANDOFF.md`;
5. `research/R8-W7-HANDOFF.md`;
6. `research/R8-W6J-HANDOFF.md`;
7. issue #105.

## 0. Project identity

Nolane HIRA is a compact non-autoregressive typed decision system, not a next-token LLM.

Long-term goals:
- encode state once/case;
- dynamic semantic schemas and changing candidate sets;
- typed choice / score / noul;
- small local-friendly parameter budget;
- transferable semantic binding;
- robust high-cardinality decisions;
- explicit calibration/reliability integrity;
- immutable negative results and sealed authorities.

## 1. Frozen encoder/runtime

A13:
- `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length 256;
- fully frozen.

HIRACore:
- 422,159 params;
- frozen in W9.

Production competitive scorer:
- shared 256 -> 128 bias-free projection;
- scalar logit scale;
- 32,769 params;
- candidate-relative IDF;
- sibling common-mode subtraction;
- weighted mean + salient-min coverage.

Exact W6e base initialization:
- HIRA SHA `d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588`;
- scorer SHA `6d5a7f2d3ed63ecd756181b1cb54e4704f68e5f74983a897f0a68fb4d1d63d2e`;
- artifact `10805567861`.

## 2. Decisive history

W5:
- pooled/simple token probes and encoder scaling failed;
- late interaction + competitive salience became production semantic scorer.

W6/W6b:
- production typed integration;
- strong overall internal competence;
- incomplete high-K/reliability generalization.

W6c:
- calibration can fix confidence without fixing semantic hard decisions.

W6d/e:
- high overall typed scores do not imply reproducible K64 held-out-domain competence.

W6f:
- removing candidate-relative salience is dramatically worse;
- relation reranking is net helpful.

W6g:
- second-order localization found field semantic weakness.

W6h:
- shared 8,192-param residual adapter failed reproducibility.
- Important: W9 must not relabel this mechanism as new.

W6i:
- representation bridge remained unresolved;
- canonical/factorized schema renderings did not rescue.

W6j:
- stable internal high-K target `COARSE_CONJUNCTION_LIMIT`.

W7:
- explicit factor smooth-AND mechanism failed: `CONJUNCTIVE_COARSE_FAIL`.
- same-data free-form scorer retune was much stronger.

W7b:
- typed-only / pair-only / typed+pair factorial.
- verdict `FREEFORM_RETUNE_NONREPLICATING`.
- primary K64 79.17% on AS and AT, but a preregistered replication gain gate failed.
- no stable attribution class.

W7c:
- Banking77 rows 400–799 external frozen transfer.
- verdict `PUBLIC_HIGH_K_TRANSFER_ABSENT`.
- frozen 1/400; typed-only 1/400; pair-only/primary/replica 0/400.
- retuned checkpoints were highly confident while wrong.

W8:
- exact empirical head `8f30d2b2f1d7b81f7e48ed43c9a89b1e46ead8c6`;
- authority run `36113828435`;
- outcome `STABLE_SEMANTIC_TRANSFER_LOCALIZATION`;
- stable target `GENERAL_SEMANTIC_TRANSFER_LIMIT`.

W8 pooled natural-definition K4/K16:
- frozen: 50.78% / 22.66%;
- typed-only: 51.95% / 25.39%;
- typed+pair primary: 55.86% / 28.91%;
- typed+pair replica: 54.69% / 26.56%.

All four checkpoints:
- AU general semantic transfer limit;
- AV unresolved;
- AW general semantic transfer limit;
- AX general semantic transfer limit.

Therefore W9 is authorized.

## 3. Prior-art motivation

W9 does not claim novelty for contrastive alignment.

Relevant prior work:
- NLI training for transferable sentence representations / semantic similarity;
- description-augmented dataless intent classification;
- semantic-similarity-aware contrastive zero-shot intent classification;
- parameter-efficient generalized zero-shot intent adaptation.

The HIRA-specific test is whether a tiny frozen-encoder state↔schema alignment mechanism fixes low-K unseen semantics while preserving state-once execution.

## 4. W9 hypothesis

Current shared projection is required to map two asymmetric language roles through one geometry:
- state/user utterance evidence;
- schema/intent definition text.

W8 shows that natural definitions help somewhat but not enough.

Hypothesis:
**a small semantic alignment objective can make unseen state/definition meaning transferable, and separate state/schema residual maps may help more than a shared map or ordinary projection retuning.**

This hypothesis is falsifiable against equal-data controls.

## 5. Mechanism

Base projection:
`p(x) = normalize(Wx)`

Identity-initialized low-rank bridge:
`b(x) = normalize(x + U gelu(Dx))`

Rank:
16.

Shared bridge:
- one D/U pair for both context and schema;
- 4,096 trainable params.

Asymmetric bridge:
- state/context D/U;
- schema/option D/U;
- 8,192 trainable params.

For bridge candidates:
- W6e base projection frozen;
- logit scale frozen;
- HIRACore frozen.

At initialization:
- every U is zero;
- scorer must reproduce exact frozen production logits within numeric tolerance.

## 6. Direct semantic-alignment objective

W9 training does not optimize typed HIRA decisions.

For a state and all 16 natural definitions in its training domain:

1. A13 state/definitions are cached once.
2. Base projection + candidate bridge maps token features.
3. Compute token cosine similarity.
4. Definition->state MaxSim mean.
5. State->definition MaxSim mean.
6. Pair score = mean of those two directional scores.
7. 16-way CE chooses the gold definition.

Temperature:
- fixed 0.07;
- never learned/tuned.

No:
- IDF in alignment loss;
- sibling common-mode subtraction in alignment loss;
- typed loss;
- pair-margin loss;
- HIRACore loss.

Inference/evaluation still uses the full production competitive scorer + frozen HIRACore.

## 7. Candidates

0. `frozen-w6e-control`
- 0 trainable.

1. `projection-semantic-control`
- ordinary competitive 256->128 projection trainable;
- production log_scale frozen because the W9 alignment loss uses fixed temperature 0.07;
- 32,768 trainable params;
- same direct semantic-alignment loss/data.

2. `shared-bridge-semantic-control`
- frozen production scorer;
- one shared rank-16 bridge;
- 4,096 params.

3. `asymmetric-bridge-primary`
- frozen production scorer;
- separate state/schema bridges;
- 8,192 params.

4. `asymmetric-bridge-replica`
- same as primary;
- independent seed;
- never substituted for primary after CONFIRM.

No new candidates after empirical exposure.

## 8. Fresh W9 domains

TRAIN:
- AY travel itinerary servicing, seed 281601;
- AZ education portal support, seed 281607;
- BA home-appliance repair coordination, seed 281619;
- BB insurance policy servicing, seed 281627.

DEV:
- BC event-ticket customer support, seed 282731.

Untouched CONFIRM:
- BD municipal permit assistance, seed 283841;
- BE small-business payroll support, seed 284953.

Each:
- 16 latent intents;
- 4 independently written state utterances/intent;
- one terse label;
- one natural definition;
- 64 base states.

Counts:
- TRAIN 256;
- DEV 64;
- BD 64;
- BE 64.

No W5-W8 ID/value/template reuse.

## 9. Cardinality

One deterministic 16-intent master set/base.

Nested:
- K4;
- K8;
- K16.

Primary schema representation:
- natural definition.

Secondary:
- terse label.

No structured synthetic criterion or lexical bridge in primary W9.

## 10. Training

Exactly 8 epochs.

Optimizer:
- AdamW;
- lr 3e-4;
- weight decay .01;
- no scheduler;
- no warmup;
- no AMP;
- grad clip 1.0;
- same TRAIN states and optimizer steps.

Seeds:
- projection 2101;
- shared bridge 2111;
- asymmetric primary 2129;
- asymmetric replica 2137.

## 11. DEV selection

Each candidate independently freezes on BC.

Lexicographic:
1. natural-definition K4 final;
2. natural-definition K16 final;
3. natural-definition K4 coarse;
4. natural-definition K16 MRR;
5. terse-label K4 final;
6. lower alignment CE;
7. earlier epoch.

No cross-candidate selection.

BD/BE may only materialize after all trainable candidates are frozen.

## 12. Metrics

Per candidate/domain/view/K:
- coarse/final top1/top5/MRR;
- gold margin;
- relation rescue/damage;
- probability mass max error;
- state encodes/base.

Alignment:
- 16-way top1;
- MRR;
- CE;
- positive score;
- hardest-negative score;
- margin.

Paired:
- label wrong -> definition right at K4;
- K4 right -> K16 wrong.

Integrity:
- checkpoint hashes;
- parameter counts;
- A13 provenance;
- dataset hashes;
- optimizer-step parity;
- forbidden-overlap checks.

## 13. Frozen gates

Asymmetric primary on both BD/BE:
- definition K4 >=.75;
- definition K16 >=.55;
- definition K4 coarse >=.72;
- alignment top1 >=.70;
- terse label K4 >=.45;
- prob error <=1e-6;
- state encodes/base=1.

Vs frozen on both:
- K4 gain >=+.15;
- K16 gain >=+.15;
- alignment top1 gain >=+.15.

Replica on both:
- definition K4 >=.70;
- definition K16 >=.50;
- alignment top1 >=.65.

Architecture causal vs projection on both:
- K4 >= projection +.05;
- alignment top1 >= projection +.05;
- K16 >= projection -.03.

Architecture causal vs shared on both:
- K4 >= shared +.03;
- alignment top1 >= shared +.03;
- K16 >= shared -.03.

## 14. Frozen verdicts

`ASYMMETRIC_SEMANTIC_BRIDGE_RESCUE`
- primary absolute/gain gates pass both;
- replica passes both;
- all causal architecture gates pass both.

`SEMANTIC_ALIGNMENT_RESCUE_NO_ASYMMETRY`
- asymmetry causal superiority fails;
- projection or shared control passes on both:
  - definition K4 >=.75;
  - definition K16 >=.55;
  - alignment top1 >=.70;
  - K4 gain vs frozen >=+.15;
  - K16 gain vs frozen >=+.15;
- asymmetric primary no more than .05 below the rescuing control K4.

`SEMANTIC_ALIGNMENT_PARTIAL`
- no full rescue;
- asymmetric K4 gain >=+.10 both;
- asymmetric K16 gain >=+.08 both;
- replica K4 gain >=+.07 both.

`SEMANTIC_ALIGNMENT_FAIL`
- otherwise.

No post-result gate changes.

## 15. External boundary

Only after a full internal rescue verdict may a separate external public transfer phase be opened.

It must:
- use a public dataset never previously exposed to HIRA;
- preregister exact dataset/revision/slice/schema before row access;
- freeze W9 checkpoint;
- use no public task training/retrieval/calibration;
- never reuse Banking77 0–799.

External evidence cannot retroactively change W9 verdict.

## 16. Permanent forbidden evidence

Never use for W9 fitting/selection:
- W6b/c CONFIRM;
- W6d F;
- W6e L/M;
- W6f N/O/P;
- W6g Q/R/S;
- W6h Y/Z;
- W6i AA/AB/AC;
- W6j AD/AE/AF;
- W7 AL/AM;
- W7b AS/AT;
- W8 AU/AV/AW/AX;
- Banking77 0–799;
- typed final/test;
- public campaign cells.

## 17. Current state

Completed:
- W8 authority closed and merged main `e92e1bd7fad5dea3c552ab2fc2008a46367f3afe`;
- W9 issue #105 preregistered;
- W9 branch created from exact post-W8 main;
- this handoff created before any W9 data exposure.

No W9 A13 data exists yet.

Remaining:
1. implement bridge library;
2. identity/parameter/state-once unit contracts;
3. implement fresh AY–BE authority generator;
4. implement semantic-alignment cache;
5. implement candidate trainer;
6. implement BC DEV freezer;
7. seal BD/BE evaluator;
8. freeze verdict library/tests;
9. add pre-data unit workflow;
10. run all contracts;
11. only then enable authority;
12. freeze exact authority result here before merge.

## 18. Claim discipline

Do not claim:
- W9 is novel merely because it uses contrastive loss;
- W9 works before untouched BD/BE;
- internal rescue means public transfer;
- W8 domains are fresh for W9;
- Banking77 can be reused;
- a better replica may replace primary;
- a threshold may be changed because a result is close.

A future AI must be able to continue from this file without chat memory.


### Pre-data amendment — projection control parameter exactness

Before any W9 A13 cache/training exposure, issue #105 froze one clarification:
- `projection-semantic-control` trains only the 256->128 projection;
- production `log_scale` is frozen;
- exact trainable count is 32,768.

Reason: direct semantic alignment uses fixed temperature 0.07 and does not consume the production logit scale, so leaving that scalar trainable would create a nominal trainable parameter with no gradient.


---

## 19. Authoritative W9 closure

Exact empirical head:
`26203f8f5e3f751de331a0f2aafbb5d447efc4fb`

Authority run:
`36122220588`

All jobs PASS:

`unit -> exact W6e provenance + fresh AY-BB/BC cache -> five candidates -> independent DEV-BC freeze -> untouched BD/BE CONFIRM`.

Pre-authority full-stack validation:
- implementation head `ee88189a8a7a7778bcc392424869c313e7ac29eb`;
- W9 unit `36121751275`: PASS;
- repository CI `36121751144`: PASS.

### Artifacts

- W9 fresh train/dev cache:
  - artifact `10858333406`;
  - digest `sha256:942548dfcac9807b523378f494fc2e51d1fffffe6f10c87ca141dd4d2a7e750f`.
- frozen DEV checkpoint bundle:
  - artifact `10858424139`;
  - digest `sha256:6f0eee6ae58d489f0486372f0311e0f420065a0d9499185af7904377ae6a4712`.
- authoritative BD/BE CONFIRM:
  - artifact `10858394726`;
  - digest `sha256:9e757c8425ab7b1fd3118e8f1f0c6cdcaf2b58827c2baa12d58c8a662cdad198`.

Candidate artifacts:
- projection semantic control:
  - `10858368835`;
  - `sha256:61ef65219a4e4fea354afe4391e51f4029d0942adce3ab9640b49a1702e577c1`.
- shared bridge control:
  - `10858408768`;
  - `sha256:f5e62537136dc1f42694ba26cd4203652e306a474f77910a4395a43314da03c4`.
- asymmetric primary:
  - `10858358763`;
  - `sha256:81213752dd12c2e1deac844b437b2c40574c7254e6c388d259736b9a983824b6`.
- asymmetric replica:
  - `10858118926`;
  - `sha256:d8550cfc53857cb63cafa97ccebc8e97ba89712044dcfb2b61ef322e9d4f02a6`.

### Integrity

Untouched CONFIRM:
- BD: 64 bases / 384 paired views;
- BE: 64 bases / 384 paired views;
- state encodes/base = 1.0 on both;
- probability mass max error < 2.5e-7;
- BD/BE materialized only after every candidate DEV checkpoint was frozen;
- no W7/W7b CONFIRM reuse;
- no W8 diagnostic reuse;
- no Banking77 reuse;
- no typed final/test rows;
- campaign cells 0.

Frozen outcome:

**`SEMANTIC_ALIGNMENT_FAIL`**

`external_public_phase_authorized = false`.

Do not reinterpret this result as PARTIAL.

## 20. DEV-frozen checkpoints

Frozen W6e:
- epoch 0;
- scorer SHA `6d5a7f2d3ed63ecd756181b1cb54e4704f68e5f74983a897f0a68fb4d1d63d2e`.

Projection semantic control:
- 32,768 trainable;
- selected DEV epoch 5;
- scorer SHA `8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102`.

Shared bridge control:
- 4,096 trainable;
- selected DEV epoch 2;
- scorer SHA `a0a06549dffab62296b5993f56ac6a3d44a0e7aa9065dd3e260ad2d22807b288`.

Asymmetric bridge primary:
- 8,192 trainable;
- selected DEV epoch 2;
- scorer SHA `a1149a2a26d8ce7ed65b50ad71ff293e907b60195e26b8ab36da8cbcbf44d4c6`.

Asymmetric bridge replica:
- 8,192 trainable;
- selected DEV epoch 2;
- scorer SHA `50a1d702b30d36546fde04782ccf285a7f716b4485350695f8e94c33a1f5ab09`.

Primary/replica seeds remain independent.

## 21. Untouched CONFIRM-BD

Frozen W6e:
- natural-definition K4 final: **46.875%**;
- natural-definition K4 coarse: 48.438%;
- natural-definition K16 final: **17.188%**;
- terse-label K4 final: 29.688%;
- alignment 16-way top1: **21.875%**.

Projection semantic control:
- definition K4 final: **51.563%**;
- K4 coarse: 48.438%;
- K16 final: **34.375%**;
- label K4: 40.625%;
- alignment top1: **51.563%**.

Shared bridge:
- definition K4 final: **53.125%**;
- K16 final: **25.000%**;
- alignment top1: **25.000%**.

Asymmetric primary:
- definition K4 final: **50.000%**;
- K4 coarse: 48.438%;
- definition K16 final: **20.313%**;
- label K4: 29.688%;
- alignment top1: **21.875%**.

Asymmetric replica:
- definition K4 final: **51.563%**;
- K16 final: **20.313%**;
- alignment top1: **26.563%**.

## 22. Untouched CONFIRM-BE

Frozen W6e:
- definition K4 final: **43.750%**;
- K4 coarse: 43.750%;
- definition K16 final: **17.188%**;
- label K4: 39.063%;
- alignment top1: **32.813%**.

Projection semantic control:
- definition K4 final: **62.500%**;
- K4 coarse: 56.250%;
- definition K16 final: **34.375%**;
- label K4: 40.625%;
- alignment top1: **53.125%**.

Shared bridge:
- definition K4 final: **40.625%**;
- definition K16 final: **14.063%**;
- alignment top1: **42.188%**.

Asymmetric primary:
- definition K4 final: **56.250%**;
- K4 coarse: 54.688%;
- definition K16 final: **21.875%**;
- label K4: 39.063%;
- alignment top1: **40.625%**.

Asymmetric replica:
- definition K4 final: **54.688%**;
- definition K16 final: **20.313%**;
- alignment top1: **42.188%**.

## 23. Why the frozen verdict is FAIL

The asymmetric-primary absolute gates fail on both BD and BE:
- K4 never reaches 75%;
- K16 never reaches 55%;
- alignment top1 never reaches 70%;
- terse-label K4 never reaches 45%;
- required gains over frozen do not hold.

Partial gates also fail:
- BD primary K4 gain is only +3.125 pp, below +10 pp;
- BD primary K16 gain is only +3.125 pp, below +8 pp;
- BD replica K4 gain is +4.688 pp, below +7 pp.
- BE K4 improves, but partial requires both domains.

Architecture-causal superiority is absent:
- asymmetric primary is worse than projection-semantic control on K4, K16 and alignment on both BD/BE;
- it does not establish the preregistered superiority over the shared control either.

No projection/shared full rescue exists:
- projection is the strongest semantic-alignment candidate overall, but still misses the frozen absolute rescue gates;
- shared bridge does not rescue.

Therefore:
**`SEMANTIC_ALIGNMENT_FAIL`**.

## 24. Key scientific result

W9 falsifies the specific hypothesis that a tiny identity-initialized rank-16 residual bridge, including separate state/schema maps, is enough to solve HIRA's stable general semantic transfer limit.

The strongest positive signal is instead the ordinary full projection semantic control.

Projection retuning improves alignment and K16 materially:
- BD alignment: 21.88% -> 51.56%;
- BD K16: 17.19% -> 34.38%;
- BE alignment: 32.81% -> 53.13%;
- BE K16: 17.19% -> 34.38%;
- BE K4: 43.75% -> 62.50%.

But it is not a rescue:
- BD K4 improves only to 51.56%;
- both K16 results remain 34.38%;
- alignment top1 remains ~52%;
- no external phase is authorized.

The bridge paths are notably weaker than projection retuning. This suggests the failure is not simply that state and schema need two small post-projection residual maps. A larger fraction of the shared projection geometry itself appears to need reorganization under semantic supervision.

This is evidence, not yet proof that encoder capacity is the bottleneck.

## 25. What W9 rules out / weakens

W9 weakens:
- a 4,096-param shared low-rank post-projection bridge;
- an 8,192-param asymmetric post-projection bridge;
- the claim that state/schema asymmetry is the primary causal missing mechanism;
- the claim that direct contrastive alignment alone, in these small bridge subspaces, is sufficient.

W9 does **not** establish:
- that A13 itself must be replaced;
- that 32,768 projection retuning is enough for production;
- that semantic alignment objective is useless;
- that external transfer would remain absent after a genuinely successful internal rescue.

## 26. Permanent exposed evidence after W9

AY/AZ/BA/BB TRAIN and BC DEV have been used.

BD/BE CONFIRM are now exposed.

Never use BD/BE for:
- training;
- DEV selection;
- bridge rank selection;
- choosing shared vs asymmetric;
- objective/temperature tuning;
- projection architecture search;
- seed selection;
- threshold/gate changes;
- calibration.

All prior forbidden evidence remains forbidden.

## 27. Authorized next research boundary

Because W9 is FAIL:
- **do not open external/public validation**;
- do not reuse BD/BE;
- do not simply increase bridge rank after seeing this result;
- do not call projection semantic control a rescue.

The next phase should be diagnostic before another rescue.

The most important fresh question is now:

**Is the transferable-semantic ceiling caused primarily by the frozen A13 representation itself, or by the learned 256->128 projection/competitive scoring interface layered on top of A13?**

A fresh diagnostic should compare, on wholly new domains and without training on the diagnostic rows:
1. frozen raw A13 semantic geometry;
2. frozen W6e projection geometry;
3. W9 projection-semantic geometry;
4. possibly a preregistered external frozen sentence-embedding reference as a diagnostic ceiling, not a HIRA production candidate.

It should use low-K first and paired state↔definition identities.

Do not train a new bridge until this representation-ceiling question is localized.

## 28. Continuation instructions

Before another AI proceeds:
1. merge PR #106 only after this closure-doc commit and CI are clean;
2. close issue #105;
3. create the next diagnostic lane from resulting main;
4. create a complete new handoff before fresh exposure;
5. preserve W9 run/artifact/checkpoint hashes above;
6. never reuse BD/BE;
7. keep external validation blocked because W9 did not reach a full rescue verdict.

A future AI should read this file first, then W8, W7c, W7b, W7 and W6j handoffs.
