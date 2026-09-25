# R8-W9 handoff — contrastive semantic-alignment bridge

Status: **PRE-DATA. No W9 AY–BE A13 cache, training output, DEV selection or CONFIRM result exists yet.**

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
