# R8-W7 handoff — explicit conjunctive coarse-evidence architecture

Status: **PRE-AUTHORITY EXECUTION STACK COMPLETE. No W7 empirical cache, training result, DEV selection or CONFIRM result exists yet.**

Issue: #97

Branch:
`feat/r8-w7-conjunctive-coarse`

Base main at branch creation:
`8bc2307fe462a43a544f9eaf5545b69517a5df02`

Read first when restoring the project:
1. this file;
2. `research/R8-W6J-HANDOFF.md`;
3. issue #97.

---

## 0. What Nolane HIRA is

Nolane HIRA is a compact, non-autoregressive, typed decision system.

It is not intended to be a tiny general-purpose next-token LLM.

Long-term goals:

- encode the state exactly once per case;
- accept dynamic semantic schemas instead of a fixed class vocabulary;
- support large changing candidate sets;
- support typed primitives:
  - `choice`;
  - `score`;
  - `noul`;
- remain small enough for constrained/local deployment;
- perform semantic binding and structured decision making;
- preserve probability/reliability integrity;
- make every claimed mechanism falsifiable with fresh, sealed authorities;
- separate internal mechanism evidence from public/general superiority claims.

The project values negative results. Never erase a failed lane merely because a later mechanism looks promising.

---

## 1. Frozen A13 semantic encoder

Unless a future explicitly preregistered phase changes this boundary:

- model: `microsoft/xtremedistil-l6-h256-uncased`;
- revision: `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- model weight SHA-256:
  `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length: 256;
- fully frozen.

W7 must not scale or fine-tune A13.

---

## 2. Production architecture entering W7

### CompetitiveCoarseScorer

Current production semantic binder:

- 256 -> 128 bias-free projection;
- scalar logit scale;
- exactly 32,769 trainable parameters;
- candidate-relative IDF/salience;
- token-level forward late interaction / MaxSim;
- within-option common-mode subtraction;
- weighted mean coverage plus minimum salient-token coverage.

### HIRACore

Legacy parameter count:
- 422,159.

High-level current path:

`state-once memory -> full-K competitive coarse scalar/candidate -> relation-to-state delta -> typed logits/probabilities`.

The relation stage is candidate-local against state memory. It does not provide a direct candidate-candidate reasoning layer.

### NolaneHira

Production runtime supports dynamic schemas and typed decisions while preserving one state compile/encode per case.

---

## 3. Decisive research history

This section exists so a future AI does not repeat dead ends.

### W5

W5a-W5e showed that:
- pooled/simple matching fails;
- token-aware probes alone do not rescue;
- controlled top-layer adaptation does not rescue;
- simply scaling A13 to A22 does not rescue.

Thus raw encoder size is not the proven bottleneck.

W5f late interaction gave a major semantic gain.

W5g contrastive salience:
- overall ~50.52%;
- K128 ~47.92%;
- K255 ~41.67%;
- verdict `CONTRASTIVE_SALIENCE_PARTIAL`.

W5h competitive/common-mode binding:
- verdict `BALANCED_BINDING_PARTIAL`.

W5i:
- fresh forward competitive control itself crossed the semantic rescue region;
- listwise reverse mechanism did not earn causal promotion.

This produced the current competitive scorer.

### W6 / W6b

Competitive scorer was integrated into `NolaneHira` while preserving:
- legacy state_dict compatibility;
- state-once;
- option permutation behavior;
- typed outputs.

W6b fresh typed authority:
- overall 81.50%;
- choice 80.31%;
- score 88.75%;
- K64 60%;
- noul 69.375%;
- hard Brier 0.3283;
- soft ECE 0.16373.

Verdict:
`PRODUCTION_COMPETITIVE_PARTIAL`.

### W6c

Three-temperature calibration strongly improved soft calibration but did not repair hard semantic generalization.

Verdict:
`RELIABILITY_CALIBRATION_PARTIAL`.

Conclusion:
calibration is real but not the dominant K64 semantic bottleneck.

### W6d / W6e

W6d joint HIRA+scorer:
- overall 89.167%;
- K64 54.167%;
- missed the preregistered 55% K64 gate by one case.

Verdict:
`GENERALIZATION_FAIL`.

W6e replication:
- overall remained high;
- K64 dropped to roughly 31-42% on new untouched domains;
- joint adaptation did not replicate.

Verdict:
`JOINT_GENERALIZATION_REPLICATION_FAIL`.

### W6f

Fresh N/O/P high-K rank-path localization, no training.

Found:
- candidate-relative salience is beneficial;
- uniform salience is much worse;
- relation reranking is net helpful;
- gold often remains in coarse top5;
- one-field near-neighbor errors dominate.

Pooled:
`UNRESOLVED_HIGH_K_FAILURE`.

### W6g

Fresh Q/R/S second-order localization.

Stable diagnostic:
`FIELD_SEMANTIC_COLLAPSE`.

This authorized one fresh semantic rescue.

### W6h

Compared:
- frozen control;
- 32,769-param projection retune;
- 8,192-param residual semantic adapter.

Untouched Y:
- frozen K64 47.917%;
- projection 56.250%;
- adapter 45.833%.

Untouched Z:
- frozen 37.500%;
- projection 64.583%;
- adapter 62.500%.

Adapter was not reproducible.

Verdict:
`FIELD_SEMANTIC_FAIL`.

Important bridge finding:
the frozen control already had ~95% full structured one-field pair accuracy even though W6g isolated-value retrieval was weak.

### W6i

Fresh AA/AB/AC representation bridge.

Projection-retune showed `STRUCTURED_PAIR_PROXY_MISMATCH` on AA/AB, but this did not replicate across all frozen checkpoints.

Canonical tagged text and frozen factorized mean/min were substantially worse than production text.

Cross-checkpoint result:
`REPRESENTATION_BRIDGE_UNRESOLVED`.

No local rescue authorized.

### W6j — decisive predecessor to W7

W6j moved one architectural level higher.

Exact empirical head:
`af33754719b1f70e08f9c01991ea41bfd00c496a`

Authority run:
`36072141944`

Merged W6j main:
`8bc2307fe462a43a544f9eaf5545b69517a5df02`

Artifacts:
- cache `10838263710`;
- frozen checkpoints `10838857136`;
- decomposition audit `10838774135`.

Cross-checkpoint outcome:

**`STABLE_HIGH_CARDINALITY_ARCHITECTURE`**

Stable target:

**`COARSE_CONJUNCTION_LIMIT`**

W6e joint-primary pooled:
- K64 coarse 30.729%;
- K64 final 32.813%;
- 100% of global K64 errors had >=1 genuine K2 pair loss;
- eventual wrong winner already beat gold in K2 coarse on 91.473% of global errors;
- winner K2-final -> K64-final reversal only 3.101%;
- all-63-pairs-win -> K64-fail conditional rate 0%;
- relation net +2.083 pp;
- oracle-perfect conjunction final top1 99.479%.

Projection-retune pooled:
- K64 final 50.521%;
- wrong winner beats gold in K2 coarse 89.474%;
- winner reversal 7.368%;
- oracle final 99.479%.

Semantic-adapter pooled:
- K64 final 42.188%;
- wrong winner beats gold in K2 coarse 94.595%;
- winner reversal 5.405%;
- oracle final 99.479%.

Scientific meaning:

The stable failure is **not** primarily:
- global K64 set-context rank reversal;
- relation-stage destruction;
- calibration;
- candidate-relative salience;
- raw encoder size.

The learned coarse semantic scorer itself usually prefers the eventual wrong candidate even in an isolated K2 comparison.

W7 is authorized specifically to test a learned **explicit conjunction of independently evidenced schema factors**.

---

## 4. Prior-art boundary

W7 must not claim novelty merely because it uses set/factor aggregation.

Relevant conceptual prior art:
- Deep Sets — permutation-invariant set aggregation, arXiv:1703.06114;
- late-interaction multi-vector retrieval / MaxSim family;
- analysis of late-interaction matching, arXiv:2403.13291;
- controlled set-compositional retrieval evidence, arXiv:2605.03824.

The HIRA-specific question is narrower:

**Can an explicit factor-evidence AND branch, with only +3 trainable parameters over the existing 32,769-param scorer and with A13/HIRACore frozen, causally repair the coarse-conjunction failure better than an equal-data free-form retune?**

Do not state that W7 is novel without a dedicated novelty review.

---

## 5. W7 explicit factor interface

Every diagnosis option keeps its normal free-form `criterion_text`.

In addition it exposes an ordered tuple of schema factors.

Authority factors:
1. entity/equipment;
2. location/zone;
3. anomaly/event;
4. channel/source.

Each factor is rendered as a natural:
`role + value`
phrase.

At inference the factor interface contains:
- no gold marker;
- no match bit;
- no domain identifier;
- no hidden target information.

It is simply an explicit decomposition of information already present in the option schema.

Library code must support variable factor counts even though W7 authority fixes four.

---

## 6. W7 architecture — ConjunctiveEvidenceScorer

The existing competitive free-form scorer remains the base path.

For each candidate factor:

1. factor phrase is encoded on the schema side;
2. use the **same** shared 256->128 projection as the competitive scorer;
3. compare projected factor tokens to the single cached state token sequence using token-level late interaction;
4. aggregate factor-token evidence to one scalar `e_f`.

No second state encoding is allowed.

### Factor match transformation

One shared threshold and one shared positive temperature:

`z_f = (e_f - threshold) / temperature`.

`p_f = sigmoid(z_f)`.

### Symmetric conjunction

`log_and = mean_f(log(p_f + eps))`.

Properties:
- smooth AND / geometric-mean log evidence;
- any weak required factor penalizes the candidate;
- factor order is permutation invariant;
- no per-role learned scalar can memorize a domain slot.

### Final coarse logit

`coarse = freeform_competitive_logit + alpha * log_and`

with non-negative conjunction gate `alpha`.

HIRACore relation remains unchanged/frozen in W7 primary comparison.

---

## 7. Parameter budget

Free-form control:
- 32,769 trainable.

Conjunctive scorer:
- same projection and scale;
- shared threshold: +1;
- shared temperature parameter: +1;
- shared non-negative conjunction gate: +1.

Total target:

**32,772 trainable parameters.**

W7 forbids:
- new MLP;
- new attention stack;
- per-role parameters;
- extra encoder layer;
- HIRACore unfreeze in the primary comparison.

The mechanism must be small enough that success cannot be explained by a meaningful capacity increase.

---

## 8. Frozen checkpoint initialization

All W7 trainable candidates initialize from exact W6e joint-primary:

- HIRA SHA:
  `d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588`;
- scorer SHA:
  `6d5a7f2d3ed63ecd756181b1cb54e4704f68e5f74983a897f0a68fb4d1d63d2e`;
- artifact:
  `10805567861`;
- source run:
  `35993402202`.

W6h projection-retune is **not** the W7 initialization.
It is prior evidence only.

---

## 9. W7 candidates

### Candidate 0 — frozen-w6e-control

- exact W6e HIRA/scorer;
- 0 trainable;
- reference only.

### Candidate 1 — freeform-retune-control

- HIRACore frozen;
- current competitive scorer trainable;
- exactly 32,769 trainable params;
- W7 TRAIN only;
- receives typed objective;
- receives the same diagnosis one-field pair-margin auxiliary.

This is the primary equal-data control.

### Candidate 2 — conjunctive-primary

- HIRACore frozen;
- ConjunctiveEvidenceScorer;
- exactly 32,772 trainable params;
- same cases/epochs/optimizer steps as candidate 1;
- typed objective;
- same diagnosis one-field pair-margin auxiliary;
- balanced factor-evidence loss.

### Candidate 3 — conjunctive-replica

Same as candidate 2, but:
- independent optimizer seed;
- independent DEV checkpoint selection;
- cannot replace candidate 2 after CONFIRM.

Primary causal comparison:
**candidate 2 vs candidate 1**.

---

## 10. Fresh W7 authority domains

All values, templates, role wording and case IDs must be disjoint from W5-W6j.

TRAIN:
- AG — orbital power-distribution maintenance;
- AH — pharmaceutical cleanroom control;
- AI — automated rail-yard inspection;
- AJ — deep-ocean sensor servicing.

DEV:
- AK — high-altitude weather instrumentation.

Sealed untouched CONFIRM:
- AL — microgrid thermal management;
- AM — robotic food-packaging safety.

Frozen seeds:
- AG 251347;
- AH 251353;
- AI 251359;
- AJ 251363;
- AK 252461;
- AL 253567;
- AM 254671.

Optimization seeds:
- freeform: 1901;
- conjunctive primary: 1907;
- conjunctive replica: 1913.

AL/AM may not be materialized before **all three trainable candidates** are independently frozen on DEV-AK.

---

## 11. Typed authority shape

Each state has five decisions:

1. diagnosis — choice;
2. response — choice;
3. needs_review — noul;
4. risk — score;
5. urgency — score.

Diagnosis K:
`{8,16,32,64}`.

Strata:

`K x severity{0,1,2,3} x confidence{verified,provisional,uncertain}`

= 48 strata.

TRAIN:
- 384 states;
- 1,920 typed decisions;
- 96 states/source domain;
- 8 states/stratum total.

DEV-AK:
- 192 states;
- 960 decisions;
- 4 states/stratum.

CONFIRM-AL:
- 192 / 960.

CONFIRM-AM:
- 192 / 960.

All candidates must use identical TRAIN cases and equal optimizer case steps.

---

## 12. Candidate anatomy

K64 diagnosis cases include:
- gold;
- >=12 one-field negatives;
- >=20 two-field negatives;
- >=15 three-field negatives;
- remaining far/cross-combination negatives.

Every option's factor tuple must correspond exactly to the semantic values used to render its normal free-form criterion.

This identity must be hash-checked.

Lower K must retain near-neighbor pressure.

---

## 13. Frozen optimization

Candidates 1/2/3:
- exactly 6 epochs;
- AdamW;
- lr 3e-4;
- weight decay .01;
- deterministic shuffle;
- no warmup;
- no scheduler;
- no AMP;
- no clipping;
- full-K training;
- HIRACore frozen.

### Typed objective

Frozen W6 weights:
- hard CE 1.0;
- teacher KL .5;
- hard Brier .1;
- soft Brier .5;
- ordinal MAE .2.

### Pair margin auxiliary

Candidates 1/2/3:
- diagnosis one-field gold-vs-negative margin;
- weight .25;
- margin .20.

### Balanced factor evidence loss

Candidates 2/3 only:
- factor positive iff candidate factor equals gold factor at that factor index;
- per factor, positive and negative losses are averaged separately;
- then equally combined;
- weight .50.

This is part of the W7 mechanism.
If W7 succeeds, do not later claim aggregation alone caused the gain.

No loss weight may change after first eligible W7 empirical cache/training output.

---

## 14. DEV selection

Each trainable candidate independently selects its checkpoint on AK.

Lexicographic:
1. diagnosis K64 final top1;
2. K32 final top1;
3. diagnosis choice accuracy;
4. overall typed accuracy;
5. lower mean K64 pair-loss count;
6. lower hard Brier;
7. earlier epoch.

There is no cross-candidate DEV winner.

Only after all candidate checkpoints are frozen may AL/AM be generated.

---

## 15. Mandatory metrics

Typed:
- overall;
- choice;
- noul;
- score;
- hard/soft Brier;
- raw/soft ECE;
- score MAE;
- probability max error;
- state encodes/case.

Diagnosis:
- K8/K16/K32/K64 top1/top5/MRR;
- coarse/final top1;
- relation rescue/damage.

Pair anatomy:
- gold-vs-all-negative K2 win rate;
- mean pair-loss count/base;
- loss histogram by semantic distance;
- one-field K2 accuracy.

Factor branch:
- balanced factor BCE;
- positive/negative factor accuracy;
- AUROC only if both classes exist and implementation is validated;
- minimum factor match probability for gold;
- minimum factor match probability for wrong winner;
- conjunction residual magnitude;
- base-freeform vs residual contribution.

Integrity:
- exact model hashes;
- factor/schema hashes;
- parameter counts;
- optimizer steps;
- leakage checks;
- confirm sealing;
- state-once.

---

## 16. Frozen W7 gates

### Absolute conjunctive-primary competence

Must pass on **both AL and AM**:

- overall >= .80;
- diagnosis choice >= .75;
- K32 final top1 >= .75;
- K64 final top1 >= .65;
- noul >= .70;
- score >= .75;
- probability mass max error <=1e-6;
- state encodes/case =1.0.

### Primary causal gates vs freeform-retune

Must pass separately on AL and AM:

- K64 gain >= +.10;
- K32 gain >= +.05;
- one-field K2 pair accuracy gain >= +.05;
- mean K64 pair-loss count reduction >=25%;
- overall regression <=.02;
- score regression <=.02;
- noul regression <=.02.

### Replica robustness

On both AL and AM:

- K64 >= .60;
- K64 gain vs freeform >= +.05;
- K32 >= .70;
- overall >= .78;
- one-field pair accuracy gain vs freeform >= +.03.

### Verdict

`CONJUNCTIVE_COARSE_RESCUE`:
all primary absolute + causal + replica gates pass on both AL/AM.

`CONJUNCTIVE_COARSE_PARTIAL`:
full rescue fails, but primary K64 gain >=+.05 on both domains and overall regression <=.03 on each.

`CONJUNCTIVE_COARSE_FAIL`:
otherwise.

No new verdict may be invented after exposure.

---

## 17. Forbidden evidence

Never use for W7 training/selection/tuning:

- W6b CONFIRM;
- W6c CONFIRM;
- W6d F;
- W6e L/M;
- W6f N/O/P;
- W6g Q/R/S;
- W6h Y/Z;
- W6i AA/AB/AC;
- W6j AD/AE/AF;
- typed final/test rows;
- public campaign cells.

Historical checkpoints may only be loaded as exact frozen provenance explicitly allowed above.

---

## 18. Scientific interpretation boundaries

If W7 rescues:
- this supports the W7 explicit-factor conjunction architecture **on the internal fresh authority**;
- it does not prove broad real-world superiority;
- production integration belongs in W8;
- external/public validation still required.

If W7 partial/fails:
- AL/AM become exposed and cannot be reused;
- do not modify gates;
- diagnose from frozen evidence before a new lane.

Do not call:
- the factor labels an inference oracle;
- W6j oracle probe a production mechanism;
- W7 a novel architecture without separate prior-art review.

---

## 19. Current implementation status

Completed:
- W6j authority closed and merged;
- issue #97 preregistered before W7 empirical work;
- W7 branch created from exact main;
- this handoff created.

Not implemented yet:
- W7 fresh domain generator;
- explicit factor schema/cache representation;
- `ConjunctiveEvidenceScorer`;
- factor-token projection path;
- pair/factor objectives;
- candidate trainer;
- DEV freezer;
- sealed AL/AM generator/evaluator;
- unit workflows;
- full authority workflow.

No W7 empirical data exists.

---

## 20. Immediate continuation steps

1. Implement factor interface as research/library structures without changing existing production option semantics.
2. Implement `ConjunctiveEvidenceScorer` with exact +3-param budget and identity/backward-safety tests.
3. Implement AG-AK generator only; keep AL/AM sealed.
4. Add freshness tests against W5-W6j.
5. Add candidate-set/factor identity contracts.
6. Add base-centric state-once cache including factor token artifacts.
7. Implement frozen objectives and exact optimizer-step accounting.
8. Implement independent AK selection for candidates 1/2/3.
9. Implement confirm generator so AL/AM cannot materialize outside the confirm script.
10. Add full pre-data unit/contracts.
11. Only after all pre-data contracts are green, enable authority workflow.
12. Once eligible TRAIN/DEV exposure occurs, do not change seeds/losses/gates/selection.
13. After all checkpoints freeze, expose AL/AM exactly once.
14. Freeze result into this handoff before merge.

---

## 21. Handoff discipline

At the end of every meaningful W7 work session, update this file with:

- exact branch head;
- exact runs and conclusions;
- artifacts/digests;
- implementation added;
- pre-data vs exposed status;
- empirical metrics if exposure occurred;
- frozen verdict;
- remaining work;
- forbidden next moves.

A new AI should be able to continue without relying on chat memory.


---

## 22. Current pre-authority execution checkpoint

This section supersedes the stale implementation checklist in section 19/20.

Exact fully validated implementation head before this handoff update:

`c0457e2ad75b0b59716ec2590f73cd6c3dfff6f2`

Pre-authority unit run:

`36097752382` — **PASS**

That run compiled and tested the complete pre-authority W7 stack.

Implemented since the earlier handoff state:

- `src/nmd/conjunctive_cache.py`
  - reuses exact W6b typed/state cache path;
  - state compile remains exactly one/case;
  - explicit factor phrases are schema-side encoder work and separately counted;
  - diagnosis caches store [K,4,T,256] factor tokens/masks;
  - factor target masks are derived only from candidate/gold schema-factor identity;
  - one-field negative identities are frozen from semantic signatures;
  - factor-identity and semantic-signature hashes are recorded.
- `scripts/r8_w7_build_cache.py`
  - builds only AG/AH/AI/AJ TRAIN + DEV-AK;
  - checks A13 model/revision/weight SHA;
  - checks value freshness through W6j;
  - keeps AL/AM sealed.
- `src/nmd/conjunctive_training.py`
  - freezes HIRACore for every W7 candidate;
  - free-form control trainable params = 32,769;
  - conjunctive primary/replica trainable params = 32,772;
  - exact frozen typed objective;
  - pair-margin auxiliary weight .25 / margin .20 for all trainable candidates;
  - balanced factor BCE weight .50 only for conjunctive candidates;
  - exact K2 coarse/final pair evaluation;
  - K64 pair-loss accounting;
  - DEV-AK lexicographic selection;
  - absolute, causal, replica and final verdict gates.
- `scripts/r8_w7_train_candidate.py`
  - binds exact W6e joint-primary HIRA/scorer SHAs;
  - equal TRAIN cases/steps;
  - independent optimization seeds;
  - saves per-candidate checkpoint/receipt.
- `scripts/r8_w7_freeze_candidates.py`
  - requires all four candidates;
  - verifies exact shared cache provenance;
  - verifies equal optimizer budget;
  - verifies primary/replica seed independence;
  - copies immutable DEV-frozen checkpoints;
  - refuses any prior AL/AM exposure.
- `scripts/r8_w7_confirm.py`
  - the only path allowed to call `generate_w7_confirm(..., allow_confirm=True)`;
  - materializes AL then AM only after a valid all-candidate freeze;
  - evaluates all four frozen candidates;
  - computes only the preregistered `CONJUNCTIVE_COARSE_RESCUE/PARTIAL/FAIL` verdict.
- unit contracts now cover:
  - fresh authority counts/balance/sealing;
  - K64 exact distance anatomy 1/12/20/15/16 for distances 0/1/2/3/4;
  - free-form/factor identity;
  - factor target masks;
  - deterministic factor/signature hashes;
  - exact parameter budgets and HIRA freeze;
  - verdict non-conflation;
  - confirm capability guard;
  - production competitive scorer regressions.

Pre-data ambiguity was also frozen on issue #97 before exposure:
- causal one-field K2 gain = **coarse K2**;
- causal K64 pair-loss reduction = **coarse K2 pair-loss count**;
- DEV pair-loss tie-break = **final K2 pair-loss count**;
- both coarse and final pair metrics are always reported.

No W7 A13 cache has been generated yet.
No W7 candidate has been trained yet.
DEV-AK has not been evaluated for selection yet.
CONFIRM-AL/AM remain sealed.

### Immediate continuation

1. Revalidate this handoff-only head with the W7 unit gate.
2. Only after that exact pre-data head is green, add the gated W7 authority workflow as a workflow-only commit.
3. Authority chain must be:
   `unit -> exact W6e provenance + fresh AG-AK cache -> 4 candidates -> DEV freeze -> AL/AM confirm`.
4. Exact upstream W6e artifact:
   - run `35993402202`;
   - artifact `10805567861`;
   - artifact digest `sha256:fe0f176514b1cee0964212978b839b351ba6f35260ce0a035a1346680fc855bd`;
   - HIRA SHA `d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588`;
   - scorer SHA `6d5a7f2d3ed63ecd756181b1cb54e4704f68e5f74983a897f0a68fb4d1d63d2e`.
5. Once the first eligible W7 TRAIN/DEV cache is exposed, do not change:
   - domains/seeds;
   - loss weights;
   - factor operator;
   - candidate budgets;
   - DEV selection order;
   - gates/verdicts.
6. Once AL/AM are exposed, never reuse them for tuning or another W7 mechanism.

A future AI should read this file first. It now contains enough project history, frozen evidence, implementation state, exact provenance and next actions to continue without chat memory.


---

## 22. Authoritative W7 closure

Exact empirical head:
`e4563996b1a905c777460f723bfbec1a41ac92ea`

Authority run:
`36098102118`

All jobs PASS:

`unit -> exact W6e provenance + fresh AG-AK cache -> four independent candidates -> DEV-AK freeze -> untouched AL/AM confirm`.

Artifacts:
- cache `10848700765`, digest `sha256:a8e9b3cee0886203a5129a7aea7aec79a0b923475c092569372796fcbdaa0c06`;
- conjunctive primary `10847999931`, digest `sha256:ff126340abdeb79688157685d773061e373e0f5a94d2ef4c28d7b69f0d2878a4`;
- conjunctive replica `10848258708`, digest `sha256:d43e0c910cb6c54e33c0de52d8ed92cff5df02336c5224418079045d3f4e41de`;
- freeform retune control `10848074367`, digest `sha256:c1a5a030e744eb7f1cc13580f1ba2768b6463daf8ebea7b2517384bb35ccf153`;
- freeze `10849186193`, digest `sha256:20e1fa4b25be894c080470aeaecdf6d523bcc487a28991954b1bac97c143caa4`;
- authoritative CONFIRM `10849032756`, digest `sha256:528e19457d0fb26c43c58a05522b3a507e2c5cd1e23088625837414ea1ff4f7d`.

Integrity:
- AL/AM each 192 states / 960 typed decisions;
- both generated only after every candidate DEV checkpoint was frozen;
- state encodes/case = 1.0;
- probability mass max error < 1.8e-7;
- primary and replica optimization seeds independent;
- no W6b-W6j exposed rows;
- no typed final/test rows;
- campaign cells 0.

### Frozen verdict

**`CONJUNCTIVE_COARSE_FAIL`**

- full rescue: FAIL;
- partial rescue: FAIL;
- no gate, threshold, seed, loss or operator may be changed using AL/AM.

### Untouched CONFIRM-AL

Frozen W6e control:
- overall 82.917%;
- diagnosis 40.104%;
- K32 final 35.417%;
- K64 final 18.750%;
- one-field K2 coarse 72.721%;
- mean K64 K2-coarse pair losses 12.396.

Equal-data freeform retune:
- overall **95.833%**;
- diagnosis **87.500%**;
- choice **93.229%**;
- K32 final **91.667%**;
- K64 final **81.250%**;
- K64 coarse **77.083%**;
- one-field K2 coarse **95.334%**;
- mean K64 K2-coarse pair losses **1.042**.

Conjunctive primary:
- overall 91.250%;
- diagnosis 66.146%;
- choice 83.073%;
- K32 final 68.750%;
- K64 final 43.750%;
- K64 coarse 50.000%;
- one-field K2 coarse 84.505%;
- mean K64 K2-coarse pair losses 4.896;
- factor balanced BCE 1.212;
- mean conjunction residual magnitude 0.01083.

Conjunctive replica:
- overall 91.563%;
- diagnosis 63.542%;
- choice 81.771%;
- K32 final 70.833%;
- K64 final 52.083%;
- one-field K2 coarse 81.315%;
- mean K64 pair losses 4.313.

### Untouched CONFIRM-AM

Frozen W6e control:
- overall 81.979%;
- diagnosis 35.417%;
- K32 final 33.333%;
- K64 final 10.417%;
- one-field K2 coarse 71.962%;
- mean K64 pair losses 12.375.

Equal-data freeform retune:
- overall **91.250%**;
- diagnosis **73.438%**;
- choice **86.198%**;
- K32 final **75.000%**;
- K64 final **62.500%**;
- K64 coarse **60.417%**;
- one-field K2 coarse **90.820%**;
- mean K64 K2-coarse pair losses **2.042**.

Conjunctive primary:
- overall 92.917%;
- diagnosis 74.479%;
- choice 87.240%;
- K32 final 68.750%;
- K64 final 64.583%;
- K64 coarse 58.333%;
- one-field K2 coarse 86.914%;
- mean K64 K2-coarse pair losses 3.250;
- factor balanced BCE 1.737;
- mean conjunction residual magnitude 0.00860.

Conjunctive replica:
- overall 92.396%;
- diagnosis 71.354%;
- choice 85.677%;
- K32 final 68.750%;
- K64 final 62.500%;
- one-field K2 coarse 84.158%;
- mean K64 pair losses 4.417.

### Why the primary causal hypothesis failed

Against the equal-data freeform retune, conjunctive-primary did **not** produce the preregistered causal gains.

AL:
- K64: 43.75% vs 81.25% -> **-37.50 pp**;
- K32: 68.75% vs 91.67% -> **-22.92 pp**;
- one-field K2 coarse: 84.51% vs 95.33% -> **-10.83 pp**;
- K64 pair-loss count is worse: 4.896 vs 1.042.

AM:
- K64: 64.58% vs 62.50% -> **+2.08 pp**, far below the +10 pp causal gate;
- K32: 68.75% vs 75.00% -> **-6.25 pp**;
- one-field K2 coarse: 86.91% vs 90.82% -> **-3.91 pp**;
- K64 pair-loss count is worse: 3.25 vs 2.042.

Replica does not rescue the mechanism:
- AL K64 52.08%, still far below freeform 81.25%;
- AM K64 62.50%, equal to freeform rather than +5 pp;
- one-field causal gain is negative on both.

### Scientific interpretation

W6j's `COARSE_CONJUNCTION_LIMIT` localization remains valid as a diagnosis of the frozen predecessor. W7 falsifies the specific proposed remedy:

**an explicit schema-factor MaxSim branch + shared threshold/temperature + smooth log-AND + balanced factor supervision is not a reproducible improvement over equal-data free-form projection retuning.**

The strongest positive evidence in W7 is instead the equal-data free-form retune:
- it moves frozen K64 18.75% -> 81.25% on AL;
- and 10.42% -> 62.50% on AM;
- while strongly reducing K2 pair losses.

Therefore the next diagnostic question is not “how do we make the AND branch bigger?”. The frozen result suggests that the existing shared projection can learn much of the missing conjunction behavior from the W7 training distribution without an explicit factor residual, while the explicit factor branch may interfere with or duplicate evidence already represented in the free-form option path.

Do not infer that free-form retuning is already a general production rescue. AM K64 remains below the old W7 absolute primary threshold of 65%, and W7 was designed to test the conjunction mechanism, not to promote the control.

### Forbidden next moves

AL/AM are now exposed. Never use them for:
- training;
- DEV selection;
- threshold/temperature/alpha tuning;
- loss-weight tuning;
- factor-operator search;
- seed selection;
- candidate selection.

Do not:
- increase factor-branch capacity using AL/AM feedback;
- alter the W7 gates and reclassify this as PARTIAL;
- choose the replica over the primary after CONFIRM;
- claim explicit conjunction works because AM primary is +2.08 pp;
- claim freeform retune is broadly production-ready from AL/AM alone.

### Authorized next research direction

Before any new rescue architecture, run a fresh **control-mechanism attribution** study:

1. isolate why equal-data freeform retuning improved so strongly;
2. separate projection adaptation from pair-margin supervision and ordinary typed supervision;
3. test whether the improvement is reproducible on fresh domains;
4. measure whether it repairs the W6j coarse-conjunction anatomy specifically;
5. keep HIRACore and A13 frozen;
6. use wholly fresh data; AL/AM are forbidden;
7. maintain a new handoff before empirical exposure.

A future AI should read this file, then W6j handoff, then issue #97/PR #98 before opening the next lane.
