# R8-W7 handoff — explicit conjunctive coarse-evidence architecture

Status: **PRE-DATA ARCHITECTURAL IMPLEMENTATION ACTIVE. No W7 empirical cache, training result, DEV selection or CONFIRM result exists yet.**

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
