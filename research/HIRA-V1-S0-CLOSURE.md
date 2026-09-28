# HIRA V1 S0 closure — query-conditioned semantic reset

Status: **CLOSED — HIRA_V1_S0_QUERY_REPAIR_DEV_FAIL**

Issue: #171  
PR: #172  
Base v0 main: `b28491c9fc2da868df9153a778742b9a7ea74e6e`

## 1. Purpose

S0 was the first Hira v1 research milestone after the complete Hira v0 M5 closure.

Its purpose was to test a root-cause hypothesis:

> the frozen v0 semantic coarse path underuses the actual question and therefore cannot reliably distinguish two different queries over the same state and option set.

S0 did not reopen Hira v0 evidence and did not authorize any post-DEV rescue.

## 2. S0-A — exact v0 question-blindness localization

Authority:

- run: `36382301792`
- artifact: `10952694461`
- digest: `sha256:f96fa93dfd2f640008a1ac3cc9ce40129cda9d57bbf062c6616a7b640c87a812`
- outcome: `HIRA_V1_S0_QUESTION_BLINDNESS_CONFIRMED`

Eight fresh paired states:
- 4 English;
- 4 Vietnamese;
- exact state fixed within each pair;
- exact option set fixed;
- only question changes;
- correct option changes with the question.

Exact frozen M4 observation:
- question token embeddings changed: **8/8**
- decision logits identical: **8/8**
- probabilities identical: **8/8**
- selected choice identical: **8/8**
- state encode calls: **8**

This directly confirms question blindness in the frozen v0 `coevidence_symmetric_semantic` coarse path.

It does not prove question omission was the only v0 semantic bottleneck.

## 3. S0-B0 — zero-new-parameter question conditioning

First run `36382670392` failed in unit tests before baseline exposure due to a subclass freeze-dispatch bug. No baseline score was exposed.

Qualified authority:

- run: `36382895869`
- artifact: `10952918278`
- digest: `sha256:e98d93b1db860a4e7f8a867c0789ced676f74f357fff90c964b0fb5b4daa0af9`
- outcome: `HIRA_V1_S0_PARAMETER_FREE_QUERY_BASELINE_READY`

Observed:
- added parameters: **0**
- trainable parameters: **0**
- question changed logits: **100%**
- question changed selected choice: **0%**
- accuracy: **0.375**
- paired both-correct: **0.0**
- state encode calls: **8**
- probability mass max error: **1.1920928955078125e-07**

Conclusion:

Using the question is causally relevant to the semantic geometry, but a fixed query-relevance rule is insufficient to reliably move the decision boundary.

## 4. S0-C candidate

Candidate:

`QueryConditionedCoEvidenceScorer`

Frozen base:
- exact W28 T0 projection;
- exact W34 state/schema residual adapters;
- exact W34 interaction maps;
- exact W34 co-evidence composition maps;
- A13 semantic encoder;
- HIRACore.

New trainable query path:
- `query_basis: 128 -> 8`
- `query_state: 128 -> 8`
- `query_schema: 128 -> 8`

Parameter accounting:
- W34 base candidate: **8,192**
- new query-binding parameters: **3,072**
- complete candidate: **11,264**
- S0-C optimization surface: exactly **3,072**

Identity boundary:
- query-state/query-schema paths initialize at zero;
- initial query multiplier = 1;
- with identical W28/W34 weights, initial V1 logits equal W34 exactly.

## 5. S0-C preregistered authority

TRAIN:
- 256 base states;
- 512 queries;
- 50% English / 50% Vietnamese;
- 4 fresh domains;
- K=4;
- opaque option IDs.

DEV:
- 64 base states;
- 128 queries;
- 50% English / 50% Vietnamese;
- fresh DEV templates and values;
- no exact TRAIN state/question overlap;
- no S0-A localization rows;
- no M5 final rows;
- no W29-W34 sealed rows.

Optimization:
- AdamW;
- seed 5101;
- 12 epochs;
- lr 5e-4;
- weight decay 0.01;
- gradient clip 1.0;
- CE + 0.20 paired swap-margin;
- margin 0.15.

Only the 3,072 query-binding parameters were trainable.

## 6. TRAIN/DEV authority

First TRAIN/DEV run `36383329831` failed in a Python 3.12 dynamic-import freshness test before any optimizer step or DEV exposure. Data, model, optimizer and gates were unchanged.

Qualified run:

- run: `36383487080`
- artifact: `10953344707`
- digest: `sha256:bf3c2e14dd6f08927bd5f48396c99e897cd85b6bb8b19ea75d6f2f39803972b7`
- outcome: `HIRA_V1_S0_QUERY_REPAIR_DEV_FAIL`

Selected DEV epoch:

**8**

Selected query checkpoint SHA256:

`1c00d903d8fda91fd3f905eaed2dac21dbd9e49d90422e89fb1a9b01b70575d5`

Selected DEV metrics:
- overall accuracy: **0.3828125**
- English accuracy: **0.5000000**
- Vietnamese accuracy: **0.2656250**
- worst-language accuracy: **0.2656250**
- paired both-correct: **0.0937500**
- question-swap selected-choice-change rate: **0.4218750**
- option-order flip rate: **0.0000000**
- probability mass max error: **1.1920928955078125e-07**
- full-K: **PASS**
- relation delta: **0**
- TRAIN state encode calls: **256**
- DEV state encode calls: **64**

## 7. Frozen DEV gate

Required:
- overall accuracy >= 0.80 — **FAIL**
- English accuracy >= 0.70 — **FAIL**
- Vietnamese accuracy >= 0.70 — **FAIL**
- paired both-correct >= 0.65 — **FAIL**
- question-swap choice-change >= 0.70 — **FAIL**
- option-order flip <= 0.02 — **PASS**
- probability mass error <= 1e-6 — **PASS**
- full-K — **PASS**
- relation delta = 0 — **PASS**
- trainable parameters = 3,072 — **PASS**
- state-once TRAIN/DEV — **PASS**

S0-C therefore fails scientifically, not operationally.

## 8. Learning dynamics

TRAIN mean loss decreased steadily:
- epoch 1: ~1.3807
- epoch 8: ~1.2071
- epoch 12: ~1.1644

DEV did not improve correspondingly.

Best DEV accuracy was 0.3828125 at the selected epoch, and paired query correctness remained 0.09375.

This indicates that the compact multiplicative low-rank query gate can fit the fresh TRAIN authority but does not generalize sufficiently to the fresh DEV question templates.

This is evidence against this S0 mechanism, not permission to tune against the exposed DEV set.

## 9. Scientific conclusion

S0 establishes three separate facts:

1. **The frozen v0 semantic coarse path is genuinely question-blind.**
2. **Simply injecting question relevance with zero new parameters changes logits but is not sufficient.**
3. **A learned 3,072-parameter multiplicative query-binding gate also remains insufficient on fresh DEV.**

The mechanism preserves:
- state-once;
- full-K;
- probability integrity;
- exact option-order invariance;
- frozen W28/W34/A13/HIRACore boundaries.

The remaining semantic problem requires a stronger mechanism than independent low-rank relevance gating.

A defensible next hypothesis is **query-keyed evidence extraction**: use question semantics to select/aggregate the relevant state evidence before option scoring, rather than only multiplying an already-computed state-option similarity tensor.

The low Vietnamese DEV result also leaves multilingual semantic representation as an unresolved problem; S1 must not assume query routing alone solves it.

## 10. Evidence boundary

S0 TRAIN/DEV is now exposed.

Forbidden for every future candidate:
- fitting;
- hyperparameter tuning;
- architecture ranking;
- candidate selection;
- threshold selection.

Only aggregate S0 observations may motivate S1.

S0 selected checkpoint may be retained as historical evidence but must not be retuned against S0 DEV.

## 11. S0-D decision

**S0-D SEALED CONFIRM IS NOT AUTHORIZED.**

The preregistered rule required S0-C DEV to pass first.

No Laya/Jev/M5 external benchmark reopening is authorized from S0.

## 12. Closure decision

S0-A: **CLOSED / CONFIRMED ROOT CAUSE**  
S0-B0: **CLOSED / INSUFFICIENT**  
S0-C: **CLOSED / DEV FAIL**  
S0-D: **NOT OPENED**  
Production-ready: **false**  
Laya/Jev parity: **not claimed**

S0 is complete as a scientifically useful negative result.

The next track must be separately versioned as **HIRA V1 S1** and must use wholly fresh authority data.
