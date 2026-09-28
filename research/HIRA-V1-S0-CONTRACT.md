# HIRA V1 S0 — query-conditioned semantic reset contract

Status: **ACTIVE / PREREGISTERED**

Issue: #171  
PR: #172  
Base v0 main: `b28491c9fc2da868df9153a778742b9a7ea74e6e`

## 1. Immutable v0 boundary

Hira v0 evidence is immutable.

Final v0 matched scorecard:
- WIN: 1
- LOSS: 16
- NOT_COMPARABLE: 5
- MISSING: 31
- Laya/Jev semantic parity: not established
- production_ready: false

No v1 fitting, selection or architecture choice may use:
- M5 final held-out labels;
- M5 fresh confirmatory labels;
- exposed W29-W34 sealed rows.

## 2. S0-A localization authority

Frozen exact-M4 authority:

- run: `36382301792`
- artifact: `10952694461`
- artifact digest: `sha256:f96fa93dfd2f640008a1ac3cc9ce40129cda9d57bbf062c6616a7b640c87a812`
- outcome: `HIRA_V1_S0_QUESTION_BLINDNESS_CONFIRMED`

Eight fresh paired diagnostic states:
- 4 English;
- 4 Vietnamese;
- state fixed within pair;
- options fixed within pair;
- question changes;
- gold answer changes.

Observed on exact frozen M4 wheel:
- question token embeddings changed: 8/8;
- logits identical: 8/8;
- probabilities identical: 8/8;
- selected option identical: 8/8;
- state encodes: 8/8 exactly once.

Therefore the final v0 `coevidence_symmetric_semantic` coarse path is empirically question-blind when state/options remain fixed.

This is a localization result. It does not prove question omission is the only semantic bottleneck.

## 3. V1-A architecture

Candidate:

`QueryConditionedCoEvidenceScorer`

Base:
- exact W28 T0 projection;
- exact frozen W34 state/schema residual adapters;
- exact frozen W34 interaction maps;
- exact frozen W34 co-evidence composition maps.

New query path:
- `query_basis: 128 -> 8`
- `query_state: 128 -> 8`
- `query_schema: 128 -> 8`

Parameter counts:
- W34 candidate base: 8,192
- new query binding: 3,072
- complete candidate surface: 11,264
- trainable S0-C surface: exactly **3,072**
- target complete resident model: < 14M parameters

Identity rule:
- query_state and query_schema initialize to zero;
- initial query multiplier is exactly 1.0;
- with the same W28/W34 tensors, initial V1 logits equal W34 logits exactly.

No dataset-, domain-, factor-, primitive- or language-specific head is allowed.

## 4. V1-B0 parameter-free baseline

Before learned query fitting, measure a zero-new-parameter baseline.

It:
- reuses exact W28/W34 parameters;
- projects question content tokens through frozen W28;
- uses fixed cosine-like state/question and option/question relevance;
- adds 0 resident/trainable parameters.

This baseline is diagnostic only. Its score cannot be used as a final promotion authority.

## 5. S0-C fresh TRAIN/DEV authority

### Dataset structure

Use deterministic synthetic paired-query tasks designed specifically to require the question.

Each base state contains at least two independently queryable facts.

Within each pair:
- exact state fixed;
- exact option set fixed;
- question changes;
- gold answer changes.

TRAIN:
- 256 base states;
- 512 queries;
- 4 wholly fresh template domains;
- 50% English / 50% Vietnamese.

DEV:
- 64 base states;
- 128 queries;
- wholly fresh templates/lexical values;
- 50% English / 50% Vietnamese;
- no exact state/question sentence overlap with TRAIN;
- no exact row overlap with S0-A localization.

K:
- exactly 4 logical options per query;
- full-K only;
- opaque option IDs;
- option ordering randomized deterministically.

### Optimization

Trainable:
- query_basis;
- query_state;
- query_schema.

Frozen:
- semantic encoder;
- W28 projection;
- every W34 candidate tensor;
- HIRACore;
- reliability/calibration;
- adaptive budget;
- relation refinement.

Optimizer:
- AdamW
- seed: 5101
- epochs: 12
- learning rate: 5e-4
- weight decay: 0.01
- gradient clip: 1.0

Loss:
- query-choice cross entropy;
- plus paired swap-margin loss, coefficient 0.20;
- margin: 0.15.

The swap-margin term requires each question to prefer its own gold option over the paired question's different gold option.

### DEV selection

Select one epoch only by this lexicographic order:
1. paired both-correct rate;
2. overall accuracy;
3. worst-language accuracy;
4. lower DEV loss;
5. earlier epoch.

No external benchmark target may participate.

## 6. S0-C DEV gate

The learned QCCE candidate earns `HIRA_V1_S0_QUERY_REPAIR_DEV_READY` only if:

- overall accuracy >= 0.80;
- EN accuracy >= 0.70;
- VI accuracy >= 0.70;
- paired both-correct rate >= 0.65;
- question-swap selected-choice-change rate >= 0.70;
- option-order flip rate <= 0.02;
- probability mass max error <= 1e-6;
- state encode count = exactly one per base state;
- full-K = true for every query;
- relation delta = 0;
- trainable parameter count = 3,072.

Failure is an acceptable scientific result and must not be repaired using DEV labels after exposure.

## 7. S0-D sealed confirm

If and only if S0-C DEV passes:
- create a new sealed confirm suite after TRAIN/DEV is frozen;
- use wholly fresh templates and values;
- include EN/VI;
- include near-duplicate options and simple composition;
- preregister its gates before exposure.

S0-A and S0-C rows cannot become sealed confirmation rows.

## 8. External benchmark boundary

No Laya/Jev/M5 final test rerun is authorized in S0.

External benchmark reopening requires:
1. S0 query repair qualification;
2. fresh sealed confirmation;
3. a separately preregistered v1 benchmark contract.

## 9. Promotion boundary

S0 can only establish:

`HIRA_V1_QUERY_CONDITIONING_QUALIFIED`

It cannot establish:
- Laya parity;
- Jev parity;
- multilingual production readiness;
- OOD/reliability readiness;
- global Hira v1 production readiness.
