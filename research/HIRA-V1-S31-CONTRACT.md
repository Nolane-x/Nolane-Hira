# HIRA V1 S31 contract — Matched Local-vs-Global Relation Canonicalization Court

Status: **OPEN / PREREGISTERED BEFORE S31-A0 EXPOSURE**

Issue: #245

Parent:
- S30 issue #243 / PR #244
- S30 outcome `HIRA_V1_S30_MATCHED_ADAPTATION_DEV_COMPLETE`
- merged main `8e3987cde85c7212c7bb58a32ca15e8e4bb2edd3`

## 1. Controlled question

S17-S30 repeatedly reduce local TRAIN objectives while fresh relation margins and paired view stability remain weak.

The existing S13/S17 relation canonicalization is local:
- each semantic query has K=4 options;
- same-option signatures are aligned across wording views;
- negatives are wrong options from that same query.

S31 tests whether the missing pressure is **global query-specific discrimination across semantic cases**.

> On the exact same fresh data and S17 architecture, does batch-global relation-signature contrastive training generalize better than the local K=4 signature canonicalization?

## 2. Frozen architecture for both arms

Exact S17 architecture and inference:
- pinned A13
- final attention Q/K/V/output LoRA only
- LoRA rank 8 / alpha 8 / dropout 0
- A13 LoRA **16,384**
- shared bias-free 256→128 projection **32,768**
- exact trainable total **49,152**
- original A13 frozen
- HIRACore frozen
- S13 relation expert
- S14 equal standardized full-K fusion, epsilon **1e-6**
- S15 fused-primary relation-logit detach
- S17 relation-priority norm-balanced shared-gradient rule
- state-once
- full-K
- opaque option IDs
- zero learned downstream head/router/gate/calibrator

Primary block is identical in both arms:

`decision + 0.05*option_alignment + 0.25*fused_cross_view_js`

Relation CE is identical in both arms.

## 3. Arm A — local control

Exact S17 relation block:

`0.10*relation_ce + 0.15*local_signature_canonicalization`

Local signature loss remains:
- same option aligned across canonical/paraphrase view;
- hardest wrong option within the same K=4 semantic query separated;
- margin 0.20.

## 4. Arm B — global treatment

Replace only the local signature canonicalization term.

Relation block:

`0.10*relation_ce + 0.15*global_cross_case_relation_contrastive`

For a semantic-case batch of N, relation signatures are [2N,K,D] because each case contains two semantic queries.

For each semantic query i:
- select the canonical gold relation signature `g_c[i]`;
- select the paraphrase gold relation signature `g_p[i]`;
- L2 normalize each;
- `g_c[i] <-> g_p[i]` is the positive pair;
- every `j != i` semantic query in the same batch is a negative;
- symmetric canonical→paraphrase and paraphrase→canonical InfoNCE;
- fixed temperature **0.10**.

No wrong-option signature is added to the global negative bank; within-query wrong-option supervision remains supplied by relation CE.

No learned params/state.
No inference change.

## 5. Why S31 is distinct

S31 is not:
- S6/S8 state-blind question-option InfoNCE;
- S18 fused-output paired margin;
- S19/S20 output distribution/evidence consistency;
- S26/S27 within-query relation/factor signature canonicalization;
- S28 state-anchor transport.

S31 operates on **query-conditioned state-option relation signatures** and uses **other semantic queries across the training batch as negatives**.

## 6. S31-A0

A0 is diagnostic only.

Required operator court:
- parameter count **0**
- finite output
- matched query permutation invariance
- canonical/paraphrase swap symmetry
- low loss for well-separated matched positives
- shuffled positive pairing has materially larger loss
- nonzero finite gradients to canonical and paraphrase signatures
- batch with fewer than 2 semantic queries rejected

Required runtime court:
- exact control/treatment token output identity
- exact raw/relation/fused/signature/logit/choice identity before training
- exact **49,152** trainable physical surface for both
- original A13/HIRACore frozen
- intended LoRA/projection gradients nonzero
- full-K/state-once/option-order/mass
- checkpoint roundtrip

A0 semantics may not tune S31.

## 7. Frozen matched fresh TRAIN/DEV

Only after qualified A0 + frozen interpretation + exact-head CI + explicit one-shot authorization.

- seed **52001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S31 domains
- K=4
- two state views
- two question views per semantic query
- two option views
- epochs **24**
- batch **16**
- AdamW lr **2e-4**
- weight decay **0.01**
- grad clip **1.0**

Both arms:
- exact same TRAIN rows
- exact same DEV rows
- exact same per-epoch row order
- independent runtime/optimizer state
- same selector
- same semantic gates

Freshness:
- no exact S0-S30 exposed rows
- no S31-A0 rows
- no M5 final/confirmatory rows
- no W29-W34 sealed rows

## 8. Selector and gates

Each arm independently uses exact S17 selector and v1 frontier gates.

No cross-arm epoch mixing.

Matched report:
`global - local`

Report all primary/relation/signature endpoints. Do not collapse to one scalar score.

## 9. Stop rule

After matched DEV exposure:
- no temperature tuning
- no coefficient tuning
- no alternate negative mining
- no batch-size retry
- no seed/LR/epoch retry
- no loss stacking
- no second DEV
- no gate weakening

If global treatment produces a coherent relation + fused improvement, advance only under a fresh confirmation protocol.

If evidence is split or worse, close the global-contrastive family.

Production-ready remains false.
Laya/Jev parity remains unestablished.
