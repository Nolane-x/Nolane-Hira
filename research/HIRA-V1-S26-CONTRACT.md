# HIRA V1 S26 contract — Factorized Role-Value Relation Signatures

Status: **OPEN / PREREGISTERED BEFORE S26-A0 EXPOSURE**

Issue: #235

Parent:
- S25 PR #234
- S25 closure head `216a059dcda92aedaef4c058d0ecf21219c8dae5`
- S25 outcome `HIRA_V1_S25_DECOUPLED_PROJECTIONS_DEV_FAIL`

## 1. Hypothesis

S25 shows that decoupling primary/relation projection ownership is mechanically correct but insufficient. The residual relation failure is not merely cross-view alignment:

- best same-option signature cosine reaches **0.9018671364**;
- best signature discrimination margin reaches only **0.0832758718**;
- best relation canonical margin remains **-0.0100153784**.

S13/S25 represent each relation signature by adding multiple signals into one vector:

`normalize(state_sig + option_sig + role_delta)`.

S26 tests one structural hypothesis:

> Explicitly preserving separate role-relation and value/content-relation signature blocks prevents semantically distinct relations from collapsing together while retaining cross-view canonicalization.

## 2. Inherited model surface

Keep S25 unchanged:
- A13 base encoder;
- shared last-attention LoRA **16,384**;
- primary-private bias-free 256→128 projection **32,768**;
- relation-private bias-free 256→128 projection **32,768**;
- exact physical trainable surface **81,920**;
- original A13 frozen;
- HIRACore frozen;
- S21 role-gated-content primary unchanged;
- S14 equal standardized full-K fusion unchanged;
- relation logits detached from fused-primary objective;
- S25 shared/private gradient ownership unchanged.

S26 relation operator added trainable parameters: **0**.

## 3. Factorized relation operator

Frozen temperatures:
- role **0.10**
- pair **0.10**
- contrastive **0.10**

Frozen component weighting:
- role compatibility **0.50**
- value/content compatibility **0.50**

For each query and option view:

1. project state/question/option tokens with the S25 relation-private projection;
2. derive query-conditioned state and option role anchors;
3. compute explicit role compatibility;
4. remove each side's role direction from token content;
5. score residual state↔option token pairs;
6. use the fixed pair softmax to derive state-value and option-value anchors;
7. compute explicit value/content compatibility;
8. build a factorized signature by concatenating:
   - normalized role delta;
   - normalized value delta;
9. normalize the concatenated signature;
10. average active option views;
11. score the relation with exact 0.50 role + 0.50 value/content compatibility.

No learned component weighting.
No option-specific parameter.
No option-ID or option-order input.

## 4. What S26 is not

S26 does **not** reopen:
- S11 local role→value windows;
- S12 generic direct/relation token-pair mixing;
- S13 additive relation signatures;
- S24 reliability-weighted fusion;
- S25 projection capacity/ownership.

The controlled variable is only **factorized relation representation and scoring**.

## 5. Required A0 hard-negative quadrants

Use fresh S26-A0 rows plus a deterministic synthetic court.

For the same semantic query, the relation operator must distinguish:
- Q0: correct role + correct value;
- Q1: correct role + wrong value;
- Q2: wrong role + correct-looking/same value;
- Q3: wrong role + wrong value.

Required synthetic margins:
- Q0 - Q1 > **0**
- Q0 - Q2 > **0**
- Q0 - Q3 > **0**

Also record role-only and value-only component margins separately.

## 6. A0 invariants

Must prove:
- factorized operator parameter count **0**;
- exact inherited S25 physical surface **81,920**;
- primary path identity;
- relation-private projection identity/storage contract;
- S14 fusion unchanged;
- option permutation equivariance;
- full-K;
- state-once;
- finite degenerate geometry;
- cross-private gradient leakage **0 / 0**;
- both private and shared-LoRA own gradients nonzero on controlled court;
- checkpoint ownership/replay unchanged;
- factorized signature dimension exactly **2 × relation projection dimension**;
- no hidden learned downstream state.

A0 semantic accuracy is diagnostic only.

## 7. Fresh TRAIN/DEV

Only after A0 QUALIFIED and interpretation frozen:
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S26 domains
- K=4
- two state views
- two question views
- two option views
- one preregistered seed
- 24 epochs
- batch 16
- AdamW lr 2e-4
- weight decay 0.01
- grad clip 1.0

Forbidden:
- exact S0-S25 exposed rows;
- S26-A0 rows;
- M5 final/confirmatory rows;
- W29-W34 sealed rows.

## 8. Stop rule

No post-DEV:
- role/value weighting change;
- temperature change;
- signature composition change;
- capacity change;
- seed/LR/epoch retry;
- selector change;
- fusion change;
- gate weakening.

Scientific FAIL is valid.

No sealed confirmation, multilingual transfer, or Laya/Jev benchmark until DEV_READY.
