# HIRA V1 S75 contract — Token-Level Bidirectional Evidence Binding Representation

Status: **FROZEN BEFORE A0**

Parent: S74 **Case D**, merged main `0edfb0a64823ccb7ef2cf5c4e77b9fb9696361be`.

## Scientific question

Can a zero-parameter token-level state/query/option binding representation create materially stronger fresh pairwise semantic evidence than frozen S69 by changing semantic evidence **before query-token pooling**?

## Matched arms

Reference:
- exact S69 query-gated identity interaction representation;
- representation dim **512**;
- representation trainable params **0**.

Treatment:
- Token-Level Bidirectional Evidence Binding Representation (TBER);
- representation dim **512**;
- representation trainable params **0**.

Both:
- exact S59 anti-symmetric pairwise head;
- **32,832 trainable params**;
- bit-identical head initialization;
- same gold-vs-distractor softplus pairwise objective;
- same TRAIN rows/order/optimizer/LR/weight decay;
- no teacher/pseudo-target/DEV target;
- dynamic K/full-K/state-once.

Treatment parameter advantage: **0**.

## Frozen treatment formula

All source tensors are detached and L2-normalized.

For query token q, state token s, and option-view token o:

1. `state_support_q = max_s cosine(q,s)`.
2. `option_support_jq = max_o cosine(q,o_j)`.
3. Frozen S54 joint attention:
   `a_jq = softmax((state_support_q + option_support_jq)/0.10)` over active query tokens.
4. Bidirectional signed compatibility:
   `g_jq = tanh(state_support_q/0.10) * tanh(option_support_jq/0.10)`.
5. Signed token weight:
   `u_jq = a_jq * g_jq`.
6. L1-normalize over active query tokens:
   `w_jq = u_jq / max(sum_q |u_jq|, 1e-12)`.
7. Token-level evidence:
   `e_j = normalize(sum_q w_jq * normalize(q))`.
8. Identity binding:
   `b_j = normalize(e_j + 16 * (identity_j ⊙ e_j))`.
9. Final treatment representation:
   `r_j = normalize([identity_j, b_j])`.

The constants **0.10**, **16**, and **1e-12** are frozen before A0 and may not be swept after DEV.

## Neutral-collapse contract

When the bidirectional compatibility `g_jq` is the same positive constant across all active query tokens for an option, L1 normalization cancels that constant and TBER pooling is exactly the S54 joint-attention pooling. Under that preregistered neutral construction, treatment reduces to S69 geometry up to numerical tolerance.

## A0

Mechanical only. Must prove:
- 512D vs 512D;
- 0 representation params vs 0;
- exact S59 head remains 32,832 params;
- bit-identical pairwise-head init;
- option permutation equivariance;
- option-view permutation invariance;
- inactive state/query/option tokens cannot affect output;
- finite K=3/7/255;
- dynamic K/full-K/state-once;
- treatment differs nontrivially from S69 on nontrivial evidence;
- neutral construction collapses to S69 within frozen tolerance;
- no upstream gradient;
- exact checkpoint replay;
- no fresh S75 TRAIN/DEV.

## Stop rule

After one fresh S75 DEV:
- no token-weight formula sweep;
- no temperature sweep;
- no pooling sweep;
- no interaction-scale sweep;
- no pairwise head width/rank change;
- no target/loss change;
- no retry;
- no second DEV.
