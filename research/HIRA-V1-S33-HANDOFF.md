# HIRA V1 S33 handoff — to S34 Query-Conditioned Entropic Relation Transport

S33 is frozen as:

`HIRA_V1_S33_MATCHED_QUERY_EXPLICIT_DEV_COMPLETE`

## Canonical evidence

A0:
- run `36988606786`
- artifact `11219055373`
- digest `sha256:99c4be1b4e2badc4aa5f6591d8c044377f62f8a958d7c51a0bee241a803596ea`
- outcome `HIRA_V1_S33_A0_QUERY_EXPLICIT_RELATION_READY`

Fresh matched TRAIN/DEV:
- run `36989462713`
- artifact `11220650899`
- digest `sha256:398f173ea3945b600bc9a71449ccc5667ebad13ba5f7b6a3b0621d19f30c2d85`
- scientific head `1ec79633d8b6532c2cbfc65cab61f6b043dbf931`
- outcome `HIRA_V1_S33_MATCHED_QUERY_EXPLICIT_DEV_COMPLETE`

Control selected:
- epoch **13**
- fused canonical/paraphrase **0.5104167 / 0.3567708**
- paired **0.2135417**
- question-swap **0.484375**
- relation canonical/paraphrase **0.421875 / 0.3151042**
- signature cosine/margin **0.8920794 / 0.1311185**
- gates **13/22 PASS**

Query-explicit selected:
- epoch **6**
- fused canonical/paraphrase **0.390625 / 0.2760417**
- paired **0.1666667**
- question-swap **0.5260417**
- relation canonical/paraphrase **0.4401042 / 0.3046875**
- signature cosine/margin **0.6780493 / 0.0439207**
- gates **12/22 PASS**

## Frozen conclusion

S33 is preregistered Case D.

The explicit query coordinate is active, but adding it at the end of the S13 anchor/additive pipeline causes large regressions in:
- fused decision quality
- primary quality
- cross-view agreement
- JS
- relation paraphrase behavior
- signature transport

Do not tune S33.

## S34 hypothesis

**Query-Conditioned Entropic Relation Transport**

The key change is not more capacity and not another auxiliary loss.

Preserve the complete token-level correspondence until a many-to-many transport plan has been constructed.

For each semantic query and option view:

1. project valid state, question and option tokens with the inherited shared 256->128 projection;
2. L2 normalize projected tokens;
3. state relevance:
   `r_s = max_q cosine(q, state_s)`;
4. option relevance:
   `r_o = max_q cosine(q, option_o)`;
5. state marginal:
   masked softmax(`r_s / 0.10`);
6. option marginal:
   masked softmax(`r_o / 0.10`);
7. state-option kernel:
   `K = exp(cosine(state_s, option_o) / 0.10)`;
8. run exactly **8 Sinkhorn iterations** to satisfy the two query-conditioned marginals;
9. option-view relation score:
   expected cosine under the transport plan, divided by **0.10**;
10. option relation signature:
    transport-weighted normalized `option_token - state_token` pair deltas;
11. average valid option views.

Treatment adds:
- **0 learned parameters**
- no head/router/gate
- no fixed positional window
- no anchor collapse before matching
- no S13 base-logit mixing
- no global contrastive objective

## Why this is new

It is not:
- S10 pooled question->state->option grounding;
- S11 fixed role->nearby-value window;
- S12 independently selected best pair;
- S13 additive role/residual signature;
- S21 role/content pooled factorization;
- S31/S32 contrastive-loss topology;
- S33 appended query coordinate.

The relation decision is formed from an explicit many-to-many transport matrix.

## Required S34 A0

Before fresh DEV:
- controlled qA/qB switches transport marginals and selected option;
- state-token permutation equivariance;
- option-token permutation equivariance;
- logical-option permutation equivariance;
- masked padding invariance;
- transport row/column marginal residual <= fixed numerical tolerance;
- finite one-token / duplicate-token geometry;
- treatment params 0;
- physical trainable surface 49,152;
- original A13/HIRACore frozen;
- nonzero finite LoRA-B and projection gradients;
- full-K/state-once/probability/checkpoint mechanics.

Use wholly fresh S34 A0/TRAIN/DEV authority.
No S33 rows.
No temperature/Sinkhorn-iteration tuning after exposure.
No second DEV.
No external Laya/Jev benchmark before DEV_READY.
