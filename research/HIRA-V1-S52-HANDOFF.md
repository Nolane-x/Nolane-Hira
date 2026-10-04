# HIRA V1 S52 handoff — to S53 Token-Level Query↔Option Late Interaction

S52 closes as **Case D**.

## Why the global query-code family is exhausted

S51 showed that query-free option identity is stable but does not fix decision stability.

S52 then aligned paired query relation codes directly. It raised same-relation cosine substantially, but also raised cross-relation centroid cosine almost as much and worsened fused correctness/stability.

This suggests the failure is in **global query pooling** itself: one 256D mean/canonical vector loses or entangles the local lexical/semantic evidence that distinguishes nearby relation types.

## S53 direction

**S53 — Token-Level Query↔Option Late Interaction**

Retain:
- exact persisted S51 native authority;
- query-free state↔option identity;
- immutable shared cache;
- native trainability 0;
- one encoder/state-once.

Remove the requirement that the private correction must first compress the query into one global vector.

Use already-cached query token embeddings directly in a late-interaction relation readout.

### Proposed treatment operator

For each option identity `s_k in R^256` and each normalized query token `q_t`:

1. transform query tokens with a compact matched projection;
2. compute token↔option interaction scores;
3. aggregate with deterministic MaxSim/log-sum-exp style token evidence across the query;
4. feed the resulting option-conditioned relation evidence into the existing private correction/fusion shell.

The key property:
**different query tokens may support different semantic aspects without being forced into one shared relation centroid.**

### Controlled court

Reference:
- S51/S52-style pooled query correction baseline.

Treatment:
- token-level late-interaction readout.

Hold fixed:
- native artifact/cache
- query-free option identity
- correction capacity as closely as possible
- total added trainable capacity matched by giving the reference an equal-size pooled projection if treatment introduces trainable token projection
- same optimizer/loss/selector
- one DEV only.

### Required A0

- K=3/7/255
- option permutation equivariance
- query-token permutation behavior explicitly characterized
- padding/mask invariance
- no second encoder
- native gradient 0
- cache gradient 0
- matched parameter count
- full-K probability mass
- token-local ablation: changing one informative query token changes only the late-interaction evidence through the documented path
- no global-mean bypass in treatment
- deterministic replay
- no max/mean/logsumexp sweep after A0.

## Scientific hypothesis

If local token-level relation evidence is the missing ingredient, treatment should recover paraphrase stability without the relation-collapse behavior seen in S52.

If it does not, Hira should stop modifying query representation and move to a deeper joint state-query-option interaction family.

One S53 DEV only.
