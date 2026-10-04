# HIRA V1 S53 handoff — to S54 Joint State–Query–Option Interaction

S53 closes as **Case C**.

## Evidence chain

S51:
- query-free option identity stable;
- correctness modestly improved;
- stability did not.

S52:
- learned global query canonicalization aligned paraphrases;
- also collapsed different relations;
- correctness/stability regressed.

S53:
- token-level option-conditioned query context is highly cross-view stable;
- fused/relation selected-choice stability still worsened.

This rules out a large class of **query-only representation fixes**.

## S54 direction

**S54 — Joint State–Query–Option Interaction**

Retain:
- exact persisted S51 native authority;
- immutable shared cache;
- native trainability 0;
- one encoder/state-once;
- full-K.

Do not first reduce state and query into two independent summaries.

Instead compute relation evidence per option from a direct interaction among:
- state tokens;
- question tokens;
- option-view tokens.

### Treatment operator

Use a parameter-free triadic late-interaction score first.

For each option k:
1. normalize cached state tokens S, query tokens Q, option-view tokens O_k;
2. compute query↔state token affinity and query↔option token affinity;
3. retain only query-token evidence jointly supported by both state and option;
4. aggregate the joint support into an option-conditioned relation evidence vector/logit;
5. feed it into the same frozen correction/fusion shell.

A simple preregisterable form:

`a_t^state = max_s <q_t, s_s>`

`a_t^option(k) = max_o <q_t, o_{k,o}>`

`joint_t(k) = softmax_t[(a_t^state + a_t^option(k))/tau]`

`c_k = normalize(sum_t joint_t(k) q_t)`

This differs from S53 because query token selection now depends on **both state and option**, not option alone.

### Controlled court

Reference:
- S53 option-conditioned query↔option late interaction.

Treatment:
- joint state+option-conditioned query late interaction.

Keep:
- zero added trainable parameters in interaction operator;
- same correction 114,688;
- bit-identical correction initialization;
- same cache, optimizer, loss, selector;
- same temperature **0.10**;
- one DEV only.

## Required A0

- K=3/7/255
- option permutation equivariance
- state-token permutation invariance
- query-token permutation invariance
- option-token/view permutation invariance
- padding/mask invariance
- all-masked rejection
- full-K probability mass
- one encoder/state-once
- native/cache gradients 0
- equal parameter count
- no query-only bypass in treatment
- changing an informative state token changes treatment context/logits
- changing an informative query token changes treatment context/logits
- changing an informative option token changes treatment context/logits
- deterministic replay
- fixed temperature 0.10; no aggregation sweep.

## Interpretation

A — stability improves while correctness remains:
joint triadic evidence is the missing factorization.

B — stability improves but correctness collapses:
joint support is too selective.

C — correctness remains but stability does not improve:
factorization is still not the dominant bottleneck; move to learned joint interaction.

D — both regress:
reject parameter-free triadic late interaction.

E — full DEV_READY:
freeze and confirm before external Laya/Jev.

One S54 DEV only.
