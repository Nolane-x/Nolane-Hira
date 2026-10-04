# HIRA V1 S54 handoff — to S55 Learned Joint Relation Interaction

S54 closes as **Case C**.

## Evidence chain

S51:
- query-free option identity preserved/improved correctness;
- stability did not improve.

S52:
- learned global query canonicalization regressed correctness/stability.

S53:
- query↔option token late interaction produced highly stable context;
- selected-choice stability still worsened.

S54:
- state+query+option parameter-free joint interaction improves correctness;
- selected-choice stability still does not improve.

The consistent pattern is now clear:
**representation/context can be stable while the learned correction relation operator still maps it to unstable choices.**

## S55 direction

**S55 — Learned Joint Relation Interaction**

Retain:
- exact persisted S51 native authority;
- immutable shared cache;
- native trainability 0;
- query-free option identity;
- one encoder/state-once;
- full-K.

### Core hypothesis

The remaining instability is not merely which tokens are selected; it is the fixed bilinear relation mapping applied after context construction.

Test a compact learned joint transform that maps:
- query context,
- state support,
- option identity

into a relation code before the existing correction scorer.

### Matched-capacity court

Both arms receive the exact same added learned capacity.

Suggested fixed operator:
- concatenate or gated-sum three 256D normalized sources into a 256D relation code through a bottleneck;
- bottleneck hidden dimension **64**;
- two bias-free matrices:
  - A: 64×768
  - B: 256×64
- total learned joint params **65,536**
- zero-init B for warm-start preservation.

Reference:
- learned relation transform receives S53-style query↔option context plus zero/neutral state-support channel.

Treatment:
- exact same learned transform receives full S54 joint state+query+option evidence.

Controlled variable remains whether state-conditioned joint evidence reaches the learned relation transform, not parameter count.

Alternative simpler matched formulation is acceptable only if frozen before A0 and preserves equal 65,536-param capacity.

### Required A0

- equal learned joint params both arms
- equal total private params
- bit-identical initialization
- zero-init warm-start identity/neutral behavior
- K=3/7/255
- option/state/query token permutations
- mask/padding invariants
- full-K/probability mass
- no native/cache gradients
- learned joint gradients live
- no raw/pooled bypass
- state perturbation affects treatment learned relation code/logits
- state perturbation does not affect reference through a hidden bypass
- query/option perturbations live in both
- deterministic replay.

## Interpretation

A — stability improves while correctness remains:
learned joint relation transform solves the remaining instability.

B — stability improves but correctness collapses:
learned transform overfits/over-smooths.

C — correctness remains but stability does not improve:
move beyond correction-factorization family into explicit cross-view decision consistency at the decision level.

D — both regress:
reject learned joint relation interaction.

E — full DEV_READY:
freeze and confirm before external Laya/Jev.

One S55 DEV only.
