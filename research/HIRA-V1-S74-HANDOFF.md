# HIRA V1 S74 → S75 handoff

Parent verdict: **S74 Case D**

## What S74 established

A direct 257-param DSAC can consume relational channels and change decisions, but the current S69/S59 relational evidence harms paired correctness and stability when given direct control of final logits.

Therefore:
- reject relational DSAC as the immediate v1.0 path;
- do not reopen residual/gate/veto/DSAC tuning;
- improve the semantic evidence substrate upstream.

## Required S75 direction

**Token-Level Bidirectional Evidence Binding Representation (TBER).**

Scientific question:

> Can a zero-parameter token-level state/query/option evidence binding representation produce materially stronger fresh pairwise semantic evidence than the frozen S69 representation, without changing pairwise-head capacity or training objective?

## Reference

Exact frozen S69 pairwise representation:
- detached identity signature `i_j ∈ R^256`
- detached S54 joint context `q_j ∈ R^256`
- query-gated identity interaction
- final representation dim **512**
- representation trainable params **0**.

## Treatment concept

Construct a new detached **256D token-level evidence vector** before pairwise subtraction.

For each option j:
1. normalize query token embeddings;
2. normalize option-view token embeddings;
3. normalize state token embeddings;
4. compute query→option support per query token by max similarity over the option's view tokens;
5. compute query→state support per query token by max similarity over state tokens;
6. form a signed binding weight from the interaction between option support and state support, using a fixed bounded formula frozen before DEV;
7. weighted-pool normalized query tokens into a 256D evidence vector;
8. multiplicatively bind that evidence with the 256D identity signature;
9. normalize and concatenate with identity to 512D.

No learned representation parameters.

The treatment must not merely reuse the S69 Hadamard formula on the already-collapsed S54 context. The intervention occurs **before query-token pooling**.

## Matched pairwise head

Both arms:
- exact S59 anti-symmetric head
- **32,832 trainable params**
- bit-identical initialization
- same gold-vs-distractor pairwise softplus objective
- same TRAIN rows/order
- same optimizer/LR/weight decay
- same correction/native trajectory
- no teacher/pseudo-target.

Treatment parameter advantage: **0**.

## A0 requirements

Must prove:
- 512D both arms
- 0 representation params both arms
- 32,832 pairwise params both arms
- bit-identical head init
- permutation equivariance over options
- option-view permutation invariance
- state-token/query-token mask correctness
- finite K=3/7/255
- dynamic K/full-K/state-once
- treatment differs nontrivially from S69 on a nontrivial probe
- neutral construction can collapse to reference-equivalent geometry within tolerance where preregistered
- no upstream gradient
- exact checkpoint replay
- no fresh S75 DEV.

## Fresh S75 authority

- seed **96001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh domains
- exact S74 state/question/option overlap **0**
- K=4
- two semantic option views
- 24 epochs
- one DEV only.

## Primary readouts

1. pairwise gold-pair accuracy
2. pairwise gold-pair margin
3. pairwise canonical/paraphrase aggregate accuracy
4. paired both-correct
5. question-swap
6. cross-view agreement/JS
7. final downstream diagnostic using a frozen non-adaptive composition path only as a secondary readout.

## Frozen interpretation direction

**A** — material pairwise semantic improvement with downstream-compatible stability:
freeze TBER and open independent confirmation.

**B** — pairwise evidence improves materially but downstream remains weak:
freeze TBER; decision integration remains the bottleneck.

**C** — token-level binding is active but pairwise evidence does not materially improve:
current native token substrate lacks sufficient semantics; stop this research line rather than opening another formula sweep.

**D** — pairwise evidence materially regresses:
reject TBER.

**E** — full DEV_READY:
freeze and confirm before v1.0/external Laya-JEV evaluation.

No token-weight formula, temperature, pooling, head-width, target or loss sweep after DEV.
