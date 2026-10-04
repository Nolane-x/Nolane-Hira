# HIRA V1 S51 closure — Persisted Native Authority + Query-Free Identity

Status: **SCIENTIFICALLY CLOSED — CASE C**

Issue: #285  
PR: #286

## What S51 fixed scientifically

Unlike S49/S50, S51 produced a valid matched court:

- one persisted native authority artifact;
- exact Phase-A/Phase-B runtime hash identity;
- no native retraining in Phase B;
- native optimizer absent;
- native trainable params 0;
- bit-identical private initialization;
- one shared immutable TRAIN/DEV cache;
- reference/treatment same cache bytes;
- no cache regeneration;
- one DEV authority only.

Therefore S51 is the first valid court in this family that can classify the query-free option-identity hypothesis.

## Verdict

**Case C.**

Treatment query-free identity:
- canonical accuracy **+2.08 pp**
- paraphrase accuracy **+2.08 pp**
- paired both-correct **+4.69 pp**

but:
- fused selected-choice agreement **-0.52 pp**
- fused JS worsened **+0.004663**
- relation selected-choice agreement **-10.94 pp**
- relation JS worsened **+0.003176**

Both reference and treatment remain DEV_READY false.

## Scientific conclusion

A query-free state↔option identity is **not sufficient to solve Hira's cross-view instability**.

It can preserve or slightly improve useful correctness, but stability does not follow. The stable option identity is still consumed by a query-conditioned readout whose relation direction changes across paraphrases.

The next family should therefore operate on the **query relation representation/readout**, while retaining:
- persisted native authority;
- immutable shared evidence;
- query-free option identity as a useful semantic substrate;
- one-pass/state-once runtime.

## Stop-rule compliance

No:
- authority regeneration
- native retraining
- identity variant
- pair-temperature sweep
- query leak/blend/projector
- loss/capacity change after DEV
- selector change
- retry
- gate weakening
- second S51 DEV
- external Laya/Jev evaluation.

S51 is closed.
