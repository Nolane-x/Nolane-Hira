# HIRA V1 S50 handoff — to S51 Persisted Native Authority

S50 is closed as **INVALID REPRODUCIBILITY COURT / INCONCLUSIVE**.

## What S50 proved

S50 successfully removed one S49 confound conceptually: both private branches were designed to consume a single shared-native cache.

But the initial run failed mechanically before private training.

A governed replay then attempted to reproduce the native authority from the same:
- seed
- rows
- code
- optimizer
- fixed 24 epochs.

All **24/24** native hashes differed.

Therefore a later workflow must never reconstruct scientific authority by retraining and hoping for bitwise equality.

## S51 design principle

**Train native once. Persist it. Never reconstruct it.**

### S51 — Persisted Native Authority + Artifact-Pinned Private Court

Phase A — native authority workflow:
1. use wholly fresh S51 TRAIN rows only;
2. train one native trajectory for the preregistered fixed epoch;
3. do **not** encode or score S51 DEV;
4. upload a sealed artifact containing:
   - exact native checkpoint bytes
   - native runtime state digest
   - all required manifest/config identifiers
   - S51 TRAIN manifest/digest
   - authority receipt
   - artifact integrity file;
5. artifact becomes immutable scientific input for Phase B.

Phase B — private court workflow:
1. download the exact Phase-A artifact by run ID + artifact name + digest;
2. reconstruct runtime by loading checkpoint bytes, **not retraining**;
3. verify loaded runtime hash equals artifact authority hash;
4. encode S51 TRAIN/DEV exactly once into immutable normal tensors;
5. destroy/freeze live native runtime;
6. fork reference/treatment correction branches from bit-identical A/B/W initialization;
7. train/evaluate both against the same persisted native evidence cache;
8. one private DEV only.

## Why S51 is stronger

Native identity is structural and artifact-pinned:
- no cross-run retraining
- no cross-run numerical reproduction assumption
- no branch-specific native path
- no dependence on CPU/GPU reduction determinism for authority recovery.

The only controlled private variable remains:
- reference: question-conditioned native relation signature
- treatment: query-free state↔option identity.

## Required S51-A0

Artifact mechanics:
- checkpoint serialize/load roundtrip exact hash
- tamper detection
- manifest/digest binding
- load order independence
- loaded native parameters all frozen
- no optimizer owns native parameters
- no native training API reachable in Phase-B script
- S51 DEV absent from Phase-A authority inputs.

Cache mechanics:
- normal non-inference tensors
- detached/contiguous/requires_grad false
- cache digest stable
- branch-order replay exact.

Private branches:
- bit-identical correction initialization
- correction params 114,688 each
- identity params 0
- K=3/7/255
- full-K/probability mass
- query-free identity API has no question input
- raw query remains live only at final private readout.

## Freshness

Use wholly fresh S51 A0/TRAIN/DEV rows.
No exact S50 row may be reused.

## Stop rule

One native authority artifact.
One private DEV court.

No:
- native retraining after Phase-A artifact exists
- alternate native authority
- artifact regeneration after private exposure
- identity variant
- temperature/mixing sweep
- query leak/blend/projector
- loss/capacity/selector tuning
- scientific retry
- second S51 DEV.

If S51 still shows no gain, the query-free identity family can finally be accepted/rejected on a valid controlled court.
