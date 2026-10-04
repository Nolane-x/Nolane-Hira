# HIRA V1 S47 closure — Ordinal Pairwise Consensus

Status: **SCIENTIFICALLY CLOSED**

Issue: #277  
PR: #278

## Qualified A0

- run `37167755900`
- artifact `11290671268`
- digest `sha256:b2505aeac631498e606025ceb5848ea7dd0faf738387d9920fc672230de671c3`
- outcome `HIRA_V1_S47_A0_ORDINAL_PAIRWISE_CONSENSUS_READY`

A0 established exact magnitude independence:
- zero decision parameters
- K=3/7/255
- option permutation equivariance
- positive-affine invariance
- strictly increasing nonlinear rank-transform invariance
- extreme magnitude invariance
- majority dominance
- Copeland dominance
- deterministic private ordinal tie-break
- one encoder pass/state-once
- exact native/private ownership.

## Fresh scientific court

Run: `37168305687`  
Artifact: `11291161358`  
Digest: `sha256:9887903bcdc7e59907d0bde425389b7642bff2f91f2a7fe5e12555ad89c3db76`

Outcome:
`HIRA_V1_S47_ORDINAL_PAIRWISE_CONSENSUS_DEV_COMPLETE`

Same-checkpoint S47 minus legacy S45:
- canonical accuracy **-5.99 pp**
- paraphrase accuracy **-1.56 pp**
- paired both-correct **-7.81 pp**
- selected-choice agreement **-13.54 pp**
- cross-view JS **+0.29356** worse
- canonical margin **-4.7960**
- paraphrase margin **-1.7028**

Ordinal DEV_READY: **false**.

## Frozen interpretation

**Case C.**

S47 proves that removing score magnitude is insufficient. The experts' option ordering itself is view-sensitive.

The next family must act **before final decision aggregation**, on the view-conditioned representation/evidence generation mechanism itself.

## Stop-rule compliance

No:
- Borda-vs-Copeland sweep
- alternate pairwise rule
- tie-break expert sweep
- majority threshold
- rank temperature
- scalar weight
- learned gate
- capacity/optimizer change
- native-gradient leakage
- second encoder
- retry
- gate weakening
- second S47 DEV
- external Laya/Jev evaluation.

Frozen evidence:
- `research/HIRA-V1-S47-MATCHED-RECEIPT.json`
- `research/HIRA-V1-S47-MATCHED-RECEIPT.md`

Next family:
**S48 — Query-Quotient Option Evidence**.
