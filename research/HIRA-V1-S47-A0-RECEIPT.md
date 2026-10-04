# HIRA V1 S47 A0 receipt — Ordinal Pairwise Consensus

Status: **QUALIFIED**

Run: `37167755900`  
Artifact: `11290671268`  
Artifact digest: `sha256:b2505aeac631498e606025ceb5848ea7dd0faf738387d9920fc672230de671c3`  
Authorization head: `de7fe028abe8e931637d3a2d4ff5320b2819022e`

Outcome:
`HIRA_V1_S47_A0_ORDINAL_PAIRWISE_CONSENSUS_READY`

## Frozen surface

- native trainable: **49,152**
- correction-only: **114,688**
- treatment total: **163,840**
- decision trainable parameters: **0**
- second encoder pass: **false**
- training mechanics: exact S45
- decision family: pairwise majority → Copeland → private ordinal tie-break
- lexicographic base: `2K-1`

## Ordinal mechanics

- K=3: PASS
- K=7: PASS
- K=255: PASS
- logical-option permutation error: **0**
- independent positive-affine error: **0**
- strictly increasing nonlinear transform error: **0**
- extreme magnitude blow-up error: **0**
- two-identical-expert dominance top index: **0**
- strict Copeland dominance violations: **0**
- deterministic Copeland-cycle private tie-break: **PASS**
- probability-mass max error: **5.96e-8**
- actual shell legacy-vs-ordinal max abs: **24.3522**
- actual shell one encoder batch: **true**

## Ownership

- matched native one-step parameter max abs: **0**
- matched native one-step output max abs: **0**
- correction → native gradient: **0**
- native objective → correction gradient: **0**
- JS-only native gradient: **0**
- JS-only A/B/W gradients: **0.000253988 / 0.005846448 / 0.289338678**
- W → B → A warm-start remains live

## A0-only ordinal diagnostics

Canonical pairwise votes:
- unanimous: **0.6927083**
- 2/3 majority: **0.3072917**
- exact tied pair: **0**

Paraphrase pairwise votes:
- unanimous: **0.6093750**
- 2/3 majority: **0.3906250**
- exact tied pair: **0**

These A0 semantic values are diagnostic only and are not model-selection evidence.

## Interpretation

S47 mechanically qualifies as a genuinely magnitude-free decision family.

No raw score magnitude can affect the decision once expert pairwise orderings are formed. The operator is zero-parameter, permutation-equivariant, invariant to monotonic rank-preserving transforms, and preserves the existing one-pass/native-private ownership contract.

Fresh S47 TRAIN/DEV remains separately gated.
