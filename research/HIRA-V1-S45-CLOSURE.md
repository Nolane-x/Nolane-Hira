# HIRA V1 S45 closure — Cross-View Consistent Private Correction

Status: **SCIENTIFICALLY CLOSED**

Issue: #273  
PR: #274

## Qualified A0

- run `37128007831`
- artifact `11275950524`
- digest `sha256:90ece8fe112a14c44b29e03d217efa372f857b47d1bfa52cd58a208e75ed924e`
- outcome `HIRA_V1_S45_A0_CROSS_VIEW_CONSISTENT_PRIVATE_CORRECTION_READY`

A0 proved:
- JS-only A/B/W liveness
- JS-only native gradient exactly zero
- exact matched native initialization/update identity
- deterministic W -> B -> A warm-start
- one encoder pass / state-once
- no added trainable parameters
- arbitrary-K, invariance, checkpoint, full-K PASS

## Fresh matched court

Run: `37128945285`  
Artifact: `11276882786`  
Digest: `sha256:b68c3bf4ab8a38ef0f7046cfe8907fc6b0fc937f62a053d6f443853b896cd6a8`

Outcome:
`HIRA_V1_S45_MATCHED_CROSS_VIEW_CONSISTENT_PRIVATE_CORRECTION_DEV_COMPLETE`

Reference selected epoch 4.  
Treatment selected epoch 7.

Treatment minus matched reference:
- relation canonical **+20.83 pp**
- relation paraphrase **+7.29 pp**
- fused canonical **+7.03 pp**
- fused paraphrase **+0.26 pp**
- paired **+5.73 pp**
- question-swap **+22.92 pp**
- relation agreement **0.00 pp**
- relation JS **+0.03762** worse
- fused agreement **-8.07 pp**
- fused JS **+0.04796** worse

Native runtime remains bitwise-identical across all 24 epochs.

## Frozen interpretation

**Case C.**

S45 falsifies the hypothesis that adding the already-frozen cross-view JS penalty to the detached private correction objective is sufficient to make the correction expert wording-stable.

The correctness source remains real. The remaining problem is downstream decision robustness under an unstable-but-useful private expert.

## Stop-rule compliance

No:
- JS coefficient sweep
- temperature sweep
- alternate divergence
- capacity/activation change
- learned fusion gate/weight
- native-gradient leakage
- second encoder
- seed/LR/epoch/batch retry
- gate weakening
- second S45 DEV
- external Laya/Jev evaluation.

Frozen evidence:
- `research/HIRA-V1-S45-MATCHED-RECEIPT.json`
- `research/HIRA-V1-S45-MATCHED-RECEIPT.md`

Next family:
**S46 — Robust Three-Expert Evidence Consensus**.
