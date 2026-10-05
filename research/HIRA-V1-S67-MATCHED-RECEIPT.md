# HIRA V1 S67 matched scientific receipt — Safe Oracle-Alpha Responsibility

Status: **FROZEN / CASE B**

Unique scientific run: `37336779415`  
Scientific head: `2d74cab9ad6357ac8f2bc6224df46d97dd7e1f4f`  
Outcome emitted by the unique fresh court:
`HIRA_V1_S67_SAFE_ORACLE_ALPHA_DEV_COMPLETE`

A0 authority:
- run `37335108984`
- artifact `11356196983`
- digest `sha256:14a77818c8e2158bdb3aff1b2cc864ef71fe47c1a451ce32d1d10cd234ffb793`.

## Post-DEV workflow failure

The TRAIN/DEV step itself completed successfully and emitted the full scientific receipt.

The subsequent verifier failed only because it looked up stale key:
`parent_s65`

The S67 receipt correctly contains:
`parent_s66`.

Therefore:
- DEV **was exposed exactly once**;
- the emitted receipt is the sole S67 scientific authority;
- no scientific rerun is authorized;
- artifact upload was skipped after verifier failure;
- there is **no GitHub Actions artifact** for run `37336779415`;
- candidate checkpoint SHA-256 values remain recorded in the receipt, but checkpoint bytes are not reconstructed or regenerated.

The raw receipt recovered verbatim from the unique run log is frozen in:
`research/HIRA-V1-S67-MATCHED-RECEIPT.json`.

## Selected checkpoints

Reference epoch: **4**  
Treatment epoch: **4**

## Reference selected DEV

- canonical accuracy **0.4869791667**
- paraphrase accuracy **0.3541666667**
- paired both-correct **0.1979166667**
- question-swap **0.5104166667**
- selected-choice agreement **0.40625**
- cross-view JS **0.1064823602**
- pairwise gold-pair accuracy **0.5546875**
- mean pairwise gold margin **0.0117667243**
- mean alpha **0.0895455247**
- alpha std **0.0015517632**.

## Treatment selected DEV

- canonical accuracy **0.4869791667**
- paraphrase accuracy **0.3541666667**
- paired both-correct **0.1979166667**
- question-swap **0.5104166667**
- selected-choice agreement **0.40625**
- cross-view JS **0.1064817603**
- pairwise gold-pair accuracy **0.5546875**
- mean pairwise gold margin **0.0117667243**
- mean alpha **0.0894643525**
- alpha std **0.0015596756**.

## Treatment minus reference

- canonical accuracy **0.00 pp**
- paraphrase accuracy **0.00 pp**
- paired both-correct **0.00 pp**
- question-swap **0.00 pp**
- selected-choice agreement **0.00 pp**
- cross-view JS **-0.000000599927** better
- canonical gold margin **-0.00000419312**
- paraphrase gold margin **+0.00000744437**
- relation metrics **unchanged**.

## TRAIN oracle evidence

Epoch 24 oracle target distribution:
- 0.00: **0.7415364583**
- 0.25: **0.0839843750**
- 0.50: **0.0397135417**
- 0.75: **0.0231119792**
- 1.00: **0.1116536458**

Additional:
- non-binary target fraction **0.1468098958**
- canonical/paraphrase oracle disagreement **0.4941406250**
- canonical mean target **0.1669921875**
- paraphrase mean target **0.1726888021**.

The supervision is therefore demonstrably multi-level and view-specific. Its failure to transfer materially cannot be attributed to target collapse.

## Frozen interpretation

**Case B.**

S67 learns a genuine multi-level TRAIN policy, yet selected DEV transfer is essentially zero.

Per the preregistered interpretation, **label granularity is no longer the main bottleneck**. S68 must change the reliability evidence / decision mechanism rather than refine the target again.
