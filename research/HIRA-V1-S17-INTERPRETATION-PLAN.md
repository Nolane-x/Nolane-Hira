# HIRA V1 S17 interpretation plan

Status: **FROZEN AFTER QUALIFIED A0, BEFORE TRAIN/DEV RESULT**

A0:
- run `36574252783`
- artifact `11034934999`
- digest `sha256:4a29beb1cf410f3f98585367d6a0dc0dc6f73be37aee64b91e4030d814f0f0fe`
- outcome `HIRA_V1_S17_A0_IDENTITY_READY`

## Controlled question

Can equal directional influence between primary and relation objectives recover relation discrimination and fusion synergy without changing model capacity, inference, or fitted loss coefficients?

## Outcome classes

### A. Relation + fusion improve materially
Supports gradient-magnitude domination as a major remaining bottleneck.

### B. Relation improves strongly, fusion degrades
Equal direction protects semantics but primary adaptation requires a different non-learned Pareto rule.

### C. Fusion improves, relation remains weak
Norm imbalance was not the main cause of relation semantic weakness; representation geometry remains the bottleneck.

### D. Both degrade
Raw objective magnitude carried useful optimization information and equal-direction balancing is harmful.

### E. TRAIN improves but fresh DEV collapses
Generalization/lexical-template transfer remains dominant.

### F. DEV_READY
Only frozen gates authorize sealed confirmation.

## Read order

1. relation canonical accuracy and signed margin
2. fused paired correctness / canonical accuracy
3. raw primary accuracy
4. question-swap / fused cross-view agreement
5. signature cosine + discrimination margin
6. per-epoch raw norm ratio, normalized conflict rate, reference/final norm

## Forbidden after DEV

Do not:
- alter balance reference scale
- introduce non-equal direction weights
- tune epsilon, loss coefficients, temperatures, seed or LR
- retry templates
- reuse S17 DEV
- reopen Laya/Jev unless DEV_READY
