# HIRA V1 S16 interpretation plan

Status: **FROZEN AFTER QUALIFIED A0, BEFORE TRAIN/DEV RESULT**

A0:
- run `36564603152`
- artifact `11031676959`
- digest `sha256:cefd83bc1decb2b6e7ebbbfe94ff8fab37d68724f9d3bced095373b4843774b1`
- outcome `HIRA_V1_S16_A0_IDENTITY_READY`

A0 real shared-gradient probe had cosine **+0.1281** and no conflict, demonstrating the surgery's no-op behavior on already compatible gradients.

## Primary hypothesis

S15 showed that logit-level detach is insufficient because primary and relation objectives still update the same LoRA/projection surface.

S16 tests whether anti-aligned shared-surface updates actually occur across fresh TRAIN and whether removing only their conflicting primary component allows:
- relation semantics to remain stronger than S15;
- S14-like fusion synergy to return;
- primary/fusion quality to continue improving without adding capacity.

## Outcome classes

### A. Nontrivial conflict rate + relation and fusion improve
Supports shared-surface destructive interference as a real bottleneck.

### B. Nontrivial conflict rate + relation improves but fusion degrades
Surgery protects relation semantics but costs primary adaptation. A later track may need a parameter-free Pareto/balancing rule, not more capacity.

### C. Nontrivial conflict rate + fusion improves but relation stays weak
Conflict resolution helps primary synergy but the relation representation itself remains underpowered/generalization-limited.

### D. Conflict rate is near zero throughout
S15 degradation cannot be attributed mainly to pairwise primary-vs-relation anti-alignment under this loss partition. Do not switch projection rule after seeing DEV.

### E. Conflict is frequent but neither surface improves
The chosen relation-priority projection is insufficient; shared representation capacity/geometry rather than simple first-order conflict is likely the bottleneck.

### F. DEV_READY
Only the frozen gate may open sealed confirmation.

## Read order

1. conflict rate and pre/post gradient diagnostics
2. relation canonical accuracy/margin
3. fused paired correctness / canonical accuracy
4. fused margin / question-swap
5. signature cosine/margin
6. raw triadic diagnostics

## Forbidden post-result actions

Do not:
- alter surgery epsilon;
- change from relation-priority to symmetric/random PCGrad;
- change loss partition;
- tune coefficients, seed, LR or templates;
- select a different gradient rule from exposed DEV;
- reuse S16 DEV;
- reopen Laya/Jev unless DEV_READY.
