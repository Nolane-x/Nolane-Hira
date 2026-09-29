# HIRA V1 S14 — preregistered interpretation plan

Status: **FROZEN AFTER QUALIFIED A0, BEFORE TRAIN/DEV RESULT**

Qualified A0:
- original scientific exposure run `36527407197`
- reproducibility/artifact run `36527874581`
- artifact `11015447511`
- digest `sha256:3fccd10b3bde77a4396cec491d8a01a36c129e7d917dedfc688ff03fb0832976`
- outcome `HIRA_V1_S14_A0_IDENTITY_READY`

This note does not modify architecture, data, optimizer, fusion, objectives, selection order or DEV gates.

## Frozen A0 baseline

Canonical accuracy:
- raw triadic: **0.25**
- relation: **0.40625**
- fused: **0.375**

Paraphrase accuracy:
- raw triadic: **0.40625**
- relation: **0.40625**
- fused: **0.40625**

Other diagnostics:
- relation cross-view agreement: **0.8125**
- fused cross-view agreement: **0.5625**
- fused canonical margin: **-0.3138212**
- triadic RMS: **0.00013414**
- relation RMS: **0.47132118**
- expert top-1 agreement: **0.359375**

## Primary hypothesis

S13 showed relation evidence stronger than the legacy primary path.
S14 tests whether a zero-parameter symmetric normalized fusion can train the two surfaces into complementary experts and turn relation evidence into better primary choices.

## Outcome classes

### A. True fusion synergy

Pattern:
- fused DEV accuracy exceeds both raw experts;
- fused signed margin becomes positive;
- paired correctness and question-swap improve;
- expert agreement rises without collapsing both experts to the same weak surface;
- relation/signature metrics remain strong.

Interpretation:
- primary evidence arbitration was a real bottleneck;
- symmetric zero-parameter fusion is supported.

Only frozen DEV_READY opens sealed confirmation.

### B. Weak-expert amplification

Pattern:
- relation expert remains materially stronger;
- triadic evidence remains near-flat or unstable;
- fused performance stays below relation or fused margin is worse;
- fusion does not generalize despite relation learning.

Interpretation:
- per-expert RMS normalization promotes a low-information triadic expert to equal voting strength.

Do not tune epsilon or fusion weights on exposed S14 DEV.
A future track may test a preregistered structural neutralization / relation-first rule on wholly fresh evidence.

### C. Partial arbitration gain

Pattern:
- fused clearly improves raw triadic;
- fused does not exceed relation expert;
- primary paired correctness remains below gate.

Interpretation:
- fusion transfers some relation signal but equal symmetric arbitration is not sufficient.

### D. Expert interference

Pattern:
- fused objective increases triadic strength but degrades relation accuracy, signature separation or cross-view relation stability.

Interpretation:
- forcing shared adaptation for two normalized experts creates destructive interference.

Do not add capacity or retune coefficients inside S14.

### E. TRAIN improves, fresh DEV collapses

Pattern:
- fused TRAIN objective improves strongly;
- fresh DEV fused accuracy/margin/agreement stagnate or regress.

Interpretation:
- fusion optimization is learnable but still template/lexical overfits.

No DEV-driven retry.

### F. Fused metrics improve without semantic relation improvement

Pattern:
- fused accuracy rises;
- relation binding/signature metrics remain weak or worsen.

Interpretation:
- normalization/arbitration may explain gains rather than stronger semantic grounding.

Do not claim the semantic hypothesis is solved from fused accuracy alone.

## Diagnostic priority

1. fused paired both-correct
2. fused canonical accuracy
3. fused question-swap choice-change
4. fused signed margin
5. fused cross-view selected-choice agreement
6. raw triadic vs relation vs fused accuracy
7. relation binding accuracy/margin
8. relation signature cosine/margin
9. expert top-1 agreement and evidence-scale diagnostics

## Forbidden after DEV exposure

Do not:
- change 0.5 / 0.5 fusion weights;
- tune fusion epsilon;
- add a learned gate;
- tune role/pair/contrastive temperatures;
- tune loss coefficients;
- retry seed/LR/templates;
- weaken gates;
- select a different hypothesis using S14 DEV;
- reuse S14 DEV rows in S15;
- reopen Laya/Jev unless `HIRA_V1_S14_EVIDENCE_FUSION_DEV_READY` is achieved.
