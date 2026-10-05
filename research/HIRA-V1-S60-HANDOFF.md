# HIRA V1 S60 → S61 handoff

S60 is **Case B**.

What survived:
- fused evidence remains the correctness anchor;
- explicit pairwise evidence remains useful;
- bounded residual composition avoids S59 catastrophic replacement;
- one global alpha is too weak to deliver meaningful stability gains.

## S61 target

**Confidence-Adaptive Bounded Hybrid Pairwise-Evidence Composition**

The S61 gate must vary residual influence per query, not globally.

Constraints:
- fused logits remain the base decision;
- pairwise residual remains bounded;
- pairwise-only replacement remains impossible;
- gate training uses TRAIN labels only;
- gate inputs must be preregistered confidence/disagreement diagnostics, not DEV-derived features;
- gradients from gate calibration remain isolated from correction/head/native/cache;
- capacity must remain very small;
- K-agnostic, permutation-invariant gate features;
- one fresh S61 DEV only.

Primary scientific question:

> Can Hira apply the stabilizing pairwise residual strongly only when fused evidence is uncertain or pairwise/fused evidence is suitably reliable, preserving fused correctness while gaining materially more cross-view stability than S60's global alpha?

S61 must be preregistered before A0 or fresh TRAIN/DEV exposure.
