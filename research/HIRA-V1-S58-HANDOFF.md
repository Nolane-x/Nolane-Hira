# HIRA V1 S58 handoff — to S59 Explicit Learned Pairwise Decision Head

S58 closes as **Case B**.

## Evidence chain

S56:
- whole-distribution consistency strongly improved stability;
- correctness collapsed.

S57:
- selective self-anchored pairwise consistency improved fused agreement;
- unsafe anchors still damaged correctness/discrimination.

S58:
- self-anchor removed completely;
- frozen S57 reference teacher, cross-view consensus and wrong-gold filtering used;
- fused agreement still improved **+3.65 pp** and fused JS improved **-0.010048**;
- canonical accuracy fell **-6.25 pp**;
- paired both-correct fell **-6.77 pp**;
- question-swap discrimination fell **-14.06 pp**;
- canonical relation accuracy fell **-11.46 pp**.

Therefore:
**the next bottleneck is the target mechanism itself, not just anchor safety.**

## S59 direction

**S59 — Explicit Learned Pairwise Decision Head**

Do not:
- reuse teacher pseudo-targets;
- use student-self anchors;
- tune S58 teacher thresholds;
- re-open S58 DEV.

The S59 head should learn pairwise option preference directly from fresh gold-supervised TRAIN data.

Required structural properties:
- anti-symmetric pair score: swapping option i/j flips sign;
- option-permutation equivariant;
- full-K aggregation;
- no second encoder;
- state-once;
- teacher-free at inference and training;
- explicit parameter accounting;
- native/cache ownership unchanged;
- no raw score-magnitude shortcut that can bypass pairwise representation;
- deterministic tie handling;
- K=3/7/255 mechanical court.

Scientific target:
**learn the decision boundary instead of inheriting it from a fixed teacher.**

The design must be preregistered before S59-A0 and fresh S59 TRAIN/DEV exposure. One S59 DEV only.
