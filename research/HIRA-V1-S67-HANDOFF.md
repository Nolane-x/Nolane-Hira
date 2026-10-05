# HIRA V1 S67 → S68 handoff

Parent verdict: **S67 Case B**

## Strongest new evidence

S67 provided rich TRAIN supervision:
- five alpha target levels;
- **14.68%** non-binary examples;
- **49.41%** canonical/paraphrase target disagreement.

Yet selected DEV changed:
- accuracy: **0**
- paired correctness: **0**
- agreement: **0**
- JS: only **-5.999e-7**.

At selected S67 epoch, the pairwise head itself had:
- gold-pair accuracy **0.5546875**
- mean gold-pair margin **0.0117667**.

This is now a more plausible bottleneck than gate-target granularity.

## Structural limitation of the S59 pairwise head

S59 computes:
1. detached option representation `r_j=[identity_j, joint_context_j]`;
2. pair difference `d_ij=r_i-r_j`;
3. fixed learned comparison metric `A,u`;
4. antisymmetric score.

The subtraction can cancel the shared state/query semantic frame. The same comparison metric is then used for every query/state.

If pairwise meaning is query-conditioned, a scalar downstream gate cannot recover information the pairwise comparator never represented.

## Required S68 question

> Can a parameter-matched **context-modulated antisymmetric pairwise comparator** improve fresh pairwise semantic evidence by allowing the shared state/query frame to change how option differences are interpreted, while preserving exact antisymmetry and full-K/state-once behavior?

## Required S68 family

Keep:
- S59 detached 512D per-option representation;
- antisymmetric pairwise matrix as the decision object;
- exact gold-vs-distractor pair supervision;
- no teacher;
- no native/correction gradient;
- dynamic K;
- one encoder/state-once.

Create a matched pair:

**Reference**
- pair difference path identical to S59;
- receives a context modulation vector forced to zero.

**Treatment**
- derives a detached set/query context from the option set, e.g. mean of the 512D representations;
- passes it through a **fixed** low-dimensional projection;
- uses a small learned multiplicative modulation of the 64 hidden pairwise comparison channels;
- multiplication only, no additive pair bias, so antisymmetry remains exact.

Both arms must have exactly the same trainable parameter count and bit-identical initialization. Treatment parameter advantage must be zero.

## Key hypothesis

S68 changes **the pairwise evidence metric itself**, not:
- gate target;
- alpha lattice;
- downstream target smoothing;
- selector;
- native encoder.

Primary scientific readout must include fresh:
- pairwise gold-pair accuracy/margin;
- final canonical/paraphrase correctness;
- paired both-correct;
- question-swap;
- agreement/JS.

A useful S68 must show that better pairwise evidence actually transfers downstream, not merely improve TRAIN pair loss.

## Discipline

S68 must preregister:
- exact modulation architecture and capacity;
- A0;
- fresh authority;
- one DEV only;
- frozen interpretation.

Do not reopen S62–S67 target refinements.
