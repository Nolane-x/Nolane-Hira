# HIRA V1 S69 contract — Query-Gated Identity Interaction Representation

Status: **FROZEN / PRE-A0**

Issue: #325

Parent:
- S68 merged main `4cfdbc6fe100715bd950c9a0e2f504f71c486e9c`
- fresh run `37389861004`
- artifact `11381470793`
- digest `sha256:0156bb57125f6b8c4f380649505803ed76cf49b8edf1c5bd4774fd8790386eaa`
- verdict **Case C**.

## Controlled question

Can a zero-parameter interaction between option identity and joint state/query/option context expose pairwise semantic geometry that exact S59 concatenation does not?

## Reference

Exact S59:
`normalize([identity, joint_context])`.

## Treatment

With detached `identity, joint_context in R^256`:

`x = joint_context + 16 * (identity ⊙ joint_context)`

`x = normalize(x)`

`representation = normalize([identity, x])`.

The factor 16 is exactly `sqrt(256)`; it is not tunable.

## Capacity

Representation params:
- reference 0
- treatment 0.

Pairwise head:
- exact S59 A [64,512]
- exact S59 u [64]
- **32,832 trainable params per arm**
- seed **80059**
- no bias.

Treatment parameter advantage: **0**.

## Ownership

Representation inputs are detached.
Pairwise loss may update only A/u.
No gradient into cache/native/correction/identity/context.

## A0

Must prove:
- reference bit-identical to S59 representation;
- treatment 512D, finite, detached;
- interaction scale exactly 16;
- treatment differs on non-degenerate probes;
- zero-identity probe collapses treatment to reference;
- option permutation equivariance;
- exact matched A/u state and capacity;
- antisymmetry exact;
- diagonal zero;
- K=3/7/255;
- finite probability mass;
- A/u gradients live;
- no upstream gradients;
- checkpoint replay exact;
- fresh S69 TRAIN/DEV exposed=false.

## Fresh court

After A0 + exact pre-DEV CI only:
- seed 90001
- TRAIN 768
- DEV 192
- 12 fresh S69 domains
- exact S68 overlap 0
- K=4
- 24 epochs
- one DEV.

Downstream composition stays matched and retains S66 per-view responsibility. No S62-S67 target reopening.

## Stop rule

No interaction-scale/formula/normalization sweep, learned representation adapter, dimension change, head width/rank change, optimizer/loss change, retry, second DEV or external Laya/Jev evaluation.
