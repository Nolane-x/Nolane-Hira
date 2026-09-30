# HIRA V1 S23 handoff — to S24 Reliability-Weighted Full-K Fusion

S23 is frozen as:

`HIRA_V1_S23_NEUTRAL_BISECTOR_DEV_FAIL`

Canonical authority:
- A0 run `36721891033`, artifact `11101960494`
- TRAIN/DEV run `36730706512`
- artifact `11106195105`
- digest `sha256:61a6b8ce293b6f43c6bcc6bcf8da2cf57950092c05eeb37009417d4a94a078af`
- selected epoch **15**
- checkpoint `f1ff5234b9e055843fa5ef2653a90abedfab21db07ca4c87e741440e52558d04`

Selected DEV:
- fused canonical **0.6432291667**
- fused paraphrase **0.609375**
- paired **0.4166666667**
- question-swap **0.75**
- fused agreement **0.6770833333**
- fused margin **+0.0783118053**
- primary canonical/paraphrase **0.5651041667 / 0.5625**
- relation canonical/paraphrase **0.6875 / 0.6276041667**
- relation canonical margin **+0.1510315314**
- relation paraphrase margin **+0.1357324384**
- signature cosine **0.8025678347**
- signature margin **0.0245468643**

Optimization:
- mean conflict **0.6545138889**
- selected conflict **0.7291666667**
- epoch24 conflict **0.9375**
- no projection coefficient/state

## Stop rule applied

Do **not** create another optimizer-priority/projection variant.

The optimizer-priority family is closed.

## S24 target

**Reliability-Weighted Full-K Fusion**

Base:
- S23 role-gated content primary
- S13 relation expert
- S23 neutral-bisector optimizer
- exact 49,152 trainable physical params
- same losses except fusion implementation
- no new learned head/router/calibrator

Controlled fusion change only.

For expert standardized logits `z_p, z_r`:
- reliability is a deterministic function of each expert's standardized top1-top2 gap;
- nonnegative expert reliabilities are normalized to weights;
- `fused = w_p*z_p + w_r*z_r`.

Frozen design constraints before A0:
- choose one explicit reliability formula and epsilon without looking at S24 DEV;
- expert-swap symmetric;
- option-permutation equivariant;
- equal reliability -> exact 0.5/0.5 S14 fusion;
- flat expert must be neutral/finitely handled;
- full-K retained;
- 0 learned params/state;
- no fitted calibration/temperature.

Motivation:
- S23 relation canonical **68.75%**
- S23 fused canonical **64.32%**
- S23 primary canonical **56.51%**

Equal fusion can suppress the stronger expert.

Use wholly fresh S24 A0/TRAIN/DEV authority.
No S23 DEV reuse.
No Laya/Jev before DEV_READY.
