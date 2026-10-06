# HIRA V1 S74 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #335

Parent S73:
- merged main `45ff1bc301972fc8aa04c1456200283f65616937`
- scientific run `37480212536`
- artifact `11420533826`
- digest `sha256:0f08d05bea219333495529c6ae2c54d65fd9a276dd5f783adff0b0ac3925982f`
- verdict **Case C**.

Frozen mechanics:
- 12 input channels
- 16 hidden
- 257 params each arm
- reference relational channels zero
- treatment relational channels live
- direct final logits
- fixed CE+0.10 JS objective
- no upstream gradient
- fresh S74 TRAIN/DEV exposed=false.

A0 seed: **95001**.

Authorization marker:
`research/HIRA-V1-S74-ENABLE-A0`

Marker MUST remain absent until pre-A0 implementation passes generic CI on Python 3.10 and 3.12.
