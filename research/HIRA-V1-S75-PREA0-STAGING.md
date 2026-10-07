# HIRA V1 S75 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #337

Parent S74:
- merged main `0edfb0a64823ccb7ef2cf5c4e77b9fb9696361be`
- scientific run `37487578737`
- artifact `11423428905`
- digest `sha256:cb2d8e2718d3aa426bf3ee44795d36c25e5960ca727fb8cc6280d12cfbefdb21`
- verdict **Case D**.

Frozen S75 mechanics:
- reference = exact S69 representation;
- treatment = TBER before query-token pooling;
- representation dim 512 both;
- representation trainable params 0 both;
- S59 pairwise head 32,832 params both;
- temperature 0.10;
- identity interaction scale 16;
- norm epsilon 1e-12;
- no teacher/pseudo-target/DEV target;
- fresh S75 TRAIN/DEV exposed=false.

A0 seed: **96001**.

Authorization marker:
`research/HIRA-V1-S75-ENABLE-A0`

Marker MUST remain absent until pre-A0 implementation passes generic CI on Python 3.10 and 3.12.
