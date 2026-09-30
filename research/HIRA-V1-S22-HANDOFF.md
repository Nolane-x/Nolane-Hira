# HIRA V1 S22 handoff — to S23 Neutral Bisector Norm-Balanced Optimization

S22 is frozen as:

`HIRA_V1_S22_PRIMARY_PRIORITY_DEV_FAIL`

Canonical authority:
- A0 run `36709243180`, artifact `11093551516`
- TRAIN/DEV run `36713837067`
- artifact `11095183683`
- digest `sha256:3329c0641159d9e96b3bff32d21b6413085b5c37fa1632b2ec0fa56ed779e14e`
- selected epoch **24**
- checkpoint `82361edc7f5bb82a364248aee471559d41c95ad9b34f93092edef7b8ccc36af0`

Selected DEV:
- fused canonical **0.5729166667**
- fused paraphrase **0.546875**
- paired **0.28125**
- question-swap **0.5729166667**
- fused agreement **0.6796875**
- fused margin **-0.0605145351**
- primary canonical/paraphrase **0.5052083333 / 0.5442708333**
- relation canonical/paraphrase **0.6015625 / 0.5520833333**
- relation canonical margin **+0.0338486681**
- signature cosine **0.8385203779**
- signature margin **0.0534842378**

Key result:
- reversing conflict priority from relation->primary to primary->relation does not rescue S21;
- relation discrimination collapses substantially;
- conflict remains high (mean **0.5**, selected epoch **0.7083**).

## S23 target

**Neutral Bisector Norm-Balanced Optimization**

Keep exactly:
- S21 role-gated content primary
- role temperature 0.10
- role/content weights 0.50 / 0.50
- A13 LoRA 16,384
- shared projection 32,768
- total 49,152
- original A13/HIRACore frozen
- S13 relation expert
- S14 equal-weight standardized fusion
- S17/S21 losses
- AdamW/lr/WD/batch/epochs/clip
- no S18/S19/S20 interventions
- zero new learned params/state

Controlled optimizer change:

For nonzero `g_p, g_r`:
- `u_p = normalize(g_p)`
- `u_r = normalize(g_r)`
- **no conflict projection**
- `d = normalize(u_p + u_r)`
- `s = 0.5*(||g_p|| + ||g_r||)`
- `g = s*d`

epsilon remains **1e-12**.

A0 must prove:
- exact neutral-bisector analytical result on conflicting unit gradients;
- primary/relation exchange symmetry;
- no-conflict output identical to S17/S21 normalized sum;
- conflict branch has no projection coefficient/state;
- both-zero / primary-zero / relation-zero exact behavior;
- positive joint-scale equivariance;
- finite near-opposite handling under frozen epsilon;
- exact 49,152 physical surface;
- S21 inference and role/content court preserved;
- zero added params/state.

Use wholly fresh S23 A0/TRAIN/DEV authority.
No S22 DEV reuse.
No post-DEV optimizer/seed/LR/template tuning.
No Laya/Jev before DEV_READY.

**Stop rule:** if S23 fails to materially improve the S17/S21 frontier, do not continue inventing optimizer-priority variants; close this optimizer family and move back to representation/fusion hypotheses.
