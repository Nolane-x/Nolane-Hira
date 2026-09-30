# HIRA V1 S21 handoff — to S22 Primary-Priority Norm-Balanced Optimization

S21 is frozen as:

`HIRA_V1_S21_ROLE_GATED_CONTENT_DEV_FAIL`

Canonical authority:
- A0 run `36701343136`, artifact `11089967708`
- TRAIN/DEV run `36703344115`
- artifact `11091248529`
- digest `sha256:af58a3fd91c282dab02641bb8082e1f66b9d8bff007b6fe92fafc652ee5dcf49`
- selected epoch **13**
- checkpoint `727fd536e819cce6235e34f47abbdfd8778d68213c5b0858ae58c8b8e1aae103`

Selected DEV:
- fused canonical **0.6145833333**
- fused paraphrase **0.5651041667**
- paired **0.3697916667**
- question-swap **0.8333333333**
- fused agreement **0.6588541667**
- fused margin **+0.1440208815**
- primary canonical/paraphrase **0.5807291667 / 0.5182291667**
- primary agreement **0.6484375**
- relation canonical/paraphrase **0.6484375 / 0.5963541667**
- relation canonical margin **+0.2356135895**
- relation paraphrase margin **+0.1948801869**

Key optimization finding:
- mean gradient conflict **0.5998263889**
- selected epoch conflict **0.7291666667**
- epoch 24 conflict **0.8125**
- S17 mean conflict was only ~0.3220

S21 keeps useful architecture evidence but exposes a strong optimizer-priority mismatch.

## S22 target

**Primary-Priority Norm-Balanced Optimization**

Start from S21 architecture and S21/S17 losses.

Keep exactly:
- role-gated content primary scorer
- role temperature 0.10
- role/content weights 0.50 / 0.50
- A13 LoRA 16,384
- shared projection 32,768
- total 49,152
- original A13/HIRACore frozen
- S13 relation expert
- S14 equal-weight fusion
- fused CE / swap / option alignment / fused JS
- relation CE / signature canonicalization
- AdamW / lr / WD / clip / batch / epochs
- no new learned params

Controlled optimizer change only:

Let normalized gradients be `u_p`, `u_r`.

If `dot(u_p,u_r) < 0`:
- keep `u_p` unchanged;
- `u_r' = u_r - dot(u_r,u_p) * u_p`.

Else:
- `u_r' = u_r`.

Then:
- `d = normalize(u_p + u_r')`
- `s = 0.5*(||g_p|| + ||g_r||)`
- `g = s*d`

epsilon remains **1e-12**.

A0 must prove:
- exact analytical primary-priority projection on controlled tensors;
- post-projection dot >= -1e-7;
- primary normalized direction is unchanged on conflict;
- relation direction changes on conflict;
- no-conflict path unchanged;
- zero-gradient cases preserve surviving raw gradient;
- exact 49,152 physical surface;
- role/content A0 mechanism preserved;
- inference numerically identical to S21;
- no added learned params/state.

Use wholly fresh S22 authority.
Do not reuse S21 DEV.
Do not change role/content weights or temperature.
Do not reopen Laya/Jev before DEV_READY.
