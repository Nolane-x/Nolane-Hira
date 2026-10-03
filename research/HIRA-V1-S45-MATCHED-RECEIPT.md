# HIRA V1 S45 matched scientific receipt

Status: **FROZEN**

Scientific run: `37128945285`  
Artifact: `11276882786`  
Artifact digest: `sha256:b68c3bf4ab8a38ef0f7046cfe8907fc6b0fc937f62a053d6f443853b896cd6a8`  
Scientific head: `29b1c1c6d080f495a06aeeeeff6898da3fe666f7`

Outcome:
`HIRA_V1_S45_MATCHED_CROSS_VIEW_CONSISTENT_PRIVATE_CORRECTION_DEV_COMPLETE`

## Authority

- seed **66001**
- TRAIN **768**
- DEV **192**
- 12 fresh S45 domains
- K=4
- 24 epochs
- batch 16
- one DEV only
- no post-DEV tuning
- no second DEV
- no external Laya/Jev evaluation

## Reference selected DEV

Epoch **4**, gates **15/23**, DEV_READY **false**.

- fused canonical **0.4270833**
- fused paraphrase **0.4192708**
- paired both-correct **0.1354167**
- question-swap **0.3854167**
- fused agreement **0.5364583**
- fused JS **0.0321839**
- relation canonical **0.3671875**
- relation paraphrase **0.3255208**
- relation agreement **0.5286458**
- relation JS **0.0003091**
- raw canonical/paraphrase **0.3958333 / 0.4244792**
- native signature cosine/margin **0.9048745 / 0.0244761**

## Treatment selected DEV

Epoch **7**, gates **15/24**, DEV_READY **false**.

- fused canonical **0.4973958**
- fused paraphrase **0.4218750**
- paired both-correct **0.1927083**
- question-swap **0.6145833**
- fused agreement **0.4557292**
- fused JS **0.0801422**
- corrected relation canonical **0.5755208**
- corrected relation paraphrase **0.3984375**
- corrected relation agreement **0.5286458**
- corrected relation JS **0.0379254**
- raw canonical/paraphrase **0.4036458 / 0.4192708**
- native signature cosine/margin **0.9324135 / 0.0368176**

## Treatment minus matched reference

Correctness:
- fused canonical **+0.0703125**
- fused paraphrase **+0.0026042**
- paired both-correct **+0.0572917**
- question-swap **+0.2291667**
- relation canonical **+0.2083333**
- relation paraphrase **+0.0729167**

Consistency / stability:
- fused agreement **-0.0807292**
- fused JS **+0.0479583** worse
- relation agreement **+0.0000000**
- relation JS **+0.0376163** worse
- fused canonical margin **+0.0948750**
- fused paraphrase margin **-0.1887871**
- relation canonical margin **-0.0029087**
- relation paraphrase margin **-0.2504375**

Native path:
- raw canonical **+0.0078125**
- raw paraphrase **-0.0052083**
- native runtime trajectory bitwise identical across all 24 epochs
- LoRA state max delta **0**
- projection state max delta **0**
- native pre-W logits max delta **0**
- native signatures max delta **0**

## Same-epoch counterfactual

At treatment-selected epoch **7**, native runtime hashes are exactly equal.

Treatment minus same-epoch native reference:
- fused canonical **+0.1119792**
- fused paraphrase **+0.0546875**
- relation canonical **+0.1744792**
- relation paraphrase **+0.0729167**
- fused agreement **-0.0937500**
- fused JS **+0.0376767**
- relation agreement **-0.1692708**
- relation JS **+0.0376239**

## Frozen interpretation

**Case C — correctness remains, but loss-level cross-view consistency does not materially improve the private correction decision.**

The S45 consistency term is mechanically live and isolated, yet fresh DEV shows:
- substantial relation correctness remains;
- the intended relation JS does not recover;
- fused JS worsens;
- fused selected-choice agreement worsens;
- paraphrase margins remain the weak side.

Therefore the next stage must change **decision/fusion family**, not tune:
- JS coefficient
- temperature
- divergence
- capacity
- optimizer
- native gradients
- encoder count.

No second S45 DEV is authorized.
