# HIRA V1 S39 handoff — to S40 Reference-Anchored Joint Bilinear Co-Adaptation

S39 is frozen as:

`HIRA_V1_S39_MATCHED_GRADIENT_ISOLATED_DEV_COMPLETE`

Preregistered interpretation:
**Case C — transport recovers but correctness gain largely disappears.**

## Canonical evidence

A0:
- run `37089648986`
- artifact `11261657743`
- digest `sha256:afc20e60cabac8de5db33eeee15574207d7e0592547a1787e4980cc0b81f6e20`

Fresh matched DEV:
- run `37092496411`
- artifact `11263399222`
- digest `sha256:7c11deed7760bddbb559aec5e608469e8fb6e2787b2e7d22b416b5f394311574`
- scientific head `2b8b789ad2729f7ee185f83185e54fffeb29ad83`

Both arms selected epoch **7** and have identical native-runtime fingerprints.

Pure W same-epoch deltas:
- relation canonical **+0.0729167**
- relation paraphrase **+0.0104167**
- fused canonical **+0.0078125**
- fused paraphrase **-0.0208333**
- relation agreement **+0.0572917**
- fused agreement **+0.1015625**
- fused JS **+0.0078839**
- signature cosine **0**
- signature discrimination margin **0**

## Residual scientific picture

S38:
- co-adaptation -> large correctness gain
- transport geometry/regression failure

S39:
- hard isolation -> transport preserved
- most correctness gain disappears

Therefore the missing ingredient is not more readout capacity.
The next question is whether **co-adaptation can be constrained rather than eliminated**.

## S40 hypothesis

**Reference-Anchored Joint Bilinear Co-Adaptation**

Matched control/reference:
- exact S35/S17 native runtime trajectory
- same initialization
- same rows/order
- no W
- detached reference logits/signatures exported on each treatment batch

Treatment:
- exact S38 full bilinear readout
- W shape **256x256**
- added params **65,536**
- joint runtime + W training enabled
- no partial detach of the treatment path

New mechanism:
- treatment native pre-W logits/signatures are anchored to the matched detached reference trajectory
- anchor constrains representation transport while allowing correctness co-adaptation
- reference runtime is not learned from treatment gradients
- no extra learned parameters

Required S40-A0 must prove:
- zero-init treatment/control evaluation identity before optimization
- exact W capacity 65,536
- reference path fully detached
- joint correctness gradient into W and treatment LoRA is nonzero
- anchor gradient into treatment native geometry is nonzero
- anchor gradient into reference runtime is exactly zero
- anchor penalizes synthetic native drift
- identical input/reference batch ownership
- logical-option equivariance
- question/padding invariance
- arbitrary K
- projection independence
- checkpoint/probability/full-K/state-once mechanics

Fresh S40 authority only.
No S39 rows.
One DEV only.
No post-DEV anchor-weight tuning.
No Laya/Jev benchmark before a separately confirmed DEV_READY candidate.
