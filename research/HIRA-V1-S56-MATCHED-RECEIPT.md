# HIRA V1 S56 matched scientific receipt — Explicit Cross-View Decision Consistency

Status: **FROZEN / CASE B**

Scientific run: `37265241846`  
Artifact: `11326048054`  
Artifact digest: `sha256:7284f9e9d4bd2e29ddb7269726b12147f636c62b2df0a51b697a5c1fee181003`  
Scientific head: `4585d67bce42f960142542200c4022e14ed3ea88`

Outcome:
`HIRA_V1_S56_CROSS_VIEW_DECISION_CONSISTENCY_DEV_COMPLETE`

## Matched authority

- parent S51 native run `37192490832`
- parent native artifact `11299783210`
- native runtime digest `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- native trainable params **0**
- native optimizer absent
- native retraining **false**
- S56-A0 run `37215444292`
- S56-A0 artifact `11307997189`
- shared TRAIN cache digest `d6be7f0c26ccd4a59b520678d57e2da1ff63927cdc27c5ef77fc78d11fe9ad9c`
- shared DEV cache digest `c6e5b67187a0643f5743d199a648599ffaf99c310b1b6bcec388de2cbf1e4ab4`
- same cache bytes **true**
- private state-view encodes **0**

Matched private surface:
- correction params **114,688 / arm**
- added trainable params **0**
- initialization bit-identical **true**

Controlled variable:
- reference decision coefficient **0**
- reference ordering coefficient **0**
- treatment decision coefficient **0.10**
- treatment ordering coefficient **0.05**
- standardization epsilon **1e-6**
- ordering active threshold **0.25**
- ordering margin floor **0.05**

## Selected checkpoints

Reference selected epoch: **23**  
Treatment selected epoch: **14**

## Reference

- fused canonical accuracy **0.4609375**
- fused paraphrase accuracy **0.4557291666666667**
- paired both-correct **0.15104166666666666**
- fused selected-choice agreement **0.5859375**
- fused JS **0.04280481316770116**
- canonical relation accuracy **0.4921875**
- paraphrase relation accuracy **0.4635416666666667**
- relation agreement **0.484375**
- relation JS **0.038445474191879235**

## Treatment

- fused canonical accuracy **0.3255208333333333**
- fused paraphrase accuracy **0.3567708333333333**
- paired both-correct **0.0625**
- fused selected-choice agreement **0.6510416666666666**
- fused JS **0.029140747850760818**
- canonical relation accuracy **0.3307291666666667**
- paraphrase relation accuracy **0.3567708333333333**
- relation agreement **0.6536458333333334**
- relation JS **0.0005532469937558441**

## Treatment minus reference

Stability:
- fused selected-choice agreement **+6.51 pp**
- fused JS **-0.013664**
- relation selected-choice agreement **+16.93 pp**
- relation JS **-0.037892**

Correctness/discrimination:
- fused canonical **-13.54 pp**
- fused paraphrase **-9.90 pp**
- paired both-correct **-8.85 pp**
- canonical relation accuracy **-16.15 pp**
- paraphrase relation accuracy **-10.68 pp**
- fused canonical margin **-0.503614**
- fused paraphrase margin **-0.282180**

## Frozen interpretation

**Case B — stability improves materially, but correctness/discrimination materially collapses.**

The treatment does exactly what S56 was designed to test:
- fused selected-choice agreement improves by **+6.51 pp**;
- relation selected-choice agreement improves by **+16.93 pp**;
- fused JS decreases by **0.013664**;
- relation JS decreases by **0.037892**.

But the same pressure collapses useful discrimination:
- fused canonical accuracy drops **13.54 pp**;
- fused paraphrase accuracy drops **9.90 pp**;
- paired both-correct drops **8.85 pp**;
- canonical relation accuracy drops **16.15 pp**;
- paraphrase relation accuracy drops **10.68 pp**;
- fused canonical margin becomes dramatically worse.

Therefore explicit distribution-level consistency is **causally useful for stability**, but the current objective is too smoothing and sacrifices task correctness.

This is stronger evidence than prior S51–S55 failures: the stability bottleneck can be directly moved, but not safely by making whole distributions similar.

The next family should preserve **discrete pairwise ordering / ordinal preferences** across views rather than forcing full-distribution agreement.

No S56 coefficient/threshold sweep or second DEV is authorized.
