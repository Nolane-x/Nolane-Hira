# HIRA V1 S6-A0 — A13 LoRA identity receipt

Status: **CLOSED — IDENTITY / LOCALIZATION ONLY**

Run: `36406203754`  
Artifact: `10962577478`  
Digest: `sha256:13ad5013cd2e4873c5801e4441602de6997e50305ccac39284886647268c4976`

Outcome:

`HIRA_V1_S6_A0_IDENTITY_READY`

Observed:
- 16 fresh English base states / 32 paired queries;
- K=4;
- two semantic views per option;
- A13 token output identity: **true**;
- A13 pooled output identity: **true**;
- attention mask identity: **true**;
- exact final triadic logit identity: **1.0**;
- exact selected-choice identity: **1.0**;
- accuracy: **0.3125**;
- paired both-correct: **0.0**;
- option-order flip: **0.0**;
- LoRA parameter count: **16,384**;
- LoRA trainable parameters during A0: **0**;
- original A13 trainable parameters: **0**;
- runtime trainable parameters: **0**;
- state encode calls: 16/16;
- full-K: PASS;
- relation refinement: OFF;
- probability max mass error: `1.1920928955078125e-07`.

Interpretation:
- zero-initialized S6 LoRA is an exact functional identity in eval mode;
- any subsequent S6 movement is attributable to learned LoRA updates in the four final-layer attention linears;
- A0 is permanently exposed and forbidden for S6 fitting or model selection.
