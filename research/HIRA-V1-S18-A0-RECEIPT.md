# HIRA V1 S18 A0 receipt — Paired Both-View Margin Consistency

Status: **QUALIFIED**

Issue: #217  
A0 run: `36582445491`  
Artifact: `11040940678`  
Artifact digest: `sha256:cb78b283fbcf1514e88aa8ad08f35c2bcad0a47a90302a8fd97e47640a3af452`  
Head exposed by A0: `244b5f3c2efd273b13dfc551713b95377bebc460`

Outcome:

`HIRA_V1_S18_A0_IDENTITY_READY`

## Identity / capacity

- A13 token identity: PASS
- A13 pooled identity: PASS
- exact inherited raw-triadic logits: **1.0**
- exact inherited raw-triadic choices: **1.0**
- S18 vs S14/S17 inference max abs: **0.0**
- candidate physical surface: **49,152**
- runtime trainable during A0: **0**
- original A13 trainable: **0**
- canonicalizer-added params: **0**
- fusion-added params: **0**
- paired-margin-added params: **0**
- full-K: PASS
- raw option-order flip: **0.0**
- fused option-order flip: **0.0**
- fused max probability-mass error: **1.1920928955e-07**

## Paired-margin mechanics

Frozen:
- margin: **0.20**
- coefficient: **0.25**

Proved:
- loss is zero when both views satisfy margin: PASS
- satisfied view receives zero hinge gradient: PASS
- violating view receives nonzero gradient: PASS
- when both violate, canonical receives gradient: PASS
- when both violate, paraphrase receives gradient: PASS

Synthetic probe:
- one-view-violating loss: **0.0750000030**
- both-view-violating loss: **0.1550000012**

## Shared-surface probe

- parameter count: **49,152**
- primary norm: **9.5384893417**
- relation norm: **0.3966580033**
- raw norm ratio primary/relation: **24.04713698**
- normalized pre-dot: **-0.0323093645**
- normalized post-dot: **-5.56e-09**
- norm-balancing reference scale: **4.9675736427**

This confirms S17 norm balancing remains active and relation-priority conflict projection still satisfies the operator contract.

## Fresh semantic A0 baseline

- raw triadic canonical accuracy: **0.4375**
- raw triadic paraphrase accuracy: **0.34375**
- relation canonical accuracy: **0.28125**
- relation paraphrase accuracy: **0.4375**
- fused canonical accuracy: **0.4375**
- fused paraphrase accuracy: **0.4375**
- paired both-correct: **0.0625**
- fused cross-view agreement: **0.78125**
- fused canonical signed margin: **-0.6364381313**
- same-option signature cosine: **0.7108100057**
- signature same-vs-wrong margin: **-0.0020399301**

A0 is diagnostic only and was not used for model selection.

TRAIN/DEV remains closed until exact current code CI is green.
