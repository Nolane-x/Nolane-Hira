# HIRA V1 S13 A0 receipt — Cross-View Relation Canonicalization

Status: **QUALIFIED**

Issue: #201  
A0 run: `36524047158`  
Artifact: `11013757416`  
Artifact digest: `sha256:d2aeeec672dd7787aab3f3fe68799e83d303a99b74b4ebf8926ed75e44fbf8ba`  
Head exposed by A0: `396f8c7514e6c7535932c43bb5e8237ab7732c26`

Outcome:

`HIRA_V1_S13_A0_IDENTITY_READY`

## Identity and mechanics

- A13 token identity: PASS
- A13 pooled identity: PASS
- exact inherited primary logits: **1.0**
- exact inherited primary selected choices: **1.0**
- physical candidate surface: **49,152**
- runtime trainable parameters during A0: **0**
- canonicalizer-added learned parameters: **0**
- full-K: PASS
- primary option-order flip: **0.0**
- signature option-permutation max abs: **0.0**
- canonicalizer-logit permutation max abs: **0.0**
- state encode calls: **32 / 32 expected**
- max probability mass error: **1.1920928955078125e-07**

## Fresh A0 localization baseline

Primary:
- all-view accuracy: **0.046875**
- canonical paired both-correct: **0.0**
- primary cross-view selected-choice agreement: **0.6875**
- primary cross-view mean JS: **9.609673057298096e-09**

Relation binding:
- canonical accuracy: **0.28125**
- paraphrase accuracy: **0.4375**
- canonical signed gold margin: **-0.31197112798690796**
- paraphrase signed gold margin: **-0.24683591723442078**
- cross-view relation-choice agreement: **0.53125**

Canonical relation signatures:
- mean same-option cross-view cosine: **0.720313310623169**
- minimum same-option cross-view cosine: **0.3168613314628601**
- mean same-option vs strongest-wrong margin: **-0.0024688527919352055**

Role/pair diagnostics:
- canonical state-role entropy: **0.8966233339160681**
- canonical state-role max weight: **0.13379479642026126**
- canonical option-role entropy: **0.8772387076169252**
- canonical option-role max weight: **0.2241000053472817**
- canonical pair entropy: **0.7586800064891577**
- canonical mean best-pair score: **0.7534320876002312**

## Interpretation

A0 supports the existence of a partially stable cross-view relation signal:
same-option signatures average cosine ~0.72 without any S13 training.

However semantic identity is not yet discriminative:
the same-option signature is, on average, **not separated** from the strongest wrong option because the signed margin is slightly negative.

This is exactly the preregistered S13 target.

A0 was not used for model selection.
No S13 TRAIN/DEV result had been exposed when the architecture and gates were frozen.
