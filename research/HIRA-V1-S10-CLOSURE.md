# HIRA V1 S10 closure — Query-Conditioned State-to-Option Grounding

Status: **CLOSED — DEV FAIL / SCIENTIFIC NEGATIVE RESULT**

Issue: #191  
PR: #192  
Branch: `feat/hira-v1-s10-state-option-grounding`

## 1. Authority

A0:
- run: `36429235962`
- artifact: `10973275125`
- outcome: `HIRA_V1_S10_A0_IDENTITY_READY`

Fresh TRAIN/DEV:
- run: `36430015114`
- artifact: `10974197119`
- artifact digest: `sha256:fd5326ffab4c474acfab830df470f9ddc412b212e8d8e3687fa3a34fb354c104`
- outcome: `HIRA_V1_S10_GROUNDING_DEV_FAIL`
- selected epoch: **4**
- selected checkpoint SHA256: `de425041ccc3fb9c9c75a6256b94dfde97e135e4667984860bb1c7c451aca9c9`

No post-DEV tuning was performed.
No sealed confirmation was opened.
No multilingual probe was opened.
No Laya/Jev benchmark was reopened.

## 2. Frozen model surface

S10 kept the exact S8/S9 trainable surface:
- A13 final-layer attention LoRA: **16,384 params**
- shared bias-free 256->128 projection: **32,768 params**
- total trainable: **49,152 params**
- original A13 trainable: **0**
- HIRACore trainable: **0**
- learned downstream scorer params: **0**
- grounding-added params: **0**

The controlled change was supervision only: replace state-blind question->option InfoNCE with parameter-free question->state-evidence->option grounding.

## 3. Selected DEV result

Primary decision:
- canonical accuracy: **0.2682291667**
- paraphrase accuracy: **0.3098958333**
- canonical paired both-correct: **0.0572916667**
- canonical question-swap choice-change: **0.359375**
- cross-view selected-choice agreement: **0.359375**
- cross-view mean JS: **7.0923e-09**

Grounding:
- canonical grounding accuracy: **0.4036458333**
- paraphrase grounding accuracy: **0.3671875**
- canonical signed grounding gold-vs-max-wrong margin: **-0.8021047898**
- paraphrase signed grounding margin: **-0.7292090220**
- grounding cross-view selected-choice agreement: **0.2317708333**
- canonical normalized attention entropy: **0.5202357521**
- canonical mean max attention weight: **0.4582131381**

Mechanical invariants:
- option-order flip: **0.0**
- max probability mass error: **1.7881393433e-07**
- full-K: PASS
- state-once: PASS
- relation delta: **0.0**

## 4. Training dynamics

The grounding objective was learnable on TRAIN:
- TRAIN mean grounding loss epoch 1: **1.380856**
- TRAIN mean grounding loss epoch 24: **0.373064**

Fresh DEV did not follow:
- best canonical grounding accuracy across epochs: **0.4296875** at epoch 11
- best canonical decision accuracy: **0.2916667** at epoch 14
- best question-swap change: **0.5520833** at epoch 21
- best signed grounding margin was already epoch 1 at **-0.084033** and then became substantially more negative

Attention became much more concentrated than the early DEV state, but correct semantic separation did not emerge.

## 5. Preregistered interpretation

S10 matches primarily the preregistered combination **Outcome C + Outcome E**:

### Outcome C — attention sharpens but grounding stays wrong
- attention max weight rises strongly;
- normalized entropy falls;
- signed gold margin remains negative and worsens;
- primary accuracy remains weak.

### Outcome E — TRAIN grounding improves but fresh DEV grounding collapses
- TRAIN grounding loss improves substantially;
- fresh DEV grounding remains weak and unstable;
- wording-view grounding agreement remains poor.

Therefore:
- simply sharpening attention further is not authorized;
- retuning the S10 temperatures or grounding coefficient from DEV is not authorized;
- widening model capacity is not justified by S10;
- the next hypothesis must address **evidence identity / role alignment** and preserve token-level role/value structure.

## 6. Scientific conclusion

S10 rejects the hypothesis that a single pooled question-conditioned state evidence vector, contrasted against pooled option vectors, is sufficient to convert S9's query sensitivity into correct semantic decisions under the current 49,152-parameter surface.

The strongest remaining hypothesis is that the model loses the distinction between:
1. **which role/field the question selects**, and
2. **which value in the state fills that role**,

during pooling.

The next track should preserve this role->value structure through the grounding/decision operator rather than increasing capacity.

Production-ready remains false.
Laya/Jev parity remains unestablished.
