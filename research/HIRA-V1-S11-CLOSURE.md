# HIRA V1 S11 closure — Role-Preserving Evidence Binding

Status: **CLOSED — DEV FAIL / SCIENTIFIC NEGATIVE RESULT**

Issue: #195  
PR: #196  
Branch: `feat/hira-v1-s11-role-preserving-binding`

## 1. Authority

A0:
- run: `36440258349`
- artifact: `10978171127`
- digest: `sha256:95115c357f9b9174bf4f0c2b0beb3f5d5829829e432e00116165e5dcee0f080b`
- outcome: `HIRA_V1_S11_A0_IDENTITY_READY`

Fresh TRAIN/DEV:
- run: `36441386657`
- artifact: `10979667976`
- artifact digest: `sha256:8780134b6d05d46aeeb9c60b77f1bb3ec07f9cfb76bec1eda5c52d708bf6de2b`
- outcome: `HIRA_V1_S11_ROLE_BINDING_DEV_FAIL`
- selected epoch: **24**
- checkpoint SHA256: `1112c1085633b897aa7de2f373401c1f41d095d2c18a3c5cf866bea31f891f55`

No post-DEV tuning was performed.
No sealed confirmation was opened.
No multilingual probe was opened.
No Laya/Jev benchmark was reopened.

## 2. Frozen model surface

S11 kept the exact S8-S10 trainable surface:
- A13 final-layer attention LoRA: **16,384 params**
- shared bias-free 256->128 projection: **32,768 params**
- total trainable: **49,152 params**
- original A13 trainable: **0**
- HIRACore trainable: **0**
- learned downstream scorer params: **0**
- binding-added params: **0**

Fixed parameter-free operator:
- role temperature: 0.10
- contrastive temperature: 0.10
- symmetric value window: 4

## 3. Selected DEV result

Primary decision:
- canonical accuracy: **0.3489583333**
- paraphrase accuracy: **0.28125**
- canonical paired both-correct: **0.125**
- canonical question-swap choice-change: **0.6458333333**
- cross-view selected-choice agreement: **0.2526041667**
- cross-view mean JS: **1.523782026e-08**

Role-preserving binding:
- canonical binding accuracy: **0.4192708333**
- paraphrase binding accuracy: **0.3385416667**
- canonical signed gold-vs-max-wrong margin: **-0.9078732127**
- paraphrase signed margin: **-1.1724244654**
- binding cross-view selected-choice agreement: **0.2057291667**
- canonical mean binding loss: **1.8049540818**
- canonical role normalized entropy: **0.7444132517**
- canonical role max weight: **0.2625867178**

Mechanical invariants:
- option-order flip: **0.0**
- max probability mass error: **1.1920928955078125e-07**
- full-K: PASS
- state-once: PASS
- relation delta: **0.0**
- exact trainable surface: PASS

## 4. Training dynamics

TRAIN binding loss:
- epoch 1: **1.3743776381**
- epoch 24: **0.3916894998**

Fresh DEV:
- best canonical binding accuracy: **0.5234375** at epoch 7
- best canonical primary accuracy: **0.3802083333** at epoch 7
- best question-swap choice-change: **0.6927083333** at epoch 7
- best signed binding margin: **-0.1130183774** at epoch 6

After early improvement, TRAIN binding continued improving while fresh DEV signed separation and cross-view stability deteriorated.

Selection correctly remained governed by the preregistered ordering. Epoch 24 won because paired both-correct reached **0.125**, the highest selected priority, despite weaker binding separation.

## 5. Preregistered interpretation

S11 matches primarily **Outcome C + Outcome E**.

### Outcome C — role localizes but value binding stays wrong

Relative to fresh A0:
- role entropy fell from ~0.8971 to ~0.7444;
- role max weight rose from ~0.1286 to ~0.2626;
- binding accuracy rose from 0.28125 to 0.41927.

But:
- signed binding margin remained negative and became strongly negative;
- cross-view binding agreement fell to 0.20573;
- correct semantic value separation did not emerge.

Thus the model increasingly identifies a role region, but the fixed local-window transport does not reliably bind that role to the correct value.

### Outcome E — TRAIN improves but fresh DEV collapses

TRAIN binding loss falls strongly and monotonically, while fresh DEV peaks early and then loses signed separation/generalization.

This is not evidence for increasing window size, changing temperature, changing coefficient, seed retry, or widening model capacity.

## 6. Scientific conclusion

S11 rejects the hypothesis that a **fixed local positional kernel** is sufficient to preserve the missing role->value relation under the current 49,152-parameter surface.

The evidence narrows the bottleneck:
- query/role localization is learnable;
- token-level value matching is available;
- the missing piece is the **relation connecting the selected role token/span to the semantically associated value token/span** across different word orders/templates.

The next clean hypothesis is an explicit parameter-free **role-value relation graph / pairwise span binding** that does not assume a fixed token-distance window.

Production-ready remains false.
Laya/Jev parity remains unestablished.
