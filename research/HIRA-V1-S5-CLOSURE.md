# HIRA V1 S5 closure — Semantic Projection Relearning

Status: **CLOSED — HIRA_V1_S5_PROJECTION_DEV_FAIL**

Issue: #181  
PR: #182  
Branch: `feat/hira-v1-s5-projection-relearning`  
Base main: `d928e0683c6d73b488f44c760c4512b8caff9472`

## 1. Purpose

S5 followed `HIRA_V1_S4_CLOSED_DEV_FAIL`.

S4 showed that a shared 16,384-parameter residual adapter before frozen W28 did not generalize on fresh DEV. S5 therefore tested the remaining shared relation bottleneck directly: relearn the single 256->128 semantic projection used identically for state, question and option tokens while keeping A13 and HIRACore frozen.

No S0-S4/M5/sealed row was used for S5 fitting or selection.

## 2. S5-A0 — exact W28 identity authority

Authority:
- run: `36400104371`
- artifact: `10960266188`
- digest: `sha256:0ba8a60dd46a8cae47ed7098c558cce9fa7e8b4e137231116bb458943a386607`
- outcome: `HIRA_V1_S5_A0_IDENTITY_READY`

Fresh English localization:
- 16 base states / 32 paired queries;
- K=4;
- two semantic views per option;
- projection initialized from exact W28 T0;
- projection parameter count: 32,768;
- projection trainable: 0;
- exact logit identity vs parameter-free W28 triadic: **1.0**;
- exact selected-choice identity: **1.0**;
- accuracy: **0.3125**;
- paired both-correct: **0.0625**;
- option-order flip: **0.0**;
- frozen option-view InfoNCE: **0.9803502075374126**;
- state-once/full-K/relation-delta-zero: PASS.

A0 was localization/identity only and was never eligible for model selection.

## 3. S5-A learned projection candidate

Frozen:
- A13 semantic encoder;
- HIRACore;
- reliability/calibration;
- relation refinement;
- adaptive budget.

Trainable:
- one shared bias-free `256 -> 128` semantic projection;
- exactly **32,768 parameters**;
- initialized from exact W28 T0 SHA256 `1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f`.

Downstream:
- parameter-free coordinate triadic state × question × option scoring;
- no W34;
- no learned decision head;
- no dataset/domain/language-specific head.

Auxiliary representation objective:
- two option semantic views;
- symmetric InfoNCE;
- temperature 0.10;
- coefficient 0.10.

## 4. Fresh TRAIN / DEV authority

Authority:
- run: `36400648104`
- artifact: `10960895474`
- digest: `sha256:b6cb1154ccb5fe0fce38481f82bdb00881e6d4c277fa66a4e2cad0a3a1fd06c6`
- outcome: `HIRA_V1_S5_PROJECTION_DEV_FAIL`

Frozen setup:
- seed: 10501;
- AdamW;
- 30 epochs;
- lr: 3e-4;
- weight decay: 0.01;
- grad clip: 1.0;
- CE + 0.25 paired swap-margin;
- swap margin: 0.20;
- option-view InfoNCE coefficient: 0.10;
- InfoNCE temperature: 0.10;
- 512 fresh English TRAIN states / 1,024 queries;
- 128 fresh English DEV states / 256 queries;
- eight fresh domains;
- K=4;
- exactly two semantic views per option.

No prior-track/S5-A0/M5/W29-W34 sealed row was used.

## 5. Selected DEV result

Selected epoch:

**29**

Selected checkpoint SHA256:

`1babafd79bc2c97c438856c6943741f5f1bcc9d874664d214bcc050f04141560`

Selected DEV:
- accuracy: **0.3125**
- paired both-correct: **0.0625**
- question-swap choice-change: **0.21875**
- option-order flip: **0.0**
- mean alignment loss: **0.20885108326910995**
- mean decision loss: **1.4325551940128207**
- max probability mass error: **1.1920928955078125e-07**
- full-K: PASS
- relation delta: 0
- TRAIN state encode: 512/512
- DEV state encode: 128/128

Mechanical isolation:
- projection trainable params: 32,768 exact
- encoder trainable: 0
- HIRACore trainable: 0
- learned downstream scorer params: 0

## 6. Frozen DEV gate

Failed quality gates:
- accuracy >= 0.85 — **FAIL**
- paired both-correct >= 0.75 — **FAIL**
- question-swap choice-change >= 0.80 — **FAIL**

Passed mechanical gates:
- option-order flip <= 0.02 — PASS
- probability mass error <= 1e-6 — PASS
- full-K — PASS
- relation delta = 0 — PASS
- state-once — PASS
- trainable projection params exactly 32,768 — PASS
- encoder/HIRACore frozen — PASS

No threshold or hyperparameter was weakened after DEV exposure.

## 7. Learning dynamics and interpretation

S5 did learn the auxiliary representation objective:
- A0 frozen option-view InfoNCE: ~0.98035;
- TRAIN alignment loss rapidly fell below 0.1 and later into the ~0.01–0.04 range.

Decision learning moved much less:
- TRAIN CE fell only from ~1.3863 to ~1.3743;
- selected DEV accuracy reached 0.3125;
- selected paired both-correct reached 0.0625;
- selected question-swap choice-change remained 0.21875.

Therefore a relearned shared W28-style projection can strongly align the two option views on TRAIN without producing transferable state-question-option decision semantics on fresh DEV.

This removes another frozen-representation explanation. Across S0-S5, the strongest remaining common frozen semantic substrate is now the **A13 encoder itself**.

This does not prove A13 is the only bottleneck, but it is the next defensible component to adapt.

## 8. Evidence boundary

Because DEV failed:
- no post-DEV S5 tuning is authorized;
- no learning-rate/epoch retry;
- no InfoNCE coefficient/temperature retry;
- no projection-width retry;
- no threshold weakening;
- no sealed confirm;
- no Vietnamese probe;
- no M5/Laya/Jev reopening.

S5 A0 and DEV rows are permanently exposed and forbidden for future fitting, selection, architecture ranking or hyperparameter tuning.

## 9. Next research direction — S6

Recommended hypothesis:

# Limited A13 Semantic Encoder Adaptation

S6 must use wholly fresh evidence and adapt the semantic frontend itself rather than adding another external scorer.

Initial defensible scope:
- keep the exact A13 architecture/tokenizer;
- keep state-once and dynamic full-K;
- freeze most of A13;
- unfreeze only a tightly bounded late encoder surface, or insert trainable low-rank adapters inside the last encoder block;
- use a simple shared projection/triadic downstream path;
- no domain-specific head;
- preregister fresh TRAIN/DEV before exposure;
- retain explicit representation-alignment objective.

S6 must not reuse S5 rows or tune from S5 DEV.

## 10. Closure decision

S5-A0: **CLOSED — identity/localization only**  
S5-A TRAIN/DEV: **CLOSED — DEV FAIL**  
S5 sealed confirm: **NOT OPENED**  
S5 multilingual probe: **NOT OPENED**  
Production-ready: **false**  
Laya/Jev parity: **not claimed**

S5 is complete as a scientifically useful negative result that shifts the primary bottleneck hypothesis from external representation adapters/projections toward the A13 semantic encoder itself.
