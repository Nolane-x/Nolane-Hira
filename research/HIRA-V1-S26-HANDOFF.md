# HIRA V1 S26 handoff — to S27 Blockwise Cross-View Factor Canonicalization

S26 is frozen as:

`HIRA_V1_S26_FACTORIZED_RELATION_DEV_FAIL`

Canonical authority:
- A0 run `36849791728`
- A0 artifact `11155305741`
- A0 digest `sha256:1cfe67edb48649a3cf7931403f1b3abd50b47d0030f05c0242a18b11f52564be`
- TRAIN/DEV run `36851831847`
- artifact `11155554912`
- artifact digest `sha256:a64bdbcbb0e217fc454f2a15653b5374bc0110ed2d05452e27f7a9b1158bd349`
- selected epoch **17**
- checkpoint `ba99c7e4c40ccc85fd2577a1eec636138f1277556068945f296d5f09b0db88c5`

Selected DEV:
- fused canonical **0.609375**
- fused paraphrase **0.3307291667**
- paired **0.3072916667**
- question-swap **0.6145833333**
- fused agreement **0.2734375**
- fused JS **0.1250924325**
- fused canonical margin **+0.1865183748**
- fused paraphrase margin **-0.5469577868**
- primary canonical/paraphrase **0.4765625 / 0.4166666667**
- relation canonical/paraphrase **0.5416666667 / 0.2135416667**
- relation canonical margin **+0.0418145005**
- relation paraphrase margin **-0.7501472371**
- relation agreement **0.2057291667**
- signature cosine **0.5707240601**
- signature discrimination margin **0.0249184022**

Best residual probes:
- relation canonical reaches **0.5651041667** and positive margin **+0.0795259426**
- fused canonical margin passes at **+0.1865183748**
- paraphrase relation margin is negative at every epoch; best only **-0.0218745681**
- signature cosine never exceeds **0.6428080897**
- relation cross-view agreement never exceeds **0.3776041667**
- fused JS never reaches the <=0.05 gate; best **0.0842635958**

Mechanics:
- exact surface **81,920**
- zero S26 relation params
- signature width **256**
- full-K/state-once PASS
- option-order PASS
- cross-private leakage **0 / 0**
- private/shared gradients remain nonzero

## Stop rule applied

Do not retry S26 with different role/value weights, temperatures, width, fusion, seed, LR or epoch count.

## S27 target

**Blockwise Cross-View Factor Canonicalization**

Why:
S26 obtains canonical discrimination but does not transport role/value factors across equivalent wording.

Freeze before A0:
- inherit S26 inference exactly;
- inherit 81,920 physical surface;
- no new learned state;
- preserve S21 primary and S14 fusion;
- preserve S25 gradient ownership;
- preserve role/value score weights 0.50/0.50;
- preserve all inference temperatures;
- change only relation-signature canonicalization loss.

Blockwise loss:
- split 256D signature into 128D role block + 128D value block;
- normalize each block independently;
- compute same-option alignment + wrong-option separation for role block;
- compute same-option alignment + wrong-option separation for value block;
- fixed block weighting **0.50 / 0.50**;
- existing total canonicalization coefficient **0.15**;
- existing separation margin **0.20**.

Required A0:
- exact S26 inference identity;
- zero added parameters;
- identical-view block losses near zero;
- role-only mismatch activates role block;
- value-only mismatch activates value block;
- block-swap/option-permutation sanity;
- relation-private/shared-LoRA gradients nonzero;
- cross-private leakage remains zero;
- full-K/state-once/checkpoint replay preserved.

Use wholly fresh S27 authority.
No S26 DEV rows may enter S27 TRAIN/DEV.
No Laya/Jev until DEV_READY.
