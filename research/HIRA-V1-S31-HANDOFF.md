# HIRA V1 S31 handoff — to S32 Global Query×Option Relation Matrix Canonicalization

S31 is frozen as:

`HIRA_V1_S31_MATCHED_RELATION_DEV_COMPLETE`

## Canonical authority

A0:
- run `36970587382`
- artifact `11211836648`
- digest `sha256:6520619fbcb1fb74464770848022d8658ac234f909451ed2338e819be3c12862`
- outcome `HIRA_V1_S31_A0_GLOBAL_RELATION_CONTRASTIVE_READY`

Fresh matched TRAIN/DEV:
- run `36973026624`
- artifact `11213780623`
- digest `sha256:3c5d24679a96dda2bd49d2a09252f66e7eb57eb5931adbe2dcf1dcb254b3acb3`
- scientific head `7ab2e2b3696af119b00f9b8d5a20243fdc4853b3`
- outcome `HIRA_V1_S31_MATCHED_RELATION_DEV_COMPLETE`

Local selected:
- epoch **20**
- checkpoint `fe99252f4ab48123ef95d39f092879b2688055168ab77ba3766fa906ff0f0fd1`
- fused canonical/paraphrase **0.484375 / 0.3723958333**
- paired **0.2239583333**
- relation canonical/paraphrase **0.3671875 / 0.3255208333**
- relation margins **-0.1254987555 / -0.1419940343**
- signature cosine/discrimination **0.9415669243 / 0.1938003438**
- gates **15/22 PASS**

Global gold-only selected:
- epoch **24**
- checkpoint `576deb92ba06897f2c45182f10a0d4dfd84523391618c09b0bd5a754aa17bd2b`
- fused canonical/paraphrase **0.5286458333 / 0.53125**
- paired **0.234375**
- relation canonical/paraphrase **0.4166666667 / 0.4557291667**
- relation margins **-0.0427276517 / -0.0189892376**
- signature cosine/discrimination **0.6501545608 / 0.1934766614**
- gates **14/22 PASS**

## Residual

Global query-specific pressure materially improves semantic discrimination, especially paraphrase, but gold-only supervision leaves K-1 option relations without cross-view global identity.

This creates the dominant new failure:
- local same-option signature cosine **0.9416**
- global gold-only **0.6502**
- best global gold-only across 24 epochs **0.7729**

Do not tune S31.

## S32 target

**Global Query×Option Relation Matrix Canonicalization**

Base:
- exact S17 attention-only architecture
- LoRA 16,384
- projection 32,768
- total 49,152
- original A13 frozen
- HIRACore frozen
- same S13/S14/S15/S17 shell

Matched fresh arms:

### Control — S31 gold-only global
For each semantic query:
- canonical gold signature <-> paraphrase gold signature positive
- other semantic-query gold signatures negatives
- temperature 0.10
- coefficient 0.15

### Treatment — all query×option global
For every query-option pair:
- canonical (q,k) <-> paraphrase (q,k) positive
- every other (q',k') relation pair is a negative
- flatten [Q,K,D] to [Q*K,D]
- symmetric InfoNCE
- temperature 0.10
- coefficient 0.15
- zero learned parameters/state

A0 must directly distinguish the operators:
- perturb only a non-gold relation pair;
- gold-only loss remains exactly unchanged;
- all-option loss increases materially;
- matched query+option permutations preserve all-option loss;
- swapping canonical/paraphrase views preserves loss;
- non-gold signatures receive finite nonzero gradients under all-option loss;
- inference remains exactly identical;
- exact 49,152 surface and mechanics PASS.

Use one wholly fresh matched S32 authority.
No S31 DEV rows.
No post-DEV operator/temperature/coefficient/negative tuning.
No second DEV.
No Laya/Jev reopening until DEV_READY.
