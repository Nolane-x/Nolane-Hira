# HIRA V1 S57 A0 receipt — Discrete Pairwise Ranking Consistency

Status: **QUALIFIED**

Run: `37266891469`  
Artifact: `11326757222`  
Digest: `sha256:fcfbececa46f5bac0e1fe75ae51348d1b4c543b5b7dc3166410c148a35dd75ce`  
Authorization head: `8394339694c9dfece088711f13b75ea4b767c93c`

Outcome:
`HIRA_V1_S57_A0_DISCRETE_PAIRWISE_RANKING_CONSISTENCY_READY`

## Pairwise mechanics

- anchor sign detached: **true**
- matching strong ranking loss: **0.248735**
- controlled sign-flip loss: **1.633284**
- sign-flip disagreement fraction: **1.0**
- shared-offset invariance error: **1.49e-8**
- positive-scale invariance error: **1.49e-8**

## Gold protection

- wrong gold anchor directions filtered: **2**
- wrong-gold filtered fraction: **0.166667**
- correct-gold active directions retained: **12**
- non-gold active-direction fraction: **0.600000**

## Anti-collapse / permutation

- flat weighted ordinal auxiliary: **0**
- flat active anchor fraction: **0**
- uniform top1-top2 gap: **0**
- uniform logit RMS: **0**
- option-permutation loss error: **1.49e-8**

## Runtime ownership

- reference auxiliary exact zero: **true**
- treatment auxiliary value: **0.016934**
- treatment auxiliary gradient L1: **0.387725**
- treatment active directional-anchor fraction: **0.770833**
- treatment gold-filtered direction fraction: **0.142029**
- private params: **114,688 / arm**
- added params: **0**
- native trainable params: **0**
- identity params: **0**
- K=3/7/255 PASS
- max probability-mass error: **1.19e-7**

A0 is mechanical only and was not used for model selection.
Fresh S57 TRAIN/DEV remains separately gated.
