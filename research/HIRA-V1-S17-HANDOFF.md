# HIRA V1 S17 handoff — to S18 Paired Worst-View Margin Consistency

S17 is frozen as:

`HIRA_V1_S17_NORM_BALANCED_GRADIENT_DEV_FAIL`

Canonical authority:
- run `36576470197`
- artifact `11039530381`
- digest `sha256:79d86431618e7f832d38a4c2e46942c7af391788660635aa5eab13523602a2d1`
- selected epoch **23**
- checkpoint `b0a292524c2bbf7b8eb56d583cbe3196aa739ffbca6eccea80240ef057c3851e`

Selected DEV:
- fused canonical **0.7213541667**
- fused paraphrase **0.5546875**
- paired both-correct **0.5104166667**
- question-swap **0.984375**
- fused agreement **0.5286458333**
- fused margin **0.3138313380**
- relation canonical **0.640625**
- relation paraphrase **0.6380208333**
- relation margin **0.2073315941**
- signature cosine **0.8373316179**
- signature margin **0.0794288889**

TRAIN:
- relation binding loss **1.40478781 -> 0.50746648**
- canonicalization loss **0.30457011 -> 0.15378285**

## Key result

S17 changes the frontier.

The relation surface is now discriminative, margins are positive, and question routing is excellent.

The remaining bottleneck is paired wording stability:
- canonical view substantially outperforms paraphrase;
- only ~51% of paired semantic cases are correct on both views;
- top-1 cross-view agreement is only ~53%.

## S18 target

**Paired Worst-View Margin Consistency**

Keep:
- S17 norm-balanced shared-gradient optimizer
- exact 49,152 trainable physical params
- original A13 and HIRACore frozen
- S14 equal-weight full-K inference fusion
- 0 learned heads/routers
- all existing semantic diagnostics
- wholly fresh S18 authority

Add one parameter-free paired margin term to the primary objective.

For each semantic query:
- canonical fused gold-vs-hardest-wrong margin: `m_c`
- paraphrase fused gold-vs-hardest-wrong margin: `m_p`

`m_pair = min(m_c, m_p)`

`L_pair = relu(0.20 - m_pair)`

Use a fixed preregistered coefficient before A0/TRAIN exposure.

The coefficient should not be selected from S17 DEV.

S18 must verify:
- both views receive gradient when the worst view violates the margin;
- if both margins exceed 0.20, paired penalty is exactly zero;
- option permutation equivariance;
- inference unchanged;
- no new parameters.

Do not:
- reuse S17 DEV;
- retune S17/S18 after DEV;
- widen capacity;
- add learned consensus head;
- reopen Laya/Jev before DEV_READY.
