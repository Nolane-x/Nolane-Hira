# HIRA V1 S44 matched scientific receipt

Status: **FROZEN**

Scientific run: `37123003224`  
Artifact: `11274687323`  
Artifact digest: `sha256:a3b5c1c2ec16211c0f9f3123fccd7e93b53d934bd37371af94741616eed748cc`  
Scientific head: `18f94cb05d48fa7b114ca3db08af8e3a10d5dd93`

Outcome:
`HIRA_V1_S44_MATCHED_PRIVATE_CORRECTION_REPRESENTATION_DEV_COMPLETE`

## Authority

- seed 65001
- TRAIN 768 semantic cases
- DEV 192 semantic cases
- 12 fresh domains
- K=4
- 24 epochs
- batch 16
- one DEV only
- no post-DEV tuning
- no second DEV
- no external Laya/Jev evaluation

## Reference selected DEV

Epoch **17**, gates **15/23**, DEV_READY **false**.

- fused canonical: **0.5286458333**
- fused paraphrase: **0.4817708333**
- paired both-correct: **0.2656250000**
- question-swap change: **0.6666666667**
- fused agreement: **0.5182291667**
- fused JS: **0.0360378101**
- relation canonical: **0.3255208333**
- relation paraphrase: **0.3020833333**
- relation agreement: **0.5703125000**
- raw primary canonical: **0.5130208333**
- raw primary paraphrase: **0.4557291667**
- native signature cosine: **0.8880016953**
- native signature margin: **0.1563350391**

## Treatment selected DEV

Epoch **17**, gates **15/24**, DEV_READY **false**.

- fused canonical: **0.5833333333**
- fused paraphrase: **0.4895833333**
- paired both-correct: **0.3281250000**
- question-swap change: **0.7187500000**
- fused agreement: **0.5546875000**
- fused JS: **0.0670550802**
- corrected relation canonical: **0.5026041667**
- corrected relation paraphrase: **0.4244791667**
- corrected relation agreement: **0.4244791667**
- raw primary canonical: **0.5130208333**
- raw primary paraphrase: **0.4557291667**
- native signature cosine: **0.8880016953**
- native signature margin: **0.1563350391**

## Treatment minus matched reference

Correctness:
- fused canonical **+0.0546875000**
- fused paraphrase **+0.0078125000**
- paired both-correct **+0.0625000000**
- question-swap **+0.0520833333**
- relation canonical **+0.1770833333**
- relation paraphrase **+0.1223958333**

Native preservation:
- raw primary canonical **+0.0000000000**
- raw primary paraphrase **+0.0000000000**
- native signature cosine **+0.0000000000**
- native signature margin **+0.0000000000**

Cross-view / fusion:
- fused agreement **+0.0364583333**
- fused JS **+0.0310172701** (worse)
- relation agreement **-0.1458333333**
- fused canonical margin **-0.0122348122**
- fused paraphrase margin **+0.0781303213**

## Native trajectory identity

All 24 epoch runtime fingerprints are identical between reference and treatment.

Max absolute treatment-reference difference:
- LoRA state: **0**
- projection state: **0**
- native pre-W logits: **0**
- native signatures: **0**

The same-epoch counterfactual at treatment-selected epoch 17 is therefore exact.

## Frozen interpretation

**Case A — native transport remains exact and the private nonlinear correction representation restores measurable correctness.**

The S44 representation-ownership hypothesis is supported:
- native transport/primary representation can remain untouched;
- a detached private nonlinear representation learns substantial relation correctness;
- fused correctness improves without native geometry degradation.

S44 is **not DEV_READY**. The remaining bottleneck is no longer native representation ownership. It is now concentrated in the private correction branch's cross-view consistency and downstream fusion:
- corrected relation agreement falls by 0.1458333;
- fused JS worsens by 0.0310173;
- paraphrase fused gain is small despite large relation-level gain.

No second S44 DEV is authorized.
