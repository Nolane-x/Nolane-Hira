# HIRA V1 S34 closure — Query-Conditioned Entropic Relation Transport

Status: **CLOSED — DEV FAIL / PREREGISTERED CASE D / ENTROPIC TRANSPORT REJECTED**

Issue: #251
PR: #252

## Canonical authority

A0:
- run `36998495974`
- artifact `11222861913`
- digest `sha256:239c2c5964a30d4ed8ca67a04e6b9ed5f4aea90efb80609368d126487fe044d9`
- authority head `de28e3f2cb51542970d07b761b85e8f36914e261`
- outcome `HIRA_V1_S34_A0_ENTROPIC_RELATION_TRANSPORT_READY`

Fresh matched TRAIN/DEV:
- run `37000789652`
- artifact `11225542519`
- digest `sha256:18228340feeee036e354d26eb183840b59525968aa6d312e0e8d429985cfba0e`
- scientific head `10753c1cbd1ee91e89a357284779300ea53e6ee7`
- outcome `HIRA_V1_S34_MATCHED_TRANSPORT_DEV_COMPLETE`

No second DEV.
No post-DEV temperature / Sinkhorn-iteration / epsilon / kernel / signature tuning.
No external Laya/Jev evaluation.

## Frozen setup

Both arms:
- exact S17 final-attention LoRA **16,384**
- shared bias-free 256->128 projection **32,768**
- exact trainable surface **49,152**
- original A13 frozen
- HIRACore frozen
- S14 equal standardized full-K fusion
- S15 relation-logit detach
- S17 norm-balanced gradients
- local relation CE + local cross-view signature canonicalization
- TRAIN **768**
- DEV **192**
- 12 fresh domains
- seed **55001**
- 24 epochs
- batch 16
- AdamW 2e-4 / wd .01 / clip 1.0

Control:
- exact S13 CrossViewRelationCanonicalizer.

Treatment:
- zero-parameter query-conditioned entropic state<->option transport;
- temperatures **0.10 / 0.10 / 0.10 / 0.10**;
- exactly **12** Sinkhorn iterations;
- epsilon **1e-12**;
- no S13 base-logit mixing.

## A0 mechanism result

Treatment was mechanically valid:
- qA selected option **0**
- qB selected option **1**
- state-marginal intervention max abs **0.9999091625**
- option-marginal intervention max abs **0.4999546111**
- transport-plan intervention max abs **0.9998811483**
- controlled row/column residual **5.9604645e-8 / 5.9604645e-8**
- real row/column residual **4.3176115e-5 / 1.1920929e-7**
- all frozen residual gates <= **1e-4**
- state-token / option-token / logical-option permutation errors **0**
- masked-padding errors **0**
- degenerate geometry finite
- operator params **0**
- physical surface **49,152**
- real LoRA-B and projection gradients nonzero
- checkpoint/full-K/mass mechanics PASS

DEV failure is therefore not a dead operator or invalid Sinkhorn implementation.

## Selected DEV — control

Selected epoch: **22**
Checkpoint:
`c8ee82e7c353b26c9b867221403fdc4efa474fa6ce80e1226bc83aadfc0afd8f`

Fused:
- canonical **0.4973958333**
- paraphrase **0.5390625**
- paired **0.1302083333**
- question-swap **0.296875**
- agreement **0.546875**
- JS **0.0423981325**
- canonical margin **-0.0265654686**
- paraphrase margin **+0.0095928079**

Primary:
- canonical **0.3463541667**
- paraphrase **0.4505208333**
- agreement **0.6067708333**

Relation:
- canonical **0.4557291667**
- paraphrase **0.4010416667**
- canonical margin **-0.0579672518**
- paraphrase margin **-0.1091988807**
- agreement **0.7135416667**

Signatures:
- cosine **0.9228848269**
- discrimination **0.1886656036**

Gate:
- PASS **15/22**
- FAIL **7/22**
- DEV_READY false

## Selected DEV — entropic transport

Selected epoch: **8**
Checkpoint:
`254098af66d5c1873308987b4bb15cac616f60fa1abeefc86424b13f5c05e4fa`

Fused:
- canonical **0.3463541667**
- paraphrase **0.3098958333**
- paired **0.0572916667**
- question-swap **0.15625**
- agreement **0.5130208333**
- JS **0.0324152214**
- canonical margin **-0.3227872021**
- paraphrase margin **-0.3853705920**

Primary:
- canonical **0.3177083333**
- paraphrase **0.3098958333**
- agreement **0.6953125**

Relation:
- canonical **0.25**
- paraphrase **0.2552083333**
- canonical margin **-0.0658348128**
- paraphrase margin **-0.0637264624**
- agreement **0.65625**

Signatures:
- cosine **0.8784866333**
- discrimination **0.1775760831**

Gate:
- PASS **14/22**
- FAIL **8/22**
- DEV_READY false

## Exact selected delta — transport minus control

Regressions:
- fused canonical **-0.1510416667**
- fused paraphrase **-0.2291666667**
- paired **-0.0729166667**
- question-swap **-0.140625**
- fused agreement **-0.0338541667**
- canonical margin **-0.2962217336**
- paraphrase margin **-0.3949633998**
- relation canonical **-0.2057291667**
- relation paraphrase **-0.1458333333**
- raw paraphrase **-0.140625**
- relation agreement **-0.0572916667**
- signature cosine **-0.0443981936**
- signature discrimination **-0.0110895205**

Only selected fused JS improves:
- **-0.0099829111** (lower is better)

This is not a coherent semantic or transport gain.

## Best observed DEV

Control across 24 epochs:
- fused canonical **0.5078125**, epoch 13
- fused paraphrase **0.5390625**, epoch 18
- paired **0.1302083333**, epoch 22
- question-swap **0.296875**, epoch 22
- agreement **0.6692708333**, epoch 21
- minimum JS **0.0278655925**, epoch 10
- relation canonical **0.4661458333**, epoch 24
- relation paraphrase **0.4192708333**, epoch 19
- signature cosine **0.9309395899**, epoch 11
- signature discrimination **0.1908431637**, epoch 20

Transport across 24 epochs:
- fused canonical **0.3463541667**, epoch 8
- fused paraphrase **0.3307291667**, epoch 11
- paired **0.0572916667**, epoch 8
- question-swap **0.3125**, epoch 19
- agreement **0.6979166667**, epoch 23
- minimum JS **0.0164049476**, epoch 20
- relation canonical **0.2916666667**, epoch 4
- relation paraphrase **0.2890625**, epoch 1
- relation agreement **0.7864583333**, epoch 24
- signature cosine **0.9431386391**, epoch 22
- signature discrimination **0.3137776007**, epoch 23

The transport arm can eventually produce stable/discriminative relation signatures at isolated epochs, but this does not translate into correct fresh semantic decisions. The frozen selector correctly does not promote those isolated metrics over poor paired/fused semantics.

## Optimization dynamics

Control epoch 1 -> 24:
- total TRAIN loss **1.7900038287 -> 0.8839546219**
- relation block **0.1906998834 -> 0.1132205403**
- canonicalization **0.3316837301 -> 0.0363329221**
- LoRA-B norm **0.3722412884 -> 2.9752771854**
- mean conflict **0.2751736111**

Transport epoch 1 -> 24:
- total TRAIN loss **1.7629591549 -> 1.5588015268**
- relation block **0.1905281742 -> 0.1405627665**
- canonicalization **0.3287295780 -> 0.0127249275**
- LoRA-B norm **0.3572124541 -> 3.6611299515**
- mean conflict **0.4210069444**

Treatment optimizes its invariance objective, but semantic decision learning remains much weaker.

## Preregistered interpretation

Observed behavior matches **Case D — little/no coherent gain**.

The treatment does not satisfy Case B because selected transport/invariance metrics do not coherently improve; isolated best-epoch signature metrics cannot override the frozen selector.

Therefore:

> Preserving a many-to-many transport plan inside the same shared W28-style 128D relation geometry is not sufficient.

S34 rejects entropic transport as the v1 solution.

## Closed directions

Do not create S34b by:
- changing any temperature
- changing Sinkhorn iterations
- changing epsilon
- changing kernel
- mixing S13 logits
- choosing a different exposed epoch
- seed/LR/epoch/batch retry
- stacking global contrastive objectives
- weakening gates

## Next residual

S5 directly relearned the shared 256->128 projection and failed.
S7 jointly adapted A13+projection and improved semantics but still generalized weakly.
S13-S34 relation mechanisms all consume the same compact shared-projection geometry.

The remaining clean localization question is now:

> Is the shared 128D W28-style relation projection itself discarding relation information that remains present in the adapted A13 256D token space?

This has not been directly tested as a matched relation-operator court while keeping the primary S17 path unchanged.

## S35 direction

**S35 — Native A13 Relation Geometry Court**

Control:
- exact S13 relation operator using the shared 256->128 projection.

Treatment:
- zero-parameter S13-equivalent relation operator directly in native adapted A13 **256D token space**;
- no relation projection;
- primary S17 scorer still uses the normal shared 128D projection;
- no new learned tensor/state;
- relation signature width **256**;
- same role/pair/contrastive temperatures as S13;
- same local relation CE/signature canonicalization coefficients;
- exact S17 selector/gates.

A0 must prove:
- native treatment learned params **0**
- exact physical trainable surface still **49,152**
- primary logits unchanged
- treatment relation outputs independent of W28 projection perturbation
- control relation outputs change under the same projection perturbation
- option/state/question token permutation behavior matches the S13 semantic contract
- full-K/state-once/probability/checkpoint mechanics
- real LoRA-B gradients nonzero
- projection still receives primary-block gradients
- treatment relation block gives **zero direct gradient** to the shared projection by construction

Use wholly fresh S35 authority.
No S34 rows.
No projection-width/native-normalization tuning after exposure.
No second DEV.

Production-ready remains false.
Laya/Jev parity remains unestablished.
