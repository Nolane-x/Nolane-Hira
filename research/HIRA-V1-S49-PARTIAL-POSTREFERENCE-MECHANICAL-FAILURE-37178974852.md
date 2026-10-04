# HIRA V1 S49 partial-post-reference mechanical failure receipt

Status: **REFERENCE DEV CONSUMED / TREATMENT NOT STARTED / TREATMENT-ONLY MECHANICAL CONTINUATION AUTHORIZED**

Scientific run: `37178974852`  
Job: `111367459293`  
Scientific head: `7ece9b16fd5ccfd5b93d9ff9eb1cb73ff15105ab`

## What completed

Before failure:
- install PASS
- S49 contracts PASS
- canonical S49-A0 PASS
- M4 integrity PASS
- reference question-conditioned-signature arm completed **24/24 epochs**
- reference DEV metrics were emitted for all 24 epochs.

Reference selected checkpoint recovered with the exact frozen S17/S35 selector:
- selected epoch **6**
- fused canonical **0.5182291666666666**
- fused paraphrase **0.4505208333333333**
- paired both-correct **0.15104166666666666**
- fused agreement **0.7265625**
- fused JS **0.02367013997475927**
- relation canonical **0.4609375**
- relation paraphrase **0.4244791666666667**
- relation agreement **0.7447916666666666**
- relation JS **0.006237854637826483**
- signature cosine **0.9602368026971817**
- signature same-vs-wrong margin **0.01653378348176678**

Reference final native runtime hash:
`91c2cbf10ccf81dede162a0173fe0a4133df973e5a18c6d5267bc70736cc2338`

All 24 reference native runtime hashes are frozen in:
`research/HIRA-V1-S49-RECOVERED-REFERENCE.json`.

## Failure

Treatment failed during construction **before treatment TRAIN_BEGIN**:

`TypeError: QueryFreeIdentityPrivateCorrectionFork.__init__() got an unexpected keyword argument 'native_dimension'`

Cause:
the S49 subclass exposed only `train_correction` and `pair_temperature`, while the frozen S45 factory passes the complete S44 constructor contract:
`native_dimension`, `hidden_dimension`, norm epsilons, residual scale, adapter seed, and `train_correction`.

This is a constructor-compatibility bug only. It does not alter:
- query-free identity equations
- pair temperature
- A/B/W shapes
- initialization seed
- loss coefficients
- optimizer
- rows/order
- checkpoint selector.

## Governance consequence

This is **not** a pre-DEV abort because reference DEV is already exposed.

Therefore:
- reference DEV budget is consumed;
- the reference arm MUST NOT be rerun;
- a second full matched S49 run is forbidden;
- treatment had not started and exposed no treatment DEV;
- because the treatment family, data, seed, loss, selector, and identity equations were all frozen before reference exposure, one **treatment-only mechanical continuation** is authorized.

The continuation must:
1. run treatment only;
2. use the exact same S49 TRAIN/DEV rows and seed 70001;
3. make no semantic change beyond constructor compatibility;
4. require the treatment native runtime hash at **every one of 24 epochs** to equal the frozen reference hash at the same epoch;
5. compare its selected DEV only against the frozen recovered reference selected DEV;
6. never rerun reference;
7. never tune after exposure;
8. permit no second treatment continuation for scientific weakness.

If the continuation fails before treatment TRAIN_BEGIN for another purely mechanical reason, a new governance decision is required. Once treatment TRAIN_BEGIN occurs, treatment DEV authority is consumed.
