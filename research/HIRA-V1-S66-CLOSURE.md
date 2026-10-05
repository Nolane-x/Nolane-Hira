# HIRA V1 S66 closure

Status: **CLOSED / CASE B**

Fresh authority:
- run `37331468679`
- artifact `11354228532`
- digest `sha256:f3c6322eec8cce02a698807037a4c05e48c32b2addc5709a116a4384338379b1`
- scientific head `e4b3a18b26288efda9128611444ef2de906c68bf`.

S66 tested whether a shared pair label was hiding which view should accept the pairwise residual.

It was.

The treatment learned:
- a much broader alpha distribution;
- about **30.5%** canonical/paraphrase responsibility disagreement by epoch 24;
- lower mean alpha than the shared-label reference;
- a small but correctly directed JS improvement.

But selected DEV:
- canonical accuracy unchanged
- paraphrase accuracy unchanged
- paired both-correct unchanged
- question-swap unchanged
- agreement unchanged
- JS improved only **0.00064353**.

Frozen verdict: **Case B**.

Scientific conclusion:
binary per-view responsibility exposes real latent structure, but a 0/1 target at one fixed full-strength probe remains too coarse to teach how much pairwise residual a view should accept.

Authorized next direction:
retain S66 per-view counterfactual semantics, but upgrade supervision granularity to a preregistered TRAIN-only **safe oracle-alpha responsibility target**. Do not sweep S66.
