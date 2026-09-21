# R10 handoff — A13 empirical connection

Status: **in progress**.

R9 is merged and provides a trainable end-to-end HIRA runtime. R10's first obligation is to remove the last checkpoint-evidence blocker for A13.

## Frozen A13

- model: `microsoft/xtremedistil-l6-h256-uncased`
- revision: `4226d9e4d2c08703e5cb0491b479bfc6a1607181`
- expected hidden size: 256
- expected layers: 6
- expected HIRA params: 422,159

Do not use `main`, `latest`, or an unpinned cache for empirical receipts.

## Proof workflow

`scripts/a13_proof.py` must:

1. materialize the exact revision on a networked runner;
2. SHA-256 the primary weights and tokenizer/config assets;
3. load those local bytes with Transformers;
4. record model dimensions and parameter counts;
5. run real A13 embeddings;
6. connect A13 to the R9 state-once/schema/HIRA runtime;
7. emit `artifacts/a13-proof/receipt.json`.

The HIRA head is still random in this proof. Any decision probabilities in the receipt are integration-only and **must not** be interpreted as semantic quality.

## After proof passes

- preserve the artifact receipt;
- update the A13 source ledger with actual weight SHA-256;
- run a frozen baseline before relation training;
- then begin non-leaky relation/NLI training;
- only after immutable benchmark receipts may the R8 Laya/Jev scorecards receive real HIRA values.
