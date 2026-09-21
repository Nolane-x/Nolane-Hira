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


## A13 proof result — PASS

GitHub Actions run `35618805415` successfully materialized and loaded the frozen revision.

Verified:
- `model.safetensors`: 51,015,826 bytes
- weight SHA-256: `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`
- config SHA-256: `42e05354ddaef257585aa57ead3f2ecd066f39adf9f5b31c421191a6d72d2fec`
- vocab SHA-256: `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`
- hidden size: 256
- hidden layers: 6
- total base-model params: 12,750,080
- pooler params: 65,792
- encoder params without pooler: 12,684,288
- HIRA params: 422,159
- real pooled output shape: `[3, 256]`
- real token output shape: `[3, 8, 256]`
- end-to-end A13 -> StateMemory -> Schema -> HIRA full-K forward: PASS

The decision probabilities in that receipt come from an **untrained HIRA relation head** and remain integration-only evidence.

## Next state

The checkpoint-materialization blocker is CLOSED. The next scientific blocker is semantic learning:
1. frozen A13 baseline;
2. relation/NLI warm-up on corpora that do not contaminate frozen XNLI/MASSIVE direct lanes;
3. typed-decision distribution training;
4. R8 scorecard execution.
