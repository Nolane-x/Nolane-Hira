# HIRA V1 S31 A0 harness non-result — run 36969870003

Status: **NON-RESULT / HARNESS-ONLY ABORT**

Issue: #245
PR: #246

## Run

- run `36969870003`
- head `2259da40830562d49acd2895dc46fb7013991165`
- artifact: **none**
- qualified receipt: **none**
- outcome: **none**

## Failure

The S31 unit/compile court passed completely.

The A0 job then:
- installed the exact source successfully;
- verified frozen M4 integrity successfully;
- entered the S31-A0 script.

The script aborted in the controlled synthetic operator-gradient diagnostic at:

`gl.backward()`

with:

`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

Cause:
- `main()` is decorated with `@torch.inference_mode()`;
- the direct synthetic gradient court did not locally disable inference mode;
- therefore autograd was disabled before the diagnostic backward pass.

This is a harness-context bug. It does not modify or challenge the S31 operator, model, data, loss, temperature, coefficient, negative bank, selector or gates.

## Permitted correction

Only:
- wrap the direct synthetic gradient diagnostic with
  `torch.inference_mode(False), torch.enable_grad()`.

Frozen science remains:
- global temperature **0.10**
- outer coefficient **0.15**
- gold-signature cross-case negatives
- exact S17 architecture and optimizer shell
- exact fresh authority
- matched local/global design

Fix commit:
- `4236dab3176ef4a69b6ff68a119cd4c41029acee`

A single replacement S31-A0 authority may run only after exact-head generic CI passes.

The failed run is not scientific evidence and must never be cited as an S31 A0 result.
