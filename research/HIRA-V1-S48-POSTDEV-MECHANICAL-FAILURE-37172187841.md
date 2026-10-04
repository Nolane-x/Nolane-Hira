# HIRA V1 S48 post-DEV mechanical failure receipt — run 37172187841

Status: **POST-DEV MECHANICAL FAILURE / SCIENTIFIC BUDGET CONSUMED**

Run: `37172187841`  
Head: `dbfea4876d9592e69777eebb906f5f18b14272fe`

## What completed

Before failure:
- install PASS
- contracts PASS
- canonical S48-A0 PASS
- M4 integrity PASS
- raw-query arm: **24/24 epochs completed**
- quotient-query arm: **24/24 epochs completed**
- DEV metrics were emitted for every epoch in both arms
- selected-checkpoint evidence is recoverable from the sealed log.

## Failure

After both scientific arms completed, the post-training diagnostic raised:

`RuntimeError: S48 quotient diagnostic count changed`

Cause:
the diagnostic concatenates canonical and paraphrase quotient rows, each of which already contains two queries per semantic case. The diagnostic count is therefore `4 * semantic_cases`, not `2 * semantic_cases`.

This bug is downstream of scientific training/DEV exposure.

## Governance consequence

This failure does **not** preserve the one-shot DEV budget.

No fresh rerun is authorized.

The scientific verdict must be recovered from run `37172187841` itself and frozen. A mechanical code repair may be committed for correctness/reproducibility, but must not be used to rerun S48 DEV.
