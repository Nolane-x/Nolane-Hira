# HIRA V1 S52 interpretation plan — frozen before S52-A0/DEV exposure

Status: **FROZEN**

Issue: #287

## Controlled variable

Reference and treatment are architecturally identical.

Reference auxiliary coefficient:
`0.0`

Treatment auxiliary coefficient:
`0.10`

Everything else is matched.

## Evidence hierarchy

1. exact sealed S51 native artifact binding;
2. native optimizer absent / native params frozen;
3. reference/treatment shared cache identity;
4. bit-identical correction + query-canonicalizer initialization;
5. equal private parameter count;
6. query-code paraphrase consistency/separation;
7. relation selected-choice behavior;
8. fused correctness/stability.

If items 1–5 fail, A–E classification is forbidden.

## Primary DEV reporting

Reference/treatment:
- selected private epoch
- fused canonical/paraphrase accuracy
- paired both-correct
- question-swap
- fused selected-choice agreement
- fused JS
- fused margins
- relation canonical/paraphrase accuracy
- relation selected-choice agreement
- relation JS
- relation margins
- option-order flip
- probability mass
- state-view encodes (=0).

Query-code diagnostics:
- same-relation A1/A2 cosine
- same-relation B1/B2 cosine
- mean same-relation cosine
- A-vs-B centroid cosine
- relation separation margin
- raw-query vs canonicalized-query cosine
- canonicalizer residual norm.

## Frozen interpretation

**A** — stability materially improves while useful correctness remains/improves:
paired-view query relation canonicalization is useful.

**B** — stability materially improves but correctness materially collapses:
canonicalization over-smooths useful relation detail.

**C** — correctness remains but stability does not materially improve:
instability lies deeper than query relation code.

**D** — correctness and stability both materially regress:
reject this family.

**E** — full DEV_READY:
freeze immediately and open separate fresh confirmation before external Laya/Jev evaluation.

No scalar winner score.

## Stop rule

After one S52 DEV:
- no auxiliary coefficient/margin sweep
- no architecture sweep
- no raw-query bypass
- no native retraining
- no identity change
- no loss/capacity change
- no selector change
- no retry
- no gate weakening
- no second DEV.
