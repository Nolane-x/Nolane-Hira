# Nolane HIRA

Nolane HIRA is a research-first implementation of a tiny, non-autoregressive typed decision engine.

The design target is **RD-SD-HIRA**: encode a state once, compile/cache decision schemas, preserve probability mass over every logical option, refine only uncertain options with a small relation core, and keep OOD/calibration as explicit fail-closed authorities.

## Current status

- **Research:** OPEN. R6 architecture is frozen enough to build; R7 starts implementation.
- **Claims:** no Laya/JEV parity, multilingual parity, final speedup, or novelty claim yet.
- **First proof model:** A13 reference (~13.1M active params including HIRA).
- **Compression target after proof:** A7-FE24 (~7.62M active params before dedicated OOD module).
- **Mandatory rival:** public ~6M dynamic router family.

## Non-negotiable invariants

1. State semantic encoding happens once per state.
2. K counts logical options, never exemplar rows.
3. Unselected options keep their coarse logits; full-K probability mass is preserved.
4. Dynamic schemas cannot use sparse-only early exit.
5. OOD is not inferred from max softmax confidence.
6. Calibration is held-out, versioned, and separate from OOD authority.
7. Option IDs are opaque routing keys; meaning comes from descriptions/aliases/exemplars.
8. Failed runs are immutable evidence and must not be overwritten.

## Development

```bash
python -m pip install -e .
pytest -q
python scripts/preflight.py
```

Read `docs/ARCHITECTURE.md`, `docs/BUILD_ORDER.md`, and `research/R7-RESEARCH-BASELINE.md` before changing model semantics.
