# R7 handoff

Start here after any context loss.

1. Read `research/R7-RESEARCH-BASELINE.md`.
2. Read `docs/ARCHITECTURE.md` and `docs/BUILD_ORDER.md`.
3. Run `pytest -q` and `python scripts/preflight.py`.
4. Do **not** begin A7 compression before A13 proof passes.
5. Do **not** use A22 unless the capacity trigger fires.
6. Do **not** infer OOD from max softmax confidence.
7. Preserve failed runs and receipts.
8. Next empirical task: materialize Banking77/CLINC150/MASSIVE/XNLI raw bytes, pin revisions, and replay controls before A13 training.

Scientific status remains OPEN. No parity or novelty claim is authorized.
