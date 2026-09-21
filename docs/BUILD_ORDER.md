# Build order

1. Keep unit/preflight checks green.
2. Pin raw datasets and replay R3-R5 controls locally.
3. Pin A13 checkpoint/revision and verify loader parity on a fixture.
4. Freeze `compile_state` / `compile_schema` interfaces.
5. Integrate HIRA, sparse sidecar, logical pooling, receipts, and OOD authority.
6. Run frozen A13 baseline.
7. Run NLI/relation warm-up and teacher-objective ablation.
8. Add dynamic-schema episodes and anti-shortcut attacks.
9. Add Choice/Score/Noul mixture.
10. Add MASSIVE + XNLI EN/VI lane.
11. Train OOD authority and calibrators only after decision weights freeze.
12. Run A6 rival; run A22 only if capacity trigger fires.
13. Run matched Laya quality/systems protocol.
14. If A13 passes, activate A7-FE tokenizer/embedding compression tournament.
15. Quantize/export only after quality freeze.

Every experiment emits an immutable run receipt. Never overwrite a failed run ID.
