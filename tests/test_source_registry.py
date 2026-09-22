import importlib.util
import json
from pathlib import Path

P = Path(__file__).parents[1] / "benchmarks" / "audit_sources.py"
s = importlib.util.spec_from_file_location("audit_sources", P)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)


def test_registry_has_no_mutable_or_blocked_final_sources():
    reg = json.loads(
        (
            Path(__file__).parents[1]
            / "benchmarks"
            / "source_registry.json"
        ).read_text()
    )
    out = m.audit(reg)
    assert out["invalid"] == []
    assert out["blocked"] == []
    assert out["pinned"] == len(reg["sources"])
