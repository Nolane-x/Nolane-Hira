import importlib.util
from pathlib import Path

P=Path(__file__).parents[1]/"benchmarks"/"laya_fullboard.py"
s=importlib.util.spec_from_file_location("fb",P); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)

def test_t4_best_and_compare():
    src={"suites":{"x":{"a":{"calibrated":{"accuracy":.7,"ece":.2}},"b":{"accuracy":.8,"ece":.3}}}}
    board=m.best_laya_t4(src)
    assert board["t4::x::accuracy"]["target"]==.8
    assert board["t4::x::ece"]["target"]==.2
    out=m.compare_board(board,{"candidate":"h","metrics":{"t4::x::accuracy":.81,"t4::x::ece":.2}})
    assert out["counts"]=={"WIN":1,"TIE":1,"LOSS":0,"MISSING":0}

def test_cpu51_best():
    src={"part_a":{"by_model":{"a":{"per_language":{"vi":{"accuracy":.2,"ece":.5}}},"b":{"per_language":{"vi":{"accuracy":.3,"ece":.4}}}}}}
    b=m.best_laya_cpu51(src)
    assert b["cpu51::vi::accuracy"]["target"]==.3
    assert b["cpu51::vi::ece"]["target"]==.4
