import importlib.util
from pathlib import Path

P=Path(__file__).parents[1]/"benchmarks"/"scorecard.py"
spec=importlib.util.spec_from_file_location("scorecard",P)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_compare():
    assert m.compare(.9,.8,"higher")=="WIN"
    assert m.compare(.7,.8,"higher")=="LOSS"
    assert m.compare(.1,.2,"lower")=="WIN"
    assert m.compare(.3,.2,"lower")=="LOSS"
    assert m.compare(.2,.2,"lower")=="TIE"

def test_missing_and_counts():
    targets={"targets":[{"id":"a","target":1,"direction":"higher"},{"id":"b","target":1,"direction":"lower"}]}
    out=m.build_scorecard(targets,{"candidate":"x","metrics":{"a":2}})
    assert out["counts"]=={"WIN":1,"TIE":0,"LOSS":0,"MISSING":1}
