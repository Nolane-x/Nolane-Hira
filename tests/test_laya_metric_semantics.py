import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/"benchmarks"/"laya_metric_semantics.py"
s=importlib.util.spec_from_file_location("lm",P); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)

def test_hard_metrics_exact_small_fixture():
    probs=np.array([[.8,.2],[.4,.6]],float); gold=[0,1]
    z=m.hard_metrics(gold,probs)
    assert z["accuracy"]==1.0
    assert abs(z["brier"]-0.2)<1e-12
    assert abs(z["nll"]-((-np.log(.8)-np.log(.6))/2))<1e-12

def test_soft_semantics_are_distribution_dot_and_sse():
    p=np.array([[.7,.3]]); g=np.array([[.6,.4]])
    z=m.soft_distribution_metrics(p,g)
    assert abs(z["soft_acc"]-.54)<1e-12
    assert abs(z["brier_soft"]-.02)<1e-12
    assert abs(z["tv"]-.1)<1e-12

def test_ordinal_expected_score():
    z=m.ordinal_metrics(np.array([[0.,.5,.5]]),[1.5])
    assert z["score_mae"]==0.0 and z["within_1"]==1.0
