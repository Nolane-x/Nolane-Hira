import torch

from nmd.v1_confidence_adaptive_bounded_hybrid import (
    ConfidenceAdaptiveBoundedHybridGate,
)
from nmd.v1_reliability_supervised_adaptive_gate import (
    S62_ALPHA_PROBE,
    S62_TARGET_TOLERANCE,
    fixed_bounded_probe,
    train_only_reliability_target,
    reliability_gate_loss,
)


EXAMPLES={
    "beneficial":(
        [-1.376891016960144,3.2005667686462402,-3.3355374336242676],
        [1.6032288074493408,-2.462509870529175,1.481863260269165],
        [1.7417689561843872,-0.2375815510749817,0.0183677077293396],
        [1.1653449535369873,1.4833534955978394,0.8274328708648682],
        0,
    ),
    "ceharm_jshelp":(
        [-0.4818168878555298,3.1734812259674072,0.6368277668952942],
        [0.08502231538295746,1.7102296352386475,2.0455708503723145],
        [1.8026907444000244,-0.3109073042869568,-3.2117626667022705],
        [-0.013404175639152527,-0.6213452219963074,-1.9260790348052979],
        2,
    ),
    "cehelp_jsharm":(
        [-1.0522860288619995,-2.656233310699463,2.790268898010254],
        [0.40249666571617126,0.8953961730003357,0.6669978499412537],
        [-1.0904462337493896,1.466923713684082,-0.07216861844062805],
        [1.1611074209213257,1.6940511465072632,-0.61772221326828],
        1,
    ),
    "bothharm":(
        [0.9856889247894287,-3.4627974033355713,-0.8162581324577332],
        [0.49354082345962524,1.432938814163208,-0.5824704766273499],
        [2.2066073417663574,4.542449474334717,-2.0895113945007324],
        [-1.6886796951293945,-0.3825035095214844,-0.11528395116329193],
        2,
    ),
}


def _row(name):
    fc,fp,pc,pp,g=EXAMPLES[name]
    return (
        torch.tensor([fc]),
        torch.tensor([fp]),
        torch.tensor([pc]),
        torch.tensor([pp]),
        torch.tensor([g],dtype=torch.long),
    )


def test_s62_constants_are_frozen():
    assert S62_ALPHA_PROBE==0.35
    assert S62_TARGET_TOLERANCE==1e-8


def test_s62_pareto_target_four_cases():
    expected={
        "beneficial":1.0,
        "ceharm_jshelp":0.0,
        "cehelp_jsharm":0.0,
        "bothharm":0.0,
    }
    for name,want in expected.items():
        y,d=train_only_reliability_target(*_row(name))
        assert float(y.item())==want
        assert y.requires_grad is False
        if name=="beneficial":
            assert bool(d["correctness_safe"].item())
            assert bool(d["stability_better"].item())
        elif name=="ceharm_jshelp":
            assert not bool(d["correctness_safe"].item())
            assert bool(d["stability_better"].item())
        elif name=="cehelp_jsharm":
            assert bool(d["correctness_safe"].item())
            assert not bool(d["stability_better"].item())


def test_s62_mixed_batch_contains_both_target_classes():
    rows=[_row("beneficial"),_row("ceharm_jshelp")]
    args=[
        torch.cat([rows[0][i],rows[1][i]],dim=0)
        for i in range(4)
    ]
    gold=torch.cat([rows[0][4],rows[1][4]],dim=0)
    y,_=train_only_reliability_target(*args,gold)
    assert sorted(y.tolist())==[0.0,1.0]


def test_s62_reliability_bce_gradients_reach_only_gate():
    fc,fp,pc,pp,g=_row("beneficial")
    fc.requires_grad_(True)
    fp.requires_grad_(True)
    pc.requires_grad_(True)
    pp.requires_grad_(True)
    gate=ConfidenceAdaptiveBoundedHybridGate()
    loss,_=reliability_gate_loss(gate,fc,fp,pc,pp,g)
    grads=torch.autograd.grad(
        loss,(gate.w,gate.b,fc,fp,pc,pp),allow_unused=True
    )
    assert grads[0] is not None and float(grads[0].abs().sum())>0
    assert grads[1] is not None and float(grads[1].abs().sum())>0
    assert all(x is None or float(x.abs().sum())==0.0 for x in grads[2:])


def test_s62_probe_is_bounded_and_finite_for_k3_k7_k255():
    gen=torch.Generator().manual_seed(62001)
    for k in (3,7,255):
        fused=torch.randn(2,k,generator=gen)
        pair=torch.randn(2,k,generator=gen)
        probe=fixed_bounded_probe(fused,pair)
        assert probe.shape==fused.shape
        assert bool(torch.isfinite(probe).all())
        assert float((torch.softmax(probe,-1).sum(-1)-1).abs().max())<=1e-6


def test_s62_flat_probe_is_finite():
    fused=torch.zeros(2,7)
    pair=torch.zeros(2,7)
    out=fixed_bounded_probe(fused,pair)
    assert bool(torch.isfinite(out).all())
