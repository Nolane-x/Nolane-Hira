import torch
from nmd.hira import HIRACore, count_parameters

def test_hira_contract():
    m=HIRACore()
    assert count_parameters(m)==422_159
    q=torch.randn(2,256); seg=torch.randn(2,5,256); opt=torch.randn(2,11,256); typ=torch.tensor([0,2])
    out=m(q,seg,opt,typ,forced_budget=4)
    assert out.probabilities.shape==(2,11)
    assert torch.allclose(out.probabilities.sum(-1),torch.ones(2),atol=1e-6)
    assert out.selected_indices.shape==(2,4)
    assert (out.tail_mass>=0).all()
