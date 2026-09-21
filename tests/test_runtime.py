import torch
from nmd.contracts import LogicalOption
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder


def options():
    return [
        LogicalOption("opaque_a", "the payment was declined"),
        LogicalOption("opaque_b", "the cash withdrawal is missing"),
        LogicalOption("opaque_c", "the transfer is still pending"),
    ]


def test_registered_schema_cache_and_state_once():
    torch.manual_seed(0)
    enc = TrainableSemanticEncoder(vocab_size=512, d_model=256, n_layers=1, n_heads=4)
    model = NolaneHira(enc, HIRACore(dropout=0.0))
    model.eval()
    schema1, receipt1 = model.compile_schema(
        primitive="choice", question_text="what happened?", options=options(), use_cache=True
    )
    schema2, receipt2 = model.compile_schema(
        primitive="choice", question_text="what happened?", options=options(), use_cache=True
    )
    assert not receipt1.cache_hit and receipt2.cache_hit
    assert schema1.schema_hash == schema2.schema_hash

    memory = model.compile_state("my card payment was rejected yesterday")
    before = enc.state_encode_calls
    a = model.forward_compiled(memory, schema1, forced_budget=2)
    b = model.forward_compiled(memory, schema2, forced_budget=2)
    assert enc.state_encode_calls == before
    assert torch.allclose(a.probabilities, b.probabilities)
    assert a.probabilities.shape == (3,)


def test_noul_requires_two_options():
    enc = TrainableSemanticEncoder(vocab_size=256, d_model=256, n_layers=1, n_heads=4)
    model = NolaneHira(enc, HIRACore(dropout=0.0))
    schema, _ = model.compile_schema(
        primitive="noul",
        question_text="is it fraudulent?",
        options=[LogicalOption("x", "false"), LogicalOption("y", "true")],
        use_cache=False,
    )
    memory = model.compile_state("the transaction looks legitimate")
    out = model.forward_compiled(memory, schema, forced_budget=2)
    assert out.value.ndim == 0
    assert 0 <= out.value.item() <= 1


def test_opaque_ids_change_cache_identity_not_semantics():
    torch.manual_seed(4)
    enc = TrainableSemanticEncoder(vocab_size=512, d_model=256, n_layers=1, n_heads=4)
    model = NolaneHira(enc, HIRACore(dropout=0.0))
    model.eval()
    a = [
        LogicalOption("billing", "the payment was declined"),
        LogicalOption("transfer", "the transfer is pending"),
    ]
    b = [
        LogicalOption("x7", "the payment was declined"),
        LogicalOption("q2", "the transfer is pending"),
    ]
    sa, _ = model.compile_schema(primitive="choice", question_text="what happened?", options=a)
    sb, _ = model.compile_schema(primitive="choice", question_text="what happened?", options=b)
    assert sa.schema_hash != sb.schema_hash
    assert torch.allclose(sa.option_embeddings, sb.option_embeddings)
    memory = model.compile_state("the transfer has not completed")
    pa = model.forward_compiled(memory, sa, forced_budget=2).probabilities
    pb = model.forward_compiled(memory, sb, forced_budget=2).probabilities
    assert torch.allclose(pa, pb)


def test_entering_train_mode_clears_registered_schema_cache():
    enc = TrainableSemanticEncoder(vocab_size=256, d_model=256, n_layers=1, n_heads=4)
    model = NolaneHira(enc, HIRACore(dropout=0.0))
    model.eval()
    _, r1 = model.compile_schema(
        primitive="choice", question_text="what happened?", options=options(), use_cache=True
    )
    _, r2 = model.compile_schema(
        primitive="choice", question_text="what happened?", options=options(), use_cache=True
    )
    assert not r1.cache_hit and r2.cache_hit
    model.train()
    model.eval()
    _, r3 = model.compile_schema(
        primitive="choice", question_text="what happened?", options=options(), use_cache=True
    )
    assert not r3.cache_hit
