from nmd.v1_s24_authority import generate_s24_cases, validate_s24_partitions

def test_s24_authority_partition_shape_and_fresh_views():
    train=generate_s24_cases("train"); dev=generate_s24_cases("dev")
    validate_s24_partitions(train,dev)
    assert len(train)==768 and len(dev)==192
    assert len({r.domain for r in train})==12
    assert all(len(r.option_texts)==4 and len(r.option_aliases)==4 for r in (*train,*dev))

def test_s24_authority_train_dev_text_isolation():
    train=generate_s24_cases("train"); dev=generate_s24_cases("dev")
    train_text={x for r in train for x in (r.state_a,r.state_b,r.question_a1,r.question_a2,r.question_b1,r.question_b2,*r.option_texts,*r.option_aliases)}
    dev_text={x for r in dev for x in (r.state_a,r.state_b,r.question_a1,r.question_a2,r.question_b1,r.question_b2,*r.option_texts,*r.option_aliases)}
    assert not (train_text & dev_text)

def test_s24_authority_is_deterministic():
    assert generate_s24_cases("train")==generate_s24_cases("train")
    assert generate_s24_cases("dev")==generate_s24_cases("dev")
