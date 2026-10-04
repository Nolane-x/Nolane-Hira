from pathlib import Path


def test_s50_trainer_compiles_and_native_phase_is_train_only_fixed_24():
    path=Path("scripts/hira_v1_s50_train_dev.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")

    start=source.index("def _train_shared_native")
    end=source.index("\ndef _expanded_ids",start)
    native=source[start:end]

    assert "NATIVE_EPOCHS=24" in source
    assert "for epoch in range(1,NATIVE_EPOCHS+1)" in native
    assert "dev_rows" not in native
    assert "_arm_evaluate" not in native
    assert "dev_scored_before_private_phase" in native
    assert '"dev_scored_before_private_phase":False' in native
    assert '"dev_encoded_before_private_phase":False' in native
    assert "fixed-epoch24" in native


def test_s50_cache_generation_uses_one_native_output_call_per_batch():
    source=Path("scripts/hira_v1_s50_train_dev.py").read_text(encoding="utf-8")
    start=source.index("def _freeze_pair")
    end=source.index("\ndef _cache_digest",start)
    block=source[start:end]
    assert block.count("s45a0._native_outputs(runtime,rows)")==1
    assert "canonical=freeze_shared_native_evidence" in block
    assert "paraphrase=freeze_shared_native_evidence" in block
    assert "triadic_logits=raw_c" in block
    assert "triadic_logits=raw_p" in block


def test_s50_private_phase_uses_cache_only_and_no_native_optimizer():
    source=Path("scripts/hira_v1_s50_train_dev.py").read_text(encoding="utf-8")
    start=source.index("def _train_private_branch")
    end=source.index("\ndef _metric_deltas",start)
    block=source[start:end]
    assert "reference_private_logits" not in block or "_private_loss" in source
    assert "torch.optim.AdamW(params" in block
    assert "runtime" not in block
    assert "_private_metrics(op,kind,dev_cache)" in block
    assert "s17._selection_key(epoch,metrics)" in block


def test_s50_private_loss_and_eval_reconstruct_fused_from_cache():
    source=Path("scripts/hira_v1_s50_train_dev.py").read_text(encoding="utf-8")
    assert "fused_private_logits(evidence,relation" in source
    assert "reference_private_logits(op,evidence)" in source
    assert "treatment_private_logits(op,evidence)" in source
    assert "symmetric_js_divergence(relation_c,relation_p)" in source
    assert "F.cross_entropy(relation_c,canonical.gold)" in source


def test_s50_main_destroys_live_native_runtime_before_private_optimizers():
    source=Path("scripts/hira_v1_s50_train_dev.py").read_text(encoding="utf-8")
    start=source.index("def main")
    main=source[start:]
    cache_pos=main.index("train_cache,train_cache_digest=_materialize_cache")
    delete_pos=main.index("del runtime")
    ref_pos=main.index("reference_op=PrivateCorrectionRepresentationFork")
    assert cache_pos < delete_pos < ref_pos
    assert '"native_runtime_live_graph":False' in main
    assert '"native_optimizer_parameter_count":0' in main
    assert '"dev_scored_before_private_phase":False' in main
    assert '"cache_regenerated_after_dev":False' in main
    assert '"native_retrained_per_branch":False' in main


def test_s50_matched_workflow_is_one_shot_a0_bound():
    path=Path(".github/workflows/hira-v1-s50-shared-native-forked-private-train-dev.yml")
    source=path.read_text(encoding="utf-8")
    assert "feat/hira-v1-s50-shared-native-forked-readouts" in source
    assert "research/HIRA-V1-S50-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37183981097" in source
    assert "hira-v1-s50-a0-shared-native-forked-readouts" in source
    assert "scripts/hira_v1_s50_train_dev.py" in source
    assert "tests/test_v1_s50_matched_harness.py" in source
    assert 'assert r["seed"]==71001' in source
    assert 'assert n["fixed_epoch"]==24' in source
    assert 'assert n["dev_scored_before_private_phase"] is False' in source
    assert 'assert p["native_optimizer_parameter_count"]==0' in source
    assert 'assert c["reference_treatment_same_cache_bytes"] is True' in source
    assert 'assert r["second_dev_run_performed"] is False' in source


def test_s50_mechanical_replay_is_hash_locked_and_private_dev_unexposed():
    source=Path("scripts/hira_v1_s50_train_dev.py").read_text(encoding="utf-8")
    assert '--recovered-native' in source
    assert "hira-v1-s50-recovered-native-preprivate-abort-v1" in source
    assert "37185080959" in source
    assert "observed!=frozen" in source
    assert "mechanical replay native trajectory mismatch" in source
    assert '"native_24_hashes_exact":True' in source
    assert '"failed_run_private_dev_scored":False' in source


def test_s50_mechanical_replay_workflow_is_separately_marker_gated():
    path=Path(".github/workflows/hira-v1-s50-mechanical-replay.yml")
    source=path.read_text(encoding="utf-8")
    assert "research/HIRA-V1-S50-ENABLE-MECHANICAL-REPLAY" in source
    assert "--recovered-native research/HIRA-V1-S50-RECOVERED-NATIVE.json" in source
    assert "run-id: 37183981097" in source
    assert 'replay=r["mechanical_replay"]' in source
    assert 'assert replay["native_24_hashes_exact"] is True' in source
    assert 'assert replay["failed_run_private_dev_scored"] is False' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
