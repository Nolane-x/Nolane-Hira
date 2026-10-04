from pathlib import Path


def test_s51_phase_a_is_structurally_train_only():
    path=Path("scripts/hira_v1_s51_native_authority.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")

    assert "generate_s51_train_cases" in source
    assert "generate_s51_dev_cases" not in source
    assert "PrivateCorrectionRepresentationFork" not in source
    assert "QueryFreeIdentityPrivateCorrectionFork" not in source
    assert "_train_private_branch" not in source
    assert '"dev_generated":False' in source
    assert '"dev_encoded":False' in source
    assert '"dev_scored":False' in source
    assert '"private_correction_constructed":False' in source
    assert "S51_NATIVE_FIXED_EPOCH+1" in source
    assert "make_native_authority_payload" in source
    assert "save_native_authority" in source
