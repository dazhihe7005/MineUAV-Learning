"""Read-only L9 evidence audit; old failure cannot become a completed run."""

import shutil
from pathlib import Path

import pytest


def test_original_l9_failure_is_hash_verified_and_recounted_without_training():
    from .analyze_termination_repair import audit_original_l9
    result = audit_original_l9()
    assert result["overall_result"] == "FAIL"
    assert result["completed_train_episodes"] == 706
    assert result["terminal_counts"] == {"success": 0, "collision": 656, "time_limit": 50,
                                         "physical_failure": 0, "sensor_failure": 0}
    assert result["test_episode_count"] == 0
    assert result["original_raw_file_count"] == 11
    assert result["original_sources_verified_against_base_git_objects"] is True
    assert result["specific_original_physical_trigger"] == "UNRESOLVED_NOT_RECORDED"
    assert result["milestone"]["actual_execution_step"] == 100096
    assert result["milestone"]["ppo_diagnostics"]["train/clip_fraction"] == 0


def test_changed_original_evidence_cannot_pass_audit(tmp_path):
    from .analyze_termination_repair import audit_original_l9, ORIGINAL_RAW
    for path in ORIGINAL_RAW.iterdir():
        shutil.copy2(path, tmp_path / path.name)
    target = tmp_path / "train_episodes.jsonl"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="evidence identity"):
        audit_original_l9(raw_directory=tmp_path)


def test_old_complete_marker_cannot_reclassify_original_l9_as_success(tmp_path):
    from .analyze_termination_repair import audit_original_l9
    (tmp_path / "formal.complete.json").write_text("{}")
    with pytest.raises(RuntimeError, match="completion"):
        audit_original_l9(raw_directory=tmp_path)
