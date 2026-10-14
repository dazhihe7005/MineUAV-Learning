"""Offline L9R identity/authorization checks; tripwire forbids PPO construction."""

from pathlib import Path

import pytest

from .ppo_formal_scheduler import load_l6_prereg
from .ppo_training_orchestration import FormalPPOTrainingDriver, l8_execution_protocol
from .ppo_training_runner import protocol_hash


UPSTREAM = Path("/tmp/mineuav-mujoco-lidar-audit")
PREREG = Path(__file__).resolve().parents[1] / "reports/mujoco_lidar_ppo_training_prereg_seed0.json"
OLD_PROTOCOL = "fe07816f48bd4183165db482d8c93de50c637a32d13d86eccb47667d85e8cef6"


def test_repair_protocol_preserves_parent_and_disallows_training():
    from .ppo_termination_repair_protocol import repair_execution_protocol
    prereg = load_l6_prereg(PREREG)
    parent = l8_execution_protocol(prereg)
    repaired = repair_execution_protocol(prereg)
    assert protocol_hash(parent) == OLD_PROTOCOL
    assert repaired["parent_l8_execution_protocol_sha256"] == OLD_PROTOCOL
    assert protocol_hash(repaired) != OLD_PROTOCOL
    assert repaired["formal_retrain_authorized"] is False
    assert repaired["reward_version"] == "NAV_V2_COLLISION_PREFLIGHT_V1"
    assert repaired["n_steps"] == 128 and repaired["batch_size"] == 64
    for key in parent:
        assert repaired[key] == parent[key]
    assert {"terminal_accounting.py", "failure_evidence.py", "ppo_training_orchestration.py",
            "ppo_formal_validation.py", "ppo_formal_seed0_health.py"} <= set(repaired["repair_source_sha256"])


def test_repair_protocol_wrong_expected_identity_and_source_drift_fail_closed(tmp_path, monkeypatch):
    from . import ppo_termination_repair_protocol as module
    prereg = load_l6_prereg(PREREG)
    with pytest.raises(RuntimeError, match="identity"):
        module.verify_repair_protocol(prereg, expected_sha256="0" * 64)
    source = tmp_path / "terminal_accounting.py"
    source.write_text("frozen test source")
    monkeypatch.setattr(module, "_repair_source_paths", lambda: (source,))
    expected = protocol_hash(module.repair_execution_protocol(prereg))
    module.verify_repair_protocol(prereg, expected_sha256=expected)
    source.write_text("changed test source")
    with pytest.raises(RuntimeError, match="identity"):
        module.verify_repair_protocol(prereg, expected_sha256=expected)


def test_formal_entry_cannot_start_repaired_unapproved_execution(tmp_path, monkeypatch):
    from . import ppo_training_orchestration as module
    output = tmp_path / "unauthorized_formal_output"
    driver = FormalPPOTrainingDriver(PREREG, UPSTREAM, output)
    def optimizer_tripwire(*args, **kwargs):
        raise RuntimeError("TEST_TRIPWIRE: unapproved execution reached PPO construction")
    monkeypatch.setattr(module, "make_ppo", optimizer_tripwire)
    with pytest.raises(PermissionError, match="not authorized"):
        driver.run_formal(allow_formal_training=True)
    assert not output.exists()


def test_published_repair_protocol_roundtrip_preserves_type_and_source_identity():
    import json
    from .ppo_termination_repair_protocol import verify_repair_protocol
    path = PREREG.parent / "mujoco_lidar_ppo_termination_repair.json"
    identity = json.loads(path.read_text())["frozen_repair_identity"]
    assert isinstance(identity["protocol"]["ent_coef"], float)
    assert protocol_hash(identity["protocol"]) == identity["protocol_sha256"]
    verify_repair_protocol(load_l6_prereg(PREREG), expected_sha256=identity["protocol_sha256"])
