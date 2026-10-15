"""Ensure regression scope never creates a second learning attempt."""
import importlib
import importlib.util


def test_l9s_scope_restores_safe_old_rejections_but_never_extra_training():
    name = "lidar_integration.verify_repaired_integration"
    assert importlib.util.find_spec(name) is not None, "L9S guarded scope missing"
    m = importlib.import_module(name)
    prefix = "mujoco/lidar_integration/test_ppo_formal_seed0_entry.py::"
    for node in ("test_formal_preflight_rejects_wrong_authorization_seed_budget_or_protocol[changes0-PermissionError]",
                 "test_formal_preflight_rejects_existing_output_without_deleting_it",
                 "test_formal_preflight_rejects_low_disk_and_low_memory"):
        assert m.exclusion_reason(prefix + node) is None
    for node in ("test_l9_driver_writes_real_post_optimizer_health_snapshot",
                 "test_frozen_l8_physical_reason_contract_reproduces_formal_stop_without_training"):
        assert m.exclusion_reason(prefix + node)
