"""Prevent the offline regression command from invoking learning/update code."""

import pytest
import torch


def test_offline_scope_excludes_update_and_superseded_history_expectations():
    from .verify_termination_repair import exclusion_reason
    assert exclusion_reason("mujoco/lidar_integration/test_ppo_preflight.py::"
                            "test_actor_critic_optimizer_and_checkpoint_roundtrip")
    assert exclusion_reason("mujoco/lidar_integration/test_ppo_formal_seed0_entry.py::"
                            "test_frozen_l8_physical_reason_contract_reproduces_formal_stop_without_training")
    assert exclusion_reason("mujoco/lidar_integration/test_failure_evidence.py::"
                            "test_health_nonfinite_position_is_recorded_as_unavailable_not_guessed_or_raised") is None


def test_training_tripwire_prevents_update_before_parameter_changes():
    from .verify_termination_repair import no_training_updates
    value = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.SGD([value], lr=.1)
    value.grad = torch.tensor([1.])
    with no_training_updates() as calls:
        with pytest.raises(AssertionError, match="forbids"):
            optimizer.step()
    assert value.item() == 1. and len(calls) == 1
