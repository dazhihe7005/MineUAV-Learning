import pytest
from .test_interfaces import module

def test_metric_recomputation_and_no_completed_port_only_claim():
    a=module("analyze_phase1")
    rows=[{"time_s":0.,"position_enu_m":[0,0,.1],"velocity_origin_enu_m_s":[0,0,0],
           "quaternion_enu_flu":[1,0,0,0],"ctrl_omega_squared":[0]*4,"saturated":False},
          {"time_s":1.,"position_enu_m":[.03,.04,.6],"velocity_origin_enu_m_s":[.3,.4,0],
           "quaternion_enu_flu":[1,0,0,0],"ctrl_omega_squared":[1]*4,"saturated":False}]
    metrics=a.physical_metrics(rows)
    assert metrics["peak_speed_m_s"]==pytest.approx(.5)
    assert metrics["max_horizontal_offset_m"]==pytest.approx(.05)
    assert metrics["height_max_m"]==.6
    with pytest.raises(ValueError):a.validate_smoke({"complete":True,"px4_console":"TCP connected"})

def test_relative_cli_input_is_resolved_for_manifest():
    a=module("analyze_phase1")
    assert a.artifact_name("mujoco/models/rotor_config.json")=="mujoco/models/rotor_config.json"
