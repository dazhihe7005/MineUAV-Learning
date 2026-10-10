"""Passive, source-introspected CPython observation of the unchanged PI cascade.

No controller replacement, monkeypatch, local-variable writes or optimizer.
Line probes copy the real clamp branch and pre-limit helper intermediates.
P is explicitly an algebraic diagnostic from recorded gains/error, not an
independently stored controller variable. Raw data never enter observations.
"""
import ast
import inspect
import sys
import textwrap

import numpy as np


def probe_lines(function):
    source, first = inspect.getsourcelines(function)
    nodes = list(ast.walk(ast.parse(textwrap.dedent(''.join(source)))))
    def line(predicate):
        matches = [n for n in nodes if predicate(n)]
        if len(matches) != 1:
            raise ValueError('controller source no longer matches unique telemetry probe')
        return first + matches[0].lineno - 1
    return dict(
        clamp_xy=line(lambda n: isinstance(n, ast.AugAssign) and ast.unparse(n.target)=='candidate[:2]' and isinstance(n.op, ast.Mult)),
        clip_z=line(lambda n: isinstance(n, ast.Assign) and ast.unparse(n.targets[0])=='candidate[2]' and isinstance(n.value, ast.Call)),
        after_clamp=line(lambda n: isinstance(n, ast.Assign) and ast.unparse(n.targets[0])=='candidate_extra'))


class PassiveTelemetry:
    """Single-thread diagnostic context; leaves every production source unchanged."""
    def __init__(self, env):
        from velocity_command_controller import velocity_error_to_acceleration
        self.env = env
        self.control_rows = []
        self.active = None
        self.frame = None
        self.compute_code = env.velocity_controller.compute.__func__.__code__
        self.allocator_code = env.allocator.allocate.__func__.__code__
        self.helper_code = velocity_error_to_acceleration.__code__
        self.lines = probe_lines(env.velocity_controller.compute)
        source, first = inspect.getsourcelines(velocity_error_to_acceleration)
        nodes = ast.walk(ast.parse(textwrap.dedent(''.join(source))))
        matches = [n for n in nodes if isinstance(n, ast.Assign) and ast.unparse(n.targets[0])=='horizontal_norm']
        if len(matches) != 1: raise ValueError('pre-limit helper probe missing')
        self.pre_limit_line = first + matches[0].lineno - 1
        apply = getattr(env, '_apply_rotor_command', None)
        self.apply_code = apply.__func__.__code__ if apply else None

    def __enter__(self):
        if sys.gettrace() is not None:
            raise RuntimeError('refuse to interfere with an existing trace/debugger')
        sys.settrace(self._trace)
        return self

    def __exit__(self, *exc):
        sys.settrace(None)
        self.frame = self.active = None

    def _trace(self, frame, event, arg):
        loc = frame.f_locals
        if frame.f_code is self.compute_code and loc.get('self') is self.env.velocity_controller:
            if event == 'call':
                c = self.env.velocity_controller
                self.frame = frame
                self.active = dict(time_s=float(self.env.data.time), policy_step=self.env.episode_steps,
                    position_m=self.env.data.qpos[:3].copy(), target_m=self.env.target_position.copy(),
                    measured_velocity_m_s=np.asarray(loc['velocity'], float).copy(),
                    commanded_velocity_m_s=np.asarray(loc['velocity_command'], float).copy(),
                    actual_quaternion_wxyz=np.asarray(loc['quaternion'], float).copy(),
                    actual_angular_velocity_body=np.asarray(loc['angular_velocity'], float).copy(),
                    cached_qacc_world_m_s2=self.env.data.qacc[:3].copy(),
                    integral_before_m=c.integral_error, integral_clamp_xy=False, integral_clamp_z=False,
                    freeze_before_xy=c.anti_windup_freeze_count_xy, freeze_before_z=c.anti_windup_freeze_count_z,
                    success_streak_at_entry=self.env.success_streak)
            elif event == 'line':
                if frame.f_lineno == self.lines['clamp_xy']:
                    self.active['integral_clamp_xy'] = True  # actual branch entered
                elif frame.f_lineno == self.lines['clip_z']:
                    self.active['integral_z_before_clip_m'] = float(loc['candidate'][2])
                elif frame.f_lineno == self.lines['after_clamp']:
                    before = self.active.get('integral_z_before_clip_m', float(loc['candidate'][2]))
                    self.active['integral_clamp_z'] = bool(before != loc['candidate'][2])
                    self.active['bounded_trial_integral_m'] = loc['candidate'].copy()
            elif event == 'return':
                if arg is not None:
                    c = self.env.velocity_controller
                    row = self.active
                    row.update(velocity_error_m_s=loc['error'].copy(), integral_after_m=c.integral_error,
                        i_contribution_m_s2=c.integral_acceleration_world,
                        p_contribution_m_s2_derived=c.kv * loc['error'],
                        anti_windup_trial_m_s2=loc['raw'].copy(),
                        anti_windup_freeze_xy=c.anti_windup_freeze_count_xy > row['freeze_before_xy'],
                        anti_windup_freeze_z=c.anti_windup_freeze_count_z > row['freeze_before_z'],
                        freeze_count_xy=c.anti_windup_freeze_count_xy, freeze_count_z=c.anti_windup_freeze_count_z,
                        desired_quaternion_wxyz=arg.desired_quaternion_wxyz.copy(),
                        desired_rotation_body_to_world=arg.desired_rotation_body_to_world.copy(),
                        desired_thrust_vector_world_n=arg.thrust_vector_world.copy(),
                        desired_thrust_n=float(np.linalg.norm(arg.thrust_vector_world)),
                        desired_wrench_body=arg.desired_wrench_body.copy(),
                        output_after_limiting_m_s2=arg.acceleration_world.copy(),
                        yaw_target_rad=c.yaw_target)
                    pre = row['output_before_limiting_m_s2']
                    row['output_limited_xy'] = bool(np.any(pre[:2] != arg.acceleration_world[:2]))
                    row['output_limited_z'] = bool(pre[2] != arg.acceleration_world[2])
                    self.control_rows.append(row)
                self.frame = self.active = None
            return self._trace
        if (frame.f_code is self.helper_code and frame.f_back is self.frame
                and loc.get('integral_acceleration') is not None):
            if event == 'line' and frame.f_lineno == self.pre_limit_line:
                self.active['output_before_limiting_m_s2'] = loc['acceleration'].copy()
            return self._trace
        if frame.f_code is self.allocator_code and loc.get('self') is self.env.allocator:
            if event == 'return' and arg is not None:
                if not self.control_rows or self.control_rows[-1]['time_s'] != float(self.env.data.time):
                    raise ValueError('allocator output has no aligned controller update')
                self.control_rows[-1].update(allocator_achieved_wrench=arg.achieved_wrench.copy(),
                    rotor_command_u=arg.u.copy(), allocator_saturated=arg.saturated,
                    commanded_thrust_n=float(arg.achieved_wrench[0]), allocator_error=arg.error.copy())
            return self._trace
        if frame.f_code is self.apply_code and loc.get('self') is self.env:
            if event == 'return':
                self.control_rows[-1]['actual_ctrl_u'] = self.env.data.ctrl.copy()
            return self._trace
        return None

    def arrays(self, physics_force_times, physics_forces):
        if not self.control_rows: raise ValueError('no control updates observed')
        fields = set(self.control_rows[0])
        if any(set(row) != fields for row in self.control_rows):
            raise ValueError('missing/unaligned telemetry field')
        arrays = {name:np.asarray([row[name] for row in self.control_rows]) for name in fields}
        times = arrays['time_s']; tick = np.rint(times / .002).astype(int)
        if np.any(tick < 0) or np.any(tick >= len(physics_force_times)):
            raise ValueError('control/physics force timestamp outside recorded trace')
        np.testing.assert_allclose(np.asarray(physics_force_times)[tick], times, atol=1e-10, rtol=0)
        arrays['external_force_world_n'] = np.asarray(physics_forces)[tick].copy()
        # Force is indexed AFTER the real before-physics hook, never from its stale buffer at compute entry.
        return {'control_'+name:value for name,value in arrays.items()}


def verify_passive(old, new):
    if not set(old).issubset(new): raise ValueError('baseline signal missing')
    differences = {}
    bitwise = True
    for name in old:
        a, b = np.asarray(old[name]), np.asarray(new[name])
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('baseline shape/dtype changed: '+name)
        exact = a.tobytes() == b.tobytes()
        if not exact:
            if a.dtype.kind not in 'fc' or not np.allclose(a, b, atol=1e-12, rtol=0):
                raise ValueError('telemetry changed baseline trajectory: '+name)
        bitwise &= exact
        differences[name] = dict(bitwise_equal=exact, max_abs_difference=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.)
    return dict(bitwise_equal=bool(bitwise), tolerance=dict(atol=1e-12, rtol=0), signals=differences)
