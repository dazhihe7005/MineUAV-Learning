"""Open-loop, bounded four-rotor allocation for the estimated MineUAV v2 model.

Input wrench is [Fz, Tx, Ty, Tz]. Output u_i is omega_i**2 in (rad/s)**2.
This module is not a feedback controller and does not model motor lag.
"""

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "mine_uav_dynamics_v2.xml"
CONFIG = ROOT / "models" / "rotor_config.json"
MAX_TESTED_RPM = 8500.0


def load_v2_mixing_matrix(model_path: Path = MODEL, config_path: Path = CONFIG) -> np.ndarray:
    """Construct B from compiled model geometry and the shared rotor config."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    signs = config["rotation_signs"]
    if model.nu != 4 or len(signs) != 4 or any(s not in (-1, 1) for s in signs):
        raise ValueError("Expected four combined rotor actuators and four ±1 rotation signs")
    k_f = float(config["k_f_n_per_rad_s_squared"])
    k_m = float(config["k_m_nm_per_rad_s_squared"])
    com = model.body("mine_uav").ipos.copy()
    arms = np.array([model.site(f"motor_{i}_site").pos - com for i in range(1, 5)])
    for i, sign in enumerate(signs):
        expected_gear = [0, 0, k_f, 0, 0, sign * k_m]
        if not np.allclose(model.actuator_gear[i], expected_gear, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Actuator {i + 1} is stale relative to rotor_config.json")
    return np.array([
        np.full(4, k_f),
        arms[:, 1] * k_f,
        -arms[:, 0] * k_f,
        np.asarray(signs) * k_m,
    ])


@dataclass(frozen=True)
class AllocationResult:
    requested_wrench: np.ndarray
    u: np.ndarray
    achieved_wrench: np.ndarray
    error: np.ndarray
    rpm: np.ndarray
    omega_rad_s: np.ndarray
    feasible: bool
    saturated: bool
    active_limits: tuple[str, ...]
    weighted_error_norm: float


class ControlAllocator:
    """Exact allocation when feasible; box-constrained least squares otherwise."""

    def __init__(self, B: np.ndarray, max_rpm: float = MAX_TESTED_RPM):
        matrix = np.asarray(B, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 4:
            raise ValueError("B must be a finite, full-rank 4x4 matrix")
        if not math.isfinite(max_rpm) or max_rpm <= 0:
            raise ValueError("max_rpm must be finite and positive")
        self.B = matrix.copy()
        self.max_rpm = float(max_rpm)
        self.u_max = (self.max_rpm * 2 * math.pi / 60) ** 2
        # Wrench components have different units and natural authority.
        # Normalize each residual by its row's max absolute rotor authority.
        self._scale = np.sum(np.abs(self.B), axis=1) * self.u_max

    @classmethod
    def from_v2_model(cls) -> "ControlAllocator":
        return cls(load_v2_mixing_matrix())

    def _result(self, desired: np.ndarray, u: np.ndarray) -> AllocationResult:
        bounded = np.clip(u, 0.0, self.u_max)
        achieved = self.B @ bounded
        error = achieved - desired
        normalized_error = float(np.linalg.norm(error / self._scale))
        tolerance = 1e-8 * self.u_max
        active = []
        for i, speed_squared in enumerate(bounded, start=1):
            if speed_squared <= tolerance:
                active.append(f"motor_{i}_lower")
            elif self.u_max - speed_squared <= tolerance:
                active.append(f"motor_{i}_upper")
        feasible = normalized_error < 1e-9
        omega = np.sqrt(bounded)
        return AllocationResult(
            desired.copy(), bounded, achieved, error,
            omega * 60 / (2 * math.pi), omega, feasible,
            bool(active) or not feasible, tuple(active), normalized_error,
        )

    def allocate(self, desired_wrench: np.ndarray) -> AllocationResult:
        desired = np.asarray(desired_wrench, dtype=float)
        if desired.shape != (4,) or not np.isfinite(desired).all():
            raise ValueError("desired_wrench must be four finite values [Fz, Tx, Ty, Tz]")

        exact = np.linalg.solve(self.B, desired)
        tolerance = 1e-10 * self.u_max
        if np.all(exact >= -tolerance) and np.all(exact <= self.u_max + tolerance):
            return self._result(desired, exact)

        weighted_B = self.B / self._scale[:, None]
        weighted_target = desired / self._scale
        best_u = None
        best_cost = math.inf
        # 0=free, 1=lower bound, 2=upper bound. Four rotors yield only 81
        # candidate active sets, so no optimization dependency is needed.
        for state in itertools.product((0, 1, 2), repeat=4):
            candidate = np.zeros(4)
            free = [i for i, status in enumerate(state) if status == 0]
            upper = [i for i, status in enumerate(state) if status == 2]
            candidate[upper] = self.u_max
            if free:
                rhs = weighted_target - weighted_B @ candidate
                candidate[free] = np.linalg.lstsq(weighted_B[:, free], rhs, rcond=None)[0]
                if np.any(candidate[free] < -tolerance) or np.any(candidate[free] > self.u_max + tolerance):
                    continue
            candidate = np.clip(candidate, 0.0, self.u_max)
            cost = float(np.linalg.norm(weighted_B @ candidate - weighted_target) ** 2)
            if cost < best_cost:
                best_cost, best_u = cost, candidate
        if best_u is None:
            raise RuntimeError("No feasible box-constrained allocation candidate")
        return self._result(desired, best_u)
