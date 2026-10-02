"""Fit F = k_f * omega^2 to official U7 KV490 + 15*5CF static data."""

import argparse
import csv
import json
import math
from pathlib import Path


DATA = Path(__file__).resolve().parents[1] / "references" / "tmotor_u7_kv490_15x5cf_static_tests.csv"
STANDARD_GRAVITY = 9.80665  # Convert gram-force test readings to newtons.


def fit(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No static-test rows")
    omega2 = []
    forces = []
    for row in rows:
        rpm = float(row["rpm"])
        thrust_g = float(row["thrust_g"])
        if not (math.isfinite(rpm) and math.isfinite(thrust_g) and rpm > 0 and thrust_g >= 0):
            raise ValueError(f"Invalid RPM or thrust: {row}")
        omega2.append((rpm * 2 * math.pi / 60) ** 2)
        forces.append(thrust_g * STANDARD_GRAVITY / 1000)
    k_f = sum(x * force for x, force in zip(omega2, forces)) / sum(x * x for x in omega2)
    mean_force = sum(forces) / len(forces)
    sse = sum((force - k_f * x) ** 2 for x, force in zip(omega2, forces))
    sst = sum((force - mean_force) ** 2 for force in forces)
    return {
        "source_url": rows[0]["source_url"],
        "reference_motor": "U7 KV490",
        "reference_propeller": "T-MOTOR 15*5CF",
        "samples": len(rows),
        "fit_method": "least_squares_through_origin_F=k_f*omega_squared",
        "k_f_n_per_rad_s_squared": k_f,
        "r_squared": 1 - sse / sst if sst else 1.0,
        "gravity_for_gram_force_conversion_m_s2": STANDARD_GRAVITY,
        "measured_rpm_range": [min(float(row["rpm"]) for row in rows), max(float(row["rpm"]) for row in rows)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = fit(args.data)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
