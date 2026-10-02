"""Reproduce the 15x5CF RPM–torque fit and compare it with the model estimate."""

import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "references" / "tmotor_15x5cf_mn5212_kv340_static_tests.csv"
USED_K_M = 8.6e-7  # ESTIMATED_FROM_TMOTOR_15x5CF_STATIC_TESTS; not a U7 KV490 published coefficient.
PROVENANCE = "ESTIMATED_FROM_TMOTOR_15x5CF_STATIC_TESTS"


def fit(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Static-test CSV has no data rows")
    omega_squared = []
    torques = []
    for row in rows:
        rpm = float(row["rpm"])
        torque = float(row["torque_nm"])
        if not (math.isfinite(rpm) and math.isfinite(torque) and rpm > 0 and torque >= 0):
            raise ValueError(f"Invalid RPM–torque row: {row}")
        omega_squared.append((rpm * 2 * math.pi / 60) ** 2)
        torques.append(torque)
    # Constrained fit tau = k_m * omega^2, with zero torque at zero RPM.
    coefficient = sum(x * y for x, y in zip(omega_squared, torques)) / sum(x * x for x in omega_squared)
    mean_torque = sum(torques) / len(torques)
    residual = sum((y - coefficient * x) ** 2 for x, y in zip(omega_squared, torques))
    total = sum((y - mean_torque) ** 2 for y in torques)
    return {
        "provenance": PROVENANCE,
        "reference_motor": "MN5212 KV340",
        "reference_propeller": "T-MOTOR 15*5CF",
        "source_url": rows[0]["source_url"],
        "samples": len(rows),
        "through_origin_fit_k_m": coefficient,
        "used_k_m": USED_K_M,
        "difference_percent": abs(USED_K_M - coefficient) / coefficient * 100,
        "fit_r_squared": 1 - residual / total if total else 1.0,
        "unit": "N*m/(rad/s)^2",
        "note": "Used value is a conservative user-specified estimate, not a direct U7 KV490 manufacturer coefficient.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = fit(args.data)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for key, value in report.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
