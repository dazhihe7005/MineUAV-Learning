"""L9S offline regression entry: no learning or optimizer calls permitted.

Seven original early-rejection cases can run unchanged. The other fifteen
historical tests remain deselected; actual optimization is the unique Pilot.
"""
import argparse
import json
import resource
import time
from pathlib import Path

import pytest

from . import verify_termination_repair as old
from .ppo_repaired_pilot import publish_json

SAFE_OLD_ENTRY = {
    "test_formal_preflight_rejects_wrong_authorization_seed_budget_or_protocol",
    "test_formal_preflight_rejects_existing_output_without_deleting_it",
    "test_formal_preflight_rejects_low_disk_and_low_memory",
}


def exclusion_reason(nodeid):
    name = nodeid.rsplit("::", 1)[-1].split("[")[0]
    if "test_ppo_formal_seed0_entry.py" in nodeid and name in SAFE_OLD_ENTRY:
        return None
    return old.exclusion_reason(nodeid)


class Scope(old.OfflineScope):
    def pytest_collection_modifyitems(self, config, items):
        keep, remove = [], []
        for item in items:
            reason = exclusion_reason(item.nodeid)
            if reason:
                remove.append(item)
                self.removed.append({"nodeid": item.nodeid, "reason": reason})
            else:
                keep.append(item)
        items[:] = keep
        config.hook.pytest_deselected(items=remove)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.evidence.exists():
        raise FileExistsError("test evidence cannot be overwritten")
    scope, begun = Scope(), time.perf_counter()
    with old.no_training_updates() as calls:
        code = pytest.main(["-q", "--tb=short", *old.PATHS], plugins=[scope])
    evidence = {"exit_code": int(code), "complete": int(code) in (0, 1),
        "wall_seconds": time.perf_counter()-begun,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
        "minimum_mem_available_mib": min(scope.mem_samples) if scope.mem_samples else None,
        "forbidden_calls": calls, "results": scope.results, "deselected": scope.removed,
        "actual_optimizer_updates": 0,
        "scope": "related Bridge, LiDAR, audit, Shadow, targeted legacy controllers; no full-repository run"}
    publish_json(args.evidence, evidence)
    print("L9S_REGRESSION=" + json.dumps({k:v for k,v in evidence.items()
                                       if k not in ("results", "deselected")}))
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
