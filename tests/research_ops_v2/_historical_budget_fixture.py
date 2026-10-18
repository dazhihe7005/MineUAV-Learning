"""Explicit R28 reader double for single-process historical unit tests ONLY.

Not imported by runtime code. Does not authorize execution, alter the ledger,
filter current attempts, or replace proof checks. Current-ledger rejection is
tested separately outside this context. No training/model imports are allowed.
"""
import copy
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from research_ops.storage import OpsError, read_json, sha256
from research_regression.authorized_rl.budget import BudgetStore

R28_REPORT_SHA256 = '54af04e2001f6612fea6a20d6af1837243a3a7b23c728ddf3d61d3b0921b35fc'


@contextmanager
def r28_budget_reader(root):
    root = Path(root).resolve()
    path = root/'mujoco/reports/autonomous_rl_held_goal_r28.json'
    try:
        if sha256(path) != R28_REPORT_SHA256:
            raise OpsError('historical R28 report identity differs')
        snapshot = read_json(path)['budget_after']
    except (OSError, KeyError) as error:
        raise OpsError('missing historical R28 report identity') from error
    original = BudgetStore.status
    expected_home = root/'.research_local/authorized_rl_v1'

    def historical_reader(store):
        if store.home.resolve() == expected_home:
            return copy.deepcopy(snapshot)
        return original(store)

    with patch.object(BudgetStore, 'status', new=historical_reader):
        yield copy.deepcopy(snapshot)
