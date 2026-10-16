"""Best-effort SIGTERM cleanup; never raises asynchronously inside journal writes.

Not a SIGKILL/power-loss guarantee, a shutdown inhibitor, or a resume mechanism.
"""
import signal
import time
from pathlib import Path

from research_ops.storage import OpsError


class DeferredTermination:
    def __enter__(self):
        self.observation = None
        self.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.previous = signal.signal(signal.SIGTERM, self._observe)
        return self

    def _observe(self, number, _frame):
        # No filesystem IO, raising, budget operation, or child work in the handler.
        if self.observation is None:
            self.observation = {'signal_number': number, 'signal_name': 'SIGTERM',
                                'observed_unix_time_ns': time.time_ns(),
                                'observed_monotonic_ns': time.monotonic_ns(),
                                'boot_id': self.boot_id}

    def snapshot(self):
        return dict(self.observation) if self.observation is not None else 'NOT_OBSERVED'

    def checkpoint(self):
        if self.observation is not None:
            raise OpsError('SIGTERM requested; stop at supervisor transaction boundary')

    def __exit__(self, *_):
        signal.signal(signal.SIGTERM, self.previous)
