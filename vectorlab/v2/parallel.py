"""Independent pieces of one trace, run on several processor cores.

Tracing each shape and fitting each outline doesn't depend on any other,
and most of that work is many small steps in Python, which threads can't
run side by side. Worker processes can. The arrays every task reads (the
partition, the padded image) are sent to each worker once, when it starts.

Processes are started with "spawn", which is safe on every platform but
re-imports the caller's main module in each worker. A script that traces
with workers > 1 must keep its top-level code under
``if __name__ == "__main__":``, as the command-line tool and server do.
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

_context = None


def _start(context):
    global _context
    _context = context
    # One process per core already; threads inside each would only compete.
    from . import fitting

    fitting.THREADS = 1


def _run(task):
    function, batch, shared = task
    return [function(_context, item, *shared) for item in batch]


def worker_count(requested):
    """0 or None means one per core."""
    if not requested:
        return os.cpu_count() or 1
    return max(1, int(requested))


class Workers:
    """Runs function(context, item, *shared) for many items, in order, on a pool of processes.

    With one worker, or too few items to be worth starting processes for,
    everything runs here.
    """

    def __init__(self, count, context, min_items=1000):
        self.count, self.context, self.min_items = count, context, min_items
        self._pool = None

    def map(self, function, items, *shared, force=False):
        items = list(items)
        if self.count <= 1 or (len(items) < self.min_items and not force):
            return [function(self.context, item, *shared) for item in items]
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.count,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_start,
                initargs=(self.context,),
            )
        # A few batches per worker keeps them all busy without a message per item.
        size = max(1, -(-len(items) // (self.count * 4)))
        batches = [(function, items[start : start + size], shared) for start in range(0, len(items), size)]
        return [result for batch in self._pool.map(_run, batches) for result in batch]

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
