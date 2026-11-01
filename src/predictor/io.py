"""Safe file writing.

The bot reads model and data files while background jobs (hourly counter-picker update,
nightly retraining) rewrite them. Writing to a temporary file and then swapping it in with
os.replace means a reader sees either the old file or the new one, never half a file.
"""

import os
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_write(path: Path):
    """with atomic_write(path) as tmp: write to tmp -> moved over `path` when done."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        yield tmp
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:  # Windows: a reader has the old file open for a moment
                if attempt == 19:
                    raise
                time.sleep(0.25)
    finally:
        if tmp.exists():
            tmp.unlink()
