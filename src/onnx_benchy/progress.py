"""Progress-bar helpers (tqdm, disabled under --quiet/--no-progress/non-tty)."""

from __future__ import annotations

from contextlib import contextmanager

from tqdm import tqdm


@contextmanager
def bar(show: bool, *args, **kwargs):
    if not show:
        kwargs["disable"] = True
    with tqdm(*args, **kwargs) as p:
        yield p
