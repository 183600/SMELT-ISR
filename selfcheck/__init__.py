"""Independent self-check routines split out of ``run_self_test``."""

from __future__ import annotations

from pathlib import Path

from exp._runner import run_isolated

from . import _manifest


def run_self_test(base_globals):
    here = Path(__file__).resolve().parent
    paths = [str(here / name) for name in _manifest.PARTS]
    return run_isolated(
        paths,
        base_globals,
        _manifest.OWNED,
        _manifest.ENTRY,
    )
