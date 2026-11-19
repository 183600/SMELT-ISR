"""Independent self-check routines split out of ``run_self_test``."""

from __future__ import annotations

import os

from exp._runner import run_isolated

from . import _manifest


def run_self_test(base_globals):
    here = os.path.dirname(__file__)
    paths = [os.path.join(here, name) for name in _manifest.PARTS]
    return run_isolated(
        paths,
        base_globals,
        _manifest.OWNED,
        _manifest.ENTRY,
    )
