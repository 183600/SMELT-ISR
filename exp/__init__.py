"""SMELT-ISR implementation split into coherent source parts."""

from __future__ import annotations

import os as _os

from . import _manifest as _manifest

_HERE = _os.path.dirname(_os.path.abspath(__file__))
# ``__file__`` keeps pointing at the repository entry point so runtime code
# that fingerprints the script, runs git from the repository root, or spawns
# child runs keeps the pre-split behaviour.
__file__ = _os.path.join(_os.path.dirname(_HERE), "experiment.py")

for _part_name in _manifest.CORE_PARTS:
    _part_path = _os.path.join(_HERE, _part_name)
    with open(_part_path, encoding="utf-8") as _part_handle:
        _part_source = _part_handle.read()
    exec(compile(_part_source, _part_path, "exec"), globals())

del _os, _part_name, _part_path, _part_handle, _part_source
