"""SMELT-ISR implementation split into coherent source parts."""

from __future__ import annotations

import sys
from pathlib import Path

from . import _manifest as _manifest

# NOTE: Module-level seed setting removed to prevent silent RNG state
# reset on import. Use the centralized seed function at the experiment
# entry point instead (e.g., repro.set_all_seeds / seed_everything).

_HERE = Path(__file__).resolve().parent
# ``__file__`` keeps pointing at the repository entry point so runtime code
# that fingerprints the script, runs git from the repository root, or spawns
# child runs keeps the pre-split behaviour.
__file__ = str(_HERE.parent / "experiment.py")

# Pre-populate commonly imported parts so that relative imports from other
# parts get the combined namespace with all names, not separate modules with
# incomplete imports. This avoids NameError for names defined in other parts.
this_module = sys.modules[__name__]
for _part_name in (
    "part_021_comparison",
    "part_022_comparison_protocol",
    "part_023_factorial_helpers",
    "part_024_window_comparison",
    "part_025_statistics",
    "part_026_equivalence",
    "part_027_aggregation",
    "part_028_contrasts",
    "part_029_ablation",
    "part_030_factorial",
    "part_031_ablation_coverage",
):
    sys.modules[f"exp.{_part_name}"] = this_module

for _part_name in _manifest.CORE_PARTS:
    _part_path = _HERE / _part_name
    with _part_path.open(encoding="utf-8") as _part_handle:
        _part_source = _part_handle.read()
    # nosec B102: exec() is used intentionally to compose experiment parts
    # from local project files only, never from user input.
    exec(compile(_part_source, str(_part_path), "exec"), globals())  # nosec B102

del _part_name, _part_path, _part_handle, _part_source
