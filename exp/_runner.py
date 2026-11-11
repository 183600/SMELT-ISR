"""Shared execution helper for source parts produced by the refactor."""

from __future__ import annotations

import os
# Reproducibility: Set random seeds
import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass


class _ReturnSignal(Exception):
    """Internal signal used to emulate ``return`` across split parts."""

    def __init__(self, value=None):
        super().__init__("split return")
        self.value = value


class _IsolatedNamespace(dict):
    """A function-local namespace that reads live module globals.

    Names assigned by the original function (``owned``) stay in this dict.
    Every other lookup falls back to the original module globals, so state
    mutated by helpers (seed manifests, caches, environment snapshots) is seen
    exactly as before the split.  Explicit writes through ``globals()`` to a
    non-owned name are redirected back to the module globals.
    """

    def __init__(self, base, owned):
        super().__init__()
        self._base = base
        self._owned = set(owned)
        self._owned.update({"_SC_BASE", "_SC_RETURN"})
        self._loading = True
        self["__name__"] = "exp._isolated"
        self["_SC_BASE"] = base
        self["_SC_RETURN"] = _ReturnSignal

    def __missing__(self, key):
        return self._base[key]

    def __setitem__(self, key, value):
        if (
            self._loading
            or key in self._owned
            or key.startswith("__")
            or key.startswith("_SC_")
        ):
            dict.__setitem__(self, key, value)
        else:
            # Explicit ``globals()[...] = ...`` writes must reach the module,
            # because called helpers read their own global namespace.
            self._base[key] = value

    def finish_loading(self):
        self._loading = False


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def run_isolated(part_paths, base_globals, owned_names, entry_calls, initial=None):
    """Execute ``entry_calls`` in a fresh namespace for one invocation."""

    namespace = _IsolatedNamespace(base_globals, owned_names)
    for path in part_paths:
        absolute = os.path.abspath(path)
        try:
            code = compile(_read(absolute), absolute, "exec")
        except SyntaxError as exc:
            raise RuntimeError(
                f"Failed to compile experiment part {absolute!r}: {exc}"
            ) from exc
        try:
            exec(code, namespace)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to execute experiment part {absolute!r}: {exc}"
            ) from exc
    namespace.finish_loading()
    if initial:
        for key, value in initial.items():
            namespace[key] = value

    try:
        for name in entry_calls:
            namespace[name]()
    except _ReturnSignal as signal:
        return signal.value
    return None
