"""Backward-compatible entry point for the SMELT-ISR experiment package."""

from __future__ import annotations

import sys as _sys
import types as _types

import exp as _exp


class _CompatModule(_types.ModuleType):
    """Expose the package namespace while keeping live attribute updates."""

    def __getattr__(self, name):
        try:
            return getattr(_exp, name)
        except AttributeError:
            raise AttributeError(
                f"module {self.__name__!r} has no attribute {name!r}"
            ) from None

    def __setattr__(self, name, value):
        if name.startswith("__") and name.endswith("__"):
            super().__setattr__(name, value)
        else:
            setattr(_exp, name, value)

    def __delattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            super().__delattr__(name)
        else:
            delattr(_exp, name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(_exp)))


_sys.modules[__name__].__class__ = _CompatModule


if __name__ == "__main__":
    if "--self-test" in _sys.argv:
        _exp.run_self_test()
    else:
        _exp.main()
