"""Fast regression tests for the methodological fixes.

Run with: python -m unittest discover -s tests -v
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

try:
    import torch
except ImportError as error:  # pragma: no cover
    raise unittest.SkipTest(
        f"torch is required for these tests: {error}"
    )


def _load_experiment():
    path = Path(__file__).resolve().parents[1] / "experiment.py"
    spec = importlib.util.spec_from_file_location(
        "smelt_experiment", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["smelt_experiment"] = module
    spec.loader.exec_module(module)
    return module


experiment = _load_experiment()


def _tiny_config(**overrides):
    values = dict(
        seq_len=16,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        n_layers=4,
        n_experts=4,
        expert_hidden=16,
        top_k=2,
        k1=2,
        k2=2,
        mode="smelt",
    )
    values.update(overrides)
    return experiment.Config(**values)


