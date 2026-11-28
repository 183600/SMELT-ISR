"""对比实验与消融方法学增强的回归测试。

覆盖：

- DerSimonian-Laird 随机效应元分析（cross-experiment pooling）：
  汇总估计、τ²、I²、Cochran's Q、预测区间；
- ``meta_analyze_contrasts`` 从 ``--compare`` 对比记录构建元分析
  输入，剔除混杂/缺标准误的对比并记录在 ``excluded`` 里；
- ``--compare`` 接入元分析后，报告写入 ``meta_analysis`` 字段并打印
  ``[meta]`` 小节，默认关闭不改变既有输出。

Run with: python -m unittest discover -s tests -v
"""

from _core_support import *

import contextlib
import io
import json
import tempfile
from pathlib import Path

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


def _meta_rows():
    """三个模式 × 三个种子，能产出多个可汇总的两两对比。"""
    rows = []
    for mode, offset in (("local_base", 0.0), ("A", -0.30), ("B", -0.15)):
        for seed in (41, 42, 43):
            rows.append(
                {
                    "run": f"{mode}{seed}",
                    "mode": mode,
                    "seed": seed,
                    "val_loss": 3.0 + offset + (seed % 3) * 0.05,
                    "protocol": {
                        "comparison_family": "neural",
                        "data_sha256": "sha-abc",
                    },
                    "protocol_recorded_keys": [],
                }
            )
    return rows


class MetaAnalysisTests(unittest.TestCase):
    def test_der_simonian_laird_pooling(self):
        effects = [-0.30, -0.20, -0.40]
        variances = [0.01, 0.02, 0.015]
        result = experiment.meta_analyze(effects, variances)
        self.assertEqual(result["schema"], "smelt-isr/meta-analysis/v1")
        self.assertEqual(result["method"], "derSimonian-Laird random effects")
        self.assertEqual(result["k"], 3)
        # 随机效应汇总应落在各效应之间，且为负（改善方向，val_loss 越低越好）。
        estimate = result["random_dl"]["estimate"]
        self.assertIsInstance(estimate, float)
        self.assertLess(estimate, 0.0)
        self.assertGreaterEqual(result["heterogeneity"]["tau_squared"], 0.0)
        self.assertGreaterEqual(result["heterogeneity"]["i_squared"], 0.0)
        self.assertLessEqual(result["heterogeneity"]["i_squared"], 1.0)
        self.assertIsInstance(result["prediction_interval"]["low"], float)
        self.assertIsInstance(result["prediction_interval"]["high"], float)
        self.assertLess(result["prediction_interval"]["low"], estimate)
        self.assertGreater(result["prediction_interval"]["high"], estimate)

    def test_meta_analyze_rejects_insufficient_studies(self):
        with self.assertRaises(ValueError):
            experiment.meta_analyze([-0.3], [0.01])

    def test_meta_analyze_rejects_nonpositive_variance(self):
        with self.assertRaises(ValueError):
            experiment.meta_analyze([-0.3, -0.2], [0.01, 0.0])

    def test_meta_analyze_contrasts_builds_from_compare_records(self):
        contrasts = [
            {
                "baseline": "local_base",
                "variant": "A",
                "delta": -0.30,
                "se": 0.1,
                "confounded": False,
                "n_pairs": 3,
            },
            {
                "baseline": "local_base",
                "variant": "B",
                "delta": -0.15,
                "se": 0.1,
                "confounded": False,
                "n_pairs": 3,
            },
            {
                "baseline": "local_base",
                "variant": "A",
                "delta": -0.30,
                "se": 0.1,
                "confounded": True,
                "n_pairs": 3,
            },
            {
                "baseline": "local_base",
                "variant": "C",
                "se": 0.1,
                "confounded": False,
                "n_pairs": 3,
            },  # 缺 delta，应被剔除
        ]
        result = experiment.meta_analyze_contrasts(contrasts)
        self.assertEqual(result["k"], 2)
        self.assertEqual(len(result["excluded"]), 2)
        excluded_reasons = {e["reason"] for e in result["excluded"]}
        self.assertIn("confounded_contrast", excluded_reasons)
        self.assertIn("missing_effect", excluded_reasons)
        self.assertLess(result["random_dl"]["estimate"], 0.0)

    def test_meta_analyze_contrasts_requires_min_two_studies(self):
        contrasts = [
            {
                "baseline": "local_base",
                "variant": "A",
                "delta": -0.30,
                "se": 0.1,
                "confounded": False,
            }
        ]
        with self.assertRaises(ValueError):
            experiment.meta_analyze_contrasts(contrasts)


class CompareMetaAnalysisIntegrationTests(unittest.TestCase):
    def _run_report(self, meta_analysis):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report = experiment.print_comparison_analysis(
                _meta_rows(), factors=["mode"], metric="val_loss",
                meta_analysis=meta_analysis,
            )
        return report, output.getvalue()

    def test_meta_analysis_default_off(self):
        report, _ = self._run_report(False)
        self.assertIsNone(report.get("meta_analysis"))

    def test_meta_analysis_on_writes_field(self):
        report, rendered = self._run_report(True)
        meta = report.get("meta_analysis")
        self.assertIsNotNone(meta)
        self.assertIsInstance(meta, dict)
        self.assertEqual(meta.get("schema"), "smelt-isr/meta-analysis/v1")
        if meta.get("available", True):
            self.assertIsInstance(meta["random_dl"]["estimate"], float)
            self.assertIn("heterogeneity", meta)
            self.assertIn("prediction_interval", meta)
        # 终端应打印 [meta] 小节。
        self.assertIn("[meta]", rendered)

    def test_json_report_includes_meta_analysis(self):
        report, _ = self._run_report(True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "comparison_analysis.json"
            experiment.write_comparison_report_file(path, report)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("meta_analysis", payload)
