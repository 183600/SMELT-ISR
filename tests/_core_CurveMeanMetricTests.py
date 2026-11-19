from _core_support import *
from _core_support import _load_experiment, _tiny_config

class CurveMeanMetricTests(unittest.TestCase):
    """RQ1 收敛速度的曲线时均指标回归测试。"""

    def _run(
        self,
        root,
        name,
        points,
        *,
        seed=1,
        mode="A",
        failed=False,
    ):
        run_dir = Path(root) / name
        run_dir.mkdir()
        with (run_dir / "metrics.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for step, loss in points:
                handle.write(
                    json.dumps({
                        "step": step,
                        "val_loss": loss,
                    })
                    + "\n"
                )
        row = {
            "run": name,
            "path": str(run_dir),
            "mode": mode,
            "seed": seed,
        }
        if failed:
            row["numerical_status"] = "failed"
        return row

    def test_uses_common_window_and_trapezoid_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                self._run(
                    tmp,
                    "a",
                    [
                        (10, 4.0),
                        (20, 2.0),
                        (25, 3.0),
                        (30, 1.0),
                    ],
                ),
                self._run(
                    tmp,
                    "b",
                    [(20, 3.0), (30, 2.0)],
                    mode="smelt",
                ),
            ]
            report = experiment.annotate_curve_metrics(rows)
        self.assertTrue(report["ok"])
        self.assertEqual(report["start_step"], 20.0)
        self.assertEqual(report["end_step"], 30.0)
        self.assertEqual(report["end_rule"], "min_last_validation_step")
        self.assertTrue(report["end_data_dependent"])
        self.assertAlmostEqual(
            rows[0][experiment._CURVE_MEAN_METRIC], 2.25
        )
        self.assertAlmostEqual(
            rows[1][experiment._CURVE_MEAN_METRIC], 2.5
        )
        self.assertEqual(report["runs"][0]["trapezoid_segments"], 2)
        json.dumps(report, allow_nan=False)

    def test_preregistered_end_interpolates_without_extrapolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                self._run(
                    tmp,
                    "a",
                    [(10, 4.0), (20, 2.0), (30, 1.0)],
                ),
                self._run(
                    tmp,
                    "b",
                    [(20, 3.0), (30, 2.0)],
                ),
            ]
            report = experiment.annotate_curve_metrics(
                rows, target_step=25
            )
            values = [
                row[experiment._CURVE_MEAN_METRIC]
                for row in rows
            ]
            beyond = experiment.annotate_curve_metrics(
                rows, target_step=35
            )
        self.assertTrue(report["ok"])
        self.assertEqual(
            report["end_rule"], "preregistered_fixed_step"
        )
        self.assertFalse(report["end_data_dependent"])
        self.assertAlmostEqual(values[0], 1.75)
        self.assertAlmostEqual(values[1], 2.75)
        self.assertFalse(beyond["ok"])
        self.assertEqual(len(beyond["excluded_runs"]), 2)
        # 超出曲线范围的共同终点不能靠外推伪造指标。
        for row in rows:
            self.assertNotIn(
                experiment._CURVE_MEAN_METRIC, row
            )

    def test_failed_and_short_runs_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                self._run(
                    tmp,
                    "a",
                    [(10, 4.0), (20, 2.0), (30, 1.0)],
                ),
                self._run(
                    tmp,
                    "b",
                    [(10, 5.0), (20, 3.0), (30, 2.0)],
                    failed=True,
                ),
                self._run(
                    tmp,
                    "c",
                    [(10, 5.0), (15, 4.0)],
                ),
            ]
            report = experiment.annotate_curve_metrics(rows)
        self.assertTrue(report["ok"])
        self.assertEqual(report["start_step"], 10.0)
        self.assertEqual(report["end_step"], 15.0)
        self.assertAlmostEqual(
            rows[0][experiment._CURVE_MEAN_METRIC], 3.5
        )
        self.assertAlmostEqual(
            rows[2][experiment._CURVE_MEAN_METRIC], 4.5
        )
        self.assertNotIn(
            experiment._CURVE_MEAN_METRIC, rows[1]
        )
        self.assertEqual(
            [item["run"] for item in report["excluded_runs"]],
            ["b"],
        )

    def test_huge_finite_losses_do_not_overflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                self._run(
                    tmp,
                    "a",
                    [(10, 1.0e308), (20, 1.0e308)],
                )
            ]
            report = experiment.annotate_curve_metrics(rows)
        self.assertTrue(report["ok"])
        self.assertTrue(
            math.isfinite(rows[0][experiment._CURVE_MEAN_METRIC])
        )
        json.dumps(report, allow_nan=False)

    def test_matched_flops_curve_skips_overflowed_cumulative(self):
        # 有限但极大的前向 FLOPs 乘上训练倍率/步数后可能溢出
        # 为 inf；inf 不能成为曲线横坐标，否则共同计算量、
        # 倍率会变成 inf/NaN 并写出非法 JSON。
        with tempfile.TemporaryDirectory() as tmp:
            huge = self._run(
                tmp, "huge", [(10, 1.0), (20, 0.5)]
            )
            huge["flops_per_forward"] = 1.0e308
            normal = self._run(
                tmp, "normal", [(10, 1.2), (20, 0.9)]
            )
            normal["flops_per_forward"] = 1.0e3
            rows = [huge, normal]
            report = experiment.annotate_compute_matched_metrics(
                rows
            )
        self.assertTrue(report["ok"])
        self.assertEqual(
            [item["run"] for item in report["runs"]],
            ["normal"],
        )
        self.assertNotIn(
            experiment._COMPUTE_MATCHED_METRIC, huge
        )
        self.assertTrue(
            any(
                item["run"] == "huge"
                for item in report["excluded_runs"]
            )
        )
        json.dumps(report, allow_nan=False)

    def test_common_step_rejects_out_of_range_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                self._run(
                    tmp,
                    "a",
                    [(10, 4.0), (20, 2.0), (30, 1.0)],
                )
            ]
            report = experiment.annotate_common_step_metrics(
                rows, target_step=10 ** 400
            )
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "有限正数" in advisory
                for advisory in report["advisories"]
            )
        )
        self.assertNotIn(
            experiment._COMMON_STEP_METRIC, rows[0]
        )

    def test_trapezoid_mean_rejects_unrepresentable_window(self):
        # 两端各自有限，但差值可溢出为 inf；此时不能把
        # width/inf=0 的项求和成看似合法的 0.0 损失。
        result = experiment._curve_trapezoid_mean(
            [(-1.0e308, 1.0), (0.0, 2.0), (1.0e308, 3.0)],
            -1.0e308,
            1.0e308,
        )
        self.assertIsNone(result)

    def test_comparison_report_exposes_curve_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                self._run(
                    tmp,
                    "a",
                    [(100, 5.0), (200, 4.0), (300, 3.5)],
                    seed=1,
                    mode="smelt",
                ),
                self._run(
                    tmp,
                    "b",
                    [(100, 5.2), (200, 4.1), (300, 3.6)],
                    seed=2,
                    mode="A",
                ),
            ]
            for row in rows:
                row["val_loss"] = 3.5
                row["selected_model"] = "best_val"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                report = experiment.print_comparison_analysis(
                    rows,
                    metric=experiment._CURVE_MEAN_METRIC,
                    common_step=250,
                )
        self.assertIn("[convergence]", output.getvalue())
        self.assertTrue(report["curve_mean"]["ok"])
        self.assertEqual(len(report["curve_mean"]["runs"]), 2)
        json.dumps(report, allow_nan=False)

    def test_metric_is_guardrailed_as_derived_analysis(self):
        self.assertIn(
            experiment._CURVE_MEAN_METRIC,
            experiment.COMPARISON_METRIC_KEYS,
        )
        self.assertNotIn(
            experiment._CURVE_MEAN_METRIC,
            experiment._SELECTION_METRIC_KEYS,
        )
        self.assertNotIn(
            experiment._CURVE_MEAN_METRIC,
            experiment._BASELINE_TUNING_METRIC_KEYS,
        )
        self.assertFalse(
            experiment.metric_higher_is_better(
                experiment._CURVE_MEAN_METRIC
            )
        )
        self.assertEqual(
            experiment._TRIVIAL_REFERENCE_FIELDS[
                experiment._CURVE_MEAN_METRIC
            ],
            "unigram_loss",
        )
        self.assertIn(
            "convergence",
            experiment.hypothesis_metric_mentions("收敛速度更快"),
        )
        self.assertIn(
            experiment._CURVE_MEAN_METRIC,
            experiment._hypothesis_supported_metrics(
                "convergence"
            ),
        )
        self.assertIn(
            experiment._CURVE_MEAN_METRIC,
            experiment._hypothesis_supported_metrics("loss"),
        )
        report = experiment.selection_bias_report(
            [{"run": "a", "mode": "A", "seed": 1}],
            experiment._CURVE_MEAN_METRIC,
        )
        self.assertFalse(report["applicable"])
