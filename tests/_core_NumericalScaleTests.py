from _core_support import *
from _core_support import _load_experiment, _tiny_config

class NumericalScaleTests(unittest.TestCase):
    def test_aggregate_interval_omits_unrepresentable_bounds(self):
        # 合法但极大的有限均值/标准差会让 t 区间端点越过 float
        # 上限；报告必须省略该区间，而不是写入 JSON 无法表示的
        # ±inf。
        rows = [
            {
                "run": "r1",
                "mode": "A",
                "seed": 1,
                "val_loss": 1.0e308,
                "n_params": 1.0e308,
            },
            {
                "run": "r2",
                "mode": "A",
                "seed": 2,
                "val_loss": 1.0,
                "n_params": 1.0,
            },
        ]
        aggregate = experiment.aggregate_runs(rows, ["mode"])[0]
        self.assertNotIn("val_loss_ci95_low", aggregate)
        self.assertNotIn("val_loss_ci95_high", aggregate)
        self.assertIsNone(aggregate["ci95"]["val_loss"])
        self.assertNotIn("ci95_low", aggregate["costs"]["n_params"])
        self.assertNotIn("ci95_high", aggregate["costs"]["n_params"])
        json.dumps(aggregate, allow_nan=False)

    def test_cost_ratio_overflow_is_not_written(self):
        # 两个成本均值都有限，但真实倍率越过 float 上限：报告
        # 不能写 inf，同时权衡判定仍应把该字段视为材料成本。
        def cell(mean):
            return {
                "costs": {
                    "n_params": {
                        "mean": mean,
                        "std": 1.0,
                        "n": 3,
                    }
                }
            }

        cost, verdict = experiment._cost_comparison(
            cell(1.0e-308),
            cell(1.0e308),
            "improvement",
            True,
        )
        self.assertNotIn("n_params", cost["ratios"])
        self.assertEqual(
            cost["ratio_overflow_fields"], ["n_params"]
        )
        self.assertIn("n_params", cost["material_cost_fields"])
        self.assertEqual(verdict, "gain_with_cost")
        json.dumps(cost, allow_nan=False)
        # 溢出倍率在判定函数里也必须等价于「远高于基线」，且
        # 回归 + 成本溢出仍是严格劣于基线。
        self.assertEqual(
            experiment._cost_verdict_from_ratios(
                {},
                "regression",
                True,
                ratio_overflow_fields=["n_params"],
            ),
            "dominated_regression",
        )

    def test_aggregate_cost_cv_omits_unrepresentable_ratio(self):
        # 成本均值接近 0、离散度极大时变异系数越过 float 上限；
        # 聚合记录不能写 inf。
        rows = [
            {
                "run": f"r{seed}",
                "mode": "A",
                "seed": seed,
                "n_params": value,
            }
            for seed, value in enumerate(
                (1.0e-308, 1.0e308, -1.0e308), 1
            )
        ]
        aggregate = experiment.aggregate_runs(rows, ["mode"])[0]
        self.assertIsNone(
            aggregate["costs"]["n_params"][
                "coefficient_of_variation"
            ]
        )
        json.dumps(aggregate, allow_nan=False)

    def test_contrast_intervals_omit_unrepresentable_bounds(self):
        test = {
            "effect": -1.0e308,
            "se": 1.0e308,
            "df": 2.0,
        }
        self.assertEqual(experiment._contrast_interval(test), {})
        item = dict(
            test,
            p_value=0.01,
            significant=True,
        )
        experiment._attach_family_adjusted_intervals([item])
        self.assertNotIn("ci_family_adjusted_low", item)
        self.assertNotIn("ci_family_adjusted_high", item)
        json.dumps(item, allow_nan=False)

    def test_compare_cells_filters_bad_std_and_samples(self):
        cells = [
            {
                "label": "A",
                "keys": {"mode": "A"},
                "samples": {"val_loss": []},
                "val_loss_mean": 1.0,
                "val_loss_std": 1.0e308,
                "n_runs": 1,
                "effective_seed_count": 1,
                "costs": {},
            },
            {
                "label": "B",
                "keys": {"mode": "B"},
                # 损坏记录：samples 不是 dict，std 不是有限值。
                "samples": [],
                "val_loss_mean": 2.0,
                "val_loss_std": float("inf"),
                "n_runs": 1,
                "effective_seed_count": 1,
                "costs": {},
            },
        ]
        contrasts = experiment.compare_cells_to_baseline(
            cells, "A", "val_loss"
        )
        self.assertEqual(len(contrasts), 1)
        contrast = contrasts[0]
        self.assertEqual(contrast["std_baseline"], 1.0e308)
        self.assertIsNone(contrast["std_variant"])
        self.assertEqual(contrast["n_baseline"], 0)
        self.assertEqual(contrast["n_variant"], 0)
        json.dumps(contrast, allow_nan=False)

    def test_attrition_bounds_omit_overflowing_shift_and_sign(self):
        cell = {
            "label": "a",
            "n_runs": 1,
            "samples": {"val_loss": [{"value": 1.0e308}]},
            "numerical_failures": [],
        }
        bounds = experiment._attrition_bounds(
            {"a": cell},
            {"a": 1.0},
            "val_loss",
            -1.0e308,
        )
        self.assertEqual(bounds["verdict"], "complete_cases")
        self.assertIsNone(bounds["max_abs_shift_from_observed"])
        # 观测效应与界方向相反时不能声称方向稳健；修复前
        # complete_cases 分支无条件写 True。
        self.assertFalse(bounds["sign_robust"])
        json.dumps(bounds, allow_nan=False)

        zero = experiment._attrition_bounds(
            {
                "a": dict(
                    cell,
                    samples={"val_loss": [{"value": 0.0}]},
                )
            },
            {"a": 1.0},
            "val_loss",
            0.0,
        )
        self.assertFalse(zero["sign_robust"])

    def test_attrition_bounds_cancelling_terms_do_not_overflow(self):
        def cell(label, value):
            return {
                "label": label,
                "n_runs": 1,
                "samples": {"val_loss": [{"value": value}]},
                "numerical_failures": [],
            }

        bounds = experiment._attrition_bounds(
            {
                "a": cell("a", 1.0e308),
                "b": cell("b", 1.0e308),
                "c": cell("c", 1.0e308),
            },
            {"a": 1.0, "b": 1.0, "c": -1.0},
            "val_loss",
            1.0e308,
        )
        self.assertEqual(bounds["verdict"], "complete_cases")
        self.assertEqual(bounds["effect_min"], 1.0e308)
        self.assertEqual(bounds["effect_max"], 1.0e308)
        self.assertTrue(bounds["sign_robust"])
        json.dumps(bounds, allow_nan=False)

    def test_curve_interpolation_avoids_endpoint_difference_overflow(self):
        value, method = experiment._interpolate_curve(
            [(1.0, -1.0e308), (2.0, 1.0e308)],
            1.5,
        )
        self.assertEqual(value, 0.0)
        self.assertEqual(method, "linear_between_evals")

        value, method = experiment._interpolate_curve(
            [(1.0, 1.0e308), (2.0, 1.0e308)],
            1.5,
        )
        self.assertEqual(value, 1.0e308)
        self.assertEqual(method, "linear_between_evals")
        json.dumps(value, allow_nan=False)

    def test_contrast_delta_overflow_is_treated_as_missing(self):
        def cell(mode, mean):
            return {
                "label": mode,
                "keys": {"mode": mode},
                "samples": {"val_loss": []},
                "val_loss_mean": mean,
                "val_loss_std": 1.0,
                "n_runs": 0,
                "effective_seed_count": 0,
                "costs": {},
            }

        cells = [cell("A", 1.0e308), cell("B", -1.0e308)]
        contrast = experiment.factor_contrasts(
            cells, ["mode"], "val_loss"
        )[0]
        self.assertIsNone(contrast["delta"])
        self.assertEqual(contrast["verdict"], "missing_metric")
        json.dumps(contrast, allow_nan=False)

        contrast = experiment.compare_cells_to_baseline(
            cells, "A", "val_loss"
        )[0]
        self.assertIsNone(contrast["delta"])
        self.assertEqual(contrast["verdict"], "missing_metric")
        json.dumps(contrast, allow_nan=False)

    def test_contrast_noise_ratio_overflow_is_treated_as_missing(self):
        def cell(mode, mean, std):
            return {
                "label": mode,
                "keys": {"mode": mode},
                "samples": {"val_loss": []},
                "val_loss_mean": mean,
                "val_loss_std": std,
                "n_runs": 0,
                "effective_seed_count": 0,
                "costs": {},
            }

        contrast = experiment.factor_contrasts(
            [
                cell("A", 0.0, 1.0e-154),
                cell("B", 1.0e308, 1.0e-154),
            ],
            ["mode"],
            "val_loss",
        )[0]
        self.assertEqual(contrast["delta"], 1.0e308)
        self.assertIsNone(contrast["delta_over_noise"])
        self.assertIsNone(contrast["hedges_g"])
        json.dumps(contrast, allow_nan=False)

        reference = experiment.compare_cells_to_baseline(
            [cell("A", 0.0, 1.0e-154), cell("B", 1.0e308, 1.0e-154)],
            "A",
            "val_loss",
        )[0]
        self.assertEqual(reference["delta"], 1.0e308)
        self.assertIsNone(reference["delta_over_noise"])
        self.assertIsNone(reference["hedges_g"])
        json.dumps(reference, allow_nan=False)

    def test_window_analysis_cross_seed_mean_does_not_overflow(self):
        def payload(loss_sum):
            return {
                "split": "val",
                "data_sha256": "d",
                "split_sha256": "s",
                "seq_len": 4,
                "batch_size": 1,
                "batches": 2,
                "windows": [
                    {
                        "tokens": 1,
                        "loss_sum": loss_sum,
                        "correct": 0,
                    },
                    {
                        "tokens": 1,
                        "loss_sum": loss_sum,
                        "correct": 0,
                    },
                ],
            }

        def cell(losses):
            return {
                "window_payloads": [
                    {
                        "seed": seed,
                        "payload": payload(loss),
                    }
                    for seed, loss in enumerate(losses, 1)
                ],
            }

        result = experiment._cell_window_paired_analysis(
            cell([0.0, 0.0, 0.0]),
            cell([1.0e308, 1.0e308, 1.0e308]),
        )
        self.assertTrue(result["available"])
        self.assertTrue(
            math.isfinite(result["mean_loss_diff"])
        )
        self.assertEqual(result["mean_loss_diff"], 1.0e308)
        self.assertEqual(result["mean_loss_diff_spread"], 0.0)

        spread = experiment._cell_window_paired_analysis(
            cell([0.0, 1.0e308, -1.0e308]),
            cell([1.0e308, 0.0, 0.0]),
        )
        self.assertTrue(spread["available"])
        self.assertIsNone(spread["mean_loss_diff_spread"])
        json.dumps(spread, allow_nan=False)

    def test_window_payload_scalar_is_not_iterated(self):
        # 损坏的旧聚合记录可能把 window_payloads 写成标量；
        # 逐窗口分析必须按「无逐窗口记录」处理，而不是抛
        # TypeError 中断整份对比报告。
        result = experiment._cell_window_paired_analysis(
            {"window_payloads": 1.0e150},
            {"window_payloads": []},
        )
        self.assertFalse(result["available"])
        json.dumps(result, allow_nan=False)

    def test_window_overlap_huge_int_is_not_formatted(self):
        def payload(overlap):
            return {
                "split": "val",
                "data_sha256": "deadbeef",
                "split_sha256": "cafebabe",
                "seq_len": 4,
                "batch_size": 1,
                "batches": 2,
                "overlap_fraction": overlap,
                "windows": [
                    {"tokens": 1, "loss_sum": 1.0, "correct": 0},
                    {"tokens": 1, "loss_sum": 1.0, "correct": 0},
                ],
            }

        def cell(losses, overlap=0.0):
            return {
                "window_payloads": [
                    {
                        "seed": seed,
                        "payload": payload(overlap),
                    }
                    for seed, _ in enumerate(losses, 1)
                ]
            }

        result = experiment._cell_window_paired_analysis(
            cell([0.0, 0.0], 10 ** 400),
            cell([1.0, 1.0]),
        )
        self.assertTrue(result["available"])
        json.dumps(result, allow_nan=False)

    def test_linear_effect_no_common_seed_sum_does_not_overflow(self):
        cells = {
            "a": {
                "samples": {
                    "val_loss": [{"seed": 1, "value": 1.0}]
                },
                "val_loss_mean": 1.0e308,
            },
            "b": {
                "samples": {
                    "val_loss": [{"seed": 2, "value": 1.0}]
                },
                "val_loss_mean": -1.0e308,
            },
        }
        result = experiment._paired_linear_effect(
            cells,
            {"a": 1.0, "b": -1.0},
            "val_loss",
        )
        self.assertIsNone(result["effect"])
        self.assertEqual(result["test"], "invalid_values")
        json.dumps(result, allow_nan=False)

    def test_linear_effect_common_seeds_does_not_overflow(self):
        # 交互项系数 (+1,-1,-1,+1) 下，四个 1e308/-1e308 的
        # 真实线性组合为 0；顺序累加会在第二步溢出成 inf，
        # 必须按尺度缩放后保留真实相消。
        def cell(value):
            return {
                "samples": {
                    "val_loss": [
                        {"seed": seed, "value": value}
                        for seed in (1, 2, 3)
                    ],
                },
                "val_loss_mean": value,
            }

        cells = {
            "L_off": cell(1.0e308),
            "L_on": cell(-1.0e308),
            "M_off": cell(1.0e308),
            "M_on": cell(-1.0e308),
        }
        result = experiment._paired_linear_effect(
            cells,
            {
                "L_off": 1.0,
                "L_on": -1.0,
                "M_off": -1.0,
                "M_on": 1.0,
            },
            "val_loss",
        )
        self.assertEqual(result["effect"], 0.0)
        self.assertEqual(result["test"], "paired_t")
        self.assertEqual(
            result["paired_differences"], [0.0, 0.0, 0.0]
        )
        json.dumps(result, allow_nan=False)

    def test_factorial_reference_mean_does_not_overflow(self):
        def cell(mode, balance, mean):
            return {
                "label": f"{mode}/{balance}",
                "keys": {"mode": mode, "balance_coef": balance},
                "samples": {
                    "val_loss": [
                        {"seed": 1, "value": 1.0}
                    ]
                },
                "val_loss_mean": mean,
                "val_loss_std": 0.0,
                "n_runs": 1,
                "effective_seed_count": 1,
                "costs": {},
                "window_payloads": [],
            }

        cells = [
            cell("local_base", 0.0, 1.0e308),
            cell("local_base", 0.01, 1.0e308),
            cell("A", 0.0, -1.0e308),
            cell("A", 0.01, -1.0e308),
        ]
        report = experiment.factorial_effects(
            cells, ["mode", "balance_coef"], "val_loss"
        )
        self.assertTrue(report["effects"])
        for effect in report["effects"]:
            self.assertTrue(
                math.isfinite(effect["mean_baseline"]),
                effect["effect_name"],
            )
        json.dumps(report, allow_nan=False)

    def test_ablation_parity_ratio_overflow_is_not_written(self):
        rows = [
            {
                "run": "base",
                "mode": "local_base",
                "balance_coef": 0.0,
                "seed": 1,
                "flops_per_forward": 1.0e-308,
                "n_params": 1.0,
            },
            {
                "run": "A",
                "mode": "A",
                "balance_coef": 0.0,
                "seed": 1,
                "flops_per_forward": 1.0e150,
                "n_params": 1.0,
            },
        ]
        report = experiment.ablation_coverage(
            rows, ["mode", "balance_coef"]
        )
        entry = next(
            item
            for item in report["compute_parity"]
            if item["delayed_mode"] == "A"
        )
        self.assertIsNone(entry["ratio"])
        self.assertFalse(entry["matched"])
        json.dumps(report, allow_nan=False)

    def test_compare_runs_sanitizes_nonfinite_config_values(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(
                json.dumps({
                    "scale_guard": {
                        "tokens_seen": float("inf"),
                    },
                    "efficiency": {
                        "latency_ms_per_forward": float("inf"),
                    },
                    "n_params": 10 ** 400,
                }),
                encoding="utf-8",
            )
            (run_dir / "metrics.jsonl").write_text(
                json.dumps({
                    "step": "final",
                    "val_loss": float("inf"),
                }),
                encoding="utf-8",
            )
            rows = experiment.compare_runs([str(run_dir)])
        self.assertIsNone(
            rows[0]["scale_guard"]["tokens_seen"]
        )
        self.assertIsNone(rows[0]["latency_ms_per_forward"])
        self.assertIsNone(rows[0]["val_loss"])
        # 超大整数仍保持精确，不能因清洗被截断。
        self.assertEqual(rows[0]["n_params"], 10 ** 400)
        json.dumps(rows, allow_nan=False)

    def test_curve_analysis_omits_overflowing_relative_fields(self):
        records = []
        val_losses = [1.0e-308, 2.0e-308, 1.0e308]
        for index, val_loss in enumerate(val_losses, 1):
            records.append({
                "step": index * 2 - 1,
                "train_loss": -1.0e308,
                "grad_norm": 1.0,
                "lr": 1.0e-3,
            })
            records.append({
                "step": index * 2,
                "val_loss": val_loss,
            })
        analysis = experiment.analyze_training_curve(records)
        # 最优验证损失极小、最终验证损失极大：回升比例与
        # train-val 差值都会越过 float 上限，必须是 None。
        self.assertIsNone(analysis["val_regression_from_best"])
        self.assertIsNone(analysis["train_val_gap_last"])
        self.assertTrue(
            math.isfinite(analysis["last_train_loss"])
        )
        json.dumps(analysis, allow_nan=False)

        # 损坏的 min_delta 不应让 resume 的减法崩溃。
        self.assertTrue(
            experiment.selection_improved(
                1.0, None, False, None
            )
        )

    def test_curve_analysis_does_not_overflow_finite_losses(self):
        # 两次各自有限的 1e308 训练损失在求验证点均值时不能
        # 溢出成 inf；否则派生 gap/状态与落盘 JSON 都会失真。
        records = []
        for index in range(1, 6):
            for offset in (0, 1):
                records.append({
                    "step": index * 2 - 1 + offset,
                    "train_loss": 1.0e308,
                    "grad_norm": 1.0e-300,
                    "lr": 1.0e-3,
                })
            records.append({
                "step": index * 2,
                "val_loss": 5.0 + index * 0.01,
            })
        analysis = experiment.analyze_training_curve(
            records, baseline_loss=5.5
        )
        self.assertTrue(
            math.isfinite(analysis["last_train_loss"])
        )
        json.dumps(analysis, allow_nan=False)

    def test_curve_analysis_gradient_ratio_overflow_is_none(self):
        # 中位数极小、近期最大值极大时尖峰倍率会溢出；尖峰
        # 警告仍应触发，但字段必须是 None 而不是 inf。
        records = []
        for step in range(1, 41):
            records.append({
                "step": step,
                "train_loss": 5.0,
                "grad_norm": 1.0e-300,
                "lr": 1.0e-3,
            })
        records.append({
            "step": 41,
            "train_loss": 5.0,
            "grad_norm": 1.0e308,
            "lr": 1.0e-3,
        })
        for step in range(42, 47):
            records.append({
                "step": step,
                "train_loss": 5.0,
                "grad_norm": 1.0,
                "lr": 1.0e-3,
            })
            records.append({
                "step": step,
                "val_loss": 5.0 - step * 1.0e-4,
            })
        analysis = experiment.analyze_training_curve(records)
        self.assertTrue(analysis["grad_norm_spike_warning"])
        self.assertIsNone(analysis["grad_norm_spike_ratio"])
        json.dumps(analysis, allow_nan=False)

    def test_lag1_autocorrelation_is_scale_stable(self):
        small = torch.tensor(
            [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            dtype=torch.float64,
        )
        large = small * 1.0e300
        expected = experiment._lag1_autocorrelation(small)
        self.assertIsNotNone(expected)
        actual = experiment._lag1_autocorrelation(large)
        self.assertIsNotNone(actual)
        self.assertAlmostEqual(actual, expected, places=12)

        # 常数序列没有可定义的自相关；超大常数不能因均值溢出
        # 被误判成 rho=1。
        self.assertIsNone(
            experiment._lag1_autocorrelation(
                torch.full((8,), 1.0e308, dtype=torch.float64)
            )
        )

        # 极小但非零的尺度同样必须保持自相关：先相乘再开方会
        # 在 1e-400 量级下溢，独立开方再相乘则不会。
        for factor in (1.0e-100, 1.0e-200, 1.0e-300):
            tiny = experiment._lag1_autocorrelation(
                small * factor
            )
            self.assertIsNotNone(tiny)
            self.assertAlmostEqual(tiny, expected, places=12)

    def test_variance_decomposition_is_tiny_scale_stable(self):
        small = {1: [1.0, 2.0], 2: [1.0, 2.0]}
        expected = experiment._window_variance_components(small)
        tiny = experiment._window_variance_components(
            {1: [1.0e-200, 2.0e-200], 2: [1.0e-200, 2.0e-200]}
        )
        self.assertTrue(tiny["available"])
        self.assertAlmostEqual(
            tiny["std_error_of_mean"] / 1.0e-200,
            expected["std_error_of_mean"],
            places=12,
        )
        self.assertEqual(
            tiny["variance_share_window"],
            expected["variance_share_window"],
        )
        self.assertEqual(
            tiny["dominant_source"],
            expected["dominant_source"],
        )
        # 绝对方差小于 float 可表示范围时应明确缺失，而不是
        # 在 std_error 非零的同时报告零方差。
        self.assertIsNone(tiny["variance_of_mean"])
        json.dumps(tiny, allow_nan=False)

    def test_ngram_diagnostics_tie_tolerance_scales_with_value(self):
        report = experiment._ngram_tuning_diagnostics(
            {"order": 1, "alpha": 0.2, "val_loss": 3.0e-13},
            {"order": 2, "alpha": 0.3, "val_loss": 5.0e-13},
            "val_loss",
            [0.1, 0.2, 0.3],
            3,
            2,
            2,
        )
        self.assertFalse(
            any("并列" in message for message in report["advisories"])
        )
        json.dumps(report, allow_nan=False)

    def test_axis_value_equality_scales_with_tiny_values(self):
        self.assertFalse(
            experiment._axis_value_equal(0.0, 1.0e-13)
        )
        self.assertFalse(
            experiment._axis_value_equal(1.0e-13, 2.0e-13)
        )
        self.assertTrue(
            experiment._axis_value_equal(
                1.0e-13, 1.0e-13 * (1.0 + 1.0e-10)
            )
        )
        self.assertTrue(experiment._axis_value_equal(7, 7.0))

    def test_attrition_bounds_scale_with_tiny_effect(self):
        cell = {
            "label": "a",
            "n_runs": 1,
            "samples": {
                "val_loss": [{"value": 1.0e-13}]
            },
            "numerical_failures": [],
        }
        bounds = experiment._attrition_bounds(
            {"a": cell},
            {"a": 1.0},
            "val_loss",
            1.0e-13,
        )
        self.assertEqual(bounds["verdict"], "complete_cases")
        self.assertFalse(bounds["zero_within_bounds"])
        self.assertTrue(bounds["sign_robust"])
        json.dumps(bounds, allow_nan=False)

    def test_leave_one_out_stability_scales_with_effect(self):
        stable = experiment._leave_one_out_effect_stability(
            [1.0e-13, 2.0e-13, 3.0e-13]
        )
        self.assertTrue(stable["leave_one_out_sign_stable"])
        mixed = experiment._leave_one_out_effect_stability(
            [1.0e-13, 2.0e-13, -3.0e-13]
        )
        self.assertFalse(mixed["leave_one_out_sign_stable"])
        json.dumps(mixed, allow_nan=False)

    def test_ngram_one_se_tolerance_scales_with_metric(self):
        # 极小但非零的 loss 差异不能被固定 1e-12 容差当成并列：
        # 否则更简单的候选配置会因为一条恒定的微小劣化被误选。
        point = {
            "order": 2,
            "alpha": 0.5,
            "val_loss": 3.0e-13,
        }
        candidates = [
            point,
            {
                "order": 2,
                "alpha": 0.1,
                "val_loss": 5.0e-13,
            },
        ]
        window_values = {
            (2, 0.5): [3.0e-13] * 4,
            (2, 0.1): [5.0e-13] * 4,
        }
        report = experiment._select_ngram_candidate_one_se(
            point,
            candidates,
            window_values,
            "val_loss",
        )
        self.assertTrue(report["available"])
        self.assertTrue(report["selected_is_point_best"])
        self.assertEqual(
            report["selected"],
            {"order": 2, "alpha": 0.5},
        )
        self.assertEqual(len(report["tied"]), 1)
        json.dumps(report, allow_nan=False)

    def test_ngram_one_se_skips_unrepresentable_paired_se(self):
        point = {
            "order": 1,
            "alpha": 0.1,
            "val_loss": 0.0,
        }
        candidates = [
            point,
            {
                "order": 2,
                "alpha": 0.2,
                "val_loss": 0.0,
            },
        ]
        window_values = {
            (1, 0.1): [0.0, 0.0, 0.0],
            (2, 0.2): [1.7e308, -1.7e308, 1.7e308],
        }
        report = experiment._select_ngram_candidate_one_se(
            point,
            candidates,
            window_values,
            "val_loss",
        )
        self.assertTrue(report["available"])
        self.assertEqual(
            report["selected"],
            {"order": 1, "alpha": 0.1},
        )
        self.assertEqual(
            [row["order"] for row in report["tied"]],
            [1],
        )
        json.dumps(report, allow_nan=False)

    def test_paired_window_bootstrap_is_scale_stable(self):
        small = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]
        large = [value * 1.0e300 for value in small]
        small_ci = experiment._paired_window_bootstrap_ci(small, 7)
        large_ci = experiment._paired_window_bootstrap_ci(large, 7)
        self.assertTrue(small_ci["available"])
        self.assertTrue(large_ci["available"])
        self.assertTrue(math.isfinite(large_ci["low"]))
        self.assertTrue(math.isfinite(large_ci["high"]))
        scale = 1.0e300
        self.assertAlmostEqual(
            large_ci["low"] / scale,
            small_ci["low"],
            places=12,
        )
        self.assertAlmostEqual(
            large_ci["high"] / scale,
            small_ci["high"],
            places=12,
        )
        self.assertAlmostEqual(
            large_ci["lag1_autocorrelation"],
            small_ci["lag1_autocorrelation"],
            places=12,
        )

        # 常数超大差值不能因重采样总和溢出而给出 NaN 区间。
        constant = experiment._paired_window_bootstrap_ci(
            [1.0e308] * 8,
            7,
        )
        self.assertEqual(constant["low"], 1.0e308)
        self.assertEqual(constant["high"], 1.0e308)
        self.assertIsNone(constant["lag1_autocorrelation"])

        # AR(1) 膨胀后端点可能越过 float 上限；此时必须明确
        # 标记区间不可用，不能把 inf 写进报告/JSON。
        overflow = experiment._paired_window_bootstrap_ci(
            [1.7e308 * (1.0 - 0.1 * index) for index in range(6)],
            7,
        )
        self.assertFalse(overflow["available"])
        self.assertEqual(
            overflow["reason"], "unrepresentable_bounds"
        )
        json.dumps(overflow, allow_nan=False)

    def test_hierarchical_window_bootstrap_is_scale_stable(self):
        small = {
            1: [1.0, 2.0, 3.0],
            2: [2.0, 3.0, 4.0],
        }
        large = {
            seed: [value * 1.0e300 for value in values]
            for seed, values in small.items()
        }
        small_ci = experiment._hierarchical_window_bootstrap_ci(small)
        large_ci = experiment._hierarchical_window_bootstrap_ci(large)
        self.assertTrue(small_ci["available"])
        self.assertTrue(large_ci["available"])
        self.assertTrue(math.isfinite(large_ci["low"]))
        self.assertTrue(math.isfinite(large_ci["high"]))
        scale = 1.0e300
        self.assertAlmostEqual(
            large_ci["low"] / scale,
            small_ci["low"],
            places=12,
        )
        self.assertAlmostEqual(
            large_ci["high"] / scale,
            small_ci["high"],
            places=12,
        )

        constant = experiment._hierarchical_window_bootstrap_ci({
            1: [1.0e308, 1.0e308],
            2: [1.0e308, 1.0e308],
        })
        self.assertEqual(constant["low"], 1.0e308)
        self.assertEqual(constant["high"], 1.0e308)

    def test_equivalence_seed_estimate_handles_huge_se(self):
        contrast = {
            "mean_baseline": 1.0,
            "delta": 0.999999,
            "n_pairs": 3,
            "n_baseline": 3,
            "n_variant": 3,
            "test": "paired_t",
        }
        # se 合法但极大：所需种子数不可表示，不能抛出
        # OverflowError 中断整份对比报告。
        experiment._attach_power_equivalence(
            contrast,
            0.999999,
            {"se": 1.0e307, "df": 2.0},
            2.0,
        )
        self.assertEqual(
            contrast["equivalence_blocker"],
            "insufficient_precision",
        )
        self.assertNotIn(
            "equivalence_seeds_to_achieve",
            contrast,
        )

    def test_equivalence_margin_overflow_is_omitted(self):
        contrast = {
            "mean_baseline": 1.0,
            "delta": 1.0,
            "n_pairs": 3,
            "n_baseline": 3,
            "n_variant": 3,
            "test": "paired_t",
        }
        # t 临界值 × 1e308 溢出：不能把 inf 写进所需等价界。
        experiment._attach_power_equivalence(
            contrast,
            1.0,
            {"se": 1.0e308, "df": 2.0},
            0.02,
        )
        self.assertEqual(
            contrast["equivalence_blocker"],
            "insufficient_precision",
        )
        self.assertNotIn("equivalence_margin_needed", contrast)
        self.assertNotIn(
            "equivalence_margin_needed_relative", contrast
        )
        json.dumps(contrast, allow_nan=False)

        family = {
            "se": 1.0e308,
            "df": 2.0,
            "delta": 1.0e308,
            "mean_baseline": 1.0,
            "equivalence_margin_abs": 1.0e308,
        }
        experiment._attach_family_power(family, 3)
        self.assertNotIn(
            "equivalence_margin_needed_bonferroni", family
        )
        json.dumps(family, allow_nan=False)

    def test_paired_effect_size_omits_unrepresentable_ratio(self):
        # 差值标准差可以合法地极小；|Δ|/sd 越过 float 上限时
        # 没有可表示的有限效应量，必须按缺失处理而不是写 inf。
        result = experiment._paired_effect_size(
            1.0e308,
            {
                "test": "paired_t",
                "n_pairs": 3,
                "se": 1.0e-300,
                "df": 2.0,
            },
        )
        self.assertEqual(result, {})
        json.dumps(result, allow_nan=False)

    def test_relative_mde_omits_unrepresentable_ratio(self):
        # 基线均值可以合法地极小：绝对 MDE 有限但相对值越过
        # float 上限时省略该字段，不能把 inf 写进对比记录。
        contrast = {
            "mean_baseline": 1.0e-300,
            "delta": 1.0,
            "n_pairs": 3,
            "n_baseline": 3,
            "n_variant": 3,
            "test": "paired_t",
        }
        experiment._attach_power_equivalence(
            contrast,
            1.0,
            {"se": 1.0e300, "df": 2.0},
            None,
        )
        self.assertIsNotNone(contrast.get("mde_80"))
        self.assertNotIn("mde_80_relative", contrast)
        json.dumps(contrast, allow_nan=False)

    def test_contrast_test_marks_overflowing_differences_missing(self):
        # 两端指标各自有限时，配对差值仍可能溢出成 ±inf；
        # 不能把非有限差值写进报告，也不能让下游脆弱性分析
        # 把它当成有效观测。
        samples_a = [
            {"seed": seed, "run": f"a{seed}", "value": 1.0e308}
            for seed in (1, 2, 3)
        ]
        samples_b = [
            {"seed": seed, "run": f"b{seed}", "value": -1.0e308}
            for seed in (1, 2, 3)
        ]
        result = experiment._contrast_test(samples_a, samples_b)
        self.assertEqual(result["test"], "invalid_values")
        self.assertIsNone(result["paired_differences"])
        self.assertFalse(result["paired_differences_finite"])
        json.dumps(result, allow_nan=False)

    def test_bad_case_report_sanitizes_nonfinite_shares_and_rates(self):
        def row(name, share, wrong, tokens=1, position_tokens=1):
            return {
                "run": name,
                "mode": "A",
                "seed": 1,
                "bad_case_available": True,
                "bad_case_tokens_evaluated": tokens,
                "bad_case_wrong_tokens": wrong,
                "bad_case_error_category_counts": {
                    "ambiguous": wrong,
                },
                "bad_case_error_category_shares": {
                    "ambiguous": share,
                },
                "bad_case_error_rate_by_position": [
                    {
                        "name": "position_0",
                        "tokens": position_tokens,
                        "wrong": 1.0e308,
                    }
                ],
                "bad_case_error_rate_by_train_frequency": [],
                "bad_case_error_frequency_basis": "train_share",
                "bad_case_token_accuracy": float("inf"),
            }

        # 两个合法但极大的错误计数相加后越过 float 上限；
        # 错误率无法表示时必须是 None。非有限占比与准确率也
        # 不能原样进入报告。
        report = experiment.bad_case_comparison_report([
            row("r1", float("inf"), 1.0e308, tokens=1),
            row(
                "r2",
                -1.0e308,
                1.0e308,
                tokens=0,
                position_tokens=0,
            ),
        ])
        self.assertIsNone(report["runs"][0]["token_accuracy"])
        self.assertIsNone(
            report["runs"][0]["error_category_shares"][
                "ambiguous"
            ]
        )
        self.assertIsNone(
            report["error_rate_by_position"][0]["error_rate"]
        )
        json.dumps(report, allow_nan=False)

        # 两个有限但符号相反的极大占比，其极差越过 float
        # 上限：没有可表示的跨运行极差。
        spread = experiment.bad_case_comparison_report([
            row("r1", 1.0e308, 1.0),
            row("r2", -1.0e308, 1.0),
        ])
        self.assertIsNone(
            spread["category_share_ranges"]["ambiguous"]
        )
        json.dumps(spread, allow_nan=False)

    def test_balance_strength_rejects_oversized_int_parameters(self):
        huge = 10**400
        # 协议计数可以写成超出 float 范围的 JSON 整数；不能
        # 在转 float 时崩溃，也不能把 inf 写进报告。
        self.assertIsNone(
            experiment._balance_width_profile({
                "mode": "smelt",
                "top_k": huge,
                "balance_coef": 0.01,
            })
        )
        self.assertIsNone(
            experiment._balance_width_profile({
                "mode": "A",
                "k1": huge,
                "k2": 2,
                "balance_coef": 0.01,
            })
        )
        report = experiment.balance_strength_report([
            {
                "run": "huge",
                "protocol": {
                    "mode": "smelt",
                    "top_k": huge,
                    "balance_coef": 0.01,
                },
            }
        ])
        json.dumps(report, allow_nan=False)

    def test_balance_strength_omits_overflowing_strength_and_ratio(self):
        # coef × width 越过 float 上限：该运行无法参与强度比较，
        # 必须显式记录并跳过，而不是写 inf。
        report = experiment.balance_strength_report([
            {
                "run": "overflow",
                "protocol": {
                    "mode": "smelt",
                    "top_k": 2,
                    "balance_coef": 1.0e308,
                },
            },
            {
                "run": "normal",
                "protocol": {
                    "mode": "smelt",
                    "top_k": 2,
                    "balance_coef": 0.01,
                },
            },
        ])
        self.assertEqual(len(report["profiles"]), 1)
        self.assertEqual(
            report["invalid_profiles"][0]["run"], "overflow"
        )
        json.dumps(report, allow_nan=False)

        # 最高/最低强度之比越过 float 上限：不能写 inf。
        ratio = experiment.balance_strength_report([
            {
                "run": "low",
                "protocol": {
                    "mode": "smelt",
                    "top_k": 2,
                    "balance_coef": 5.0e-324,
                },
            },
            {
                "run": "high",
                "protocol": {
                    "mode": "A",
                    "k1": 1,
                    "k2": 2,
                    "balance_coef": 1.0e308,
                },
            },
        ])
        self.assertTrue(ratio["confounded"])
        self.assertNotIn("strength_ratio", ratio)
        json.dumps(ratio, allow_nan=False)

    def test_multiple_comparison_skips_corrupt_p_values(self):
        contrasts = [
            {"p_value": 0.01, "significant": True},
            {"p_value": 0.04, "significant": True},
            {"p_value": "bad", "significant": True},
            {"p_value": float("inf"), "significant": True},
            {"p_value": float("nan"), "significant": True},
            {"p_value": 10**400, "significant": True},
            {"p_value": -0.01, "significant": True},
            {"p_value": 1.01, "significant": True},
        ]
        experiment._apply_multiple_comparison_correction(contrasts)
        # 只对两个可用的 p 值做家族校正；损坏记录跳过而不是
        # 让整份对比报告抛 ValueError/OverflowError。
        self.assertAlmostEqual(contrasts[0]["p_value_holm"], 0.02)
        self.assertAlmostEqual(contrasts[1]["p_value_holm"], 0.04)
        for contrast in contrasts[2:]:
            self.assertNotIn("p_value_holm", contrast)
            self.assertNotIn("p_value_bh", contrast)
            self.assertTrue(contrast["significant_raw"])
            self.assertFalse(contrast["significant"])
        self.assertEqual(
            experiment._holm_bonferroni(
                [0.01, "bad", float("nan"), -0.01, 1.01, 0.04]
            ),
            [0.02, None, None, None, None, 0.04],
        )
        self.assertEqual(
            experiment._benjamini_hochberg(
                [0.01, "bad", float("inf"), -0.01, 1.01, 0.04]
            ),
            [0.02, None, None, None, None, 0.04],
        )

    def test_environment_identity_falls_back_for_minimal_snapshots(self):
        expected = {
            "schema": experiment._ENVIRONMENT_SCHEMA,
            "python": {"version": None},
            "unknown_field": "expected",
        }
        actual = {
            "schema": experiment._ENVIRONMENT_SCHEMA,
            "python": {"version": None},
            "unknown_field": "actual",
        }
        expected_identity = experiment._environment_identity_payload(
            expected
        )
        actual_identity = experiment._environment_identity_payload(
            actual
        )
        # 没有已知身份字段时不能退化成同一个空对象哈希。
        self.assertNotEqual(expected_identity, actual_identity)
        report = experiment.compare_environment_snapshots(
            expected, actual
        )
        self.assertFalse(report["ok"])

    def test_nan_factor_keys_are_stable_and_ordered(self):
        first = float("nan")
        second = float("nan")
        first_key = experiment._stable_factor_key(first)
        second_key = experiment._stable_factor_key(second)
        self.assertEqual(first_key, second_key)
        self.assertEqual(hash(first_key), hash(second_key))

        values = [
            float("nan"),
            1.0,
            float("nan"),
            float("inf"),
            0.0,
        ]
        ordered = sorted(
            values, key=experiment._factor_sort_rank
        )
        # 相同输入顺序下排序必须确定；两个 NaN 落在同一稳定位置。
        self.assertEqual(
            [experiment._factor_sort_rank(value) for value in ordered],
            sorted(
                experiment._factor_sort_rank(value)
                for value in values
            ),
        )
        self.assertEqual(
            experiment._factor_sort_rank(float("nan")),
            experiment._factor_sort_rank(float("nan")),
        )

    def test_duplicate_chunk_examples_are_deterministically_sorted(self):
        first = bytes(index % 251 for index in range(300))
        second = bytes((index * 7 + 3) % 251 for index in range(300))
        raw = bytearray(
            first + b"|" + second + b"|" + first + b"|" + second
        )
        boundary = len(first) + 1 + len(second) + 1

        class _FakeDataset:
            seq_len = 8
            group_separator = b"|"
            train_excluded_spans = []
            _raw = memoryview(raw)
            _raw_buffer = raw
            split_bounds = {
                "train": (0, boundary),
                "val": (boundary, len(raw)),
                "test": (len(raw), len(raw)),
            }

        report = experiment._content_chunk_duplicate_audit(
            _FakeDataset(), ["train", "val"]
        )
        examples = report["cross_split"]["train_val"]["examples"]
        self.assertGreaterEqual(len(examples), 2)
        digests = []
        for start, _right_start, length in examples:
            digest = hashlib.blake2b(
                bytes(raw[start:start + length]),
                digest_size=16,
            ).digest()
            digests.append(digest)
        self.assertEqual(digests, sorted(digests))

    def test_baseline_provenance_rejects_corrupt_winner(self):
        for winner in ({}, {"lr": "not-a-number"}, ["bad"]):
            error = experiment.baseline_tuning_provenance_error(
                {"mode": "dense", "winner": winner},
                mode="dense",
                lr=1.0e-4,
                steps=100,
                data_sha256="0" * 64,
            )
            self.assertIsInstance(error, str)
            self.assertIn("winner", error)

    def test_near_duplicate_threshold_huge_int_falls_back(self):
        self.assertEqual(
            experiment._near_duplicate_threshold_or_default(
                10 ** 400
            ),
            experiment._NEAR_DUP_DEFAULT_THRESHOLD,
        )

    def test_near_duplicate_plan_rejects_huge_threshold(self):
        with self.assertRaises(ValueError):
            experiment._plan_train_near_duplicate_cleaning(
                bytearray(b"aaa|bbb|aaa|"),
                b"|",
                8,
                {},
                [],
                threshold=10 ** 400,
                shingle_bytes=1,
            )

    def test_byte_dataset_rejects_huge_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.txt"
            path.write_bytes(b"a\nb\nc\nd\ne\nf\n")
            with self.assertRaises(ValueError):
                experiment.ByteDataset(
                    str(path),
                    seq_len=2,
                    near_duplicate_threshold=10 ** 400,
                    device=torch.device("cpu"),
                )

    def test_preregistration_schema_rejects_huge_threshold(self):
        payload = {
            "study_id": "extreme-numbers",
            "frozen_commit": "a" * 40,
            "primary_metric": "val_loss",
            "common_step": 100,
            "min_effect": 0.01,
            "direction": "decrease",
            "baseline_mode": "smelt",
            "comparisons": [["smelt", "A"]],
            "seeds": [42, 43, 44],
            "budget_match": "smelt",
            "steps": 200,
            "data": {
                "sha256": "b" * 64,
                "split_policy": "contiguous",
                "val_fraction": 0.1,
                "test_fraction": 0.1,
                "cleaning": {
                    "near_duplicate_threshold": 10 ** 400,
                },
            },
        }
        error = experiment.preregistration_schema_error(payload)
        self.assertIsInstance(error, str)
        self.assertIn("near_duplicate_threshold", error)

    def test_finite_gap_tolerates_huge_ints(self):
        self.assertIsNone(experiment._finite_gap(10 ** 400, 1.0))
        self.assertIsNone(experiment._finite_gap(1.0, 10 ** 400))
        self.assertEqual(
            experiment._finite_gap(1.5, 0.25), 1.25
        )

    def test_extreme_int_reports_do_not_overflow(self):
        self.assertIsNone(experiment._t_critical(10 ** 400))
        self.assertIsNone(
            experiment._percentile([1.0, 10 ** 400], 0.5)
        )
        self.assertEqual(
            experiment.distribution_shift_violations(
                {"distribution_shift": {"train/val": 10 ** 400}},
                0.1,
            ),
            [],
        )
        rows = [
            {
                "run": "a",
                "mode": "A",
                "selected_model": "best_val",
                "curve_n_eval_points": 10 ** 400,
            },
            {
                "run": "b",
                "mode": "A",
                "selected_model": "best_val",
                "curve_n_eval_points": 1,
            },
        ]
        report = experiment.selection_bias_report(
            rows, "val_loss"
        )
        self.assertIsNone(report["opportunity_imbalance"]["ratio"])
        same_scale = [
            {
                "run": "a",
                "mode": "A",
                "selected_model": "best_val",
                "curve_n_eval_points": 10 ** 400,
            },
            {
                "run": "b",
                "mode": "A",
                "selected_model": "best_val",
                "curve_n_eval_points": 10 ** 399,
            },
        ]
        report = experiment.selection_bias_report(
            same_scale, "val_loss"
        )
        self.assertEqual(
            report["opportunity_imbalance"]["ratio"], 10.0
        )

    def test_extended_split_near_duplicate_reason_is_counted(self):
        spans, stats = experiment._plan_train_near_duplicate_cleaning(
            bytearray(b"aaa|bbb|aaa|"),
            b"|",
            8,
            {"holdout": (8, 11)},
            [],
            threshold=0.5,
            shingle_bytes=1,
        )
        self.assertEqual(spans, [(0, 3)])
        self.assertEqual(stats["flagged_duplicate_of"]["holdout"], 1)

    def test_format_helpers_tolerate_huge_report_numbers(self):
        display = experiment._format_window_paired({
            "window_paired": {
                "available": True,
                "mean_loss_diff": 10 ** 400,
                "n_seeds": 2,
                "per_seed": [
                    {
                        "window_p_value": 10 ** 400,
                        "window_ar1_adjustment_applied": True,
                        "window_p_value_ar1": 10 ** 400,
                    },
                ],
                "hierarchical_bootstrap_ci95": {
                    "available": True,
                    "low": 10 ** 400,
                    "high": 10 ** 400,
                    "n_seeds": 2,
                    "replicates": 10,
                },
                "variance_components": {
                    "available": True,
                    "variance_share_seed": 10 ** 400,
                    "variance_share_window": 10 ** 400,
                    "variance_share_interaction": 10 ** 400,
                    "dominant_source": "training_seed",
                },
            },
        })
        self.assertIn("Δloss=-", display)
        effect_line = experiment._format_factorial_effect({
            "kind": "ablation",
            "design": "smelt",
            "description": "A",
            "mean_effect": 10 ** 400,
            "p_value": 10 ** 400,
            "ci95_low": 10 ** 400,
            "ci95_high": 10 ** 400,
        })
        self.assertIn("Δ=-", effect_line)

    def test_ngram_tuning_diagnostics_tolerate_huge_values(self):
        report = experiment._ngram_tuning_diagnostics(
            {"order": 1, "alpha": 0.1, "val_loss": 10 ** 400},
            {"order": 2, "alpha": 0.2, "val_loss": 10 ** 400},
            "val_loss",
            [0.1, 0.2],
            2,
            2,
            2,
        )
        self.assertIsNone(report["best"]["value"])
        self.assertIsNone(report["runner_up"]["value"])
        self.assertIsNone(report["metric_gap"])

    def test_comparison_print_tolerates_huge_autocorrelation(self):
        rows = [{
            "run": "A1",
            "mode": "A",
            "seed": 1,
            "val_loss": 1.0,
            "val_loss_window_lag1_autocorrelation": 10 ** 400,
        }]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            experiment.print_comparison_analysis(
                rows,
                factors=["mode"],
                metric="val_loss",
            )
        self.assertNotIn("10" * 50, output.getvalue())

    def test_diagnostic_prints_tolerate_huge_numbers(self):
        huge = 10 ** 400
        with contextlib.redirect_stdout(io.StringIO()):
            experiment.print_bad_cases({
                "split": "val",
                "tokens_evaluated": 1,
                "token_accuracy": 1.0,
                "top_cases": [{
                    "rank": 1,
                    "loss": huge,
                    "confidence": huge,
                    "margin": huge,
                    "target_text": "a",
                    "predicted_text": "b",
                    "category": "c",
                    "context": "d",
                }],
                "top_cases_concentration_warning": True,
                "top_cases_window_concentration": huge,
                "error_rate_by_position": [{
                    "name": "p",
                    "error_rate": huge,
                    "wrong": 1,
                    "tokens": 1,
                }],
                "calibration": {
                    "ece": huge,
                    "mean_confidence": huge,
                    "accuracy": huge,
                    "overconfidence": huge,
                },
                "error_category_counts": {"a": 1},
                "error_category_shares": {"a": huge},
            })
            experiment.print_routing_analysis({
                "mode": "x",
                "split": "val",
                "tokens_evaluated": 1,
                "middle_blocks": 1,
                "rq4_load_and_entropy": {
                    "summary": {"round1": {
                        "load_entropy": huge,
                        "load_max_share": huge,
                        "dead_experts": 0,
                        "router_entropy_mean": huge,
                    }}
                },
                "rq3_expert_overlap": {"summary": {
                    "second_round_jaccard_mean": huge,
                    "second_round_exact_match_share": huge,
                    "second_round_slot_reuse_mean": huge,
                    "token_second_round_jaccard_mean": huge,
                    "token_second_round_exact_match_share": huge,
                    "branch_self_reuse_share": huge,
                    "jaccard_chance_uniform": huge,
                    "jaccard_chance_marginal": huge,
                    "jaccard_lift_over_uniform": huge,
                    "jaccard_lift_over_marginal": huge,
                    "jaccard_adjusted_marginal": huge,
                }},
                "branch_divergence": {"summary": {
                    "branch_cosine_mean": huge,
                    "blocks_with_branches": 1,
                }},
            })
            experiment.print_routing_interventions({
                "mode": "x",
                "baseline": {"loss": huge},
                "interventions": {
                    "routing_shuffle": {
                        "loss_delta": huge,
                        "loss_ratio": huge,
                        "accuracy_delta": huge,
                    },
                    "expert_drop": {
                        "max_expert": 0,
                        "max_loss_delta": huge,
                        "min_expert": 1,
                        "min_loss_delta": huge,
                        "mean_loss_delta": huge,
                        "std_loss_delta": huge,
                        "importance_cv": huge,
                    },
                },
            })
            experiment.print_baseline_tuning_report({
                "metric": "val_loss",
                "higher_is_better": False,
                "training_config": {},
                "mode": "local_base",
                "candidates": [{
                    "lr": huge,
                    "n_runs": 1,
                    "metric_mean": huge,
                    "metric_std": huge,
                }],
                "winner": {
                    "lr": huge,
                    "n_runs": 1,
                    "metric_mean": huge,
                },
            }, Path("/tmp"))
        text = experiment._format_window_paired({
            "window_paired": {
                "available": True,
                "mean_loss_diff": 1.0,
                "n_seeds_favoring_variant": 10 ** 400,
                "n_seeds_favoring_baseline": float("inf"),
                "n_seeds": "x",
            }
        })
        self.assertIn("Δloss=+1.0000", text)

    def test_safe_ratio_keeps_representable_huge_int_ratio(self):
        self.assertEqual(
            experiment._safe_ratio(10 ** 400, 10 ** 399),
            10.0,
        )
        self.assertEqual(
            experiment._safe_ratio(10 ** 400, 10 ** 400),
            1.0,
        )
        self.assertIsNone(
            experiment._safe_ratio(10 ** 400, 1)
        )

    def test_contrast_display_tolerates_huge_numbers(self):
        display = experiment._contrast_display({
            "factor": "mode",
            "baseline_value": "smelt",
            "variant_value": "A",
            "delta": 10 ** 400,
            "p_value": 10 ** 400,
            "p_value_holm": 10 ** 400,
            "test": "paired_t",
            "n_pairs": 3,
            "delta_over_noise": 10 ** 400,
            "hedges_g": 10 ** 400,
            "cohens_dz": 10 ** 400,
            "hedges_gz": 10 ** 400,
            "ci95_low": 10 ** 400,
            "ci95_high": 10 ** 400,
            "ci_family_adjusted_low": 10 ** 400,
            "ci_family_adjusted_high": 10 ** 400,
            "ci_family_adjusted_family_size": 2,
            "window_paired": {
                "available": True,
                "mean_loss_diff": 10 ** 400,
            },
            "fragility": {
                "available": True,
                "verdict": "fragile",
                "min_seed_deletions_to_lose_significance": 1,
                "min_seed_deletions_p_value": 10 ** 400,
                "alpha": 10 ** 400,
            },
            "equivalent": True,
            "equivalence_margin_abs": 10 ** 400,
            "tost_p_value_holm": 10 ** 400,
            "significant": True,
        }, "val_loss")
        self.assertIn("mode: smelt→A", display)
        self.assertIn("Δ=-", display)

    def test_curve_print_tolerates_huge_numbers(self):
        analysis = {
            "status": "healthy",
            "n_eval_points": 2,
            "n_nonfinite_events": 0,
            "best_val_loss": 10 ** 400,
            "last_train_loss": 10 ** 400,
            "last_val_loss": 10 ** 400,
            "val_regression_from_best": 10 ** 400,
            "best_val_step": 1,
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            experiment.print_curve_analysis(analysis)
        self.assertIn("[curve]", output.getvalue())

    def test_efficiency_and_design_print_tolerate_huge_numbers(self):
        efficiency = {
            "n_params_total": 10 ** 400,
            "flops_per_forward": 10 ** 400,
            "latency_ms": {
                "n": 2,
                "median": 10 ** 400,
                "p95": 10 ** 400,
                "p99": 10 ** 400,
                "std": 10 ** 400,
                "coefficient_of_variation": 10 ** 400,
                "max": 10 ** 400,
            },
        }
        design = {
            "planned_runs": 2,
            "order_scheme": "randomized",
            "order_seed": 1,
            "plan_id": "x" * 20,
            "path": "design.json",
            "power": {
                "adequate": False,
                "expected_attrition_rate": 10 ** 400,
                "required_seeds_with_attrition": 10 ** 400,
                "required_seeds_for_t": 10 ** 400,
                "required_seeds_for_robust_sign_flip": 10 ** 400,
                "planned_seeds": 2,
                "planned_usable_seeds": 2,
                "achieved_power_at_min_effect": 10 ** 400,
                "minimum_detectable_effect_relative": 10 ** 400,
                "equivalence": {
                    "margin_relative": 10 ** 400,
                    "power_at_null_difference": 10 ** 400,
                    "margin_for_target_power": 10 ** 400,
                },
            },
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            experiment.print_efficiency(efficiency)
            experiment.print_design_summary(design)
        self.assertIn("[efficiency]", output.getvalue())
        self.assertIn("[design]", output.getvalue())

    def test_current_seed_count_tolerates_huge_counts(self):
        self.assertIsNone(
            experiment._current_seed_count({"n_pairs": 10 ** 400})
        )
        self.assertIsNone(
            experiment._current_seed_count({
                "n_baseline": 10 ** 400,
                "n_variant": 10 ** 400,
            })
        )
        self.assertAlmostEqual(
            experiment._current_seed_count({
                "n_baseline": 100,
                "n_variant": 3,
            }),
            2.0 * 100 * 3 / 103,
        )

    def test_data_audit_print_tolerates_huge_numbers(self):
        report = {
            "splits": {
                "train": {
                    "bytes": 10 ** 400,
                    "effective_bytes": 10 ** 400,
                },
            },
            "dataset_split_fractions": {"train": 10 ** 400},
            "train_cleaning": {
                "groups_total": 1,
                "groups_dropped": 0,
                "deletion_bias": {
                    "drop_fraction": 10 ** 400,
                    "byte_js": 10 ** 400,
                },
            },
            "label_coverage": {
                "val": {
                    "unseen_label_classes": 1,
                    "unseen_label_token_share": 10 ** 400,
                },
            },
            "distribution_shift": {"train/val": 10 ** 400},
            "distribution_shift_bigram": {"train/val": 10 ** 400},
            "distribution_shift_conditional": {
                "train/val": 10 ** 400,
            },
            "unseen_transition_share": {"train/val": 10 ** 400},
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            experiment.print_data_audit(report)
        self.assertIn("[data]", output.getvalue())
