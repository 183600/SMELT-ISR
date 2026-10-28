from _core_support import *
from _core_support import _load_experiment, _tiny_config

class StatisticsTests(unittest.TestCase):
    def test_exact_sign_test_drops_zero_and_reports_resolution(self):
        six = experiment._exact_sign_test(
            [1.0, 1.0, 1.0, 1.0, 1.0, -1.0]
        )
        self.assertEqual(six["n_positive"], 5)
        self.assertEqual(six["n_negative"], 1)
        self.assertEqual(six["n_nonzero"], 6)
        self.assertAlmostEqual(six["p_value"], 0.21875)
        self.assertAlmostEqual(six["min_attainable_p"], 0.03125)
        # n=5 的最小可达双侧 p=0.0625，0.05 水平上不可能显著。
        five = experiment._exact_sign_test(
            [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        self.assertAlmostEqual(five["min_attainable_p"], 0.0625)
        self.assertTrue(five["resolution_limited"])
        with_zero = experiment._exact_sign_test(
            [1.0, 0.0, 1.0, 0.0, -1.0]
        )
        self.assertEqual(with_zero["n_zero"], 2)
        self.assertEqual(with_zero["n_nonzero"], 3)

    def test_tost_power_resolves_narrow_equivalence_boundary(self):
        # 大 df、大 margin/临界值比时，正常功效表达式的转折宽度
        # 只有 upper/sqrt(df)（本例约 5e-4），固定 0.01 网格
        # 会把这一段整体跨过并低估功效。参考值由 SciPy 非中心 t
        # CDF 差（upper = margin - t_{1-alpha,df}）高精度复核。
        cases = (
            (
                3026.1544367406605,
                2.638701447487692,
                0.8863036119918171,
                0.1951440750942129,
                0.0006652944300069687,
            ),
            (
                160.86216210361707,
                1.6523599113942893,
                3.7796112570198046,
                0.00012166870603104297,
                0.005347019781185125,
            ),
            (
                69809.78603112319,
                -0.586314062797227,
                5.207860576274624,
                1.4092235790709054e-07,
                0.04845356607606455,
            ),
        )
        for df, effect, margin, alpha, expected in cases:
            power = experiment._tost_power(
                effect, 1.0, df, margin, alpha
            )
            self.assertIsNotNone(power)
            self.assertAlmostEqual(
                power / expected, 1.0, places=9
            )

    def test_fragility_detects_single_seed_dependence(self):
        # 5 个种子里 4 个差值约 1、1 个 3：整体显著，但删掉
        # 任意一个「1」后只剩 4 个观测，p 值回到 0.09。
        report = experiment._paired_result_fragility(
            [1.0, 1.0, 1.0, 1.0, 3.0],
            seeds=[1, 2, 3, 4, 5],
            alpha=0.05,
        )
        self.assertTrue(report["available"])
        self.assertFalse(
            report["significance_survives_leave_one_out"]
        )
        self.assertEqual(
            report["min_seed_deletions_to_lose_significance"], 1
        )
        self.assertEqual(report["verdict"], "fragile")
        self.assertEqual(report["n_positive"], 5)
        self.assertTrue(report["direction_consistent"])

    def test_fragility_stable_when_all_seed_deletions_remain_significant(self):
        report = experiment._paired_result_fragility(
            [1.0, 1.0, 1.0, 1.0, 1.2],
            alpha=0.05,
        )
        self.assertEqual(report["verdict"], "stable")
        self.assertTrue(
            report["significance_survives_leave_one_out"]
        )
        self.assertIsNone(
            report["min_seed_deletions_to_lose_significance"]
        )
        self.assertTrue(report["deletions_tested_exhaustively"])

    def test_fragility_marks_three_seed_design_not_evaluable(self):
        report = experiment._paired_result_fragility(
            [1.0, 1.1, 0.9],
            alpha=0.05,
        )
        self.assertEqual(report["verdict"], "not_evaluable")
        self.assertFalse(report["leave_one_out_evaluable"])

    def test_fragility_flags_magnitude_driven_direction_mix(self):
        # 7 个 +5、1 个 -1：均值显著，且删除 3 个种子后仍显著，
        # 但多数种子同向的符号检验无法解释均值效应。
        report = experiment._paired_result_fragility(
            [5.0] * 7 + [-1.0],
            alpha=0.05,
        )
        self.assertEqual(report["verdict"], "direction_mixed")
        self.assertFalse(report["direction_consistent"])

    def test_factor_contrasts_attach_fragility_and_summary(self):
        rows = []
        differences = [1.0, 1.0, 1.0, 1.0, 3.0]
        for index, difference in enumerate(differences):
            seed = index + 1
            rows.append({
                "run": f"base{seed}",
                "mode": "local_base",
                "seed": seed,
                "val_loss": 10.0,
            })
            rows.append({
                "run": f"A{seed}",
                "mode": "A",
                "seed": seed,
                "val_loss": 10.0 + difference,
            })
        aggregates = experiment.aggregate_runs(rows, ["mode"])
        contrasts = experiment.factor_contrasts(
            aggregates, ["mode"], "val_loss"
        )
        self.assertEqual(len(contrasts), 1)
        contrast = contrasts[0]
        self.assertEqual(
            contrast["paired_differences"],
            differences,
        )
        self.assertEqual(
            contrast["fragility"]["verdict"], "fragile"
        )
        summary = experiment._fragility_summary(contrasts)
        self.assertEqual(summary["verdict_counts"]["fragile"], 1)
        self.assertEqual(len(summary["fragile"]), 1)
        self.assertIn("删除少量种子", summary["advisories"][0])

    def _attrition_cell(self, label, values, failed=0):
        return {
            "label": label,
            "n_runs": len(values),
            "samples": {
                "val_loss": [
                    {
                        "seed": index + 1,
                        "run": f"{label}{index + 1}",
                        "value": value,
                    }
                    for index, value in enumerate(values)
                ]
            },
            "numerical_failures": [
                {"run": f"{label}_failed{index}"}
                for index in range(failed)
            ],
        }

    def test_attrition_bounds_complete_cases_match_observed_effect(self):
        bounds = experiment._attrition_bounds(
            {
                "baseline": self._attrition_cell(
                    "local_base", [10.0, 10.0, 10.0]
                ),
                "variant": self._attrition_cell(
                    "A", [9.0, 9.0, 9.0]
                ),
            },
            {"baseline": -1.0, "variant": 1.0},
            "val_loss",
            -1.0,
        )
        self.assertTrue(bounds["applicable"])
        self.assertEqual(bounds["verdict"], "complete_cases")
        self.assertEqual(bounds["missing_runs"], 0)
        self.assertAlmostEqual(bounds["effect_min"], -1.0)
        self.assertAlmostEqual(bounds["effect_max"], -1.0)
        self.assertTrue(bounds["sign_robust"])

    def test_attrition_bounds_tolerates_out_of_float_range_counts(self):
        huge = 10**400
        bounds = experiment._attrition_bounds(
            {
                "variant": {
                    "label": "A",
                    "n_runs": huge,
                    "samples": {
                        "val_loss": [
                            {"seed": 1, "run": "A1", "value": 1.0}
                        ]
                    },
                    "numerical_failures": [],
                }
            },
            {"variant": 1.0},
            "val_loss",
            1.0,
        )
        self.assertTrue(bounds["applicable"])
        self.assertEqual(bounds["observed_fraction"], 0.0)
        self.assertEqual(bounds["missing_runs"], huge - 1)
        self.assertAlmostEqual(bounds["effect_min"], 1.0)
        self.assertAlmostEqual(bounds["effect_max"], 1.0)

    def test_attrition_bounds_extreme_missing_can_cross_zero(self):
        # 两臂观测范围 [0, 10]；A 有 2 个失败运行。完整案例
        # 差值为 +0.1，但把 A 的缺失运行放到 0、baseline 的
        # 缺失运行放到 10 时方向可反转。
        bounds = experiment._attrition_bounds(
            {
                "baseline": self._attrition_cell(
                    "local_base", [0.0, 10.0, 0.0, 10.0], failed=2
                ),
                "variant": self._attrition_cell(
                    "A", [5.0, 5.0, 5.0, 5.0], failed=2
                ),
            },
            {"baseline": -1.0, "variant": 1.0},
            "val_loss",
            0.1,
        )
        self.assertTrue(bounds["applicable"])
        self.assertEqual(bounds["verdict"], "crosses_zero")
        self.assertEqual(bounds["missing_runs"], 4)
        self.assertTrue(bounds["zero_within_bounds"])
        self.assertFalse(bounds["sign_robust"])
        self.assertLess(bounds["effect_min"], 0.0)
        self.assertGreater(bounds["effect_max"], 0.0)

    def test_attrition_bounds_unbounded_without_observed_outcomes(self):
        bounds = experiment._attrition_bounds(
            {
                "baseline": self._attrition_cell(
                    "local_base", [], failed=4
                ),
                "variant": self._attrition_cell("A", [], failed=4),
            },
            {"baseline": -1.0, "variant": 1.0},
            "val_loss",
            None,
        )
        self.assertFalse(bounds["applicable"])
        self.assertEqual(bounds["verdict"], "unbounded")
        self.assertIn("没有任何已观测结果", bounds["reason"])

    def test_factor_contrast_attaches_attrition_bounds(self):
        rows = []
        for seed in range(1, 7):
            rows.append({
                "run": f"base{seed}",
                "mode": "local_base",
                "seed": seed,
                "val_loss": 10.0,
            })
            rows.append({
                "run": f"A{seed}",
                "mode": "A",
                "seed": seed,
                "val_loss": 9.0,
            })
        # A 的第 7 个计划运行数值失败：完整案例只覆盖 6 个，
        # 缺失敏感性必须显式附着，不能被静默丢弃。
        rows.append({
            "run": "A7",
            "mode": "A",
            "seed": 7,
            "val_loss": None,
            "numerical_status": "failed",
        })
        aggregates = experiment.aggregate_runs(rows, ["mode"])
        contrast = experiment.factor_contrasts(
            aggregates, ["mode"], "val_loss"
        )[0]
        bounds = contrast["attrition_bounds"]
        self.assertEqual(bounds["missing_runs"], 1)
        self.assertEqual(bounds["verdict"], "bounded_same_sign")
        self.assertTrue(bounds["sign_robust"])
        summary = experiment._attrition_bounds_summary(
            [contrast]
        )
        self.assertEqual(len(summary["with_missing"]), 1)
        self.assertEqual(summary["not_robust"], [])
        self.assertIn("有缺测", summary["advisories"][0])

    def test_attrition_summary_flags_nonrobust_sign_with_missing(self):
        contrast = {
            "baseline": "A",
            "variant": "B",
            "attrition_bounds": {
                "applicable": True,
                "verdict": "bounded_same_sign",
                "sign_robust": False,
                "complete_cases": False,
                "missing_runs": 1,
                "attempted_runs": 6,
                "observed_runs": 5,
                "observed_effect": 1.0,
                "effect_min": -2.0,
                "effect_max": -1.0,
                "zero_within_bounds": False,
                "observed_effect_within_bounds": False,
            },
        }
        summary = experiment._attrition_bounds_summary([contrast])
        self.assertEqual(len(summary["not_robust"]), 1)
        self.assertIn("效应方向可能反转", summary["advisories"][0])

    def test_attrition_summary_tolerates_invalid_missing_counts(self):
        for bad_count in ("bad", float("inf"), float("nan"), []):
            with self.subTest(bad_count=bad_count):
                summary = experiment._attrition_bounds_summary([{
                    "baseline": "A",
                    "variant": "B",
                    "attrition_bounds": {
                        "applicable": True,
                        "verdict": "bounded_same_sign",
                        "sign_robust": False,
                        "missing_runs": bad_count,
                    },
                }])
                self.assertEqual(summary["with_missing"], [])
                self.assertEqual(summary["not_robust"], [])

    def test_component_summary_downgrades_attrition_crossing(self):
        effect = {
            "effect_name": "delayed_merge_main",
            "delayed_mode": "A",
            "claim": "supports_component",
            "mean_effect": -0.1,
            "p_value_holm": 0.01,
            "significant": True,
            "equivalent": False,
            "n_pairs": 6,
            "fragility": {"verdict": "stable"},
            "attrition_bounds": {"verdict": "crosses_zero"},
        }
        summary = experiment._component_evidence_summary([effect])
        self.assertEqual(summary["status"], "inconclusive")
        self.assertIn("缺失", summary["reason"])
        self.assertEqual(
            len(summary["attrition_unstable_effects"]), 1
        )

    def test_component_summary_downgrades_nonrobust_attrition_sign(self):
        effect = {
            "effect_name": "delayed_merge_main",
            "delayed_mode": "A",
            "claim": "supports_component",
            "mean_effect": -0.1,
            "p_value_holm": 0.01,
            "significant": True,
            "equivalent": False,
            "n_pairs": 6,
            "fragility": {"verdict": "stable"},
            "attrition_bounds": {
                "verdict": "complete_cases",
                "sign_robust": False,
                "observed_effect_within_bounds": False,
            },
        }
        summary = experiment._component_evidence_summary([effect])
        self.assertEqual(summary["status"], "inconclusive")
        self.assertIn("符号", summary["reason"])
        self.assertEqual(
            len(summary["attrition_unstable_effects"]), 1
        )

    def test_ablation_verdict_downgrades_attrition_unstable_simple_effect(self):
        def effect(name, attrition_verdict):
            return {
                "effect_name": name,
                "delayed_mode": "A",
                "claim": "supports_component",
                "mean_effect": -0.1,
                "p_value_holm": 0.01,
                "significant": True,
                "equivalent": False,
                "n_pairs": 6,
                "fragility": {"verdict": "stable"},
                "attrition_bounds": {
                    "verdict": attrition_verdict
                },
            }

        factorial = {
            "applicable": True,
            "metric": "val_loss",
            "effects": [
                effect("delayed_merge_main", "bounded_same_sign"),
                effect(
                    "delayed_merge_at_balance_off",
                    "bounded_same_sign",
                ),
                effect(
                    "delayed_merge_at_balance_on",
                    "crosses_zero",
                ),
            ],
            "incomplete_designs": [],
            "confounded_designs": [],
            "controlled_variables_ok": True,
        }
        coverage = {
            "expected_delayed_modes": ["A"],
            "project_delayed_modes": ["A"],
            "common_step_sensitivity": {"applicable": False},
        }
        report = experiment.ablation_component_verdicts(
            factorial, coverage
        )
        component = next(
            item
            for item in report["components"]
            if item["component"] == "delayed_merge_A"
        )
        self.assertEqual(component["status"], "inconclusive")
        self.assertTrue(
            any(
                "简单效应" in reason and "缺失" in reason
                for reason in component["reasons"]
            )
        )

    def test_seed_plan_flags_robust_impossible(self):
        plan = experiment._build_seed_plan(
            [1, 2, 3, 4, 5], False
        )
        self.assertFalse(plan["robust_claim_possible"])
        self.assertEqual(
            plan["min_seeds_for_robust_sign_flip"], 6
        )
        self.assertGreaterEqual(
            experiment._sign_flip_required_seeds(0.05, 6),
            6,
        )

    def test_pseudo_replication_is_collapsed(self):
        samples_a = [
            {"seed": 1, "run": "a1", "value": 1.0},
            {"seed": 1, "run": "a1dup", "value": 1.2},
            {"seed": 2, "run": "a2", "value": 1.4},
            {"seed": 3, "run": "a3", "value": 1.6},
        ]
        samples_b = [
            {"seed": 1, "run": "b1", "value": 2.0},
            {"seed": 2, "run": "b2", "value": 2.2},
            {"seed": 3, "run": "b3", "value": 2.4},
        ]
        result = experiment._contrast_test(
            samples_a, samples_b
        )
        self.assertTrue(result["duplicate_seeds"])
        self.assertTrue(
            result["pseudo_replication"]["detected"]
        )
        # 折叠为 3 个独立种子后仍可配对，n 不因复制运行增加。
        self.assertEqual(result["n_pairs"], 3)
        self.assertEqual(result["n_baseline"], 3)
        self.assertEqual(result["n_variant"], 3)

    def test_observed_seed_sd_matches_welch_definition(self):
        # Welch se² = s²/n_a + s²/n_b；等臂 n 时 s = se·√(n/2)，
        # 不能把调和均值 n_eq 直接当成配对样本数。
        contrast = {
            "n_pairs": 0,
            "n_baseline": 3,
            "n_variant": 3,
            "se": 0.1,
        }
        self.assertAlmostEqual(
            experiment._observed_seed_sd(contrast, 0.1),
            0.1 * math.sqrt(1.5),
        )
        paired = {
            "n_pairs": 3,
            "n_baseline": 3,
            "n_variant": 3,
            "se": 0.1,
        }
        self.assertAlmostEqual(
            experiment._observed_seed_sd(paired, 0.1),
            0.1 * math.sqrt(3.0),
        )

    def test_all_failed_cell_stays_visible(self):
        rows = [
            {
                "run": f"base{seed}",
                "mode": "local_base",
                "seed": seed,
                "val_loss": 3.0 + seed / 100.0,
            }
            for seed in (1, 2, 3)
        ]
        rows += [
            {
                "run": f"Afailed{seed}",
                "mode": "A",
                "seed": seed,
                "val_loss": float("nan"),
                "numerical_status": "failed",
            }
            for seed in (1, 2, 3)
        ]
        aggregates = experiment.aggregate_runs(rows, ["mode"])
        by_mode = {
            aggregate["keys"]["mode"]: aggregate
            for aggregate in aggregates
        }
        self.assertIn("local_base", by_mode)
        failed = by_mode["A"]
        self.assertTrue(failed["all_runs_failed"])
        self.assertEqual(failed["n_runs"], 0)
        self.assertEqual(failed["n_planned_runs"], 3)
        self.assertEqual(
            {
                item["run"]
                for item in failed["numerical_failures"]
            },
            {"Afailed1", "Afailed2", "Afailed3"},
        )

    def test_failure_rate_report(self):
        rows = [
            {"run": "a1", "mode": "A", "numerical_status": "ok"},
            {"run": "a2", "mode": "A", "numerical_status": "failed"},
            {"run": "b1", "mode": "B", "numerical_status": "ok"},
        ]
        report = experiment.failure_rate_report(rows)
        arms = {arm["mode"]: arm for arm in report["arms"]}
        self.assertEqual(arms["A"]["planned_runs"], 2)
        self.assertEqual(arms["A"]["failed_runs"], 1)
        self.assertAlmostEqual(arms["A"]["failure_rate"], 0.5)

    def test_family_adjusted_intervals_are_simultaneous(self):
        # 三臂两两对比构成一个 m=3 的检验家族；逐检验 95%CI
        # 与 Holm 判定可能不一致，必须另给 Bonferroni 同时区间。
        values = {
            "local_base": [10.00, 10.03, 9.98, 10.01],
            "A": [9.50, 9.44, 9.52, 9.47],
            "B": [9.20, 9.10, 9.16, 9.12],
        }
        rows = [
            {
                "run": f"{mode}{seed}",
                "mode": mode,
                "seed": seed,
                "val_loss": value,
            }
            for mode, series in values.items()
            for seed, value in enumerate(series, start=1)
        ]
        aggregates = experiment.aggregate_runs(rows, ["mode"])
        contrasts = experiment.factor_contrasts(
            aggregates, ["mode"], "val_loss"
        )
        self.assertEqual(len(contrasts), 3)
        for contrast in contrasts:
            self.assertEqual(
                contrast["ci_family_adjusted_family_size"], 3
            )
            self.assertAlmostEqual(
                contrast["ci_family_adjusted_alpha"], 0.05 / 3
            )
            point_width = (
                float(contrast["ci95_high"])
                - float(contrast["ci95_low"])
            )
            adjusted_width = (
                float(contrast["ci_family_adjusted_high"])
                - float(contrast["ci_family_adjusted_low"])
            )
            self.assertGreaterEqual(
                adjusted_width, point_width - 1e-12
            )
            # 同时区间不含 0 必然蕴含 Holm 显著（充分条件）；
            # 反向不成立，不能反过来要求 significant 蕴含
            # 同时区间不含 0。
            if contrast["ci_family_adjusted_excludes_zero"]:
                self.assertTrue(contrast["significant"])
            self.assertTrue(
                contrast["ci_family_adjusted_decision_consistent"]
            )
        display = experiment._contrast_display(
            contrasts[0], "val_loss"
        )
        self.assertIn("95%CI", display)
        self.assertIn("同时95%CI", display)

    def test_hierarchical_window_bootstrap_and_variance_components(self):
        per_seed = {
            1: [-0.50, -0.48, -0.52, -0.51],
            2: [-0.20, -0.22, -0.18, -0.19],
            3: [-0.80, -0.78, -0.82, -0.79],
        }
        bootstrap = experiment._hierarchical_window_bootstrap_ci(
            per_seed
        )
        self.assertTrue(bootstrap["available"])
        self.assertEqual(bootstrap["n_seeds"], 3)
        self.assertLess(
            bootstrap["low"], bootstrap["high"]
        )
        self.assertLessEqual(
            bootstrap["low"], -0.5
        )
        self.assertGreaterEqual(
            bootstrap["high"], -0.5
        )
        repeated = experiment._hierarchical_window_bootstrap_ci(
            per_seed
        )
        self.assertEqual(
            repeated["low"], bootstrap["low"]
        )
        self.assertEqual(
            repeated["high"], bootstrap["high"]
        )
        single = experiment._hierarchical_window_bootstrap_ci(
            {1: [-0.5, -0.4]}
        )
        self.assertFalse(single["available"])

        components = experiment._window_variance_components(
            per_seed
        )
        self.assertTrue(components["available"])
        self.assertEqual(components["n_seeds"], 3)
        self.assertGreater(components["between_seed_sd"], 0.0)
        self.assertGreater(
            components["within_seed_sd_mean"], 0.0
        )
        self.assertAlmostEqual(
            float(components["variance_share_seed"])
            + float(components["variance_share_window"])
            + float(components["variance_share_interaction"]),
            1.0,
        )
        self.assertEqual(
            components["dominant_source"], "training_seed"
        )
        self.assertIn(
            "增加独立种子", components["advisory"]
        )
        prediction = components[
            "new_seed_mean_prediction_interval_95"
        ]
        self.assertLessEqual(
            prediction["low"],
            components["seed_mean_effect"],
        )
        self.assertGreaterEqual(
            prediction["high"],
            components["seed_mean_effect"],
        )
        # 只有一个种子时不能分离种子层方差，不能因此伪造
        # 方差占比或预测区间。
        one_seed = experiment._window_variance_components(
            {1: [-0.5, -0.4, -0.6]}
        )
        self.assertFalse(one_seed["available"])
        self.assertEqual(
            one_seed["reason"],
            "insufficient_seeds_for_variance_components",
        )
        self.assertNotIn(
            "new_seed_mean_prediction_interval_95", one_seed
        )

    def test_window_paired_analysis_reports_two_level_uncertainty(self):
        def payload(series):
            return {
                "split": "val",
                "data_sha256": "deadbeef",
                "split_sha256": "cafebabe",
                "seq_len": 8,
                "batch_size": 8,
                "batches": 1,
                "n_windows": len(series),
                "overlap_fraction": 0.0,
                "windows": [
                    {
                        "loss_sum": float(loss) * 10.0,
                        "tokens": 10,
                        "correct": 0,
                    }
                    for loss in series
                ],
            }

        base = [4.0, 4.1, 4.2, 3.9]
        variant_by_seed = {
            1: [loss - 0.5 for loss in base],
            2: [loss - 0.2 for loss in base],
            3: [loss - 0.8 for loss in base],
        }
        cell_a = {
            "window_payloads": [
                {
                    "seed": seed,
                    "run": f"base{seed}",
                    "payload": payload(base),
                }
                for seed in (1, 2, 3)
            ]
        }
        cell_b = {
            "window_payloads": [
                {
                    "seed": seed,
                    "run": f"variant{seed}",
                    "payload": payload(series),
                }
                for seed, series in variant_by_seed.items()
            ]
        }
        report = experiment._cell_window_paired_analysis(
            cell_a, cell_b
        )
        self.assertTrue(report["available"])
        self.assertEqual(report["n_seeds"], 3)
        hierarchical = report["hierarchical_bootstrap_ci95"]
        self.assertTrue(hierarchical["available"])
        self.assertEqual(hierarchical["n_seeds"], 3)
        components = report["variance_components"]
        self.assertTrue(components["available"])
        self.assertEqual(components["n_seeds"], 3)
        self.assertEqual(
            components["dominant_source"], "training_seed"
        )
        # 原始逐窗口差值只供两层 bootstrap 使用，不落进
        # per_seed 明细（否则报告体量随窗口数膨胀）。
        for entry in report["per_seed"]:
            self.assertNotIn("loss_differences", entry)
        display = experiment._format_window_paired(
            {"window_paired": report}
        )
        self.assertIn("两层 bootstrap 95%CI", display)
        self.assertIn("方差占比", display)
