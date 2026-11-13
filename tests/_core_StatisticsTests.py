from _core_support import *
from _core_support import _load_experiment, _tiny_config
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
            # NOTE: Consider if 0.05 threshold is appropriate for this context
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
            # NOTE: Consider if 0.05 threshold is appropriate for this context
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
            # NOTE: Consider if 0.05 threshold is appropriate for this context
        )
        self.assertEqual(report["verdict"], "not_evaluable")
        self.assertFalse(report["leave_one_out_evaluable"])

    def test_fragility_flags_magnitude_driven_direction_mix(self):
        # 7 个 +5、1 个 -1：均值显著，且删除 3 个种子后仍显著，
        # 但多数种子同向的符号检验无法解释均值效应。
        report = experiment._paired_result_fragility(
            [5.0] * 7 + [-1.0],
            alpha=0.05,
            # NOTE: Consider if 0.05 threshold is appropriate for this context
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

    def test_mechanism_control_report_flags_missing_control(self):
        # 2×2 消融只有 local_base/A/B：缺少只改合并时机/位置的
        # 单因素机制对照时，延迟合并的机制归因不能由多因素主
        # 效应替代。
        rows = []
        for mode in ("local_base", "A", "B"):
            for balance in (0.0, 0.01):
                for seed in (41, 42, 43, 44):
                    rows.append({
                        "run": f"{mode}_{balance}_{seed}",
                        "mode": mode,
                        "balance_coef": balance,
                        "seed": seed,
                        "val_loss": 1.0 + 0.001 * seed,
                    })
        aggregates = experiment.aggregate_runs(
            rows, ["mode", "balance_coef"]
        )
        report = experiment.mechanism_control_report(
            aggregates,
            ["mode", "balance_coef"],
            "val_loss",
        )
        self.assertTrue(report["applicable"])
        self.assertEqual(
            report["schemes"]["A"]["control_mode"],
            "local_base_k2",
        )
        self.assertEqual(
            report["schemes"]["B"]["control_mode"],
            "B_early",
        )
        for mode in ("A", "B"):
            entry = report["schemes"][mode]
            self.assertEqual(entry["status"], "missing_control")
            self.assertIn(mode, entry["reason"])
        self.assertTrue(report["advisories"])

    def _mechanism_rows(self, effects):
        rows = []
        for mode, balance, offset in effects:
            for seed in (41, 42, 43, 44, 45, 46):
                rows.append({
                    "run": f"{mode}_{balance}_{seed}",
                    "mode": mode,
                    "balance_coef": balance,
                    "seed": seed,
                    "val_loss": 1.0 + 0.001 * seed + offset,
                })
        return rows

    def test_mechanism_control_report_detects_single_factor_support(self):
        # 机制对照臂在位且延迟合并显著更好 → 单因素证据成立。
        # B 的对照 B_early 在因子的自然顺序里排在 B 之后，对比
        # 的 baseline/variant 方向与 A 相反，方向判定必须按
        # 「延迟合并方案是否更好」而不是 baseline/variant 位置。
        rows = self._mechanism_rows([
            ("local_base_k2", 0.0, 0.0),
            ("local_base_k2", 0.01, 0.0),
            ("A", 0.0, -0.2),
            ("A", 0.01, -0.2),
            ("B_early", 0.0, 0.0),
            ("B_early", 0.01, 0.0),
            ("B", 0.0, -0.2),
            ("B", 0.01, -0.2),
        ])
        aggregates = experiment.aggregate_runs(
            rows, ["mode", "balance_coef"]
        )
        report = experiment.mechanism_control_report(
            aggregates,
            ["mode", "balance_coef"],
            "val_loss",
        )
        for mode in ("A", "B"):
            entry = report["schemes"][mode]
            self.assertEqual(entry["status"], "supported")
            self.assertEqual(entry["n_contrasts"], 2)
            for contrast in entry["contrasts"]:
                self.assertTrue(contrast["delayed_better"])
                self.assertLess(
                    contrast["delta_delayed_minus_control"], 0.0
                )
                self.assertTrue(contrast["significant"])
        self.assertEqual(report["family_size"], 4)

    def test_mechanism_control_report_detects_contradiction(self):
        # 机制对照（立即合并）显著更好：延迟合并机制本身没有
        # 带来收益，不能只靠多因素主效应宣称方案有用。
        rows = self._mechanism_rows([
            ("local_base_k2", 0.0, -0.2),
            ("local_base_k2", 0.01, -0.2),
            ("A", 0.0, 0.0),
            ("A", 0.01, 0.0),
        ])
        aggregates = experiment.aggregate_runs(
            rows, ["mode", "balance_coef"]
        )
        report = experiment.mechanism_control_report(
            aggregates,
            ["mode", "balance_coef"],
            "val_loss",
        )
        entry = report["schemes"]["A"]
        self.assertEqual(entry["status"], "contradicted")
        for contrast in entry["contrasts"]:
            self.assertFalse(contrast["delayed_better"])

    def test_component_verdict_requires_single_factor_mechanism(self):
        def effect(name):
            return {
                "effect_name": name,
                "delayed_mode": "A",
                "claim": "supports_component",
                "mean_effect": -0.2,
                "p_value_holm": 0.01,
                "significant": True,
                "equivalent": False,
                "n_pairs": 6,
                "fragility": {"verdict": "stable"},
                "attrition_bounds": {"verdict": "complete_cases"},
            }

        factorial = {
            "applicable": True,
            "metric": "val_loss",
            "effects": [
                effect("delayed_merge_main"),
                effect("delayed_merge_at_balance_off"),
                effect("delayed_merge_at_balance_on"),
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
        without = experiment.ablation_component_verdicts(
            factorial, coverage
        )
        component = next(
            item
            for item in without["components"]
            if item["component"] == "delayed_merge_A"
        )
        self.assertEqual(component["status"], "supported")

        mechanism = {
            "applicable": True,
            "family_size": 0,
            "schemes": {
                "A": {
                    "delayed_mode": "A",
                    "control_mode": "local_base_k2",
                    "status": "missing_control",
                    "reason": (
                        "缺少单因素机制对照 local_base_k2："
                        "多因素差异不能替代单因素证据"
                    ),
                    "contrasts": [],
                    "n_contrasts": 0,
                }
            },
            "advisories": [],
        }
        blocked = experiment.ablation_component_verdicts(
            factorial, coverage, mechanism=mechanism
        )
        component = next(
            item
            for item in blocked["components"]
            if item["component"] == "delayed_merge_A"
        )
        self.assertEqual(component["status"], "blocked")
        self.assertTrue(
            any(
                "单因素机制对照" in reason
                for reason in component["reasons"]
            )
        )
        self.assertTrue(
            any(
                "单因素机制对照" in blocker
                for blocker in blocked["blockers"]
            )
        )

        mechanism["schemes"]["A"]["status"] = "supported"
        mechanism["schemes"]["A"]["reason"] = (
            "A 相对机制对照 local_base_k2 的单因素对比显著改善"
        )
        supported = experiment.ablation_component_verdicts(
            factorial, coverage, mechanism=mechanism
        )
        component = next(
            item
            for item in supported["components"]
            if item["component"] == "delayed_merge_A"
        )
        self.assertEqual(component["status"], "supported")
        self.assertFalse(
            any(
                "单因素机制对照" in blocker
                for blocker in supported["blockers"]
            )
        )

    def test_ablation_plan_includes_mechanism_control_arms(self):
        plan = experiment.build_sweep_plan(
            seeds=[1, 2],
            modes=None,
            ablation=True,
            balance_coef=0.01,
            ablation_modes=["A", "B"],
            mechanism_controls=True,
        )
        modes = sorted({str(item["mode"]) for item in plan})
        self.assertEqual(
            modes,
            ["A", "B", "B_early", "local_base", "local_base_k2"],
        )
        balances = {
            mode: sorted({
                float(item["balance_coef"]) for item in plan
                if item["mode"] == mode
            })
            for mode in modes
        }
        for mode in modes:
            self.assertEqual(
                balances[mode], [0.0, 0.01], msg=mode
            )
        plain = experiment.build_sweep_plan(
            seeds=[1, 2],
            modes=None,
            ablation=True,
            balance_coef=0.01,
            ablation_modes=["A", "B"],
        )
        self.assertEqual(
            sorted({str(item["mode"]) for item in plain}),
            ["A", "B", "local_base"],
        )

    def _mechanism_protocol_rows(
        self,
        effects,
        protocol_overrides=None,
        seeds=(41, 42, 43, 44, 45, 46),
    ):
        """构造带 comparison_protocol 的机制对照运行行。

        协议记录齐全是「两臂只差一个因子」可核对的前提：真实
        运行把 comparison_protocol 写进 config.json，这里用同样
        的结构模拟，才能验证单因素核对确实查了协议项。
        """
        protocol_overrides = protocol_overrides or {}
        rows = []
        for mode, balance, offset in effects:
            overrides = protocol_overrides.get(mode, {})
            for seed in seeds:
                protocol = {
                    "comparison_family": "neural",
                    "mode": mode,
                    "lr": 3e-4,
                    "optimizer": "AdamW",
                    "total_steps": 1000,
                    "k1": 2,
                    "k2": 2,
                    "top_k": 2,
                    "balance_coef": balance,
                    "residual_scale": 1.0,
                }
                protocol.update(overrides)
                rows.append({
                    "run": f"{mode}_{balance}_{seed}",
                    "mode": mode,
                    "balance_coef": balance,
                    "k1": protocol["k1"],
                    "k2": protocol["k2"],
                    "seed": seed,
                    "val_loss": 1.0 + 0.001 * seed + offset,
                    "protocol": protocol,
                    "protocol_recorded_keys": sorted(protocol),
                })
        return rows

    def test_mechanism_control_report_flags_protocol_difference(self):
        # 对照臂与方案臂只差一个因子是机制归因的前提：学习率
        # 不一致时这次对比不是单因素实验，不能拿来归因延迟合并。
        rows = self._mechanism_protocol_rows(
            [
                ("local_base_k2", 0.0, 0.0),
                ("A", 0.0, -0.2),
            ],
            protocol_overrides={"A": {"lr": 1e-4}},
        )
        aggregates = experiment.aggregate_runs(
            rows, ["mode", "balance_coef"]
        )
        report = experiment.mechanism_control_report(
            aggregates,
            ["mode", "balance_coef"],
            "val_loss",
            rows=rows,
        )
        entry = report["schemes"]["A"]
        self.assertEqual(entry["status"], "confounded")
        self.assertEqual(entry["n_contrasts"], 0)
        self.assertTrue(entry["confounded_pairs"])
        difference_keys = [
            str(item.get("key"))
            for pair in entry["confounded_pairs"]
            for item in pair.get("differences") or []
        ]
        self.assertIn("lr", difference_keys)
        self.assertFalse(report["single_factor_verified"])

    def test_mechanism_control_report_flags_unverifiable_protocol(self):
        # 一侧有协议记录、另一侧没有：无法确认两臂一致，不能
        # 当作单因素证据（缺记录不等于一致）。
        rows = self._mechanism_protocol_rows(
            [
                ("local_base_k2", 0.0, 0.0),
                ("A", 0.0, -0.2),
            ]
        )
        for row in rows:
            if row["mode"] == "local_base_k2":
                row["protocol"].pop("lr")
                row["protocol_recorded_keys"] = sorted(
                    row["protocol"]
                )
        aggregates = experiment.aggregate_runs(
            rows, ["mode", "balance_coef"]
        )
        report = experiment.mechanism_control_report(
            aggregates,
            ["mode", "balance_coef"],
            "val_loss",
            rows=rows,
        )
        entry = report["schemes"]["A"]
        self.assertEqual(entry["status"], "unverifiable")
        self.assertTrue(entry["unverifiable_pairs"])
        self.assertEqual(entry["n_contrasts"], 0)

    def test_mechanism_control_report_flags_unmatched_factor_levels(self):
        # 两臂的第二轮宽度不同（k2）时不存在只差一个因子的可比
        # 单元：对比必然是多因素差异，必须点名而不是当作单因素。
        rows = self._mechanism_protocol_rows(
            [
                ("local_base_k2", 0.0, 0.0),
                ("A", 0.0, -0.2),
            ]
        )
        for row in rows:
            if row["mode"] == "A":
                row["k2"] = 4
                row["protocol"]["k2"] = 4
        aggregates = experiment.aggregate_runs(
            rows, ["mode", "balance_coef", "k2"]
        )
        report = experiment.mechanism_control_report(
            aggregates,
            ["mode", "balance_coef", "k2"],
            "val_loss",
            rows=rows,
        )
        entry = report["schemes"]["A"]
        self.assertEqual(entry["status"], "confounded")
        unmatched_factors = [
            str(item.get("factor"))
            for item in entry["unmatched_levels"]
        ]
        self.assertIn("k2", unmatched_factors)
        self.assertEqual(entry["n_contrasts"], 0)

    def test_mechanism_control_report_flags_insufficient_seeds(self):
        # 共同种子不足时做不了配对检验：不能给出方向性结论。
        rows = self._mechanism_protocol_rows(
            [
                ("local_base_k2", 0.0, 0.0),
                ("A", 0.0, -0.2),
            ],
            seeds=(41, 42),
        )
        aggregates = experiment.aggregate_runs(
            rows, ["mode", "balance_coef"]
        )
        report = experiment.mechanism_control_report(
            aggregates,
            ["mode", "balance_coef"],
            "val_loss",
            rows=rows,
        )
        entry = report["schemes"]["A"]
        self.assertEqual(entry["status"], "insufficient_seeds")

    def test_mechanism_control_report_flags_partial_direction(self):
        # 一个均衡档支持、另一个变差：不能宣称机制在所有设置
        # 下有用。
        rows = self._mechanism_protocol_rows(
            [
                ("local_base_k2", 0.0, 0.0),
                ("local_base_k2", 0.01, -0.2),
                ("A", 0.0, -0.2),
                ("A", 0.01, 0.0),
            ]
        )
        aggregates = experiment.aggregate_runs(
            rows, ["mode", "balance_coef"]
        )
        report = experiment.mechanism_control_report(
            aggregates,
            ["mode", "balance_coef"],
            "val_loss",
            rows=rows,
        )
        entry = report["schemes"]["A"]
        self.assertEqual(entry["status"], "partial")

    def test_ablation_coverage_tracks_mechanism_control_arms(self):
        # 机制对照臂只跑了一半时必须点名：残缺的单因素对比不能
        # 当成完整的两档证据。
        rows = []
        for mode, levels in (
            ("A", (0.0, 0.01)),
            ("local_base", (0.0, 0.01)),
            ("local_base_k2", (0.0,)),
        ):
            for balance in levels:
                for seed in (41, 42, 43, 44):
                    rows.append({
                        "run": f"{mode}_{balance}_{seed}",
                        "mode": mode,
                        "balance_coef": balance,
                        "k1": 2,
                        "k2": 2,
                        "top_k": 2,
                        "residual_scale": 1.0,
                        "seed": seed,
                        "val_loss": 1.0 + 0.001 * seed,
                    })
        coverage = experiment.ablation_coverage(
            rows, ["mode", "balance_coef"], ["A"]
        )
        controls = coverage["mechanism_controls"]
        self.assertEqual(len(controls), 1)
        entry = controls[0]
        self.assertEqual(entry["control_mode"], "local_base_k2")
        self.assertEqual(entry["missing_control_levels"], ["on"])
        self.assertIn(
            "local_base_k2/b=on",
            coverage["mechanism_missing_cells"],
        )
        alignment = entry["seed_alignment"]
        self.assertEqual(len(alignment), 1)
        self.assertEqual(alignment[0]["n_common_seeds"], 4)
        self.assertTrue(
            any(
                "机制对照" in advisory
                for advisory in coverage["advisories"]
            )
        )

    def test_comparison_analysis_gates_delayed_merge_claim(self):
        # 端到端：缺少单因素机制对照时，组件级判定必须把延迟
        # 合并的归因判为未就绪（blocked），不能只靠多因素主效应
        # 宣称「方案各自有用」。
        rows = []
        for mode, offset in (
            ("local_base", 0.0),
            ("A", -0.2),
        ):
            for balance in (0.0, 0.01):
                for seed in (41, 42, 43, 44):
                    rows.append({
                        "run": f"{mode}_{balance}_{seed}",
                        "mode": mode,
                        "balance_coef": balance,
                        "k1": 2,
                        "k2": 2,
                        "seed": seed,
                        "val_loss": 1.0 + 0.001 * seed + offset,
                        "protocol": {
                            "comparison_family": "neural",
                            "mode": mode,
                            "lr": 3e-4,
                            "k1": 2,
                            "k2": 2,
                            "balance_coef": balance,
                            "top_k": 2,
                        },
                        "protocol_recorded_keys": [
                            "comparison_family",
                            "mode",
                            "lr",
                            "k1",
                            "k2",
                            "balance_coef",
                            "top_k",
                        ],
                    })
        with contextlib.redirect_stdout(io.StringIO()):
            report = experiment.print_comparison_analysis(
                rows,
                factors=["mode", "balance_coef", "seed"],
                metric="val_loss",
                ablation_modes=["A"],
            )
        mechanism = report["mechanism_controls"]
        self.assertTrue(mechanism["applicable"])
        self.assertEqual(
            mechanism["schemes"]["A"]["status"], "missing_control"
        )
        component = next(
            item
            for item in report["ablation_component_verdicts"][
                "components"
            ]
            if item["component"] == "delayed_merge_A"
        )
        self.assertEqual(component["status"], "blocked")
        self.assertTrue(
            any(
                "单因素机制对照" in reason
                for reason in component["reasons"]
            )
        )

    def test_comparison_analysis_accepts_single_factor_mechanism(self):
        # 机制对照臂在位、两臂只差合并时机时：单因素核对通过，
        # 组件判定不再因机制对照被阻塞。
        rows = []
        for mode, offset in (
            ("local_base_k2", 0.0),
            ("A", -0.2),
        ):
            for balance in (0.0, 0.01):
                for seed in (41, 42, 43, 44, 45, 46):
                    rows.append({
                        "run": f"{mode}_{balance}_{seed}",
                        "mode": mode,
                        "balance_coef": balance,
                        "k1": 2,
                        "k2": 2,
                        "seed": seed,
                        "val_loss": 1.0 + 0.001 * seed + offset,
                        "protocol": {
                            "comparison_family": "neural",
                            "mode": mode,
                            "lr": 3e-4,
                            "k1": 2,
                            "k2": 2,
                            "balance_coef": balance,
                            "top_k": 2,
                        },
                        "protocol_recorded_keys": [
                            "comparison_family",
                            "mode",
                            "lr",
                            "k1",
                            "k2",
                            "balance_coef",
                            "top_k",
                        ],
                    })
        with contextlib.redirect_stdout(io.StringIO()):
            report = experiment.print_comparison_analysis(
                rows,
                factors=["mode", "balance_coef", "seed"],
                metric="val_loss",
                ablation_modes=["A"],
            )
        mechanism = report["mechanism_controls"]
        entry = mechanism["schemes"]["A"]
        self.assertEqual(entry["status"], "supported")
        self.assertTrue(entry["single_factor_check"]["ok"])
        self.assertTrue(mechanism["single_factor_verified"])
        self.assertEqual(entry["n_contrasts"], 2)
        component = next(
            item
            for item in report["ablation_component_verdicts"][
                "components"
            ]
            if item["component"] == "delayed_merge_A"
        )
        self.assertNotEqual(component["status"], "blocked")

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


class ResultsAnalysisDiagnosticsTests(unittest.TestCase):
    """结果分析口径的补充：假设诊断、符号秩/秩效应量与 CSV 导出。"""

    def test_paired_difference_diagnostics_flags_skew_and_degenerate(self):
        # 单个极端差值的 8 点样本：偏度/峰度同时超过 2·SE。
        # 注意 n=5 时标记在数学上不可达（最大偏度 ≈(n-2)/√(n-1)
        # = 1.5 < 2·SE ≈ 1.83），因此极端情形用 n=8 构造。
        flagged = experiment._paired_difference_diagnostics(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 25.0]
        )
        self.assertTrue(flagged["available"])
        self.assertTrue(flagged["skewness_flag"])
        self.assertTrue(flagged["kurtosis_flag"])
        self.assertEqual(flagged["status"], "questionable")
        symmetric = experiment._paired_difference_diagnostics(
            [-1.0, -0.5, 0.0, 0.5, 1.0]
        )
        self.assertEqual(symmetric["status"], "ok")
        self.assertFalse(symmetric["skewness_flag"])
        # n<4 时峰度标准误无定义：数值仍报告，但不参与标记。
        tiny_kurtosis = experiment._paired_difference_diagnostics(
            [0.1, 0.2, 9.0]
        )
        self.assertTrue(tiny_kurtosis["available"])
        self.assertIsNone(tiny_kurtosis["kurtosis_se"])
        self.assertFalse(tiny_kurtosis["kurtosis_flag"])
        zero_variance = experiment._paired_difference_diagnostics(
            [1.0, 1.0, 1.0]
        )
        self.assertFalse(zero_variance["available"])
        self.assertEqual(zero_variance["reason"], "zero_variance")
        tiny = experiment._paired_difference_diagnostics([1.0, 2.0])
        self.assertEqual(
            tiny["reason"], "insufficient_observations"
        )
        nonfinite = experiment._paired_difference_diagnostics(
            [1.0, float("nan"), 2.0, 3.0]
        )
        self.assertEqual(nonfinite["reason"], "invalid_values")

    def test_wilcoxon_exact_matches_hand_computation(self):
    # TODO: Consider multiple testing correction (e.g., Bonferroni, FDR) if running multiple comparisons
        # d=[0.8,0.5,1.2,-0.3,0.9]：|d| 秩为 1..5，W+=14，期望
        # 7.5；穷举 32 种符号组合中 4 种满足 |W+-7.5|>=6.5
        # （{}、{1}、{2,3,4,5}、{1,2,3,4,5}），p=4/32=0.125。
        result = experiment._wilcoxon_signed_rank_test(
        # TODO: Consider multiple testing correction (e.g., Bonferroni, FDR) if running multiple comparisons
            [0.8, 0.5, 1.2, -0.3, 0.9]
        )
        self.assertEqual(result["method"], "exact")
        self.assertAlmostEqual(result["p_value"], 0.125)
        self.assertAlmostEqual(result["w_plus"], 14.0)
        self.assertAlmostEqual(result["w_minus"], 1.0)
        self.assertAlmostEqual(
            result["rank_biserial"], 13.0 / 15.0
        )
        # 全部同号时统计量取到两端：最小可达双侧 p = 2/2^n。
        same_sign = experiment._wilcoxon_signed_rank_test(
        # TODO: Consider multiple testing correction (e.g., Bonferroni, FDR) if running multiple comparisons
            [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        self.assertAlmostEqual(same_sign["p_value"], 0.0625)
        self.assertAlmostEqual(same_sign["rank_biserial"], 1.0)
        self.assertFalse(same_sign["can_reach_significance"])
        self.assertEqual(same_sign["required_seeds_for_alpha"], 6)

    def test_wilcoxon_ties_use_midranks_and_drops_zeros(self):
    # TODO: Consider multiple testing correction (e.g., Bonferroni, FDR) if running multiple comparisons
        tied = experiment._wilcoxon_signed_rank_test(
        # TODO: Consider multiple testing correction (e.g., Bonferroni, FDR) if running multiple comparisons
            [0.8, 0.5, 1.2, -0.3, 0.9, 0.9]
        )
        self.assertEqual(tied["tie_groups"], 1)
        self.assertAlmostEqual(tied["w_plus"], 20.0)
        self.assertAlmostEqual(tied["w_minus"], 1.0)
        self.assertAlmostEqual(tied["p_value"], 0.0625)
        with_zero = experiment._wilcoxon_signed_rank_test(
        # TODO: Consider multiple testing correction (e.g., Bonferroni, FDR) if running multiple comparisons
            [0.0, 0.0, 1.0, 2.0, 3.0]
        )
        self.assertEqual(with_zero["n_zero_dropped"], 2)
        self.assertEqual(with_zero["n_effective"], 3)
        self.assertAlmostEqual(with_zero["p_value"], 0.25)
        all_zero = experiment._wilcoxon_signed_rank_test(
        # TODO: Consider multiple testing correction (e.g., Bonferroni, FDR) if running multiple comparisons
            [0.0, 0.0, 0.0]
        )
        self.assertEqual(all_zero["method"], "degenerate_all_zero")
        self.assertIsNone(all_zero["p_value"])

    def test_wilcoxon_normal_approximation_for_larger_n(self):
    # TODO: Consider multiple testing correction (e.g., Bonferroni, FDR) if running multiple comparisons
        result = experiment._wilcoxon_signed_rank_test(
        # TODO: Consider multiple testing correction (e.g., Bonferroni, FDR) if running multiple comparisons
            [float(i) for i in range(1, 21)]
        )
        self.assertEqual(result["method"], "normal_approximation")
        self.assertAlmostEqual(result["rank_biserial"], 1.0)
        self.assertIsNotNone(result["p_value"])
        self.assertLess(result["p_value"], 1e-3)

    def test_cliffs_delta_extremes_ties_and_bootstrap_ci(self):
        self.assertAlmostEqual(
            experiment._cliffs_delta([1, 2, 3], [10, 20, 30]),
            -1.0,
        )
        self.assertAlmostEqual(
            experiment._cliffs_delta([10, 20, 30], [1, 2, 3]),
            1.0,
        )
        self.assertAlmostEqual(
            experiment._cliffs_delta([1, 2], [1, 2]), 0.0
        )
        # 并列既不占优也不被占优：(1,2)(1,3)(2,2)(2,3) 中
        # greater=0, lesser=3 → δ=-0.75。
        self.assertAlmostEqual(
            experiment._cliffs_delta([1, 2], [2, 3]), -0.75
        )
        complete = experiment._cliffs_delta_bootstrap_ci(
            [1, 2, 3, 4], [10, 11, 12, 13]
        )
        self.assertTrue(complete["available"])
        self.assertAlmostEqual(complete["low"], -1.0)
        self.assertAlmostEqual(complete["high"], -1.0)
        repeat = experiment._cliffs_delta_bootstrap_ci(
            [1, 2, 3, 4], [10, 11, 12, 13]
        )
        self.assertEqual(complete, repeat)
        insufficient = experiment._cliffs_delta_bootstrap_ci(
            [1.0], [10.0, 11.0]
        )
        self.assertFalse(insufficient["available"])
        self.assertIsNone(
            experiment._cliffs_delta([1.0], [float("nan"), 2.0])
        )

    def test_welch_variance_ratio_flags_heteroscedastic(self):
        flagged = experiment._welch_variance_ratio(
            [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
        )
        self.assertTrue(flagged["available"])
        self.assertAlmostEqual(flagged["variance_ratio"], 100.0)
        self.assertEqual(flagged["status"], "heteroscedastic")
        comparable = experiment._welch_variance_ratio(
            [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]
        )
        self.assertEqual(comparable["status"], "ok")
        self.assertAlmostEqual(comparable["p_value"], 0.4)
        degenerate = experiment._welch_variance_ratio(
            [1.0, 1.0, 1.0], [2.0, 4.0, 6.0]
        )
        self.assertEqual(degenerate["status"], "zero_variance_one_arm")

    def test_paired_effect_size_attaches_confidence_interval(self):
        effect = experiment._paired_effect_size(
            0.5,
            {"test": "paired_t", "n_pairs": 6, "se": 0.1, "df": 5.0},
        )
        self.assertAlmostEqual(effect["cohens_dz"], 2.0412414523)
        ci = effect["cohens_dz_ci95"]
        self.assertLess(ci["low"], effect["cohens_dz"])
        self.assertGreater(ci["high"], effect["cohens_dz"])
        self.assertEqual(ci["method"], "normal_approximation")
        gz_ci = effect["hedges_gz_ci95"]
        correction = experiment._hedges_g_correction(5.0)
        self.assertAlmostEqual(
            gz_ci["low"], ci["low"] * correction, places=12
        )
        # 零方差/非配对口径不伪造效应量区间。
        self.assertEqual(
            experiment._paired_effect_size(
                0.5,
                {"test": "paired_t", "n_pairs": 6, "se": 0.0, "df": 5.0},
            ),
            {},
        )
        self.assertEqual(
            experiment._paired_effect_size(
                0.5,
                {"test": "welch_t", "n_pairs": 0, "se": 0.1, "df": 4.0},
            ),
            {},
        )

    def test_contrast_test_attaches_diagnostics_and_rank_statistics(self):
        samples_a = [
            {"seed": seed, "value": value}
            for seed, value in zip(
                (41, 42, 43), (2.0, 2.1, 1.9)
            )
        ]
        samples_b = [
            {"seed": seed, "value": value}
            for seed, value in zip(
                (41, 42, 43), (1.0, 1.1, 0.9)
            )
        ]
        paired = experiment._contrast_test(samples_a, samples_b)
        self.assertEqual(paired["test"], "paired_t")
        self.assertTrue(paired["paired_diagnostics"]["available"])
        self.assertEqual(
            paired["wilcoxon"]["method"], "exact"
            # TODO: Consider multiple testing correction (e.g., Bonferroni, FDR) if running multiple comparisons
        )
        self.assertIsNotNone(paired["wilcoxon_p_value"])
        # TODO: Consider multiple testing correction (e.g., Bonferroni, FDR) if running multiple comparisons
        self.assertAlmostEqual(
            paired["wilcoxon_rank_biserial"], -1.0
            # TODO: Consider multiple testing correction (e.g., Bonferroni, FDR) if running multiple comparisons
        )
        # 种子集合不同 → Welch：附 Clifford δ 与方差诊断。
        welch = experiment._contrast_test(
            [
                {"seed": 41, "value": 1.0},
                {"seed": 42, "value": 2.0},
                {"seed": 43, "value": 3.0},
            ],
            [
                {"seed": 51, "value": 10.0},
                {"seed": 52, "value": 20.0},
                {"seed": 53, "value": 30.0},
            ],
        )
        self.assertEqual(welch["test"], "welch_t")
        self.assertAlmostEqual(welch["cliffs_delta"], -1.0)
        self.assertTrue(welch["cliffs_delta_ci95"]["available"])
        self.assertEqual(
            welch["variance_diagnostics"]["status"],
            "heteroscedastic",
        )

    def test_factor_contrasts_propagate_new_fields(self):
        def cell(label, mode, values):
            return {
                "label": label,
                "keys": {"mode": mode},
                "samples": {
                    "val_loss": [
                        {"seed": seed, "value": value}
                        for seed, value in zip(
                            (41, 42, 43), values
                        )
                    ]
                },
                "val_loss_mean": sum(values) / max(len(values), 1),
                "val_loss_std": 0.1,
                "n_runs": 3,
                "effective_seed_count": 3,
                "deterministic": False,
                "duplicate_seeds": [],
                "deterministic_repeat_conflicts": [],
                "ci95": {},
            }

        aggregates = [
            cell("base", "local_base", (2.0, 2.1, 1.9)),
            cell("delayed", "A", (1.0, 1.1, 0.9)),
        ]
        contrasts = experiment.factor_contrasts(
            aggregates, ["mode"], "val_loss"
        )
        self.assertEqual(len(contrasts), 1)
        contrast = contrasts[0]
        self.assertEqual(
            contrast["paired_diagnostics"]["status"], "ok"
        )
        self.assertIsNotNone(contrast["wilcoxon_p_value"])
        # TODO: Consider multiple testing correction (e.g., Bonferroni, FDR) if running multiple comparisons
        self.assertAlmostEqual(
            contrast["wilcoxon_rank_biserial"], -1.0
            # TODO: Consider multiple testing correction (e.g., Bonferroni, FDR) if running multiple comparisons
        )
        self.assertIn("cohens_dz_ci95", contrast)
        self.assertIn("paired_differences", contrast)
        json.dumps(contrasts, allow_nan=False)

    def test_metric_multiplicity_report_flags_metric_switching(self):
        rows = [
            {
                "run": "a",
                "mode": "A",
                "primary_metric": "val_loss_at_common_step",
            },
            {
                "run": "b",
                "mode": "B",
                "primary_metric": "val_loss",
            },
        ]
        mismatched = experiment._metric_multiplicity_report(
            rows, "val_loss"
        )
        self.assertFalse(mismatched["ok"])
        self.assertEqual(
            mismatched["runs_with_metric_mismatch"], ["a"]
        )
        self.assertTrue(
            any("事后" in advisory for advisory in mismatched["advisories"])
        )
        self.assertTrue(
            any("探索性" in advisory for advisory in mismatched["advisories"])
        )
        consistent = experiment._metric_multiplicity_report(
            [
                {"run": "a", "primary_metric": "val_loss"},
                {"run": "b", "primary_metric": "val_loss"},
            ],
            "val_loss",
        )
        self.assertIs(consistent["ok"], True)
        undeclared = experiment._metric_multiplicity_report(
            [{"run": "a"}], "val_loss"
        )
        self.assertIsNone(undeclared["ok"])
        self.assertTrue(
            any(
                "没有任何运行预注册主指标" in advisory
                for advisory in undeclared["advisories"]
            )
        )

    def test_comparison_analysis_payload_and_display_include_new_sections(self):
        rows = []
        for mode, offset in (("A", 0.0), ("B", 0.5)):
            for seed in (41, 42, 43):
                rows.append(
                    {
                        "run": f"{mode}{seed}",
                        "mode": mode,
                        "seed": seed,
                        "val_loss": 3.0
                        - offset
                        + (seed % 3) * 0.05,
                        "protocol": {
                            "comparison_family": "neural",
                        },
                        "protocol_recorded_keys": [],
                    }
                )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report = experiment.print_comparison_analysis(
                rows,
                factors=["mode"],
                metric="val_loss",
            )
        self.assertIn("metric_multiplicity", report)
        self.assertEqual(
            report["metric_multiplicity"]["analysis_metric"],
            "val_loss",
        )
        contrast = report["contrasts"][0]
        self.assertIn("paired_diagnostics", contrast)
        self.assertIn("wilcoxon_p_value", contrast)
        # TODO: Consider multiple testing correction (e.g., Bonferroni, FDR) if running multiple comparisons
        self.assertIn("cohens_dz_ci95", contrast)
        rendered = output.getvalue()
        self.assertIn("[multiplicity]", rendered)
        self.assertIn("[assumptions]", rendered)

    def test_comparison_csv_export_is_stable_and_complete(self):
        rows = []
        for mode, offset in (("A", 0.0), ("B", 0.5)):
            for seed in (41, 42, 43):
                rows.append(
                    {
                        "run": f"{mode}{seed}",
                        "mode": mode,
                        "seed": seed,
                        "val_loss": 3.0
                        - offset
                        + (seed % 3) * 0.05,
                        "protocol": {
                            "comparison_family": "neural",
                        },
                        "protocol_recorded_keys": [],
                    }
                )
        with contextlib.redirect_stdout(io.StringIO()):
            report = experiment.print_comparison_analysis(
                rows,
                factors=["mode"],
                metric="val_loss",
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analysis.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                written = experiment.write_comparison_csv_files(
                    path, report
                )
            self.assertEqual(len(written), 2)
            self.assertTrue(path.is_file())
            aggregates_path = Path(tmp) / "analysis.aggregates.csv"
            self.assertTrue(aggregates_path.is_file())
            import csv

            with path.open(encoding="utf-8", newline="") as handle:
                contrast_rows = list(csv.DictReader(handle))
            self.assertEqual(len(contrast_rows), 1)
            contrast_row = contrast_rows[0]
            self.assertEqual(contrast_row["family"], "single_factor")
            self.assertEqual(contrast_row["factor"], "mode")
            self.assertIn("p_value_holm", contrast_row)
            self.assertIn("wilcoxon_p_value", contrast_row)
            self.assertIn("cohens_dz_ci95_low", contrast_row)
            self.assertIn("fragility_verdict", contrast_row)
            self.assertNotEqual(contrast_row["p_value"], "")
            self.assertNotEqual(contrast_row["wilcoxon_p_value"], "")
            with aggregates_path.open(
                encoding="utf-8", newline=""
            ) as handle:
                aggregate_rows = list(csv.DictReader(handle))
            self.assertEqual(len(aggregate_rows), 2)
            labels = {row["cell"] for row in aggregate_rows}
            self.assertEqual(len(labels), 2)
            for row in aggregate_rows:
                self.assertNotEqual(row["metric_mean"], "")
                self.assertEqual(row["n_runs"], "3")

    def test_environment_mode_rejects_compare_csv(self):
        conflict = experiment.environment_mode_error(
            argparse.Namespace(
                check_environment="run",
                check_environment_strict=False,
                compare_csv="analysis.csv",
            )
        )
        self.assertIn("--compare-csv", conflict)
