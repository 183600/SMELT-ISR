"""结果分析稳健性扩展的回归测试。

覆盖本轮新增的结果分析能力：

- 跨指标一致性：方向统一折算后能识别「显著指标方向相反」的
  分歧；综合秩效应量按**同源指标族**汇总，不把互为单调变换的
  指标当成多份独立证据；
- 机器可读裁决：脆弱显著性、精度不足、越界等价性分别落到
  正确的裁决与下一步建议；
- 精度/效率联合判定：成本 ≥1.5× 且增益/成本区间跨 1 时标记
  成本无效率；
- 路由专家分离的 BH-FDR：运行内 E 个专家为一个族，跨种子
  一致性才判定为共识专门化；
- 评估集规模敏感性与外推风险标记；
- 扩展字段随对比分析落盘，并出现在 CSV 与 Markdown 摘要中。

Run with: python -m unittest discover -s tests -v
"""

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


def _cell(label, mode, samples, **extra):
    """构造一个最小实验单元：samples 为 {metric: [值列表]}。"""
    sample_map = {}
    means = {}
    for metric, values in samples.items():
        sample_map[metric] = [
            {"seed": seed, "value": value}
            for seed, value in zip((41, 42, 43), values, strict=False)
        ]
        means[f"{metric}_mean"] = sum(values) / max(len(values), 1)
    cell = {
        "label": label,
        "keys": {"mode": mode},
        "samples": sample_map,
        "n_runs": 3,
        "effective_seed_count": 3,
        "deterministic": False,
        "duplicate_seeds": [],
        "deterministic_repeat_conflicts": [],
        "ci95": {},
        "costs": {},
    }
    cell.update(means)
    cell.update(extra)
    return cell


class CrossMetricConsistencyTests(unittest.TestCase):
    def test_divergence_is_detected_after_direction_normalisation(self):
        # 变体在 loss 上更好、在 accuracy 上更差：两个指标都显著，
        # 方向相反，必须标成 metric-divergence。
        aggregates = [
            _cell(
                "base",
                "local_base",
                {
                    "val_loss": (2.0, 2.1, 1.9),
                    "val_token_accuracy": (0.70, 0.71, 0.69),
                },
            ),
            _cell(
                "delayed",
                "A",
                {
                    "val_loss": (1.0, 1.1, 0.9),
                    "val_token_accuracy": (0.50, 0.51, 0.49),
                },
            ),
        ]
        contrasts = experiment.factor_contrasts(
            aggregates, ["mode"], "val_loss"
        )
        contrast = contrasts[0]
        self.assertTrue(contrast["cross_metric_divergent"])
        self.assertFalse(contrast["cross_metric_consistent"])
        self.assertEqual(
            contrast["cross_metric_summary"]["marker"],
            "[metric-divergence]",
        )
        directions = contrast["metric_directions"]
        self.assertEqual(directions["val_loss"], "improves")
        self.assertEqual(directions["val_token_accuracy"], "degrades")
        self.assertIn("loss", contrast["cross_metric_summary"]["family_directions"])
        self.assertIn(
            "accuracy",
            contrast["cross_metric_summary"]["family_directions"],
        )

    def test_consistent_metrics_are_not_flagged_as_divergence(self):
        aggregates = [
            _cell(
                "base",
                "local_base",
                {
                    "val_loss": (2.0, 2.1, 1.9),
                    "val_token_accuracy": (0.50, 0.51, 0.49),
                },
            ),
            _cell(
                "delayed",
                "A",
                {
                    "val_loss": (1.0, 1.1, 0.9),
                    "val_token_accuracy": (0.70, 0.71, 0.69),
                },
            ),
        ]
        contrasts = experiment.factor_contrasts(
            aggregates, ["mode"], "val_loss"
        )
        contrast = contrasts[0]
        self.assertFalse(contrast["cross_metric_divergent"])
        self.assertTrue(contrast["cross_metric_consistent"])
        composite = contrast["cross_metric_summary"][
            "composite_cliffs_delta"
        ]
        self.assertGreater(composite, 0.0)

    def test_composite_effect_counts_metric_families_not_metrics(self):
        # val_loss / val_ppl 互为单调变换，属于同一个指标族：
        # 综合效应量按族汇总，不能把它们当成两份独立证据。
        aggregates = [
            _cell(
                "base",
                "local_base",
                {
                    "val_loss": (2.0, 2.1, 1.9),
                    "val_ppl": (4.0, 4.1, 3.9),
                    "val_macro_f1": (0.50, 0.51, 0.49),
                },
            ),
            _cell(
                "delayed",
                "A",
                {
                    "val_loss": (1.0, 1.1, 0.9),
                    "val_ppl": (2.0, 2.1, 1.9),
                    "val_macro_f1": (0.70, 0.71, 0.69),
                },
            ),
        ]
        contrasts = experiment.factor_contrasts(
            aggregates, ["mode"], "val_loss"
        )
        profile = contrasts[0]["cross_metric"]
        families = {entry["family"] for entry in profile["families"]}
        self.assertIn("loss", families)
        self.assertIn("accuracy", families)
        # 三个指标只归入两个族：综合效应量的族数不能等于指标数。
        self.assertEqual(
            contrasts[0]["cross_metric_summary"]["composite_n_families"],
            2,
        )


class ResultVerdictTests(unittest.TestCase):
    def _contrast(self, **overrides):
        contrast = {
            "baseline": "base",
            "variant": "variant",
            "significant": True,
            "equivalent": False,
            "delta": -0.5,
            "power_at_effect": 0.95,
            "fragility": {"verdict": "stable"},
            "attrition_bounds": {
                "verdict": "complete_cases",
                "sign_robust": True,
            },
            "equivalence_blocker": None,
        }
        contrast.update(overrides)
        return contrast

    def test_fragile_significance_is_inconclusive(self):
        verdict = experiment._result_verdict(
            self._contrast(fragility={"verdict": "fragile"})
        )
        self.assertEqual(verdict["verdict"], "inconclusive_fragile")
        self.assertEqual(verdict["action"], "add_seeds")

    def test_attrition_crossing_zero_downgrades_significance(self):
        verdict = experiment._result_verdict(
            self._contrast(
                attrition_bounds={
                    "verdict": "crosses_zero",
                    "sign_robust": False,
                }
            )
        )
        self.assertEqual(verdict["verdict"], "inconclusive_attrition")
        self.assertEqual(verdict["action"], "complete_runs")

    def test_underpowered_significance_is_reported(self):
        verdict = experiment._result_verdict(
            self._contrast(power_at_effect=0.4)
        )
        self.assertEqual(verdict["verdict"], "significant_underpowered")
        self.assertEqual(verdict["action"], "add_seeds")

    def test_effect_beyond_margin_rejects_equivalence(self):
        verdict = experiment._result_verdict(
            self._contrast(
                significant=False,
                equivalence_blocker="effect_exceeds_margin",
            )
        )
        self.assertEqual(verdict["verdict"], "reject_equivalence")

    def test_insufficient_precision_suggests_more_seeds(self):
        verdict = experiment._result_verdict(
            self._contrast(
                significant=False,
                equivalence_blocker="insufficient_precision",
            )
        )
        self.assertEqual(verdict["verdict"], "inconclusive_precision")
        self.assertEqual(verdict["action"], "add_seeds")

    def test_confounded_contrast_is_never_interpreted(self):
        verdict = experiment._result_verdict(
            self._contrast(confounded=True, differing_factors=["mode", "lr"])
        )
        self.assertEqual(verdict["verdict"], "confounded")
        self.assertEqual(verdict["action"], "fix_protocol")

    def test_evaluation_window_dominance_recommends_expanding_eval_set(self):
        verdict = experiment._result_verdict(
            self._contrast(
                significant=False,
                power_at_effect=0.4,
                window_paired={
                    "variance_components": {
                        "dominant_source": "evaluation_windows"
                    }
                },
            )
        )
        self.assertEqual(verdict["action"], "expand_eval_set")


class CostEfficiencyTests(unittest.TestCase):
    def test_material_cost_with_unpaid_gain_is_flagged(self):
        # 成本 2×（额外成本 1×）而相对增益只有 5%：增益/额外成本
        # = 0.05，远小于 1，额外代价明确没有赚回。
        contrast = {
            "mean_baseline": 2.0,
            "mean_variant": 1.9,
            "delta": -0.1,
            "ci95_low": -0.15,
            "ci95_high": -0.05,
            "cost": {
                "ratios": {"flops_per_forward": 2.0},
                "uncertainty": {
                    "flops_per_forward": {
                        "ratio_ci95": [1.2, 3.0],
                    }
                },
            },
        }
        assessment = experiment._cost_efficiency_assessment(contrast)
        self.assertTrue(assessment["available"])
        self.assertTrue(assessment["material_cost"])
        self.assertTrue(assessment["cost_inefficient"])
        self.assertFalse(assessment["cost_unproven"])
        self.assertEqual(assessment["marker"], "[cost-inefficient]")

    def test_uncertain_gain_at_material_cost_is_unproven(self):
        # 效应区间跨过 0，且成本倍率区间下端接近 1：增益/额外成本
        # 区间跨过 1 → 无法证明额外代价被赚回，但也不能说它一定亏。
        contrast = {
            "mean_baseline": 2.0,
            "mean_variant": 1.9,
            "delta": -0.1,
            "ci95_low": -0.3,
            "ci95_high": 0.1,
            "cost": {
                "ratios": {"flops_per_forward": 2.0},
                "uncertainty": {
                    "flops_per_forward": {
                        "ratio_ci95": [1.05, 3.0],
                    }
                },
            },
        }
        assessment = experiment._cost_efficiency_assessment(contrast)
        self.assertFalse(assessment["cost_inefficient"])
        self.assertTrue(assessment["cost_unproven"])
        self.assertEqual(assessment["marker"], "[cost-unproven]")

    def test_proportional_gain_pays_for_extra_cost(self):
        # 相对增益 75%、额外成本 50%：1:1 口径下额外代价被赚回。
        contrast = {
            "mean_baseline": 2.0,
            "mean_variant": 0.5,
            "delta": -1.5,
            "ci95_low": -1.6,
            "ci95_high": -1.4,
            "cost": {
                "ratios": {"flops_per_forward": 1.5},
                "uncertainty": {
                    "flops_per_forward": {
                        "ratio_ci95": [1.4, 1.6],
                    }
                },
            },
        }
        assessment = experiment._cost_efficiency_assessment(contrast)
        self.assertFalse(assessment["cost_inefficient"])
        self.assertFalse(assessment["cost_unproven"])
        self.assertIsNone(assessment["marker"])

    def test_cheaper_variant_has_no_tradeoff(self):
        contrast = {
            "mean_baseline": 2.0,
            "mean_variant": 1.0,
            "delta": -1.0,
            "ci95_low": -1.2,
            "ci95_high": -0.8,
            "cost": {
                "ratios": {"flops_per_forward": 0.9},
                "uncertainty": {
                    "flops_per_forward": {
                        "ratio_ci95": [0.85, 0.95],
                    }
                },
            },
        }
        assessment = experiment._cost_efficiency_assessment(contrast)
        self.assertFalse(assessment["cost_inefficient"])
        self.assertFalse(assessment["cost_unproven"])
        self.assertIsNone(assessment["marker"])
        self.assertIsNone(assessment["gain_per_cost"])

    def test_small_extra_cost_is_not_a_material_tradeoff(self):
        contrast = {
            "mean_baseline": 2.0,
            "mean_variant": 1.0,
            "delta": -1.0,
            "ci95_low": -1.2,
            "ci95_high": -0.8,
            "cost": {
                "ratios": {"flops_per_forward": 1.1},
                "uncertainty": {
                    "flops_per_forward": {
                        "ratio_ci95": [1.05, 1.15],
                    }
                },
            },
        }
        assessment = experiment._cost_efficiency_assessment(contrast)
        self.assertFalse(assessment["material_cost"])
        self.assertFalse(assessment["cost_inefficient"])
        self.assertFalse(assessment["cost_unproven"])
        self.assertIsNone(assessment["marker"])

    def test_effect_interval_is_recovered_from_se_and_df(self):
        # 对比载荷常规只有 delta/se/df：效应区间必须能从这三项
        # 还原，否则成本判定永远拿不到区间。
        contrast = {
            "mean_baseline": 2.0,
            "mean_variant": 1.9,
            "delta": -0.1,
            "se": 0.05,
            "df": 2.0,
            "cost": {
                "ratios": {"flops_per_forward": 2.0},
                "uncertainty": {
                    "flops_per_forward": {
                        "ratio_ci95": [1.8, 2.2],
                    }
                },
            },
        }
        assessment = experiment._cost_efficiency_assessment(contrast)
        self.assertIsNotNone(assessment["gain_per_cost_ci95"])
        self.assertTrue(assessment["cost_inefficient"])

    def test_missing_cost_measurement_is_reported_as_unavailable(self):
        assessment = experiment._cost_efficiency_assessment({})
        self.assertFalse(assessment["available"])


class RoutingSpecializationFdrTests(unittest.TestCase):
    def _row(self, seed, deltas):
        per_expert = {}
        for index, (delta, half_width) in enumerate(deltas):
            per_expert[str(index)] = {
                "loss_delta": delta,
                "loss_delta_ci95": {
                    "low": delta - half_width,
                    "high": delta + half_width,
                },
            }
        return {
            "run": f"A{seed}",
            "mode": "A",
            "seed": seed,
            "routing_interventions": {
                "interventions": {"expert_drop": {"per_expert": per_expert}}
            },
        }

    def test_bh_correction_runs_within_each_run(self):
        rows = [
            self._row(
                seed,
                [(0.6, 0.05)] + [(0.01, 0.5)] * 7,
            )
            for seed in (41, 42, 43)
        ]
        report = experiment._routing_specialization_fdr(rows, ["mode"])
        self.assertTrue(report["available"])
        cell = report["cells"][0]
        self.assertEqual(cell["n_runs"], 3)
        self.assertEqual(cell["n_experts"], 8)
        self.assertEqual(cell["n_consensus_specialized_experts"], 1)
        self.assertEqual(cell["consensus_specialized_experts"], ["0"])
        # 非显著专家的校正 p 值不能被算成显著。
        expert_rows = {entry["expert"]: entry for entry in cell["experts"]}
        self.assertEqual(expert_rows["1"]["n_fdr_significant"], 0)
        self.assertFalse(expert_rows["1"]["consensus_specialized"])

    def test_single_run_cannot_establish_consensus(self):
        rows = [self._row(41, [(0.6, 0.05)] + [(0.01, 0.5)] * 3)]
        report = experiment._routing_specialization_fdr(rows, ["mode"])
        cell = report["cells"][0]
        self.assertEqual(cell["n_consensus_specialized_experts"], 0)

    def test_missing_intervention_data_is_not_invented(self):
        report = experiment._routing_specialization_fdr(
            [{"run": "x", "mode": "A", "seed": 1}], ["mode"]
        )
        self.assertFalse(report["available"])
        self.assertEqual(report["cells"], [])


class WindowSizeSensitivityTests(unittest.TestCase):
    def test_subset_flipping_direction_is_sensitive(self):
        per_seed = {
            41: [-1.0] * 8 + [5.0] * 2,
            42: [-1.0] * 8 + [5.0] * 2,
            43: [-1.0] * 8 + [5.0] * 2,
        }
        report = experiment._window_size_sensitivity_analysis(per_seed)
        self.assertTrue(report["available"])
        self.assertEqual(report["verdict"], "sensitive")
        self.assertTrue(report["direction_flipped_subsets"])

    def test_stable_windows_are_reported_as_stable(self):
        per_seed = {
            seed: [-1.0 + 0.01 * index for index in range(10)]
            for seed in (41, 42, 43)
        }
        report = experiment._window_size_sensitivity_analysis(per_seed)
        self.assertTrue(report["available"])
        self.assertEqual(report["verdict"], "stable")
        self.assertEqual(report["direction_flipped_subsets"], [])

    def test_insufficient_data_does_not_fabricate_a_verdict(self):
        report = experiment._window_size_sensitivity_analysis({41: [1.0]})
        self.assertFalse(report["available"])


class ExtrapolationRiskTests(unittest.TestCase):
    def test_unseen_label_share_above_threshold_is_flagged(self):
        rows = [
            {
                "run": "A41",
                "label_coverage": {
                    "val": {"unseen_label_token_share": 0.002}
                },
            },
            {
                "run": "A42",
                "label_coverage": {
                    "val": {"unseen_label_token_share": 0.0}
                },
            },
        ]
        report = experiment._extrapolation_risk_report(rows)
        self.assertTrue(report["extrapolation_risk"])
        self.assertEqual(report["marker"], "[extrapolation-risk]")
        self.assertEqual(report["n_affected"], 1)

    def test_missing_coverage_record_is_reported_as_unknown(self):
        report = experiment._extrapolation_risk_report([{"run": "A41"}])
        self.assertFalse(report["extrapolation_risk"])
        self.assertEqual(report["runs_without_coverage_record"], ["A41"])


class ComparisonAnalysisExtensionTests(unittest.TestCase):
    def _rows(self):
        rows = []
        for mode, offset in (("A", 0.0), ("B", 0.5)):
            for seed in (41, 42, 43):
                rows.append(
                    {
                        "run": f"{mode}{seed}",
                        "mode": mode,
                        "seed": seed,
                        "val_loss": 3.0 - offset + (seed % 3) * 0.05,
                        "val_token_accuracy": 0.6
                        - offset
                        + (seed % 3) * 0.001,
                        "protocol": {
                            "comparison_family": "neural",
                            "data_sha256": "sha-abc",
                        },
                        "protocol_recorded_keys": [],
                    }
                )
        return rows

    def test_payload_contains_result_analysis_extensions(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report = experiment.print_comparison_analysis(
                self._rows(), factors=["mode"], metric="val_loss"
            )
        extensions = report["result_extensions"]
        self.assertIn("result_verdicts", extensions)
        self.assertIn("cost_efficiency", extensions)
        self.assertIn("routing_specialization_fdr", extensions)
        self.assertIn("extrapolation_risk", extensions)
        self.assertEqual(
            extensions["fingerprints"]["data_sha256"], "sha-abc"
        )
        contrast = report["contrasts"][0]
        self.assertIn("result_verdict", contrast)
        self.assertIn("cost_efficiency", contrast)
        rendered = output.getvalue()
        self.assertIn("[cross-metric]", rendered)
        self.assertIn("[verdict]", rendered)
        self.assertIn("[cost-efficiency]", rendered)

    def test_csv_export_includes_new_columns(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report = experiment.print_comparison_analysis(
                self._rows(), factors=["mode"], metric="val_loss"
            )
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "contrasts.csv"
            experiment.write_comparison_csv_files(csv_path, report)
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            columns = header.split(",")
            for column in (
                "result_verdict",
                "cross_metric_divergent",
                "cross_metric_composite_cliffs_delta",
                "cost_inefficient",
                "gain_per_cost",
            ):
                self.assertIn(column, columns)

    def test_markdown_summary_reports_verdict_and_markers(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report = experiment.print_comparison_analysis(
                self._rows(), factors=["mode"], metric="val_loss"
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.md"
            experiment.write_comparison_report_markdown(path, report)
            text = path.read_text(encoding="utf-8")
        self.assertIn("# SMELT-ISR 对比分析摘要", text)
        self.assertIn("## 追溯指纹", text)
        self.assertIn("sha-abc", text)
        self.assertIn("## 裁决分布", text)
        self.assertIn("## 对比明细", text)
        self.assertIn("## 解读边界", text)

    def test_json_report_is_written_with_companion_markdown(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report = experiment.print_comparison_analysis(
                self._rows(), factors=["mode"], metric="val_loss"
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "comparison_analysis.json"
            experiment.write_comparison_report_file(path, report)
            markdown_path = Path(tmp) / "comparison_analysis.md"
            self.assertTrue(markdown_path.exists())
            text = markdown_path.read_text(encoding="utf-8")
            self.assertIn("# SMELT-ISR 对比分析摘要", text)
            # JSON 仍是完整证据结构：扩展字段必须在 JSON 里。
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("result_extensions", payload)


if __name__ == "__main__":
    unittest.main()
