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


class CoreModelTests(unittest.TestCase):
    def test_all_modes_forward_backward(self):
        for mode in (
            "base",
            "smelt",
            "A",
            "local_base",
            "local_base_k2",
            "B",
            "B_early",
            "dense",
        ):
            cfg = _tiny_config(mode=mode)
            model = experiment.LanguageModel(cfg)
            x = torch.randint(0, 256, (2, 16))
            y = torch.randint(0, 256, (2, 16))
            output = model(x, y)
            self.assertTrue(
                math.isfinite(float(output["loss"])),
                mode,
            )
            output["loss"].backward()
            if cfg.mode != "dense":
                grad = model.blocks[0].moe.router.weight.grad
                self.assertIsNotNone(grad, mode)
                self.assertTrue(
                    bool(torch.isfinite(grad).all()), mode
                )

    def test_top1_router_receives_lm_gradient(self):
        cfg = _tiny_config(
            top_k=1,
            k1=1,
            k2=1,
            mode="base",
            balance_coef=0.0,
        )
        model = experiment.LanguageModel(cfg)
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        output = model(x, y)
        output["loss"].backward()
        grad = model.blocks[0].moe.router.weight.grad
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_k1_one_equivalences(self):
        common = dict(
            seq_len=16,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            n_layers=4,
            n_experts=4,
            expert_hidden=16,
        )
        x = torch.randint(0, 256, (2, 16))
        for left, right, cfg in (
            ("smelt", "B", dict(top_k=1, k1=1, k2=1)),
            ("local_base", "A", dict(k1=1, k2=2, top_k=2)),
        ):
            first = experiment.LanguageModel(
                experiment.Config(**common, **cfg, mode=left)
            ).eval()
            second = experiment.LanguageModel(
                experiment.Config(**common, **cfg, mode=right)
            ).eval()
            second.load_state_dict(first.state_dict())
            with torch.no_grad():
                a = first(x)["logits"]
                b = second(x)["logits"]
            torch.testing.assert_close(
                a, b, rtol=1e-5, atol=1e-6
            )

    def test_gqa_forward_backward(self):
        cfg = _tiny_config(n_kv_heads=1)
        model = experiment.LanguageModel(cfg)
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        output = model(x, y)
        output["loss"].backward()
        self.assertTrue(
            math.isfinite(float(output["loss"]))
        )
        self.assertEqual(
            model.blocks[0].attn.k_proj.weight.shape[0],
            1 * (32 // 4),
        )

    def test_delayed_merge_gate_is_applied_once(self):
        # A 的第一轮分支若先乘 gate，最终跨分支收缩又乘同一组
        # gate，会让专家贡献变成 p_i²·expert。这里用桩替换 MoE，
        # 只保留门控/合并路径：正确结果应为 x + Σ p_i·expert_i。
        cfg = _tiny_config(mode="A", k1=2, k2=2)
        block = experiment.TransformerBlock(cfg)
        block.attention_step = (
            lambda x, scale=None, attn_mask=None: x
        )
        block.norm_moe = torch.nn.Identity()

        def unmerged(x, top_k):
            if x.dim() == 3:
                experts = torch.tensor(
                    [[[[2.0, 0.0], [0.0, 4.0]]]]
                )
                gates = torch.tensor([[[0.75, 0.25]]])
                return (
                    experts,
                    gates,
                    None,
                    torch.zeros(()),
                )
            experts = torch.zeros(
                *x.shape[:-1], 2, x.size(-1)
            )
            gates = torch.full((*x.shape[:-1], 2), 0.5)
            return experts, gates, None, torch.zeros(())

        def merge(
            expert_outputs,
            top_weights,
            out_shape,
            residual=None,
            residual_scale=1.0,
        ):
            return residual

        block.moe.unmerged = unmerged
        block.moe.merge = merge
        output, _, _ = block.local_two_stage(
            torch.zeros(1, 1, 2),
            2,
            2,
            delayed=True,
        )
        torch.testing.assert_close(
            output[0, 0],
            torch.tensor([1.5, 1.0]),
        )

    def test_loop_residual_scaling(self):
        cfg = _tiny_config(residual_scaling="paper")
        self.assertAlmostEqual(
            cfg.loop_residual_scale, 0.5
        )
        global_cfg = _tiny_config(
            residual_scaling="global"
        )
        self.assertAlmostEqual(
            global_cfg.loop_residual_scale, 1.0
        )


class CostModelTests(unittest.TestCase):
    def test_a_counts_first_stage(self):
        cfg = _tiny_config(mode="A")
        profile = experiment.mode_cost_profile(cfg)
        # top_k=2, k1=2, k2=2, n_layers=4 -> middle=2
        # pre+post = 2 -> 2*2 + 2*(2 + 4) = 16
        self.assertEqual(
            profile["expert_assignments_per_token"], 16
        )

    def test_a_counts_routers_on_each_branch(self):
        # middle=2, pre=post=1, k1=2：A 的第二轮在 k1 个分支上
        # 各自路由，所以每个中间 Block 有 1+k1=3 次 Router/MoE
        # 执行，总计 1+1+2*3=8，而不是 layers+middle=6。
        profile = experiment.mode_cost_profile(
            _tiny_config(mode="A")
        )
        self.assertEqual(
            profile["router_calls_per_token"], 8
        )
        self.assertEqual(
            profile["moe_passes_per_token"], 8
        )
        early = experiment.mode_cost_profile(
            _tiny_config(mode="B_early")
        )
        self.assertEqual(
            early["analytic_formula"]["attention_executions"],
            "pre + [1 + (middle-1)*k1 + middle*k1] + post",
        )

    def test_b_counts_first_round_branch_routing(self):
        cfg = _tiny_config(mode="B")
        profile = experiment.mode_cost_profile(cfg)
        # middle=2, pre=post=1, top_k=k1=k2=2:
        # first block creates k1 branches (k1 assignments), the
        # remaining middle-1 first-round blocks route k1 experts
        # on each of the k1 branches (k1*k1 each), and the second
        # round routes k2 experts on each branch.
        self.assertEqual(
            profile["expert_assignments_per_token"],
            2 * (1 + 1) + 2 + 1 * 2 * 2 + 2 * 2 * 2,
        )

    def test_b_early_matches_b_execution_counts(self):
        b = experiment.mode_cost_profile(
            _tiny_config(mode="B")
        )
        early = experiment.mode_cost_profile(
            _tiny_config(mode="B_early")
        )
        for key in (
            "attention_executions_per_token",
            "moe_passes_per_token",
            "expert_assignments_per_token",
            "router_calls_per_token",
            "analytic_flops_per_token",
        ):
            self.assertEqual(b[key], early[key], key)

    def test_kv_cache_bytes_counts_k_and_v_once(self):
        # d_model=32, n_heads=4, n_kv_heads=2 -> head_dim=8,
        # kv_dim=16；每层 fp16 的 K+V = 16*2*2 = 64 bytes。
        bytes_per_layer = 16 * 2 * 2
        cfg = _tiny_config(mode="smelt")
        profile = experiment.mode_cost_profile(cfg)
        # 物理层仍只有 4 份；middle=2 层访问两次，pre/post 各
        # 一次 -> 共 1+2*2+1=6 次 Attention，而不是 4*2。
        self.assertEqual(
            profile["kv_cache_bytes_per_token"],
            4 * bytes_per_layer,
        )
        self.assertEqual(
            profile["kv_cache_bytes_per_token_per_visit"],
            6 * bytes_per_layer,
        )
        self.assertEqual(
            profile["kv_cache_bytes_per_token_per_visit"],
            profile["attention_executions_per_token"]
            * bytes_per_layer,
        )

        # B 有 k1=2 个独立分支：分支只出现在 middle，pre/post
        # 仍各一份。middle=2 -> 物理容量 (1+1+2*2)=6 份。
        branch_profile = experiment.mode_cost_profile(
            _tiny_config(mode="B")
        )
        self.assertEqual(
            branch_profile["kv_cache_bytes_per_token"],
            6 * bytes_per_layer,
        )
        # 访问次数：pre 1 + [首块 1 + 剩余 1 块×2 分支 +
        # 第二轮 2 块×2 分支] + post 1 = 9。
        self.assertEqual(
            branch_profile["kv_cache_bytes_per_token_per_visit"],
            9 * bytes_per_layer,
        )
        self.assertEqual(
            branch_profile[
                "kv_cache_bytes_per_token_per_visit"
            ],
            branch_profile["attention_executions_per_token"]
            * bytes_per_layer,
        )
        early = experiment.mode_cost_profile(
            _tiny_config(mode="B_early")
        )
        self.assertEqual(
            early["kv_cache_bytes_per_token"],
            branch_profile["kv_cache_bytes_per_token"],
        )
        self.assertEqual(
            early["kv_cache_bytes_per_token_per_visit"],
            branch_profile["kv_cache_bytes_per_token_per_visit"],
        )

    def test_dense_profile_counts_dense_ffn_flops(self):
        # dense 默认 H_ffn = top_k * expert_hidden = 32。
        # 每层 SwiGLU FFN = 6*D*H = 6*32*32；共 4 层。
        cfg = _tiny_config(mode="dense")
        profile = experiment.mode_cost_profile(cfg)
        breakdown = profile["analytic_flops_breakdown"]
        expected_ffn = 4 * 6 * 32 * 32
        self.assertEqual(breakdown["dense_ffn"], expected_ffn)
        self.assertEqual(
            profile["analytic_flops_per_token"],
            sum(breakdown.values()),
        )
        # 等计算口径下，稠密 FFN 的激活 FLOPs 应与 baseline
        # 单次前向激活的 top_k 个专家相同；dense 少了路由开销。
        base = experiment.mode_cost_profile(
            _tiny_config(mode="base")
        )
        self.assertEqual(
            breakdown["dense_ffn"],
            base["analytic_flops_breakdown"]["experts"],
        )
        self.assertGreater(
            base["analytic_flops_per_token"],
            profile["analytic_flops_per_token"],
        )

    def test_budget_search_reports_recommendation(self):
        cfg = _tiny_config(mode="A")
        report = experiment.budget_match_report(
            cfg, "smelt", 0.1
        )
        self.assertIn(
            report["search"]["kind"], {"k2", "k1_k2"}
        )
        self.assertIsNotNone(
            report["search"]["recommended_k2"]
        )


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


class DesignPreparationTests(unittest.TestCase):
    """先验功效、随机化区组顺序与设计计划核对。"""

    def _payload(self):
        return {
            "study_id": "smelt-isr-design-test",
            "frozen_commit": "a" * 40,
            "primary_metric": "val_loss",
            "common_step": 100,
            "min_effect": 0.01,
            "direction": "decrease",
            "baseline_mode": "smelt",
            "comparisons": [
                ["smelt", "A"],
                ["smelt", "B"],
            ],
            "seeds": [42, 43, 44],
            "budget_match": "smelt",
            "steps": 200,
            "data": {
                "sha256": "b" * 64,
                "split_policy": "contiguous",
                "val_fraction": 0.1,
                "test_fraction": 0.1,
            },
        }

    def _args(self, **overrides):
        values = dict(
            primary_metric=None,
            hypothesis_direction=None,
            hypothesis_min_effect=None,
            hypothesis_baseline=None,
            common_step=None,
            budget_match=None,
            equivalence_margin=0.01,
            steps=200,
            seeds=None,
            seed=42,
            mode="A",
            statistical_baseline=False,
            resume=None,
            evaluate_test=False,
            run_order=None,
            run_order_seed=None,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_order_sweep_plan_randomized_complete_blocks(self):
        plan = experiment.build_sweep_plan(
            seeds=[1, 2, 3],
            modes=["A", "B"],
            ablation=False,
            balance_coef=0.01,
            default_mode="B",
        )
        grouped = experiment.order_sweep_plan(
            plan, "grouped", 0
        )
        self.assertEqual(
            [item["label"] for item in grouped],
            [item["label"] for item in plan],
        )
        self.assertEqual(
            [item["order_index"] for item in grouped],
            [1, 2, 3, 4, 5, 6],
        )

        blocked = experiment.order_sweep_plan(
            plan, "blocked", 0
        )
        for block in (1, 2, 3):
            labels = [
                item["label"]
                for item in blocked
                if item["block"] == block
            ]
            self.assertEqual(
                labels, [f"A_seed{block}", f"B_seed{block}"]
            )

        randomized = experiment.order_sweep_plan(
            plan, "randomized", 7
        )
        repeated = experiment.order_sweep_plan(
            plan, "randomized", 7
        )
        self.assertEqual(
            [item["label"] for item in randomized],
            [item["label"] for item in repeated],
        )
        for block in (1, 2, 3):
            block_items = [
                item
                for item in randomized
                if item["block"] == block
            ]
            self.assertEqual(len(block_items), 2)
            self.assertEqual(
                {item["mode"] for item in block_items},
                {"A", "B"},
            )
            self.assertEqual(
                [item["position_in_block"] for item in block_items],
                [1, 2],
            )
        # 随机化不是把整组顺序倒过来：不同区组可以使用不同的
        # 臂顺序，且同一计划必须可复现。
        different = experiment.order_sweep_plan(
            plan, "randomized", 11
        )
        self.assertNotEqual(
            [item["label"] for item in randomized],
            [item["label"] for item in different],
        )

    def test_order_plan_tolerates_unhashable_seed(self):
        plan = [
            {"label": "A_bad", "mode": "A", "balance_coef": 0.0, "seed": []},
            {"label": "B_bad", "mode": "B", "balance_coef": 0.0, "seed": []},
            {"label": "A_bad2", "mode": "A", "balance_coef": 0.0, "seed": {"x": 1}},
            {"label": "B_bad2", "mode": "B", "balance_coef": 0.0, "seed": {"x": 1}},
        ]
        for scheme in ("grouped", "blocked", "randomized", "counterbalanced"):
            ordered = experiment.order_sweep_plan(plan, scheme, 3)
            self.assertEqual(len(ordered), 4, scheme)
            self.assertEqual(
                [item["order_index"] for item in ordered],
                [1, 2, 3, 4],
                scheme,
            )
            # 原始 seed 值必须保留给报告/子进程，而不是替换成
            # 内部稳定键。
            self.assertEqual(
                sorted(
                    repr(item["block"]) for item in ordered
                ),
                sorted(repr(item["seed"]) for item in ordered),
                scheme,
            )

    def test_run_order_drift_detects_monotonic_trend(self):
        rows = []
        for index, (mode, value) in enumerate(
            (
                ("A", 1.0),
                ("B", 2.0),
                ("A", 3.0),
                ("B", 4.0),
                ("A", 5.0),
                ("B", 6.0),
            ),
            start=1,
        ):
            rows.append({
                "run": f"{mode}{index}",
                "mode": mode,
                "val_loss": value,
                "design_order_index": index,
                "design_block": index,
                "design_block_position": 1,
                "design_plan_id": "plan-1",
                "design_order_scheme": "randomized",
            })
        report = experiment.run_order_drift_report(
            rows, "val_loss"
        )
        self.assertTrue(report["available"])
        self.assertGreater(
            abs(report["detrended_spearman"]), 0.9
        )
        self.assertEqual(report["verdict"], "warn")
        self.assertTrue(report["warnings"])

    def test_run_order_drift_flags_grouped_scheme(self):
        rows = [
            {
                "run": f"A{seed}",
                "mode": "A",
                "val_loss": 1.0 + seed,
                "design_order_index": seed,
                "design_block": seed,
                "design_block_position": 1,
                "design_plan_id": "plan-1",
                "design_order_scheme": "grouped",
            }
            for seed in (1, 2, 3)
        ]
        report = experiment.run_order_drift_report(
            rows, "val_loss"
        )
        self.assertEqual(report["verdict"], "warn")
        self.assertTrue(
            any("grouped" in warning for warning in report["warnings"])
        )

    def test_preregistration_power_rejects_underpowered_design(self):
        payload = self._payload()
        payload["power"] = {
            "expected_seed_sd_relative": 0.05,
            "target_power": 0.8,
            "alpha": 0.05,
        }
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("power", error)
        self.assertIn("不足", error)

        payload["power"]["underpowered_acknowledged"] = True
        payload["power"]["justification"] = "探索性试点"
        self.assertIsNone(
            experiment.preregistration_schema_error(payload)
        )
        record = experiment.preregistration_record(
            payload,
            "prereg.json",
            "c" * 64,
            [],
            self._args(),
        )
        self.assertFalse(record["power"]["adequate"])
        self.assertTrue(
            record["power"]["underpowered_acknowledged"]
        )

    def test_preregistration_power_accepts_adequate_design(self):
        payload = self._payload()
        payload["seeds"] = list(range(42, 62))
        payload["power"] = {
            "expected_seed_sd_relative": 0.002,
            "target_power": 0.8,
            "alpha": 0.05,
        }
        self.assertIsNone(
            experiment.preregistration_schema_error(payload)
        )
        plan = experiment.preregistration_power_plan(
            payload["power"], 0.01, len(payload["seeds"]), 2
        )
        self.assertTrue(plan["adequate"])
        self.assertEqual(plan["required_seeds_for_t"], 3)
        self.assertEqual(
            plan["required_seeds_for_robust_sign_flip"], 7
        )
        self.assertEqual(plan["required_seeds"], 7)

    def test_preregistration_power_accepts_pilot_calibration(self):
        payload = self._payload()
        payload["seeds"] = list(range(42, 62))
        payload["power"] = {
            "expected_seed_sd_relative": 0.002,
            "target_power": 0.8,
            "alpha": 0.05,
            "pilot_calibration": "upper_confidence_bound",
            "pilot_confidence": 0.9,
        }
        self.assertIsNone(
            experiment.preregistration_schema_error(payload)
        )
        self.assertEqual(
            experiment._pilot_calibration_setting(
                payload["power"]
            ),
            ("upper_confidence_bound", 0.9),
        )
        record = experiment.preregistration_record(
            payload,
            "prereg.json",
            "c" * 64,
            [],
            self._args(),
        )
        self.assertEqual(
            record["power"]["pilot_calibration"],
            "upper_confidence_bound",
        )
        self.assertEqual(record["power"]["pilot_confidence"], 0.9)

    def test_preregistration_power_rejects_invalid_pilot_calibration(
        self,
    ):
        payload = self._payload()
        payload["seeds"] = list(range(42, 62))
        power = {
            "expected_seed_sd_relative": 0.002,
            "target_power": 0.8,
            "alpha": 0.05,
        }
        payload["power"] = dict(
            power, pilot_calibration="after_the_fact"
        )
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("pilot_calibration", error)
        payload["power"] = dict(power, pilot_confidence=0.5)
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("pilot_confidence", error)
        payload["power"] = dict(power, pilot_confidence=True)
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("pilot_confidence", error)

    def test_preregistration_power_cannot_understate_family(self):
        payload = self._payload()
        payload["power"] = {
            "expected_seed_sd_relative": 0.002,
            "target_power": 0.8,
            "alpha": 0.05,
            "planned_comparisons": 1,
        }
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("planned_comparisons", error)

    def test_preregistration_run_order_applied_and_conflict_rejected(self):
        payload = self._payload()
        payload["run_order"] = {
            "scheme": "randomized",
            "seed": 7,
        }
        args = self._args()
        with mock.patch.object(
            sys, "argv", ["experiment.py"]
        ):
            applied, error = experiment.preregistration_apply(
                args,
                payload,
                current_commit="a" * 40,
                working_tree_dirty=False,
                current_script_sha256="c" * 64,
            )
        self.assertIsNone(error)
        self.assertIn("run_order.scheme", applied)
        self.assertEqual(args.run_order, "randomized")
        self.assertEqual(args.run_order_seed, 7)

        conflict_args = self._args(run_order="grouped")
        with mock.patch.object(
            sys, "argv", ["experiment.py", "--run-order", "grouped"]
        ):
            _, error = experiment.preregistration_apply(
                conflict_args,
                payload,
                current_commit="a" * 40,
                working_tree_dirty=False,
                current_script_sha256="c" * 64,
            )
        self.assertIsNotNone(error)
        self.assertIn("run_order", error)

    def test_design_plan_roundtrip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(
                seeds=[1, 2, 3],
                allow_underpowered=False,
                steps=10,
                batch_size=2,
                seq_len=8,
                preregistration_record=None,
                run_order="randomized",
                run_order_seed=5,
            )
            plan = experiment.build_sweep_plan(
                seeds=[1, 2, 3],
                modes=["A", "B"],
                ablation=False,
                balance_coef=0.01,
                default_mode="B",
            )
            ordered = experiment.order_sweep_plan(
                plan, "randomized", 5
            )
            design = experiment.write_design_plan(
                root, ordered, "randomized", 5, args
            )
            self.assertEqual(design["planned_runs"], 6)
            self.assertEqual(
                design["training_budget"]["total_tokens"], 960
            )
            item = next(
                entry
                for entry in ordered
                if entry["order_index"] == 3
            )
            child_args = argparse.Namespace(
                out=str(root / str(item["label"])),
                seed=item["seed"],
                mode=item["mode"],
                balance_coef=item["balance_coef"],
            )
            env = {
                "SMELT_DESIGN_PLAN_FILE": str(design["path"]),
                "SMELT_DESIGN_PLAN_SHA256": design[
                    "file_sha256"
                ],
                "SMELT_DESIGN_PLAN_ID": design["plan_id"],
                "SMELT_DESIGN_ORDER_SCHEME": design[
                    "order_scheme"
                ],
                "SMELT_DESIGN_ORDER_SEED": str(
                    design["order_seed"]
                ),
                "SMELT_DESIGN_ORDER_INDEX": str(
                    item["order_index"]
                ),
                "SMELT_DESIGN_BLOCK": str(item["block"]),
                "SMELT_DESIGN_BLOCK_POSITION": str(
                    item["position_in_block"]
                ),
            }
            with mock.patch.dict(
                os.environ, env, clear=False
            ):
                assignment = experiment.verify_design_assignment(
                    child_args
                )
            self.assertEqual(
                assignment["order_index"], item["order_index"]
            )
            self.assertEqual(
                assignment["plan_id"], design["plan_id"]
            )

            # 计划文件在开跑前被改：指纹不一致必须拒绝。
            path = Path(design["path"])
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["runs"][0]["label"] = "tampered"
            path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ, env, clear=False
            ):
                with self.assertRaises(ValueError):
                    experiment.verify_design_assignment(child_args)

    def test_design_plan_consistency_detects_missing_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good"
            bad = root / "bad"
            missing = root / "missing"
            for path, plan_id in (
                (good, "plan-1"),
                (bad, "plan-2"),
                (missing, None),
            ):
                path.mkdir()
                payload = {}
                if plan_id is not None:
                    payload["design_assignment"] = {
                        "plan_id": plan_id
                    }
                (path / "config.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            error = experiment.design_plan_consistency_error(
                [str(good)], "plan-1"
            )
            self.assertIsNone(error)
            error = experiment.design_plan_consistency_error(
                [str(good), str(bad), str(missing)], "plan-1"
            )
            self.assertIsNotNone(error)
            self.assertIn("bad", error)
            self.assertIn("missing", error)

    def test_resolve_run_order_rejects_empty_scheme(self):
        args = argparse.Namespace(
            run_order="", run_order_seed=0
        )
        with self.assertRaises(ValueError):
            experiment.resolve_run_order(args)
        self.assertEqual(
            experiment.resolve_run_order(
                argparse.Namespace(
                    run_order=None, run_order_seed=None
                )
            ),
            ("randomized", 0),
        )

    def test_design_plan_recomputes_canonical_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(
                seeds=[1, 2, 3],
                allow_underpowered=False,
                steps=10,
                batch_size=2,
                seq_len=8,
                preregistration_record=None,
                run_order="randomized",
                run_order_seed=5,
            )
            plan = experiment.build_sweep_plan(
                seeds=[1, 2, 3],
                modes=["A", "B"],
                ablation=False,
                balance_coef=0.01,
                default_mode="B",
            )
            ordered = experiment.order_sweep_plan(
                plan, "randomized", 5
            )
            design = experiment.write_design_plan(
                root, ordered, "randomized", 5, args
            )
            item = next(
                entry
                for entry in ordered
                if entry["order_index"] == 3
            )
            child_args = argparse.Namespace(
                out=str(root / str(item["label"])),
                seed=item["seed"],
                mode=item["mode"],
                balance_coef=item["balance_coef"],
            )
            path = Path(design["path"])
            payload = json.loads(
                path.read_text(encoding="utf-8")
            )
            payload["runs"][0]["label"] = "tampered"
            path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            # 同时把文件 SHA 与 plan_id 环境变量改成「与当前
            # 文件一致」：仅比对文件 SHA/字段值仍应被规范化
            # 指纹识别出来。
            env = {
                "SMELT_DESIGN_PLAN_FILE": str(path),
                "SMELT_DESIGN_PLAN_SHA256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
                "SMELT_DESIGN_PLAN_ID": payload["plan_id"],
                "SMELT_DESIGN_ORDER_SCHEME": design[
                    "order_scheme"
                ],
                "SMELT_DESIGN_ORDER_SEED": str(
                    design["order_seed"]
                ),
                "SMELT_DESIGN_ORDER_INDEX": str(
                    item["order_index"]
                ),
                "SMELT_DESIGN_BLOCK": str(item["block"]),
                "SMELT_DESIGN_BLOCK_POSITION": str(
                    item["position_in_block"]
                ),
            }
            with mock.patch.dict(
                os.environ, env, clear=False
            ):
                with self.assertRaises(ValueError):
                    experiment.verify_design_assignment(
                        child_args
                    )

    def test_design_plan_rejects_malformed_assignment_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(
                seeds=[1, 2, 3],
                allow_underpowered=False,
                steps=10,
                batch_size=2,
                seq_len=8,
                preregistration_record=None,
                run_order="randomized",
                run_order_seed=5,
            )
            plan = experiment.build_sweep_plan(
                seeds=[1, 2, 3],
                modes=["A", "B"],
                ablation=False,
                balance_coef=0.01,
                default_mode="B",
            )
            ordered = experiment.order_sweep_plan(
                plan, "randomized", 5
            )
            design = experiment.write_design_plan(
                root, ordered, "randomized", 5, args
            )
            item = next(
                entry
                for entry in ordered
                if entry["order_index"] == 3
            )
            child_args = argparse.Namespace(
                out=str(root / str(item["label"])),
                seed=item["seed"],
                mode=item["mode"],
                balance_coef=item["balance_coef"],
            )
            path = Path(design["path"])
            payload = json.loads(
                path.read_text(encoding="utf-8")
            )
            entry = next(
                run
                for run in payload["runs"]
                if run["order_index"] == item["order_index"]
            )
            entry["seed"] = None
            # 同步刷新文件指纹与 plan_id，确保拒绝理由不是
            # 「文件被改」而是「条目字段损坏」。
            payload["plan_id"] = experiment._design_plan_id(
                payload
            )
            path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            env = {
                "SMELT_DESIGN_PLAN_FILE": str(path),
                "SMELT_DESIGN_PLAN_SHA256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
                "SMELT_DESIGN_PLAN_ID": payload["plan_id"],
                "SMELT_DESIGN_ORDER_SCHEME": design[
                    "order_scheme"
                ],
                "SMELT_DESIGN_ORDER_SEED": str(
                    design["order_seed"]
                ),
                "SMELT_DESIGN_ORDER_INDEX": str(
                    item["order_index"]
                ),
                "SMELT_DESIGN_BLOCK": str(item["block"]),
                "SMELT_DESIGN_BLOCK_POSITION": str(
                    item["position_in_block"]
                ),
            }
            with mock.patch.dict(
                os.environ, env, clear=False
            ):
                with self.assertRaises(ValueError):
                    experiment.verify_design_assignment(
                        child_args
                    )

            # 浮点值不能被 int() 静默截断成合法分配。
            entry["seed"] = 1.9
            payload["plan_id"] = experiment._design_plan_id(
                payload
            )
            path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            env["SMELT_DESIGN_PLAN_SHA256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            env["SMELT_DESIGN_PLAN_ID"] = payload["plan_id"]
            with mock.patch.dict(
                os.environ, env, clear=False
            ):
                with self.assertRaises(ValueError):
                    experiment.verify_design_assignment(
                        child_args
                    )

            # 极大的 balance_coef 也不能让 float() 溢出逃逸。
            entry["seed"] = item["seed"]
            entry["balance_coef"] = 10**400
            payload["plan_id"] = experiment._design_plan_id(
                payload
            )
            path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            env["SMELT_DESIGN_PLAN_SHA256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            env["SMELT_DESIGN_PLAN_ID"] = payload["plan_id"]
            with mock.patch.dict(
                os.environ, env, clear=False
            ):
                with self.assertRaises(ValueError):
                    experiment.verify_design_assignment(
                        child_args
                    )

    def test_run_order_drift_skips_deterministic_modes(self):
        rows = [
            {
                "run": f"ngram{index}",
                "mode": "ngram",
                "val_loss": float(index),
                "design_order_index": index,
                "design_block": index,
                "design_block_position": 1,
                "design_plan_id": "plan-1",
                "design_order_scheme": "randomized",
            }
            for index in (1, 2, 3, 4)
        ]
        report = experiment.run_order_drift_report(
            rows, "val_loss"
        )
        # 旧 ngram 运行可能没有显式 deterministic 字段；模式本身
        # 已说明它不消耗随机性，不能当作独立随机重复。
        self.assertFalse(report["available"])
        self.assertEqual(report["verdict"], "unavailable")

    def test_run_order_drift_warns_missing_plan_record(self):
        rows = []
        for index in (1, 2, 3, 4):
            rows.append({
                "run": f"r{index}",
                "mode": "A" if index % 2 else "B",
                "val_loss": float(index),
                "design_order_index": index,
                "design_block": index,
                "design_block_position": 1,
                "design_plan_id": (
                    None if index == 4 else "plan-1"
                ),
                "design_order_scheme": "randomized",
            })
        report = experiment.run_order_drift_report(
            rows, "val_loss"
        )
        self.assertTrue(report["available"])
        self.assertEqual(report["verdict"], "warn")
        self.assertTrue(
            any(
                "缺少设计计划指纹" in warning
                for warning in report["warnings"]
            )
        )

    def test_planned_comparisons_counts_ablation_balance_axis(self):
        plan = experiment.build_sweep_plan(
            seeds=[1, 2, 3],
            modes=None,
            ablation=True,
            balance_coef=0.01,
            default_mode="B",
            ablation_modes=["A", "B"],
        )
        # 3 个模式 × 2 个 balance 档 = 6 个处理单元；不能只数
        # mode 而漏掉 balance 轴，否则家族规模从 15 低估成 3。
        self.assertEqual(
            experiment.planned_comparisons_for_plan(plan), 15
        )
        # 两模式消融（local_base + A）× 2 档：单因子两两组合
        # 有 6 个，但每个延迟方案的 2×2 主效应/交互/简单效应
        # 家族有 7 个检验，不能仍报 6。
        partial = experiment.build_sweep_plan(
            seeds=[1, 2, 3],
            modes=None,
            ablation=True,
            balance_coef=0.01,
            default_mode="B",
            ablation_modes=["A"],
        )
        self.assertEqual(
            experiment.planned_comparisons_for_plan(partial), 7
        )

    def test_build_sweep_execution_uses_larger_actual_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                seeds=[1, 2, 3],
                allow_underpowered=False,
                steps=10,
                batch_size=2,
                seq_len=8,
                suite_modes=["A", "B", "local_base", "local_base_k2"],
                suite_ablation=False,
                ablation_modes=None,
                balance_coef=0.01,
                mode="B",
                run_order="randomized",
                run_order_seed=5,
                preregistration_record={
                    "power": {
                        "expected_seed_sd_relative": 0.01,
                        "min_effect_relative": 0.01,
                        "target_power": 0.8,
                        "alpha": 0.05,
                        "planned_comparisons": 3,
                        "planned_seeds": 3,
                        "underpowered_acknowledged": True,
                        "justification": "test",
                    }
                },
            )
            plan, design, planned = (
                experiment.build_sweep_execution(
                    args, Path(tmp)
                )
            )
            self.assertEqual(len(plan), 12)
            # 4 个臂的两两对比是 6，大于预注册声明的 3。
            self.assertEqual(planned, 6)
            self.assertEqual(
                design["planned_comparisons"], 6
            )
            self.assertEqual(
                design["seed_plan"]["planned_comparisons"], 6
            )
            self.assertEqual(
                design["power"]["planned_comparisons"], 6
            )

    def test_design_plan_refuses_to_replace_frozen_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_args = argparse.Namespace(
                seeds=[1, 2, 3],
                allow_underpowered=False,
                steps=10,
                batch_size=2,
                seq_len=8,
                preregistration_record=None,
            )
            plan = experiment.build_sweep_plan(
                seeds=[1, 2, 3],
                modes=["A", "B"],
                ablation=False,
                balance_coef=0.01,
                default_mode="B",
            )
            ordered = experiment.order_sweep_plan(
                plan, "randomized", 5
            )
            first = experiment.write_design_plan(
                root, ordered, "randomized", 5, base_args
            )
            # 同一设计重复生成：复用已冻结文件（SHA 不变）。
            same = experiment.write_design_plan(
                root, ordered, "randomized", 5, base_args
            )
            self.assertEqual(
                first["plan_id"], same["plan_id"]
            )
            self.assertEqual(
                first["file_sha256"], same["file_sha256"]
            )
            # 不同臂集合不能静默覆盖已冻结计划。
            changed = experiment.build_sweep_plan(
                seeds=[1, 2, 3, 4],
                modes=["A", "B"],
                ablation=False,
                balance_coef=0.01,
                default_mode="B",
            )
            changed_ordered = experiment.order_sweep_plan(
                changed, "randomized", 5
            )
            with self.assertRaises(ValueError):
                experiment.write_design_plan(
                    root,
                    changed_ordered,
                    "randomized",
                    5,
                    base_args,
                )

    def test_run_order_drift_handles_unhashable_block_record(self):
        rows = []
        for index in (1, 2, 3, 4):
            rows.append({
                "run": f"r{index}",
                "mode": "A" if index % 2 else "B",
                "val_loss": (
                    10**400 if index == 2 else float(index)
                ),
                "design_order_index": (
                    10**400 if index == 3 else index
                ),
                # 损坏的旧 config 可能把区组写成对象/列表；
                # 诊断报告不能因此抛 TypeError。
                "design_block": (
                    {"bad": index}
                    if index == 1
                    else (10**400 if index == 2 else index)
                ),
                "design_block_position": 1,
                "design_plan_id": "plan-1",
                "design_order_scheme": "randomized",
            })
        report = experiment.run_order_drift_report(
            rows, "val_loss"
        )
        self.assertTrue(report["available"])
        # 指标为 10**400 的运行不可用（不是有限浮点），不进入
        # 相关性样本；剩余区组仍必须可哈希计数而不是崩溃。
        self.assertEqual(report["n_blocks"], 3)

    def test_run_order_drift_flags_mixed_schemes(self):
        rows = []
        for index, scheme in enumerate(
            ("randomized", "grouped", "blocked", "randomized"),
            start=1,
        ):
            rows.append({
                "run": f"r{index}",
                "mode": "A" if index % 2 else "B",
                "val_loss": float(index),
                "design_order_index": index,
                "design_block": index,
                "design_block_position": 1,
                "design_plan_id": "plan-1",
                "design_order_scheme": scheme,
            })
        report = experiment.run_order_drift_report(
            rows, "val_loss"
        )
        self.assertTrue(
            any(
                "多种执行顺序方案" in warning
                for warning in report["warnings"]
            )
        )

    def test_run_order_drift_zero_variance_is_insufficient(self):
        rows = []
        for index, mode in enumerate(("A", "B", "A", "B"), start=1):
            rows.append({
                "run": f"{mode}{index}",
                "mode": mode,
                "val_loss": 1.0,
                "design_order_index": index,
                "design_block": index,
                "design_block_position": 1,
                "design_plan_id": "plan-1",
                "design_order_scheme": "randomized",
            })
        report = experiment.run_order_drift_report(
            rows, "val_loss"
        )
        self.assertTrue(report["available"])
        self.assertIsNone(report["detrended_spearman"])
        self.assertEqual(report["verdict"], "insufficient")
        self.assertIn("秩方差", report["reason"])

    def test_plan_design_does_not_require_training_hypothesis(self):
        args = argparse.Namespace(
            compare=None,
            final_test_comparison=None,
            sanity_only=False,
            statistical_baseline=False,
            plan_design=True,
            resume=None,
            hypothesis="",
            skip_hypothesis_check=False,
        )
        self.assertIsNone(
            experiment.hypothesis_check_error(args)
        )

    def test_extreme_power_inputs_do_not_crash(self):
        self.assertGreaterEqual(
            experiment._sign_flip_required_seeds(
                1e-320, 1
            ),
            6,
        )
        self.assertGreaterEqual(
            experiment._sign_flip_required_seeds(
                0.05, 10**400
            ),
            6,
        )
        self.assertEqual(
            experiment._sign_flip_required_seeds(
                10**400, 1
            ),
            6,
        )
        payload = self._payload()
        payload["power"] = {
            "expected_seed_sd_relative": 0.01,
            "target_power": 0.8,
            "alpha": 0.05,
            "planned_comparisons": 10**400,
        }
        error = experiment.preregistration_schema_error(payload)
        self.assertIsInstance(error, str)
        # 直接调用功效规划也不能因超大种子数在 float()/int()
        # 处崩溃；报告保留原始种子数。
        plan = experiment.preregistration_power_plan(
            {
                "expected_seed_sd_relative": 0.01,
                "target_power": 0.8,
                "alpha": 0.05,
                "planned_comparisons": 1,
            },
            0.01,
            10**400,
            1,
        )
        self.assertEqual(plan["planned_seeds"], 10**400)
        json.dumps(plan, allow_nan=False)


    def test_counterbalanced_order_balances_positions(self):
        plan = experiment.build_sweep_plan(
            seeds=[1, 2, 3, 4, 5, 6],
            modes=["A", "B", "C"],
            ablation=False,
            balance_coef=0.01,
            default_mode="B",
        )
        ordered = experiment.order_sweep_plan(
            plan, "counterbalanced", 11
        )
        repeated = experiment.order_sweep_plan(
            plan, "counterbalanced", 11
        )
        self.assertEqual(
            [item["label"] for item in ordered],
            [item["label"] for item in repeated],
        )
        report = experiment.run_order_balance_report(ordered)
        self.assertTrue(report["available"])
        self.assertTrue(report["complete_blocks"])
        self.assertTrue(report["balanced"])
        self.assertEqual(report["max_position_imbalance"], 0)
        # 每个种子区组仍包含全部三个处理各一次，只是位置被
        # 拉丁方轮换。
        for seed in range(1, 7):
            block = [
                item
                for item in ordered
                if item["block"] == seed
            ]
            self.assertEqual(
                sorted(item["mode"] for item in block),
                ["A", "B", "C"],
            )
            self.assertEqual(
                [item["position_in_block"] for item in block],
                [1, 2, 3],
            )

    def test_counterbalanced_rejects_unequal_blocks(self):
        plan = [
            {
                "label": "A_seed1",
                "mode": "A",
                "balance_coef": 0.01,
                "seed": 1,
            },
            {
                "label": "B_seed1",
                "mode": "B",
                "balance_coef": 0.01,
                "seed": 1,
            },
            {
                "label": "A_seed2",
                "mode": "A",
                "balance_coef": 0.01,
                "seed": 2,
            },
        ]
        with self.assertRaises(ValueError) as context:
            experiment.order_sweep_plan(
                plan, "counterbalanced", 0
            )
        self.assertIn("相同的处理单元", str(context.exception))

    def test_order_balance_report_detects_position_confounding(self):
        plan = [
            {
                "label": "A_seed1",
                "mode": "A",
                "seed": 1,
                "block": 1,
                "position_in_block": 1,
            },
            {
                "label": "B_seed1",
                "mode": "B",
                "seed": 1,
                "block": 1,
                "position_in_block": 2,
            },
            {
                "label": "A_seed2",
                "mode": "A",
                "seed": 2,
                "block": 2,
                "position_in_block": 1,
            },
            {
                "label": "B_seed2",
                "mode": "B",
                "seed": 2,
                "block": 2,
                "position_in_block": 2,
            },
        ]
        report = experiment.run_order_balance_report(plan)
        self.assertTrue(report["available"])
        self.assertFalse(report["balanced"])
        self.assertEqual(report["max_position_imbalance"], 2)

    def test_preregistration_power_accounts_for_attrition(self):
        payload = self._payload()
        payload["seeds"] = list(range(42, 50))
        payload["power"] = {
            "expected_seed_sd_relative": 0.005,
            "target_power": 0.8,
            "alpha": 0.05,
            "expected_attrition_rate": 0.2,
        }
        error = experiment.preregistration_schema_error(payload)
        self.assertIsInstance(error, str)
        self.assertIn("启动种子", error)

        payload["seeds"] = list(range(42, 51))
        self.assertIsNone(
            experiment.preregistration_schema_error(payload)
        )
        plan = experiment.preregistration_power_plan(
            payload["power"],
            0.01,
            len(payload["seeds"]),
            2,
        )
        self.assertEqual(plan["required_seeds"], 7)
        self.assertEqual(
            plan["required_seeds_with_attrition"], 9
        )
        self.assertEqual(plan["planned_usable_seeds"], 7)
        self.assertTrue(plan["adequate"])

    def test_preregistration_power_reports_mde_and_sensitivity(self):
        power = {
            "expected_seed_sd_relative": 0.005,
            "target_power": 0.8,
            "alpha": 0.05,
            "planned_comparisons": 5,
        }
        plan = experiment.preregistration_power_plan(
            power, 0.01, 8, 5
        )
        self.assertTrue(plan["adequate"])
        self.assertIsNotNone(
            plan["achieved_power_at_min_effect"]
        )
        self.assertGreaterEqual(
            plan["achieved_power_at_min_effect"], 0.8
        )
        self.assertIsNotNone(
            plan["minimum_detectable_effect_relative"]
        )
        self.assertLessEqual(
            plan["minimum_detectable_effect_relative"], 0.01
        )
        sensitivity = plan["power_sensitivity"]
        self.assertEqual(
            [entry["sd_multiplier"] for entry in sensitivity],
            [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        )
        required = [
            entry["required_seeds_for_t"]
            for entry in sensitivity
            if entry["required_seeds_for_t"] is not None
        ]
        self.assertEqual(required, sorted(required))

    def test_tost_power_and_equivalence_seed_planning(self):
        # 真实差异为零时，n=3（df=2）在 1% 界内达不到 80% 功效，
        # n=4 达到；功率必须随种子数单调不减。
        power_three = experiment._tost_power(
            0.0, 0.005 / math.sqrt(3.0), 2.0, 0.01, 0.05
        )
        power_four = experiment._tost_power(
            0.0, 0.005 / math.sqrt(4.0), 3.0, 0.01, 0.05
        )
        self.assertIsNotNone(power_three)
        self.assertIsNotNone(power_four)
        self.assertLess(power_three, 0.8)
        self.assertGreaterEqual(power_four, 0.8)
        self.assertEqual(
            experiment._seeds_for_tost(
                0.005, 0.01, 0.8, 0.05
            ),
            4,
        )
        margin_needed = experiment._tost_margin_for_power(
            0.005 / math.sqrt(8.0), 7.0, 0.8, 0.05
        )
        self.assertIsNotNone(margin_needed)
        self.assertLess(margin_needed, 0.01)
        # 等价界越小，所需种子越多；界为零时不可判定。
        self.assertGreaterEqual(
            experiment._seeds_for_tost(0.005, 0.005, 0.8, 0.05)
            or 10**9,
            experiment._seeds_for_tost(0.005, 0.01, 0.8, 0.05)
            or 0,
        )
        self.assertIsNone(
            experiment._seeds_for_tost(0.005, 0.0, 0.8, 0.05)
        )

    def test_preregistration_power_equivalence_is_gated_only_when_claimed(self):
        power = {
            "expected_seed_sd_relative": 0.002,
            "target_power": 0.8,
            "alpha": 0.05,
            "planned_comparisons": 2,
            "plan_equivalence": True,
        }
        plan = experiment.preregistration_power_plan(
            power, 0.01, 20, 2, equivalence_margin=0.0005
        )
        self.assertTrue(plan["superiority_adequate"])
        self.assertFalse(plan["equivalence_adequate"])
        self.assertFalse(plan["adequate"])
        self.assertEqual(
            plan["equivalence"]["claim_planned"], True
        )
        self.assertTrue(
            all(
                entry["equivalence_adequate_at_planned_seeds"]
                is False
                for entry in plan["power_sensitivity"]
            )
        )
        self.assertTrue(
            all(
                entry["adequate_at_planned_seeds"] is False
                for entry in plan["power_sensitivity"]
            )
        )
        gate = experiment.design_power_gate_error(
            {"power": plan}
        )
        self.assertIsNotNone(gate)
        self.assertIn("等价性", gate)

        # 不主张等价时，TOST 只是信息性结果，不能反过来
        # 把显著性功效足够的设计判成不可执行。
        power["plan_equivalence"] = False
        informational = experiment.preregistration_power_plan(
            power, 0.01, 20, 2, equivalence_margin=0.0005
        )
        self.assertTrue(informational["adequate"])
        self.assertFalse(
            informational["equivalence_adequate"]
        )
        self.assertTrue(
            all(
                entry["adequate_at_planned_seeds"]
                for entry in informational["power_sensitivity"]
            )
        )
        self.assertIsNone(
            experiment.design_power_gate_error(
                {"power": informational}
            )
        )

    def test_preregistration_equivalence_claim_requires_declared_margin(self):
        payload = self._payload()
        payload["seeds"] = list(range(42, 62))
        payload["power"] = {
            "expected_seed_sd_relative": 0.002,
            "target_power": 0.8,
            "alpha": 0.05,
            "plan_equivalence": True,
        }
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("equivalence_margin", error)

        payload["equivalence_margin"] = 0.0005
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("等价性", error)

        payload["power"]["underpowered_acknowledged"] = True
        payload["power"]["justification"] = "探索性等价性试点"
        self.assertIsNone(
            experiment.preregistration_schema_error(payload)
        )
        record = experiment.preregistration_record(
            payload,
            "prereg.json",
            "c" * 64,
            [],
            self._args(),
        )
        self.assertFalse(
            record["power"]["equivalence_adequate"]
        )
        self.assertTrue(
            record["power"]["underpowered_acknowledged"]
        )

    def test_preregistration_rejects_equivalence_margin_above_min_effect(self):
        payload = self._payload()
        payload["seeds"] = list(range(42, 62))
        payload["power"] = {
            "expected_seed_sd_relative": 0.002,
            "target_power": 0.8,
            "alpha": 0.05,
            "plan_equivalence": True,
        }
        # min_effect=0.01；等价界比「最小有意义效应」还宽，
        # 会让同一效应既算有意义又算可忽略。边界相等可接受。
        payload["equivalence_margin"] = 0.011
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("equivalence_margin", error)
        self.assertIn("min_effect", error)
        payload["equivalence_margin"] = 0.01
        self.assertIsNone(
            experiment.preregistration_schema_error(payload)
        )

    def test_design_power_recomputes_equivalence_for_larger_family(self):
        power = {
            "expected_seed_sd_relative": 0.005,
            "min_effect_relative": 0.01,
            "target_power": 0.8,
            "alpha": 0.05,
            "planned_comparisons": 1,
            "planned_contrasts": ["pairwise"],
            "planned_seeds": 8,
            "plan_equivalence": True,
            "equivalence_margin_relative": 0.01,
            "underpowered_acknowledged": False,
        }
        adjusted = experiment._design_power_for_family(
            power, 3, ["pairwise", "interaction"]
        )
        self.assertEqual(adjusted["planned_comparisons"], 3)
        self.assertIn(
            "interaction", adjusted["planned_contrasts"]
        )
        self.assertEqual(
            adjusted["equivalence"]["margin_relative"], 0.01
        )
        self.assertTrue(
            adjusted["equivalence"]["claim_planned"]
        )
        # 交互的方差是两两对比的 2 倍；等价性设计也必须按
        # 更吃种子的类型重算，而不是沿用 pairwise 的结论。
        self.assertIn(
            "interaction", adjusted["contrast_power"]
        )
        self.assertIsNotNone(
            adjusted["contrast_power"]["interaction"].get(
                "required_seeds_for_equivalence"
            )
        )

    def test_preregistration_power_rejects_invalid_attrition(self):
        payload = self._payload()
        payload["power"] = {
            "expected_seed_sd_relative": 0.005,
            "target_power": 0.8,
            "alpha": 0.05,
            "expected_attrition_rate": 1.0,
        }
        error = experiment.preregistration_schema_error(payload)
        self.assertIsInstance(error, str)
        self.assertIn("expected_attrition_rate", error)

        payload["power"]["expected_attrition_rate"] = -0.1
        error = experiment.preregistration_schema_error(payload)
        self.assertIsInstance(error, str)
        self.assertIn("expected_attrition_rate", error)

    def test_contrast_power_accounts_for_factorial_variance(self):
        power = {
            "expected_seed_sd_relative": 0.01,
            "target_power": 0.8,
            "alpha": 0.05,
        }
        pairwise = experiment.preregistration_power_plan(
            power, 0.02, 6, 1, contrast_types=["pairwise"]
        )
        main = experiment.preregistration_power_plan(
            power, 0.02, 6, 1, contrast_types=["main"]
        )
        interaction = experiment.preregistration_power_plan(
            power,
            0.02,
            6,
            1,
            contrast_types=[
                "pairwise",
                "main",
                "interaction",
                "simple",
            ],
        )
        # 主效应方差是两两对比的 0.5 倍，交互是 2 倍；同样
        # 最小效应下交互必须要求更多（或相等）种子，且不能
        # 比两两对比更少。
        self.assertLessEqual(
            main["required_seeds_for_t"],
            pairwise["required_seeds_for_t"],
        )
        self.assertGreater(
            interaction["contrast_power"]["interaction"][
                "required_seeds_for_t"
            ],
            interaction["contrast_power"]["pairwise"][
                "required_seeds_for_t"
            ],
        )
        self.assertEqual(
            interaction["binding_contrast"], "interaction"
        )
        self.assertAlmostEqual(
            interaction["contrast_power"]["main"][
                "variance_factor"
            ],
            0.5,
        )
        self.assertAlmostEqual(
            interaction["contrast_power"]["interaction"][
                "variance_factor"
            ],
            2.0,
        )
        # 6 个种子对两两对比够用，但对交互不够：adequacy 必须
        # 按最吃种子的对比类型判定。
        self.assertTrue(pairwise["adequate"])
        self.assertFalse(interaction["adequate"])

    def test_preregistration_power_rejects_underpowered_interaction(self):
        payload = self._payload()
        payload["min_effect"] = 0.02
        payload["seeds"] = list(range(42, 49))
        payload["power"] = {
            "expected_seed_sd_relative": 0.01,
            "target_power": 0.8,
            "alpha": 0.05,
            "planned_contrasts": ["pairwise", "interaction"],
        }
        # 7 个种子足以做两两对比，但不足以检验方差更大的交互。
        error = experiment.preregistration_schema_error(payload)
        self.assertIsInstance(error, str)
        self.assertIn("interaction", error)
        self.assertIn("交互", error)

        payload["seeds"] = list(range(42, 50))
        self.assertIsNone(
            experiment.preregistration_schema_error(payload)
        )

        payload["power"]["planned_contrasts"] = ["pairwise"]
        self.assertIsNone(
            experiment.preregistration_schema_error(payload)
        )

    def test_preregistration_power_rejects_unknown_contrast_type(self):
        payload = self._payload()
        payload["power"] = {
            "expected_seed_sd_relative": 0.005,
            "target_power": 0.8,
            "alpha": 0.05,
            "planned_contrasts": ["pairwise", "interactoin"],
        }
        error = experiment.preregistration_schema_error(payload)
        self.assertIsInstance(error, str)
        self.assertIn("planned_contrasts", error)
        self.assertIn("interactoin", error)

    def test_planned_contrast_types_follow_plan_shape(self):
        modes_only = experiment.build_sweep_plan(
            seeds=[1, 2, 3],
            modes=["A", "B"],
            ablation=False,
            balance_coef=0.01,
            default_mode="B",
        )
        self.assertEqual(
            experiment.planned_contrast_types_for_plan(modes_only),
            ["pairwise"],
        )
        ablation = experiment.build_sweep_plan(
            seeds=[1, 2, 3],
            modes=None,
            ablation=True,
            balance_coef=0.01,
            default_mode="B",
            ablation_modes=["A"],
        )
        self.assertEqual(
            experiment.planned_contrast_types_for_plan(ablation),
            ["pairwise", "main", "interaction", "simple"],
        )
        single_arm = experiment.build_sweep_plan(
            seeds=[1, 2, 3],
            modes=None,
            ablation=False,
            balance_coef=0.01,
            default_mode="A",
        )
        self.assertEqual(
            experiment.planned_contrast_types_for_plan(single_arm),
            [],
        )

    def test_design_plan_uses_ablation_contrast_power(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                seeds=[1, 2, 3, 4, 5, 6, 7],
                allow_underpowered=False,
                steps=10,
                batch_size=2,
                seq_len=8,
                suite_modes=None,
                suite_ablation=True,
                ablation_modes=["A"],
                balance_coef=0.01,
                mode="B",
                run_order="randomized",
                run_order_seed=5,
                preregistration_record={
                    "study_id": "ablation-power-test",
                    "power": {
                        "expected_seed_sd_relative": 0.01,
                        "min_effect_relative": 0.02,
                        "target_power": 0.8,
                        "alpha": 0.05,
                        "planned_comparisons": 7,
                        "planned_seeds": 7,
                        "underpowered_acknowledged": False,
                    },
                },
            )
            plan, design, planned = (
                experiment.build_sweep_execution(
                    args, Path(tmp)
                )
            )
            self.assertEqual(len(plan), 4 * 7)
            self.assertEqual(planned, 7)
            # 计划实际会估计主效应/交互/简单效应，设计文件不能
            # 只保留预注册里默认的两两对比口径。
            self.assertEqual(
                design["planned_contrasts"],
                ["pairwise", "main", "interaction", "simple"],
            )
            power = design["power"]
            self.assertEqual(
                power["binding_contrast"], "interaction"
            )
            self.assertIn("interaction", power["contrast_power"])
            self.assertGreater(
                power["contrast_power"]["interaction"][
                    "required_seeds_for_t"
                ],
                power["contrast_power"]["pairwise"][
                    "required_seeds_for_t"
                ],
            )
            # 7 个种子对两两对比够、对交互不够：启动硬门必须
            # 在设计阶段拒绝，而不是等分析时解释不显著。
            self.assertFalse(power["adequate"])
            self.assertIsNotNone(
                experiment.design_power_gate_error(design)
            )
            self.assertIsNone(
                experiment.design_power_gate_error(
                    design, allow_underpowered=True
                )
            )

            acknowledged = dict(design)
            acknowledged_power = dict(power)
            acknowledged_power[
                "underpowered_acknowledged"
            ] = True
            acknowledged["power"] = acknowledged_power
            self.assertIsNone(
                experiment.design_power_gate_error(acknowledged)
            )

    def _write_pilot_run(
        self,
        root: Path,
        mode: str,
        seed: int,
        value: float,
        *,
        failed: bool = False,
        balance_coef: float = 0.01,
    ) -> Path:
        run_dir = root / f"{mode}_seed{seed}"
        run_dir.mkdir()
        (run_dir / "config.json").write_text(
            json.dumps({
                "config": {
                    "mode": mode,
                    "balance_coef": balance_coef,
                },
                "args": {"seed": seed},
            }),
            encoding="utf-8",
        )
        final = {
            "step": "final",
            "val_loss": value,
            "numerical_integrity_ok": True,
        }
        if failed:
            final["numerical_stop"] = True
        (run_dir / "metrics.jsonl").write_text(
            "\n".join(
                json.dumps(record)
                for record in (
                    {"step": 100, "val_loss": value},
                    final,
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return run_dir

    def test_pilot_missing_expected_protocol_is_unverified(self):
        # 正式计划无法唯一确定的协议项必须标记为未核对；
        # 试点运行彼此一致不能替代与正式计划的比对。
        rows = [{
            "run": "pilot_a",
            "protocol": {"lr": 0.01},
            "protocol_recorded_keys": ["lr"],
        }]
        report = experiment._pilot_comparability_report(
            rows, {}, {}
        )
        item = next(
            item
            for item in report["items"]
            if item["key"] == "lr"
        )
        self.assertFalse(item["expected_recorded"])
        self.assertEqual(item["status"], "not_recorded")
        self.assertIn(
            "lr",
            [entry["key"] for entry in report["unverified_keys"]],
        )
        self.assertFalse(report["verified"])
        self.assertEqual(report["status"], "unverified")

    def test_pilot_comparability_keeps_duplicate_run_names(self):
        # 不同目录可能使用相同 basename；按 run 名索引会把
        # 后一个运行覆盖前一个，从而漏掉真实的协议差异。
        rows = [
            {
                "run": "pilot",
                "protocol": {"lr": 0.001},
                "protocol_recorded_keys": ["lr"],
            },
            {
                "run": "pilot",
                "protocol": {"lr": 0.01},
                "protocol_recorded_keys": ["lr"],
            },
        ]
        report = experiment._pilot_comparability_report(
            rows, {"lr": 0.01}, {}
        )
        item = next(
            item
            for item in report["items"]
            if item["key"] == "lr"
        )
        self.assertEqual(len(item["values"]), 2)
        self.assertEqual(item["status"], "mismatch")
        self.assertIn(item, report["blocking_mismatches"])

    def test_pilot_expected_cleaning_flags_use_bool_encoding(self):
        # comparison_protocol 把清洗开关统一写成 bool；直接
        # 调用方传入 1/0 时，期望值不能保留整数类型。
        args = argparse.Namespace(
            deduplicate_train_groups=1,
            drop_train_shared_chunks=0,
        )
        expected = (
            experiment._pilot_expected_comparability_protocol(
                args, {}, "val_loss", None
            )
        )
        self.assertIs(
            expected["deduplicate_train_groups"], True
        )
        self.assertIs(
            expected["drop_train_shared_chunks"], False
        )
        rows = [{
            "run": "pilot",
            "protocol": {
                "deduplicate_train_groups": True,
                "drop_train_shared_chunks": False,
            },
            "protocol_recorded_keys": [
                "deduplicate_train_groups",
                "drop_train_shared_chunks",
            ],
        }]
        item = experiment._pilot_comparability_item(
            "deduplicate_train_groups",
            "data",
            "blocking",
            expected,
            rows,
        )
        self.assertEqual(item["status"], "match")

    def test_pilot_expected_hashes_are_case_insensitive(self):
        upper_data = "A" * 64
        upper_split = {"train": "B" * 64}
        args = argparse.Namespace(
            seeds=[1],
            data_split_sha256=upper_split,
        )
        expected = (
            experiment._pilot_expected_comparability_protocol(
                args,
                {"data": {"sha256": upper_data}},
                "val_loss",
                None,
            )
        )
        self.assertEqual(
            expected["data_sha256"], upper_data.lower()
        )
        self.assertEqual(
            expected["split_sha256"],
            {"train": upper_split["train"].lower()},
        )
        rows = [{
            "run": "pilot",
            "protocol": {
                "data_sha256": upper_data,
                "split_sha256": upper_split,
            },
            "protocol_recorded_keys": [
                "data_sha256",
                "split_sha256",
            ],
        }]
        for key in ("data_sha256", "split_sha256"):
            item = experiment._pilot_comparability_item(
                key, "data", "blocking", expected, rows
            )
            self.assertEqual(item["status"], "match")

    def test_pilot_comparability_accepts_single_row(self):
        report = experiment._pilot_comparability_report(
            {
                "run": "pilot",
                "protocol": {"lr": 0.01},
                "protocol_recorded_keys": ["lr"],
            },
            {"lr": 0.01},
            {},
        )
        item = next(
            item
            for item in report["items"]
            if item["key"] == "lr"
        )
        self.assertEqual(item["status"], "match")
        self.assertEqual(item["values"], {"pilot": 0.01})
        unverified = experiment._pilot_comparability_item(
            "lr", "training", "blocking", None, [{
                "run": "pilot",
                "protocol": {"lr": 0.01},
                "protocol_recorded_keys": ["lr"],
            }]
        )
        self.assertEqual(unverified["status"], "not_recorded")

    def test_pilot_declared_check_rejects_nonfinite_values(self):
        self.assertEqual(
            experiment._pilot_declared_check(10**400, 0.1),
            "not_evaluable",
        )
        self.assertEqual(
            experiment._pilot_declared_check(0.1, 10**400),
            "not_evaluable",
        )
        self.assertEqual(
            experiment._pilot_declared_check(
                float("nan"), 0.1
            ),
            "not_evaluable",
        )
        self.assertEqual(
            experiment._pilot_declared_check(0.2, 0.1),
            "declared_at_least_observed",
        )

    def test_chi_square_quantile_handles_small_probability(self):
        # df=2 时 CDF=1-exp(-x/2)，下侧分位数有
        # 解析解 -2*log(1-p)。极端置信水平（如
        # confidence=1-1e-15）会产生远小于 1e-12 的分位数，
        # 二分不能再用绝对容差提前停止；固定 200 步还会把
        # p=1e-100 的分位数卡在 2^-200 附近（约 1e-61），
        # 因此必须覆盖到双精度指数范围的更小概率。
        for probability in (
            1.0e-12,
            1.0e-15,
            1.0e-100,
            1.0e-200,
            1.0e-300,
        ):
            quantile = experiment._chi_square_quantile(
                probability, 2.0
            )
            expected = -2.0 * math.log1p(-probability)
            self.assertIsNotNone(quantile)
            self.assertAlmostEqual(
                quantile / expected, 1.0, places=12
            )

    def test_chi_square_quantile_handles_upper_tail(self):
        # df=2 时上侧概率 Q=exp(-x/2)，因此分位数解析解为
        # -2*log(1-p)。若通过 1-P 计算 Q，p 接近 1 时会把
        # Q 的绝对舍入误差放大成分位数误差。
        for probability in (
            0.6,
            0.9,
            1.0 - 1.0e-8,
            1.0 - 1.0e-12,
            1.0 - 1.0e-15,
        ):
            quantile = experiment._chi_square_quantile(
                probability, 2.0
            )
            expected = -2.0 * math.log1p(-probability)
            self.assertIsNotNone(quantile)
            self.assertAlmostEqual(
                quantile / expected, 1.0, places=12
            )

    def test_chi_square_quantile_handles_small_df(self):
        # df=0.5（a=0.25）的小 x 展开为
        # P(a,x)=x^a/(a*Gamma(a))*(1+O(x))，因此分位数约为
        # 2*(p*a*Gamma(a))^(1/a)。p=1e-15 时修正项在 1e-60
        # 量级，远低于断言精度；固定 200 步二分会高估约 15%。
        probability = 1.0e-15
        df = 0.5
        a = df / 2.0
        quantile = experiment._chi_square_quantile(
            probability, df
        )
        expected = 2.0 * (
            probability * a * math.gamma(a)
        ) ** (1.0 / a)
        self.assertIsNotNone(quantile)
        self.assertAlmostEqual(
            quantile / expected, 1.0, places=12
        )

    def test_pilot_expected_planned_seeds_normalize_strings(self):
        expected = (
            experiment._pilot_expected_comparability_protocol(
                argparse.Namespace(seeds=["1", "2"]),
                {},
                "val_loss",
                None,
            )
        )
        self.assertEqual(expected["planned_seeds"], [1, 2])

    def test_pilot_expected_split_hash_rejects_non_mapping(self):
        args = argparse.Namespace(
            seeds=[1],
            data_split_sha256=["not", "a", "mapping"],
            lr=1e-3,
            warmup_steps=1,
            steps=2,
            weight_decay=0.0,
            grad_clip=1.0,
            batch_size=1,
        )
        expected = (
            experiment._pilot_expected_comparability_protocol(
                args, {}, "val_loss", None
            )
        )
        self.assertNotIn("split_sha256", expected)
        protocol = experiment.build_comparison_protocol(
            _tiny_config(), args
        )
        self.assertIsNone(protocol["split_sha256"])

    def test_pilot_gate_note_only_flags_upper_when_above_declared(
        self,
    ):
        calibration = {
            "sd_basis": "upper_confidence_bound",
            "confidence_level": 0.8,
            "sd_effective_input": 0.01,
            "attrition_basis": "upper_confidence_bound",
            "attrition_effective_input": 0.1,
        }
        evidence = {
            "calibration": calibration,
            "sd_check": {
                "declared": 0.02,
                "status": "declared_at_least_observed",
                "upper_confidence_bound_status": (
                    "declared_at_least_observed"
                ),
            },
            "attrition_check": {
                "declared": 0.2,
                "status": "declared_at_least_observed",
                "upper_confidence_bound_status": (
                    "declared_at_least_observed"
                ),
            },
        }
        self.assertEqual(
            experiment._pilot_gate_note(evidence), ""
        )
        evidence["sd_check"][
            "upper_confidence_bound_status"
        ] = "declared_below_observed"
        note = experiment._pilot_gate_note(evidence)
        self.assertIn("单侧置信上界", note)
        self.assertIn("高于预注册", note)

    def test_pilot_seed_overlap_tolerates_scalar_evidence(self):
        expected = (
            experiment._pilot_expected_comparability_protocol(
                argparse.Namespace(seeds=7),
                {},
                "val_loss",
                None,
            )
        )
        self.assertEqual(expected["planned_seeds"], [7])
        report = experiment._pilot_comparability_report(
            [], {"planned_seeds": 7}, {"cells": 7}
        )
        self.assertEqual(
            report["seed_overlap"]["planned_seed_count"], 1
        )
        self.assertEqual(
            report["seed_overlap"]["reused_seed_count"], 0
        )
        overlap = experiment._pilot_seed_overlap(
            {
                "cells": [{
                    "label": "pilot",
                    "seed_values": [7],
                }],
            },
            [7],
        )
        self.assertEqual(overlap["reused_seed_count"], 1)

    def test_pilot_evidence_estimates_paired_sd_and_attrition(self):
        smelt = [1.0, 1.0005, 0.9995, 1.0002, 0.9998]
        variant = [0.995, 0.9945, 0.9948, 0.9951, 0.9947]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index, seed in enumerate(range(1, 6)):
                paths.append(
                    str(self._write_pilot_run(
                        root, "smelt", seed, smelt[index]
                    ))
                )
                paths.append(
                    str(self._write_pilot_run(
                        root,
                        "A",
                        seed,
                        variant[index],
                        failed=(seed == 5),
                    ))
                )
            evidence = (
                experiment.build_pilot_power_evidence(
                    paths,
                    "val_loss",
                    target_step=100,
                    declared_sd=0.0001,
                    declared_attrition=0.0,
                    planned_modes=["smelt", "A", "B"],
                )
            )

        self.assertEqual(evidence["cell_count"], 2)
        self.assertEqual(evidence["pair_count"], 1)
        # val_loss 是记录值而不是共同步数派生指标：即使调用方
        # 传了 target_step，也不能声称该指标取自共同步数。
        self.assertEqual(
            evidence["metric_annotation"]["kind"],
            "recorded_metric",
        )
        self.assertIsNone(
            evidence["metric_annotation"]["target_step"]
        )
        pair = evidence["pairs"][0]
        # A 的第 5 个种子数值失败，不能进入配对差值。
        self.assertEqual(pair["n_pairs"], 4)
        self.assertTrue(pair["evaluable"])
        self.assertIsNotNone(pair["paired_sd_abs"])
        self.assertGreater(
            evidence["observed_paired_sd_relative"], 0.0001
        )
        self.assertEqual(
            evidence["sd_check"]["status"],
            "declared_below_observed",
        )
        self.assertAlmostEqual(
            evidence["attrition_check"]["observed"], 0.2
        )
        self.assertEqual(
            evidence["attrition_check"]["status"],
            "declared_below_observed",
        )
        self.assertEqual(
            evidence["arm_coverage"]["missing_modes"], ["B"]
        )
        self.assertTrue(
            any(
                "未覆盖计划中的模式" in warning
                for warning in evidence["warnings"]
            )
        )

    def test_design_plan_recomputes_power_from_pilot_runs(self):
        smelt = [1.0, 1.0005, 0.9995, 1.0002, 0.9998, 1.0001]
        variant = [0.995, 0.9945, 0.9948, 0.9951, 0.9947, 0.9952]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index, seed in enumerate(range(1, 7)):
                paths.append(
                    str(self._write_pilot_run(
                        root, "smelt", seed, smelt[index]
                    ))
                )
                paths.append(
                    str(self._write_pilot_run(
                        root,
                        "A",
                        seed,
                        variant[index],
                        failed=(seed == 6),
                    ))
                )
            args = argparse.Namespace(
                seeds=[1, 2, 3, 4, 5, 6],
                allow_underpowered=False,
                steps=10,
                batch_size=2,
                seq_len=8,
                suite_modes=["smelt", "A"],
                suite_ablation=False,
                ablation_modes=None,
                balance_coef=0.01,
                mode="A",
                primary_metric="val_loss",
                common_step=100,
                run_order="randomized",
                run_order_seed=5,
                design_pilot_runs=paths,
                preregistration_record={
                    "study_id": "pilot-power-test",
                    "primary_metric": "val_loss",
                    "power": {
                        "expected_seed_sd_relative": 0.0001,
                        "min_effect_relative": 0.0005,
                        "target_power": 0.8,
                        "alpha": 0.05,
                        "planned_comparisons": 1,
                        "planned_contrasts": ["pairwise"],
                        "planned_seeds": 6,
                        "expected_attrition_rate": 0.0,
                        "underpowered_acknowledged": False,
                        "adequate": True,
                    },
                },
            )
            _, design, _ = experiment.build_sweep_execution(
                args, root / "plan"
            )

        power = design["power"]
        self.assertTrue(power["power_recomputed_for_pilot"])
        self.assertGreater(
            power["expected_seed_sd_relative"], 0.0001
        )
        self.assertAlmostEqual(
            power["expected_attrition_rate"], 1.0 / 6.0
        )
        self.assertEqual(
            design["pilot_evidence"]["sd_check"]["status"],
            "declared_below_observed",
        )
        self.assertEqual(
            design["pilot_evidence"]["attrition_check"]["status"],
            "declared_below_observed",
        )
        self.assertTrue(
            design["pilot_evidence"]["applied_to_power"]
        )
        self.assertFalse(power["adequate"])
        gate = experiment.design_power_gate_error(design)
        self.assertIsNotNone(gate)
        self.assertIn("试点观测", gate)
        self.assertIsNone(
            experiment.design_power_gate_error(
                design, allow_underpowered=True
            )
        )

    def test_design_plan_uses_declared_pilot_confidence(self):
        smelt = [1.0, 1.0005, 0.9995, 1.0002, 0.9998, 1.0001]
        variant = [0.995, 0.9945, 0.9948, 0.9951, 0.9947, 0.9952]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index, seed in enumerate(range(1, 7)):
                paths.append(
                    str(self._write_pilot_run(
                        root, "smelt", seed, smelt[index]
                    ))
                )
                paths.append(
                    str(self._write_pilot_run(
                        root, "A", seed, variant[index]
                    ))
                )
            args = argparse.Namespace(
                seeds=[1, 2, 3, 4, 5, 6],
                allow_underpowered=False,
                steps=10,
                batch_size=2,
                seq_len=8,
                suite_modes=["smelt", "A"],
                suite_ablation=False,
                ablation_modes=None,
                balance_coef=0.01,
                mode="A",
                primary_metric="val_loss",
                common_step=100,
                run_order="randomized",
                run_order_seed=5,
                design_pilot_runs=paths,
                preregistration_record={
                    "study_id": "pilot-confidence-test",
                    "primary_metric": "val_loss",
                    "power": {
                        "expected_seed_sd_relative": 0.0001,
                        "min_effect_relative": 0.0005,
                        "target_power": 0.8,
                        "alpha": 0.05,
                        "planned_comparisons": 1,
                        "planned_contrasts": ["pairwise"],
                        "planned_seeds": 6,
                        "expected_attrition_rate": 0.0,
                        "pilot_calibration": (
                            "upper_confidence_bound"
                        ),
                        "pilot_confidence": 0.8,
                    },
                },
            )
            _, design, _ = experiment.build_sweep_execution(
                args, root / "plan"
            )

        evidence = design["pilot_evidence"]
        self.assertEqual(
            design["power"]["pilot_calibration"],
            "upper_confidence_bound",
        )
        self.assertEqual(design["power"]["pilot_confidence"], 0.8)
        self.assertEqual(
            evidence["uncertainty"]["confidence_level"], 0.8
        )
        self.assertEqual(
            evidence["calibration"]["confidence_level"], 0.8
        )
        self.assertEqual(
            evidence["calibration"]["sd_basis"],
            "upper_confidence_bound",
        )
        self.assertGreater(
            evidence["calibration"]["sd_effective_input"],
            evidence["calibration"]["sd_point_estimate"],
        )

    def test_pilot_recompute_failure_is_not_reported_as_success(self):
        smelt = [1.0, 1.0005, 0.9995, 1.0002, 0.9998, 1.0001]
        variant = [0.995, 0.9945, 0.9948, 0.9951, 0.9947, 0.9952]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index, seed in enumerate(range(1, 7)):
                paths.append(
                    str(self._write_pilot_run(
                        root, "smelt", seed, smelt[index]
                    ))
                )
                paths.append(
                    str(self._write_pilot_run(
                        root, "A", seed, variant[index]
                    ))
                )
            # 旧/手工 power 记录缺少 min_effect_relative 等
            # 重算字段：试点输入仍会被上调，但功效无法重算。
            args = argparse.Namespace(
                seeds=[1, 2, 3, 4, 5, 6],
                allow_underpowered=False,
                steps=10,
                batch_size=2,
                seq_len=8,
                primary_metric="val_loss",
                common_step=100,
                design_pilot_runs=paths,
                preregistration_record={
                    "power": {
                        "expected_seed_sd_relative": 1e-6,
                        "target_power": 0.8,
                        "alpha": 0.05,
                        "planned_comparisons": 1,
                        "planned_contrasts": ["pairwise"],
                        "planned_seeds": 6,
                        "expected_attrition_rate": 0.0,
                        "adequate": True,
                    },
                },
            )
            plan = [
                {
                    "order_index": index + 1,
                    "block": seed,
                    "position_in_block": 1,
                    "label": f"smelt_seed{seed}",
                    "mode": "smelt",
                    "seed": seed,
                    "balance_coef": 0.01,
                }
                for index, seed in enumerate(range(1, 7))
            ]
            design = experiment.write_design_plan(
                root / "plan", plan, "randomized", 0, args
            )
        self.assertTrue(
            design["pilot_evidence"]["applied_to_power"]
        )
        self.assertFalse(
            design["pilot_evidence"]["power_recompute_ok"]
        )
        self.assertFalse(
            design["power"]["power_recomputed_for_pilot"]
        )
        self.assertIn(
            "power_recompute_error",
            design["pilot_evidence"],
        )
        gate = experiment.design_power_gate_error(design)
        self.assertIsNotNone(gate)
        self.assertIn("无法", gate)
        self.assertNotIn("None 个启动种子", gate)

    def test_pilot_evidence_ratios_do_not_overflow(self):
        # 一个单元内 ±1e308 与极小的值可以同时让均值和 sd
        # 有限，但相对 sd 的除法溢出；这类字段必须省略而不是
        # 写入 inf/NaN。
        smelt = [1.0e308, -1.0e308, 1.0e-308, 1.0e-308,
                 1.0e-308, 1.0e-308]
        variant = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index, seed in enumerate(range(1, 7)):
                paths.append(
                    str(self._write_pilot_run(
                        root, "smelt", seed, smelt[index]
                    ))
                )
                paths.append(
                    str(self._write_pilot_run(
                        root, "A", seed, variant[index]
                    ))
                )
            evidence = experiment.build_pilot_power_evidence(
                paths,
                "val_loss",
                declared_sd=1.0e-300,
                declared_attrition=0.0,
            )
        cell = next(
            item
            for item in evidence["cells"]
            if item["mode"] == "smelt"
        )
        self.assertIsNone(cell["seed_sd_relative"])
        self.assertIsNone(
            evidence["sd_check"]["ratio_observed_over_declared"]
        )
        json.dumps(evidence, allow_nan=False)

    def test_pilot_unassigned_failed_run_counts_toward_attrition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index, seed in enumerate(range(1, 4)):
                paths.append(
                    str(self._write_pilot_run(
                        root, "smelt", seed, 1.0 + index * 0.001
                    ))
                )
                paths.append(
                    str(self._write_pilot_run(
                        root, "A", seed, 0.99 + index * 0.001
                    ))
                )
            # 崩溃到连 config/metrics 都没写出的运行目录：无法
            # 归入处理单元，但不能静默把它的脱落忽略。
            empty = root / "crashed"
            empty.mkdir()
            paths.append(str(empty))
            evidence = experiment.build_pilot_power_evidence(
                paths,
                "val_loss",
                declared_sd=0.01,
                declared_attrition=0.0,
            )
        self.assertEqual(evidence["cell_count"], 2)
        self.assertEqual(
            evidence["attrition_check"]["unassigned_runs"], 1
        )
        self.assertEqual(
            evidence["attrition_check"]["unassigned_unusable"], 1
        )
        self.assertEqual(
            evidence["attrition_check"]["observed"], 1.0
        )
        self.assertTrue(
            evidence["attrition_check"]["unusable"]
        )
        self.assertTrue(
            any(
                "无法归入处理单元" in warning
                for warning in evidence["warnings"]
            )
        )
        design = {
            "power": {"adequate": True},
            "pilot_evidence": evidence,
        }
        self.assertIsNotNone(
            experiment.design_power_gate_error(design)
        )
        json.dumps(evidence, allow_nan=False)

    def test_design_summary_distinguishes_undeclared_pilot(self):
        design = {
            "planned_runs": 2,
            "order_scheme": "randomized",
            "order_seed": 0,
            "plan_id": "x" * 20,
            "path": "design.json",
            "power": {
                "adequate": True,
                "planned_seeds": 6,
                "required_seeds_with_attrition": 6,
                "required_seeds_for_t": 3,
                "required_seeds_for_robust_sign_flip": 6,
                "expected_attrition_rate": 0.0,
                "planned_usable_seeds": 6,
            },
            "pilot_evidence": {
                "source_run_count": 6,
                "cell_count": 2,
                "pair_count": 1,
                "metric": "val_loss",
                "metric_annotation": {
                    "kind": "recorded_metric",
                    "target_step": None,
                },
                "sd_check": {
                    "status": "not_declared",
                    "observed": 0.01,
                    "min_pairs": 3,
                },
                "attrition_check": {
                    "status": "not_declared",
                    "observed": 0.0,
                },
                "warnings": [],
            },
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            experiment.print_design_summary(design)
        text = output.getvalue()
        self.assertIn("未声明", text)
        self.assertNotIn("无法估计配对差值", text)

    def test_pilot_evidence_sanitizes_nonfinite_config_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for seed in (1, 2, 3):
                paths.append(
                    str(self._write_pilot_run(
                        root,
                        "smelt",
                        seed,
                        1.0 + seed * 0.001,
                        balance_coef=float("nan"),
                    ))
                )
                paths.append(
                    str(self._write_pilot_run(
                        root,
                        "A",
                        seed,
                        0.99 + seed * 0.001,
                    ))
                )
            evidence = experiment.build_pilot_power_evidence(
                paths, "val_loss"
            )
        cell = next(
            item
            for item in evidence["cells"]
            if item["mode"] == "smelt"
        )
        self.assertIsNone(cell["balance_coef"])
        for source in evidence["source_runs"]:
            if source["mode"] == "smelt":
                self.assertIsNone(source["balance_coef"])
        json.dumps(evidence, allow_nan=False)

    def test_pilot_labels_disambiguate_structural_axes(self):
        # 同一 mode/balance 但 k1 不同的两个处理单元不能共用
        # 标签，否则 pair 的 first/second 无法区分。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for k1 in (1, 2):
                for seed in (1, 2, 3):
                    run_dir = root / f"A_k{k1}_{seed}"
                    run_dir.mkdir()
                    (run_dir / "config.json").write_text(
                        json.dumps({
                            "config": {
                                "mode": "A",
                                "balance_coef": 0.01,
                                "k1": k1,
                            },
                            "args": {"seed": seed},
                        }),
                        encoding="utf-8",
                    )
                    value = 1.0 - 0.1 * k1 + 0.001 * seed
                    (run_dir / "metrics.jsonl").write_text(
                        "\n".join(
                            json.dumps(record)
                            for record in (
                                {"step": 10, "val_loss": value},
                                {
                                    "step": "final",
                                    "val_loss": value,
                                    "numerical_integrity_ok": True,
                                },
                            )
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    paths.append(str(run_dir))
            evidence = experiment.build_pilot_power_evidence(
                paths, "val_loss"
            )
        labels = [cell["label"] for cell in evidence["cells"]]
        self.assertEqual(len(labels), 2)
        self.assertEqual(len(set(labels)), 2)
        self.assertNotEqual(
            evidence["pairs"][0]["first"],
            evidence["pairs"][0]["second"],
        )

    def test_design_return_value_is_standard_json_safe(self):
        # power 中的非有限声明值会被 _apply 复制进 pilot_evidence
        # 的 adjusted_power_inputs；返回给调用方的设计对象也必须
        # 能按 allow_nan=False 序列化，而不能只在写文件时清洗。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index, seed in enumerate(range(1, 4)):
                paths.append(
                    str(self._write_pilot_run(
                        root, "smelt", seed, 1.0 + index * 0.001
                    ))
                )
                paths.append(
                    str(self._write_pilot_run(
                        root, "A", seed, 0.99 + index * 0.001
                    ))
                )
            args = argparse.Namespace(
                seeds=[1, 2, 3],
                allow_underpowered=False,
                steps=10,
                batch_size=2,
                seq_len=8,
                primary_metric="val_loss",
                common_step=100,
                design_pilot_runs=paths,
                preregistration_record={
                    "power": {
                        "expected_seed_sd_relative": float("inf"),
                        "expected_attrition_rate": -1.0,
                        "planned_comparisons": 1,
                        "planned_contrasts": ["pairwise"],
                        "planned_seeds": 3,
                        "target_power": 0.8,
                        "alpha": 0.05,
                        "adequate": True,
                    },
                },
            )
            plan = [
                {
                    "order_index": index + 1,
                    "block": seed,
                    "position_in_block": 1,
                    "label": f"smelt_seed{seed}",
                    "mode": "smelt",
                    "seed": seed,
                    "balance_coef": 0.01,
                }
                for index, seed in enumerate(range(1, 4))
            ]
            design = experiment.write_design_plan(
                root / "plan", plan, "randomized", 0, args
            )
        json.dumps(design, allow_nan=False)
        json.dumps(design["pilot_evidence"], allow_nan=False)

    def test_pilot_evidence_with_no_usable_cell_blocks_start(self):
        # 运行缺少 mode/seed 但指标本身有效：虽然不算「数值
        # 失败」，也没有任何单元可用于核对 sd/脱落率，不能
        # 当作功效足够放行。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "unassigned"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(
                json.dumps({
                    "config": {"mode": None, "balance_coef": 0.01},
                    "args": {"seed": None},
                }),
                encoding="utf-8",
            )
            (run_dir / "metrics.jsonl").write_text(
                json.dumps({"step": 1, "val_loss": 1.0})
                + "\n"
                + json.dumps({
                    "step": "final",
                    "val_loss": 1.0,
                    "numerical_integrity_ok": True,
                })
                + "\n",
                encoding="utf-8",
            )
            evidence = experiment.build_pilot_power_evidence(
                [str(run_dir)], "val_loss"
            )
        self.assertEqual(evidence["cell_count"], 0)
        self.assertFalse(
            evidence["attrition_check"]["unusable"]
        )
        design = {
            "power": {"adequate": True},
            "pilot_evidence": evidence,
        }
        gate = experiment.design_power_gate_error(design)
        self.assertIsNotNone(gate)
        self.assertIn("没有", gate)
        self.assertIsNone(
            experiment.design_power_gate_error(
                design, allow_underpowered=True
            )
        )
        # 即使没有 power 块，零可用单元的试点证据也不能静默
        # 放行；--allow-underpowered 仍是显式逃生口。
        self.assertIsNotNone(
            experiment.design_power_gate_error(
                {"pilot_evidence": evidence}
            )
        )
        self.assertIsNone(
            experiment.design_power_gate_error(
                {"pilot_evidence": evidence},
                allow_underpowered=True,
            )
        )

    def test_pilot_evidence_all_runs_failed_blocks_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for seed in (1, 2, 3):
                paths.append(str(self._write_pilot_run(
                    root, "smelt", seed, 1.0 - seed * 0.001
                )))
                paths.append(str(self._write_pilot_run(
                    root,
                    "A",
                    seed,
                    0.99 - seed * 0.001,
                    failed=True,
                )))
            evidence = experiment.build_pilot_power_evidence(
                paths,
                "val_loss",
                target_step=100,
                declared_sd=0.01,
                declared_attrition=0.0,
            )
        self.assertEqual(
            evidence["attrition_check"]["observed"], 1.0
        )
        self.assertTrue(evidence["attrition_check"]["unusable"])
        design = {
            "power": {
                "adequate": True,
                "planned_seeds": 6,
                "planned_comparisons": 1,
            },
            "pilot_evidence": evidence,
        }
        gate = experiment.design_power_gate_error(design)
        self.assertIsNotNone(gate)
        self.assertIn("不可用", gate)
        self.assertIsNone(
            experiment.design_power_gate_error(
                design, allow_underpowered=True
            )
        )
        # --allow-underpowered 是显式的探索逃生口，即使设计
        # 计划没有 power 块也必须生效（与旧行为一致）。
        no_power = {"pilot_evidence": evidence}
        self.assertIsNotNone(
            experiment.design_power_gate_error(no_power)
        )
        self.assertIsNone(
            experiment.design_power_gate_error(
                no_power, allow_underpowered=True
            )
        )


class RoutingTests(unittest.TestCase):
    def test_chance_levels_are_reported(self):
        a = torch.tensor([[0, 1], [2, 3], [0, 2]])
        b = torch.tensor([[1, 2], [2, 0], [1, 3]])
        metrics = experiment._routing_set_metrics(
            a, b, 4
        )
        self.assertIsNotNone(metrics)
        self.assertGreaterEqual(
            metrics["jaccard_chance_uniform"], 0.0
        )
        self.assertGreaterEqual(
            metrics["jaccard_chance_marginal"], 0.0
        )
        self.assertIsNotNone(
            metrics["jaccard_lift_over_marginal"]
        )

    def test_routing_set_bool_does_not_mutate_or_fail_on_views(self):
        indices = torch.tensor([[-1, 99]], dtype=torch.int64)
        experiment._routing_set_bool(indices, 5)
        self.assertEqual(indices.tolist(), [[-1, 99]])
        # stride-0 的 expand 视图不能做原地 clamp_；分支创建时
        # 的 a_expanded 正是这种布局。
        base = torch.tensor([[0, 1]], dtype=torch.int64)
        expanded = base.expand(2, 2)
        flags = experiment._routing_set_bool(expanded, 4)
        self.assertEqual(flags.shape, (2, 4))
        self.assertEqual(base.tolist(), [[0, 1]])

    def test_branch_creation_overlap_aligns_each_branch_slot(self):
        cfg = experiment.Config(
            mode="B",
            n_experts=4,
            d_model=4,
            n_heads=1,
            n_layers=4,
            seq_len=1,
            expert_hidden=2,
            top_k=2,
            k1=2,
            k2=2,
        )
        probe = experiment.RoutingProbe(cfg, batch_size=2)
        first = {
            "indices": torch.tensor([[[0, 1]], [[2, 3]]]),
            "input_shape": [2, 1, 4],
        }
        second = {
            "indices": torch.tensor(
                [[0, 2], [1, 3], [2, 0], [3, 1]]
            ),
            "input_shape": [4, 1, 4],
        }
        overlap = probe._pair_overlap(
            first, second, cfg.middle_start
        )
        # 每个分支只与它自己携带的第一轮专家比较：
        # {0}->{0,2}, {1}->{1,3}, {2}->{2,0}, {3}->{3,1}。
        self.assertAlmostEqual(
            overlap["branch_level"][
                "second_round_jaccard_mean"
            ],
            0.5,
        )
        self.assertAlmostEqual(
            overlap["branch_level"]["first_round_recall_mean"],
            1.0,
        )
        self.assertEqual(overlap["branch_self_reuse_share"], 1.0)

    def test_routing_indices_keep_large_expert_ids(self):
        class _FakeMoE:
            n_experts = 40000

        cfg = experiment.Config(
            mode="base",
            n_experts=40000,
            d_model=2,
            n_heads=1,
            n_layers=4,
            seq_len=1,
            expert_hidden=1,
            top_k=1,
            k1=1,
            k2=1,
        )
        probe = experiment.RoutingProbe(cfg, batch_size=1)
        module = _FakeMoE()
        probe._block_index[id(module)] = 0
        logits = torch.zeros(1, 40000)
        logits[0, 39999] = 10.0
        probe.record(
            module,
            torch.zeros(1, 1, 2),
            torch.tensor([[0, 39999]]),
            torch.tensor([[0.5, 0.5]]),
            logits,
        )
        probe._materialize_events()
        self.assertEqual(
            probe.events[0]["indices"].tolist(),
            [[0, 39999]],
        )

    def test_routing_shuffle_seed_uses_stable_module_index(self):
        first = experiment._routing_shuffle_seed(7, 0)
        self.assertEqual(
            first, experiment._routing_shuffle_seed(7, 0)
        )
        self.assertEqual(
            experiment._routing_shuffle_seed(7, 1),
            experiment._routing_shuffle_seed(7, 1),
        )
        self.assertNotEqual(
            first, experiment._routing_shuffle_seed(8, 0)
        )
        self.assertNotEqual(
            first, experiment._routing_shuffle_seed(7, 1)
        )

    def test_routing_shuffle_is_distinct_and_advances(self):
        generator = torch.Generator().manual_seed(0)
        first = experiment._routing_shuffle_indices(
            4, torch.Size([64, 2]), generator
        )
        second = experiment._routing_shuffle_indices(
            4, torch.Size([64, 2]), generator
        )
        # 每个 token 的 top-k 必须是无放回的不同专家。
        self.assertTrue(
            bool(
                (first[:, 0] != first[:, 1]).all()
            )
        )
        self.assertTrue(
            bool(
                (second[:, 0] != second[:, 1]).all()
            )
        )
        # 第二次调用不能复用第一次的随机 assignment。
        self.assertFalse(torch.equal(first, second))

    def test_routing_shuffle_reproducible_across_fresh_models(self):
        class _FixedEvalDataset:
            def __init__(self, seq_len):
                self.seq_len = seq_len

            def fixed_eval_batch(
                self,
                split,
                batch_index,
                batches,
                batch_size,
                device,
            ):
                generator = torch.Generator().manual_seed(0)
                x = torch.randint(
                    0,
                    256,
                    (batch_size, self.seq_len),
                    generator=generator,
                )
                y = torch.randint(
                    0,
                    256,
                    (batch_size, self.seq_len),
                    generator=generator,
                )
                return x, y

        cfg = _tiny_config(mode="A", seq_len=8, n_layers=4)
        experiment.set_seed(123)
        first_model = experiment.LanguageModel(cfg)
        experiment.set_seed(123)
        second_model = experiment.LanguageModel(cfg)
        dataset = _FixedEvalDataset(cfg.seq_len)

        first = experiment.analyze_routing_interventions(
            first_model,
            dataset,
            batch_size=2,
            batches=1,
            device=torch.device("cpu"),
            amp_dtype=None,
            seed=7,
        )
        second = experiment.analyze_routing_interventions(
            second_model,
            dataset,
            batch_size=2,
            batches=1,
            device=torch.device("cpu"),
            amp_dtype=None,
            seed=7,
        )
        self.assertEqual(
            first["interventions"]["routing_shuffle"]["loss"],
            second["interventions"]["routing_shuffle"]["loss"],
        )
        self.assertEqual(
            first["interventions"]["routing_shuffle"][
                "token_accuracy"
            ],
            second["interventions"]["routing_shuffle"][
                "token_accuracy"
            ],
        )

        other_seed = experiment.analyze_routing_interventions(
            first_model,
            dataset,
            batch_size=2,
            batches=1,
            device=torch.device("cpu"),
            amp_dtype=None,
            seed=8,
        )
        self.assertNotEqual(
            first["interventions"]["routing_shuffle"]["loss"],
            other_seed["interventions"]["routing_shuffle"]["loss"],
        )


class ReproducibilityTests(unittest.TestCase):
    def test_compare_report_file_writes_analysis_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reports" / "analysis.json"
            with contextlib.redirect_stdout(io.StringIO()):
                written = experiment.write_comparison_report_file(
                    path,
                    {
                        "metric": "val_loss",
                        "fragility_summary": {
                            "verdict_counts": {"stable": 1}
                        },
                    },
                )
            self.assertEqual(written, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["metric"], "val_loss")
            self.assertEqual(
                payload["fragility_summary"]["verdict_counts"],
                {"stable": 1},
            )
            self.assertIn("generated_at", payload)

    def test_set_seed_records_effective_manifest(self):
        experiment.set_seed(321)
        manifest = experiment._LAST_SEED_MANIFEST
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest["seed"], 321)
        self.assertEqual(manifest["torch_initial_seed"], 321)
        self.assertTrue(manifest["python_random"])
        if torch.cuda.is_available():
            self.assertTrue(manifest["cuda_initial_seeds"])
            self.assertTrue(
                all(
                    value == 321
                    for value in manifest[
                        "cuda_initial_seeds"
                    ]
                )
            )

    def test_optimizer_coverage_detects_missing_params(self):
        model = experiment.LanguageModel(_tiny_config())
        optimizer = torch.optim.AdamW(
            model.token_embedding.parameters(),
            lr=1e-3,
        )
        report = experiment._probe_optimizer_coverage(
            model, optimizer
        )
        self.assertFalse(report["ok"])
        self.assertTrue(report["missing_params"])

        full = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
        )
        self.assertTrue(
            experiment._probe_optimizer_coverage(
                model, full
            )["ok"]
        )

    def test_fresh_out_dir_guard_protects_existing_tracking(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            self.assertIsNone(
                experiment.fresh_out_dir_error(
                    out_dir,
                    resume_path=None,
                    allow_overwrite=False,
                )
            )
            (out_dir / "metrics.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            error = experiment.fresh_out_dir_error(
                out_dir,
                resume_path=None,
                allow_overwrite=False,
            )
            self.assertIsNotNone(error)
            self.assertIn("metrics.jsonl", error)
            self.assertIsNone(
                experiment.fresh_out_dir_error(
                    out_dir,
                    resume_path=None,
                    allow_overwrite=True,
                )
            )
            self.assertIsNone(
                experiment.fresh_out_dir_error(
                    out_dir,
                    resume_path="last.pt",
                    allow_overwrite=False,
                )
            )

    def test_fresh_out_dir_guard_refuses_unreadable_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with mock.patch.object(
                Path,
                "iterdir",
                side_effect=PermissionError("denied"),
            ):
                error = experiment.fresh_out_dir_error(
                    out_dir,
                    resume_path=None,
                    allow_overwrite=False,
                )
            self.assertIsNotNone(error)
            self.assertIn("无法读取", error)
            self.assertIsNone(
                experiment.fresh_out_dir_error(
                    out_dir,
                    resume_path=None,
                    allow_overwrite=True,
                )
            )

    def test_tracking_header_fields_are_self_contained(self):
        class _Dataset:
            data_sha256 = "data-sha"
            split_sha256 = {"train": "train-sha"}
            effective_train_sha256 = "effective-sha"

        class _Args:
            seed = 7

        header = experiment.tracking_header_fields(
            _Args(),
            _Dataset(),
            {
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "script_sha256": "script-sha",
                "git_commit": "commit-sha",
                "git_dirty": False,
                "pythonhashseed_at_start": "7",
                "rng_seed_manifest": {"seed": 7},
                "deterministic_algorithms": True,
            },
            comparison_protocol_sha256="protocol-sha",
        )
        self.assertEqual(header["seed"], 7)
        self.assertEqual(header["data_sha256"], "data-sha")
        self.assertEqual(
            header["data_split_sha256"]["train"],
            "train-sha",
        )
        self.assertEqual(
            header["data_effective_train_sha256"],
            "effective-sha",
        )
        self.assertEqual(header["script_sha256"], "script-sha")
        self.assertEqual(
            header["comparison_protocol_sha256"],
            "protocol-sha",
        )
        self.assertEqual(
            header["pythonhashseed_at_start"], "7"
        )
        self.assertEqual(
            header["rng_seed_manifest"], {"seed": 7}
        )

    def test_tracking_provenance_prefers_original_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "config.json").write_text(
                json.dumps({
                    "original_run_metadata": {
                        "timestamp_utc": "2026-01-01T00:00:00+00:00",
                        "script_sha256": "original-script",
                        "git_commit": "original-commit",
                        "git_dirty": False,
                    },
                    "run_metadata": {
                        "timestamp_utc": "2026-01-02T00:00:00+00:00",
                        "script_sha256": "resumed-script",
                        "git_commit": "resumed-commit",
                        "git_dirty": True,
                    },
                    "resume_count": 2,
                    "comparison_protocol_sha256": "protocol-sha",
                }),
                encoding="utf-8",
            )
            provenance = experiment.load_tracking_provenance(
                run_dir
            )
            self.assertEqual(
                provenance["run_started_utc"],
                "2026-01-01T00:00:00+00:00",
            )
            self.assertEqual(
                provenance["original_script_sha256"],
                "original-script",
            )
            self.assertEqual(
                provenance["original_git_commit"],
                "original-commit",
            )
            self.assertEqual(provenance["resume_count"], 2)
            self.assertEqual(
                provenance["comparison_protocol_sha256"],
                "protocol-sha",
            )

    def test_tracking_provenance_defaults_without_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            provenance = experiment.load_tracking_provenance(
                Path(tmp)
            )
            self.assertIsNone(provenance["run_started_utc"])
            self.assertEqual(provenance["resume_count"], 0)

    def test_non_expert_digest_tracks_parameter_updates(self):
        model = experiment.LanguageModel(_tiny_config())
        before = experiment._non_expert_parameter_digest(
            model
        )
        self.assertEqual(
            before,
            experiment._non_expert_parameter_digest(model),
        )
        with torch.no_grad():
            model.token_embedding.weight.add_(1.0)
        self.assertNotEqual(
            before,
            experiment._non_expert_parameter_digest(model),
        )


class RunAuditTests(unittest.TestCase):
    """运行级复现审计：跨文件身份必须一致，证据不足要显式披露。"""

    def _build_run(
        self,
        root,
        *,
        metadata_overrides=None,
        config_overrides=None,
    ):
        root.mkdir()
        snapshot = experiment.build_environment_snapshot()
        environment_bytes = json.dumps(
            snapshot, ensure_ascii=False, indent=2
        ).encode("utf-8")
        (root / "environment.json").write_bytes(environment_bytes)
        environment_sha = hashlib.sha256(
            environment_bytes
        ).hexdigest()
        identity = {
            "seed": 7,
            "data_sha256": "b" * 64,
            "data_split_sha256": {
                "train": "1" * 64,
                "val": "2" * 64,
                "test": "3" * 64,
            },
            "script_sha256": "d" * 64,
            "environment_id": snapshot["environment_id"],
            "comparison_protocol_sha256": "c" * 64,
        }
        metadata = {
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "command": (
                "python experiment.py --data corpus.txt --out run"
            ),
            "cwd": "/project",
            "git_commit": "a" * 40,
            "git_dirty": False,
            "git_diff_sha256": None,
            "script_sha256": identity["script_sha256"],
            "environment_id": identity["environment_id"],
            "pythonhashseed_at_start": "7",
            "rng_seed_manifest": {"seed": 7},
            "deterministic_algorithms": True,
            "deterministic_algorithms_warn_only": False,
            "cudnn_deterministic": True,
            "allow_tf32_matmul": False,
            "allow_tf32_cudnn": False,
        }
        if metadata_overrides:
            metadata.update(metadata_overrides)
        config = {
            "config_schema": (
                experiment._ENVIRONMENT_CONFIG_SCHEMA
            ),
            "args": {"seed": 7, "mode": "smelt"},
            "data": {
                "sha256": identity["data_sha256"],
                "split_sha256": identity["data_split_sha256"],
                "effective_train_sha256": "4" * 64,
            },
            "comparison_protocol_sha256": (
                identity["comparison_protocol_sha256"]
            ),
            "run_metadata": metadata,
            "environment": {
                "schema": experiment._ENVIRONMENT_SCHEMA,
                "environment_id": identity["environment_id"],
                "file": "environment.json",
                "file_sha256": environment_sha,
            },
            "sanity_check": {
                "reproducibility": {
                    "repeats": 2,
                    "identical": True,
                    "mismatches": [],
                }
            },
        }
        if config_overrides:
            config.update(config_overrides)
        (root / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        records = [
            {"step": "reference_baselines", **identity},
            {"step": 1, "val_loss": 2.0},
            {
                "step": "final",
                **identity,
                "final_loss": 1.0,
            },
        ]
        (root / "metrics.jsonl").write_text(
            "".join(
                json.dumps(record) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        return config, identity, metadata

    def _codes(self, report, key):
        return {
            item["code"]
            for item in report[key]
            if isinstance(item, dict)
        }

    def test_audit_accepts_consistent_run_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(root)
            report = experiment.audit_run_directory(root)
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["errors"], [])
            self.assertIn(
                "manifest_missing",
                self._codes(report, "warnings"),
            )

            experiment.write_artifact_manifest(root)
            strict = experiment.audit_run_directory(
                root, strict=True
            )
            self.assertTrue(strict["ok"], strict)
            self.assertEqual(strict["errors"], [])
            self.assertEqual(strict["warnings"], [])

    def test_audit_does_not_require_full_identity_on_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(root)
            metrics_path = root / "metrics.jsonl"
            records = [
                json.loads(line)
                for line in metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            records[-1].pop("environment_id")
            records[-1].pop("comparison_protocol_sha256")
            metrics_path.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            experiment.write_artifact_manifest(root)
            report = experiment.audit_run_directory(
                root, strict=True
            )
            self.assertTrue(report["ok"], report)

    def test_audit_detects_metrics_from_another_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(root)
            metrics_path = root / "metrics.jsonl"
            records = [
                json.loads(line)
                for line in metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            records[0]["script_sha256"] = "e" * 64
            metrics_path.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            self.assertFalse(report["ok"])
            self.assertIn(
                "record_identity_mismatch",
                self._codes(report, "errors"),
            )

    def test_audit_detects_duplicate_or_misplaced_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(root)
            metrics_path = root / "metrics.jsonl"
            records = [
                json.loads(line)
                for line in metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            metrics_path.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in records + [records[0]]
                ),
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            self.assertFalse(report["ok"])
            self.assertIn(
                "multiple_reference_baselines",
                self._codes(report, "errors"),
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(root)
            metrics_path = root / "metrics.jsonl"
            records = [
                json.loads(line)
                for line in metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            metrics_path.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in [records[1], records[0]]
                ),
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            self.assertFalse(report["ok"])
            self.assertIn(
                "metrics_header_not_first",
                self._codes(report, "errors"),
            )

    def test_audit_discloses_missing_test_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            config, identity, _ = self._build_run(root)
            metrics_path = root / "metrics.jsonl"
            records = [
                json.loads(line)
                for line in metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            records.append(
                {"step": "test", **identity, "test_loss": 1.0}
            )
            metrics_path.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            warning_codes = self._codes(report, "warnings")
            self.assertIn(
                "test_ledger_missing", warning_codes
            )
            self.assertNotIn(
                "test_ledger_entry_missing", warning_codes
            )

            local_ledger = root / "test_access_ledger.jsonl"
            local_ledger.write_text(
                json.dumps(
                    {"out_dir": str(root.resolve())}
                )
                + "\n",
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            warning_codes = self._codes(report, "warnings")
            self.assertNotIn(
                "test_ledger_missing", warning_codes
            )
            self.assertNotIn(
                "test_ledger_entry_missing", warning_codes
            )

            local_ledger.unlink()
            global_ledger = Path(tmp) / "global_test_ledger.jsonl"
            global_ledger.write_text(
                json.dumps(
                    {"out_dir": str(root.resolve())}
                )
                + "\n",
                encoding="utf-8",
            )
            config["args"]["test_ledger"] = str(global_ledger)
            (root / "config.json").write_text(
                json.dumps(
                    config, ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            warning_codes = self._codes(report, "warnings")
            self.assertNotIn(
                "test_ledger_missing", warning_codes
            )
            self.assertNotIn(
                "test_ledger_entry_missing", warning_codes
            )

    def test_audit_matches_relative_test_ledger_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            root = base / "run"
            _, identity, _ = self._build_run(
                root,
                metadata_overrides={"cwd": str(base)},
            )
            metrics_path = root / "metrics.jsonl"
            records = [
                json.loads(line)
                for line in metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            records.append(
                {"step": "test", **identity, "test_loss": 1.0}
            )
            metrics_path.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            (root / "test_access_ledger.jsonl").write_text(
                json.dumps({"out_dir": "run"}) + "\n",
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            warning_codes = self._codes(report, "warnings")
            self.assertNotIn(
                "test_ledger_entry_missing", warning_codes
            )

    def test_final_comparison_normalizes_ledger_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "run"
            ledger = Path(tmp) / "ledger.jsonl"
            ledger.write_text(
                json.dumps({"out_dir": str(candidate)}) + "\n"
                + json.dumps({
                    "out_dir": os.path.relpath(
                        candidate, Path.cwd()
                    )
                })
                + "\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                report = experiment.print_final_test_comparison(
                    [], str(ledger)
                )
            counts = report["ledger"]["per_candidate"]
            self.assertEqual(list(counts.values()), [2])
            self.assertIn("重复解封", output.getvalue())

    def test_audit_detects_crash_between_ledger_and_seal(self):
        # 访问账本在 test_started/封条之前写入；进程若在这之间
        # 崩溃，只有账本文件能证明测试集已解封。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(root)
            (root / "test_access_ledger.jsonl").write_text(
                json.dumps(
                    {"out_dir": str(root.resolve())}
                )
                + "\n",
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            warning_codes = self._codes(report, "warnings")
            self.assertIn("test_seal_missing", warning_codes)
            self.assertIn(
                "test_started_without_summary",
                warning_codes,
            )
            self.assertNotIn(
                "test_ledger_missing", warning_codes
            )

    def test_audit_detects_crash_in_global_test_ledger(self):
        # 全局 --test-ledger 同样先于 test_started/封条写入；
        # 进程在这之间崩溃时运行目录里没有本地账本，审计必须
        # 从全局账本里找到本目录的条目，而不是当成从未解封。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            config, _, _ = self._build_run(root)
            global_ledger = Path(tmp) / "global_test_ledger.jsonl"
            global_ledger.write_text(
                json.dumps(
                    {"out_dir": str(root.resolve())}
                )
                + "\n",
                encoding="utf-8",
            )
            config["args"]["test_ledger"] = str(global_ledger)
            (root / "config.json").write_text(
                json.dumps(
                    config, ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            warning_codes = self._codes(report, "warnings")
            self.assertIn("test_seal_missing", warning_codes)
            self.assertIn(
                "test_started_without_summary",
                warning_codes,
            )
            self.assertNotIn(
                "test_ledger_missing", warning_codes
            )
            self.assertNotIn(
                "test_ledger_entry_missing", warning_codes
            )

            # 账本里只有其它候选的访问时，不能把本目录误判成
            # 已解封；但既然配置了全局账本且可读，仍应指出本
            # 目录没有对应访问记录。
            global_ledger.write_text(
                json.dumps(
                    {"out_dir": str(Path(tmp) / "other")}
                )
                + "\n",
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            warning_codes = self._codes(report, "warnings")
            self.assertNotIn("test_seal_missing", warning_codes)
            self.assertNotIn(
                "test_started_without_summary",
                warning_codes,
            )

    def test_audit_flags_unreadable_test_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _, identity, _ = self._build_run(root)
            metrics_path = root / "metrics.jsonl"
            records = [
                json.loads(line)
                for line in metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            records.append(
                {"step": "test", **identity, "test_loss": 1.0}
            )
            metrics_path.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            local_ledger = root / "test_access_ledger.jsonl"
            local_ledger.write_text(
                json.dumps(
                    {"out_dir": str(root.resolve())}
                )
                + "\n",
                encoding="utf-8",
            )
            real_open = open
            ledger_key = str(local_ledger)

            def _selective_open(path, *args, **kwargs):
                if str(path) == ledger_key:
                    raise PermissionError("denied")
                return real_open(path, *args, **kwargs)

            with mock.patch(
                "builtins.open", new=_selective_open
            ):
                report = experiment.audit_run_directory(root)
            warning_codes = self._codes(report, "warnings")
            self.assertIn(
                "test_ledger_unreadable", warning_codes
            )
            self.assertNotIn(
                "test_ledger_entry_missing", warning_codes
            )

    def test_audit_detects_started_without_summary(self):
        for step_record in (
            {"step": "test_started", "split": "test"},
            {
                "step": "bad_cases",
                "split": "test",
                "wrong_tokens": 1,
            },
        ):
            with self.subTest(step=step_record["step"]):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "run"
                    self._build_run(root)
                    metrics_path = root / "metrics.jsonl"
                    records = [
                        json.loads(line)
                        for line in metrics_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                    ]
                    records.append(step_record)
                    metrics_path.write_text(
                        "".join(
                            json.dumps(record) + "\n"
                            for record in records
                        ),
                        encoding="utf-8",
                    )
                    report = experiment.audit_run_directory(root)
                    warning_codes = self._codes(
                        report, "warnings"
                    )
                    self.assertIn(
                        "test_seal_missing", warning_codes
                    )
                    self.assertIn(
                        "test_started_without_summary",
                        warning_codes,
                    )
                    self.assertIn(
                        "test_ledger_missing", warning_codes
                    )

    def test_audit_detects_repeated_test_unsealing(self):
        for step_record, expected_code in (
            (
                {"step": "test_started", "split": "test"},
                "multiple_test_started",
            ),
            (
                {
                    "step": "bad_cases",
                    "split": "test",
                    "wrong_tokens": 1,
                },
                "multiple_test_bad_cases",
            ),
        ):
            with self.subTest(step=step_record["step"]):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "run"
                    self._build_run(root)
                    metrics_path = root / "metrics.jsonl"
                    records = [
                        json.loads(line)
                        for line in metrics_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                    ]
                    records.extend([step_record, step_record])
                    metrics_path.write_text(
                        "".join(
                            json.dumps(record) + "\n"
                            for record in records
                        ),
                        encoding="utf-8",
                    )
                    report = experiment.audit_run_directory(root)
                    self.assertFalse(report["ok"])
                    self.assertIn(
                        expected_code,
                        self._codes(report, "errors"),
                    )

    def test_audit_detects_duplicate_test_ledger_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(root)
            entries = (
                json.dumps({"out_dir": str(root.resolve())})
                + "\n"
            ) * 2
            (root / "test_access_ledger.jsonl").write_text(
                entries, encoding="utf-8"
            )
            report = experiment.audit_run_directory(root)
            self.assertFalse(report["ok"])
            self.assertIn(
                "multiple_test_ledger_entries",
                self._codes(report, "errors"),
            )

            config, _, _ = self._build_run(
                Path(tmp) / "allowed",
                config_overrides={
                    "args": {
                        "seed": 7,
                        "mode": "smelt",
                        "allow_repeat_test_access": True,
                    }
                },
            )
            allowed_root = Path(tmp) / "allowed"
            (allowed_root / "test_access_ledger.jsonl").write_text(
                (
                    json.dumps(
                        {"out_dir": str(allowed_root.resolve())}
                    )
                    + "\n"
                ) * 2,
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(allowed_root)
            self.assertTrue(report["ok"], report)
            self.assertIn(
                "test_ledger_repeated_access",
                self._codes(report, "warnings"),
            )

    def test_audit_ignores_ledger_entries_for_other_test_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _, identity, _ = self._build_run(root)
            metrics_path = root / "metrics.jsonl"
            records = [
                json.loads(line)
                for line in metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            records.append(
                {"step": "test", **identity, "test_loss": 1.0}
            )
            metrics_path.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            (root / "test_access_ledger.jsonl").write_text(
                json.dumps(
                    {
                        "out_dir": str(root.resolve()),
                        "data_sha256": "e" * 64,
                        "test_split_sha256": "3" * 64,
                        "test_fraction": 0.05,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            warning_codes = self._codes(report, "warnings")
            self.assertIn(
                "test_ledger_entry_missing", warning_codes
            )
            self.assertEqual(
                report["checks"]["test_seal"]["ledger_matches"], 0
            )

    def test_audit_detects_sanity_repeat_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(
                root,
                config_overrides={
                    "sanity_check": {
                        "reproducibility": {
                            "repeats": 2,
                            "identical": False,
                            "mismatches": ["final_loss"],
                        }
                    }
                },
            )
            report = experiment.audit_run_directory(root)
            self.assertFalse(report["ok"])
            self.assertIn(
                "sanity_repeat_mismatch",
                self._codes(report, "errors"),
            )

    def test_audit_strict_treats_evidence_gaps_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(
                root,
                metadata_overrides={
                    "pythonhashseed_at_start": None,
                    "deterministic_algorithms": False,
                },
            )
            report = experiment.audit_run_directory(root)
            self.assertTrue(report["ok"], report)
            warning_codes = self._codes(report, "warnings")
            self.assertIn(
                "pythonhashseed_unset", warning_codes
            )
            self.assertIn(
                "nondeterministic_algorithms", warning_codes
            )
            strict = experiment.audit_run_directory(
                root, strict=True
            )
            self.assertFalse(strict["ok"])

    def test_audit_checks_external_source_and_data_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(root)
            source = Path(tmp) / "experiment.py"
            source.write_text("# archived copy\n", encoding="utf-8")
            data = Path(tmp) / "corpus.txt"
            data.write_text("corpus\n", encoding="utf-8")
            report = experiment.audit_run_directory(
                root,
                source_path=source,
                data_path=data,
            )
            self.assertFalse(report["ok"])
            error_codes = self._codes(report, "errors")
            self.assertIn(
                "source_sha_mismatch", error_codes
            )
            self.assertIn("data_sha_mismatch", error_codes)

    def test_audit_detects_preregistration_edited_after_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            config, identity, _ = self._build_run(root)
            prereg = Path(tmp) / "preregistration.json"
            payload = {
                "study_id": "smelt-isr-audit-test",
                "frozen_commit": "a" * 40,
                "primary_metric": "val_loss",
                "common_step": 100,
                "min_effect": 0.01,
                "direction": "decrease",
                "baseline_mode": "smelt",
                "comparisons": [
                    ["smelt", "A"],
                    ["smelt", "B"],
                ],
                "seeds": [42, 43, 44],
                "budget_match": "smelt",
                "steps": 200,
                "frozen_script_sha256": identity[
                    "script_sha256"
                ],
                "data": {
                    "sha256": identity["data_sha256"],
                    "split_policy": "contiguous",
                    "val_fraction": 0.1,
                    "test_fraction": 0.1,
                },
            }
            prereg.write_text(
                json.dumps(payload), encoding="utf-8"
            )
            _, prereg_sha = experiment.load_preregistration(
                prereg
            )
            config["preregistration"] = {
                **{
                    key: payload[key]
                    for key in (
                        "study_id",
                        "frozen_commit",
                        "primary_metric",
                        "common_step",
                        "direction",
                        "min_effect",
                        "baseline_mode",
                        "comparisons",
                        "seeds",
                        "budget_match",
                    )
                },
                "file_sha256": prereg_sha,
            }
            (root / "config.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(
                root, preregistration_path=prereg
            )
            self.assertTrue(report["ok"], report)

            payload["min_effect"] = 0.5
            prereg.write_text(
                json.dumps(payload), encoding="utf-8"
            )
            report = experiment.audit_run_directory(
                root, preregistration_path=prereg
            )
            self.assertFalse(report["ok"])
            self.assertIn(
                "preregistration_sha_mismatch",
                self._codes(report, "errors"),
            )

    def test_audit_mode_requires_run_and_rejects_training(self):
        self.assertIsNotNone(
            experiment.audit_mode_error(
                argparse.Namespace(audit_report="report.json")
            )
        )
        self.assertIsNotNone(
            experiment.audit_mode_error(
                argparse.Namespace(
                    audit_run="run",
                    out="other",
                )
            )
        )
        for name, value in (
            ("plan_design", True),
            ("suite_modes", "smelt,A"),
            ("suite_ablation", True),
            ("paper_cell", "smelt-200M"),
        ):
            with self.subTest(conflict=name):
                self.assertIsNotNone(
                    experiment.audit_mode_error(
                        argparse.Namespace(
                            audit_run="run",
                            **{name: value},
                        )
                    )
                )
        self.assertIsNone(
            experiment.audit_mode_error(
                argparse.Namespace(audit_run="run")
            )
        )


class NearDuplicateDataTests(unittest.TestCase):
    """近重复数据审计/清洗：候选生成、精确复核与 train-only 边界。"""

    @staticmethod
    def _printable(tag: str, size: int) -> bytes:
        raw = hashlib.sha256(tag.encode("utf-8")).digest()
        out = bytearray()
        counter = 0
        while len(out) < size:
            raw = hashlib.sha256(
                raw + str(counter).encode("ascii")
            ).digest()
            out.extend(32 + (value % 95) for value in raw)
            counter += 1
        return bytes(out[:size])

    @classmethod
    def _near_pair(cls, size: int = 640):
        core = (
            b"alpha beta gamma delta epsilon zeta eta theta "
            * 12
        )[: size - 64]
        filler = cls._printable("near-filler", size - len(core))
        base = core + filler
        variant = base.replace(b"alpha", b"omega", 1)
        return base, variant

    def test_bottom_k_sketch_uses_distinct_shingle_hashes(self):
        # 重复 shingle 的哈希不能占满 bottom-k 草图：MinHash
        # 的集合口径要求每个值只计一次，否则高度重复内容的
        # 草图会丢掉其他真实值并造成近重复候选漏检。
        self.assertEqual(
            experiment._near_duplicate_bottom_k(
                [5] * 20, 8
            ),
            (5,),
        )
        self.assertEqual(
            experiment._near_duplicate_bottom_k(
                [9, 3, 9, 3, 7, 7, 1], 3
            ),
            (1, 3, 7),
        )

    def test_planner_drops_near_duplicate_train_group(self):
        base, variant = self._near_pair()
        separator = b"\n\n"
        unrelated = [
            self._printable(f"doc-{index}", 640)
            for index in range(2)
        ]
        raw = bytearray(
            separator.join([base, *unrelated, variant])
        )
        train_end = len(base) + len(unrelated[0]) + 2 * len(
            separator
        )
        eval_ranges = {"val": (train_end, len(raw))}

        excluded, stats = (
            experiment._plan_train_near_duplicate_cleaning(
                raw,
                separator,
                train_end,
                eval_ranges,
                [],
                threshold=0.8,
                shingle_bytes=8,
                apply_cleaning=True,
            )
        )

        self.assertEqual(excluded, [(0, len(base))])
        self.assertEqual(stats["groups_dropped"], 1)
        self.assertEqual(stats["groups_flagged"], 1)
        self.assertEqual(
            stats["flagged_duplicate_of"],
            {"val": 1, "test": 0, "train": 0},
        )
        self.assertGreaterEqual(
            stats["examples"][0][3], 0.8
        )
        self.assertLessEqual(
            stats["examples"][0][3], 1.0
        )

        # 只读审计报告同一匹配，但不返回删除区间。
        audit_excluded, audit_stats = (
            experiment._plan_train_near_duplicate_cleaning(
                raw,
                separator,
                train_end,
                eval_ranges,
                [],
                threshold=0.8,
                shingle_bytes=8,
                apply_cleaning=False,
            )
        )
        self.assertEqual(audit_excluded, [])
        self.assertEqual(audit_stats["groups_dropped"], 0)
        self.assertEqual(audit_stats["groups_flagged"], 1)
        self.assertGreater(
            audit_stats["projected_train_bytes_if_flagged_dropped"],
            0,
        )

    def test_planner_rejects_invalid_parameters(self):
        raw = bytearray(b"abc" * 100)
        for kwargs in (
            {"threshold": 0.0},
            {"threshold": 1.5},
            {"shingle_bytes": 0},
            {"shingle_bytes": 65},
            {"sketch_size": 1},
        ):
            with self.assertRaises(ValueError):
                experiment._plan_train_near_duplicate_cleaning(
                    raw,
                    b"\n\n",
                    len(raw),
                    {"val": (0, 0)},
                    [],
                    **kwargs,
                )

    def test_dataset_cleaning_only_changes_train(self):
        base, variant = self._near_pair()
        groups = [
            self._printable("u1", 640),
            base,
            self._printable("u2", 640),
            self._printable("u3", 640),
            variant,
            self._printable("v2", 640),
            self._printable("t1", 640),
            self._printable("t2", 640),
        ]
        payload = b"\n\n".join(groups)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "near.txt"
            path.write_bytes(payload)

            plain = experiment.ByteDataset(
                str(path),
                seq_len=8,
                val_fraction=0.25,
                test_fraction=0.25,
                group_separator=b"\n\n",
            )
            cleaned = experiment.ByteDataset(
                str(path),
                seq_len=8,
                val_fraction=0.25,
                test_fraction=0.25,
                group_separator=b"\n\n",
                drop_train_near_duplicate_groups=True,
                near_duplicate_threshold=0.8,
                near_duplicate_shingle_bytes=8,
            )

        self.assertEqual(
            plain.split_bounds, cleaned.split_bounds
        )
        self.assertEqual(
            plain.split_sha256, cleaned.split_sha256
        )
        self.assertEqual(
            cleaned.train_excluded_spans, [(642, 1282)]
        )
        self.assertEqual(
            cleaned.train_cleaning["cleaning_kinds"],
            ["near_duplicate"],
        )
        self.assertEqual(
            cleaned.train_cleaning["near_duplicates"][
                "flagged_duplicate_of"
            ],
            {"val": 1, "test": 0, "train": 0},
        )
        self.assertEqual(
            cleaned.effective_train_bytes,
            plain.effective_train_bytes - 640,
        )

    def test_audit_does_not_delete_and_is_deterministic(self):
        base, variant = self._near_pair()
        groups = [
            self._printable("u1", 640),
            base,
            self._printable("u2", 640),
            self._printable("u3", 640),
            variant,
            self._printable("v2", 640),
            self._printable("t1", 640),
            self._printable("t2", 640),
        ]
        payload = b"\n\n".join(groups)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.txt"
            path.write_bytes(payload)
            dataset = experiment.ByteDataset(
                str(path),
                seq_len=8,
                val_fraction=0.25,
                test_fraction=0.25,
                group_separator=b"\n\n",
                near_duplicate_audit=True,
            )
            audit_one = experiment.audit_data_splits(dataset)
            audit_two = experiment.audit_data_splits(dataset)

        self.assertEqual(dataset.train_excluded_spans, [])
        self.assertEqual(
            audit_one["near_duplicates"]["groups_flagged"], 1
        )
        self.assertEqual(
            audit_one["near_duplicates"],
            audit_two["near_duplicates"],
        )
        self.assertTrue(
            any(
                "近重复" in warning
                for warning in audit_one["warnings"]
            )
        )

    def test_test_content_is_not_used_for_near_dedup_by_default(self):
        base, variant = self._near_pair()
        groups = [
            base,
            self._printable("u1", 640),
            self._printable("u2", 640),
            self._printable("u3", 640),
            self._printable("v1", 640),
            self._printable("v2", 640),
            variant,
            self._printable("t2", 640),
        ]
        payload = b"\n\n".join(groups)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sealed.txt"
            path.write_bytes(payload)
            sealed = experiment.ByteDataset(
                str(path),
                seq_len=8,
                val_fraction=0.25,
                test_fraction=0.25,
                group_separator=b"\n\n",
                drop_train_near_duplicate_groups=True,
            )
            leaky = experiment.ByteDataset(
                str(path),
                seq_len=8,
                val_fraction=0.25,
                test_fraction=0.25,
                group_separator=b"\n\n",
                drop_train_near_duplicate_groups=True,
                clean_train_against_test=True,
            )

        self.assertEqual(sealed.train_excluded_spans, [])
        self.assertFalse(
            sealed.train_cleaning["test_used_for_cleaning"]
        )
        self.assertEqual(
            leaky.train_cleaning["cleaning_eval_splits"],
            ["test", "val"],
        )
        self.assertTrue(
            leaky.train_cleaning["test_used_for_cleaning"]
        )
        self.assertEqual(
            leaky.train_excluded_spans, [(0, 640)]
        )

    @staticmethod
    def _mutate_byte(content: bytes, position: int) -> bytes:
        """把一个可打印字节改成另一个可打印字节，制造近重复。"""
        value = content[position]
        replacement = 32 + ((value - 32 + 1) % 95)
        return (
            content[:position]
            + bytes([replacement])
            + content[position + 1:]
        )

    def test_eval_near_duplicate_audit_reports_effective_groups(self):
        separator = b"\n\n"
        val_base = self._printable("eval-val-base", 640)
        val_variant = self._mutate_byte(val_base, 300)
        val_other = self._printable("eval-val-other", 640)
        test_base = self._printable("eval-test-base", 640)
        test_variant = self._mutate_byte(test_base, 320)
        test_other = self._printable("eval-test-other", 640)

        val_raw = separator.join(
            [val_base, val_variant, val_other]
        )
        test_raw = separator.join(
            [test_base, test_variant, test_other]
        )
        raw = bytearray(val_raw + separator + test_raw)
        val_range = (0, len(val_raw))
        test_range = (
            len(val_raw) + len(separator),
            len(raw),
        )

        report = experiment._eval_near_duplicate_audit(
            raw,
            separator,
            {"val": val_range, "test": test_range},
            threshold=0.8,
            shingle_bytes=8,
        )

        self.assertEqual(
            report["schema"],
            "smelt-isr/eval-near-duplicate/v1",
        )
        self.assertEqual(report["splits"], ["val", "test"])
        self.assertTrue(report["test_content_used"])
        self.assertTrue(report["exact_verification"])

        val_info = report["per_split"]["val"]
        self.assertEqual(val_info["groups_total"], 3)
        self.assertEqual(val_info["groups_scanned"], 3)
        self.assertEqual(val_info["within_pairs"], 1)
        self.assertEqual(val_info["clusters"], 2)
        self.assertEqual(val_info["effective_groups"], 2)
        self.assertEqual(val_info["redundant_groups"], 1)
        self.assertEqual(val_info["largest_cluster_groups"], 2)
        self.assertEqual(val_info["effective_fraction"], 2 / 3)
        self.assertTrue(
            val_info["examples"][0][4] >= 0.8
        )

        test_info = report["per_split"]["test"]
        self.assertEqual(test_info["groups_scanned"], 3)
        self.assertEqual(test_info["within_pairs"], 1)
        self.assertEqual(test_info["effective_groups"], 2)

        # 两组近重复内容分别来自不同随机基串，跨 split 不应
        # 被误报为近重复。
        self.assertEqual(
            report["cross_split"]["val_test"]["pairs"], 0
        )

    def test_eval_near_duplicate_audit_detects_cross_split_only_with_test(self):
        separator = b"\n\n"
        val_base = self._printable("cross-val-base", 640)
        test_variant = self._mutate_byte(val_base, 200)
        raw = bytearray(
            val_base + separator + test_variant
        )
        val_range = (0, len(val_base))
        test_range = (
            len(val_base) + len(separator),
            len(raw),
        )

        val_only = experiment._eval_near_duplicate_audit(
            raw,
            separator,
            {"val": val_range},
            threshold=0.8,
            shingle_bytes=8,
        )
        self.assertFalse(val_only["test_content_used"])
        self.assertEqual(val_only["splits"], ["val"])
        self.assertEqual(
            val_only["cross_split"], {}
        )

        with_test = experiment._eval_near_duplicate_audit(
            raw,
            separator,
            {"val": val_range, "test": test_range},
            threshold=0.8,
            shingle_bytes=8,
        )
        self.assertTrue(with_test["test_content_used"])
        self.assertGreaterEqual(
            with_test["cross_split"]["val_test"]["pairs"], 1
        )
        self.assertEqual(
            with_test["per_split"]["val"]["within_pairs"], 0
        )
        self.assertEqual(
            with_test["per_split"]["test"]["within_pairs"], 0
        )

    def test_eval_near_duplicate_audit_is_read_only_and_deterministic(self):
        separator = b"\n\n"
        base = self._printable("audit-base", 640)
        variant = self._mutate_byte(base, 100)
        groups = [
            *[
                self._printable(f"audit-train-{index}", 640)
                for index in range(6)
            ],
            base,
            variant,
            self._printable("audit-val-other", 640),
            self._mutate_byte(base, 300),
            self._mutate_byte(variant, 400),
            self._printable("audit-test-other", 640),
        ]
        payload = separator.join(groups)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval-near.txt"
            path.write_bytes(payload)
            dataset = experiment.ByteDataset(
                str(path),
                seq_len=8,
                val_fraction=0.25,
                test_fraction=0.25,
                group_separator=separator,
                near_duplicate_audit=True,
                near_duplicate_threshold=0.8,
                near_duplicate_shingle_bytes=8,
            )
            sealed_one = experiment.audit_data_splits(dataset)
            sealed_two = experiment.audit_data_splits(dataset)
            unsealed = experiment.audit_data_splits(
                dataset,
                include_test=True,
            )
            excluded_after = list(dataset.train_excluded_spans)

        self.assertEqual(excluded_after, [])
        self.assertEqual(sealed_one, sealed_two)
        self.assertIsNotNone(
            sealed_one["evaluation_near_duplicates"]
        )
        self.assertEqual(
            sealed_one["evaluation_near_duplicates"]["splits"],
            ["val"],
        )
        self.assertFalse(
            sealed_one["evaluation_near_duplicates"][
                "test_content_used"
            ]
        )
        self.assertNotIn(
            "test",
            sealed_one["evaluation_near_duplicates"][
                "per_split"
            ],
        )
        self.assertEqual(
            sealed_one["evaluation_near_duplicates"][
                "cross_split"
            ],
            {},
        )
        self.assertTrue(
            any(
                "评估集内部" in warning
                for warning in sealed_one["warnings"]
            )
        )

        self.assertEqual(
            unsealed["evaluation_near_duplicates"]["splits"],
            ["val", "test"],
        )
        self.assertTrue(
            unsealed["evaluation_near_duplicates"][
                "test_content_used"
            ]
        )
        self.assertIn(
            "test",
            unsealed["evaluation_near_duplicates"][
                "per_split"
            ],
        )
        self.assertGreaterEqual(
            unsealed["evaluation_near_duplicates"][
                "cross_split"
            ]["val_test"]["pairs"],
            1,
        )
        self.assertTrue(
            any(
                "跨 split 近重复评估群组" in warning
                for warning in unsealed["warnings"]
            )
        )

    def test_eval_near_duplicate_audit_rejects_invalid_parameters(self):
        raw = bytearray(b"abc" * 100)
        for kwargs in (
            {"threshold": 0.0},
            {"threshold": 1.5},
            {"shingle_bytes": 0},
            {"shingle_bytes": 65},
            {"sketch_size": 1},
        ):
            with self.assertRaises(ValueError):
                experiment._eval_near_duplicate_audit(
                    raw,
                    b"\n\n",
                    {"val": (0, len(raw))},
                    **kwargs,
                )
        with self.assertRaises(ValueError):
            experiment._eval_near_duplicate_audit(
                raw,
                b"",
                {"val": (0, len(raw))},
            )


class ProcessTests(unittest.TestCase):
    def test_set_seed_updates_child_hashseed_on_reseed(self):
        if experiment._PYTHONHASHSEED_AT_START is not None:
            self.skipTest("PYTHONHASHSEED was fixed before import")
        import os

        saved_env = os.environ.get("PYTHONHASHSEED")
        saved_rng = experiment._capture_rng_state()
        try:
            os.environ.pop("PYTHONHASHSEED", None)
            experiment.set_seed(101)
            self.assertEqual(
                os.environ.get("PYTHONHASHSEED"), "101"
            )
            experiment.set_seed(102)
            self.assertEqual(
                os.environ.get("PYTHONHASHSEED"), "102"
            )
        finally:
            experiment._restore_rng_state(saved_rng)
            if saved_env is None:
                os.environ.pop("PYTHONHASHSEED", None)
            else:
                os.environ["PYTHONHASHSEED"] = saved_env

    def test_require_pythonhashseed_rejects_non_fixed_values(self):
        previous = experiment._PYTHONHASHSEED_AT_START
        try:
            for value in (None, "", "random", "abc", "-1", "4294967296"):
                experiment._PYTHONHASHSEED_AT_START = value
                self.assertIsNotNone(
                    experiment.pythonhashseed_error(True),
                    value,
                )
            for value in ("0", "42", " 7 "):
                experiment._PYTHONHASHSEED_AT_START = value
                self.assertIsNone(
                    experiment.pythonhashseed_error(True),
                    value,
                )
            experiment._PYTHONHASHSEED_AT_START = ""
            self.assertIsNone(
                experiment.pythonhashseed_error(False)
            )
        finally:
            experiment._PYTHONHASHSEED_AT_START = previous

    def test_split_policy_default_is_contiguous(self):
        cfg = _tiny_config()
        self.assertEqual(
            experiment.paper_fidelity_report(cfg)[
                "paper"
            ],
            "arXiv:2609.01343v1",
        )

    def test_test_ledger_repeat_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            key = {
                "data_sha256": "d",
                "test_split_sha256": "s",
                "test_fraction": 0.05,
            }
            experiment.append_test_access_ledger(
                ledger, {**key, "out_dir": str(Path(tmp) / "a")}
            )
            check = experiment.check_test_access_ledger(
                ledger, key, Path(tmp) / "b"
            )
            self.assertTrue(check["repeat_access"])
            check_same = experiment.check_test_access_ledger(
                ledger, key, Path(tmp) / "a"
            )
            self.assertFalse(check_same["repeat_access"])
            self.assertEqual(check_same["matching_accesses"], 1)
            self.assertEqual(
                len(check_same["same_candidates"]), 1
            )
            self.assertTrue(
                experiment._test_access_repeat_blocked(check_same)
            )
            self.assertIn(
                "本目录已登记过",
                experiment._repeat_test_access_error(check_same),
            )

    def test_test_ledger_missing_key_is_conservative(self):
        # 账本条目缺少测试集指纹时无法证明它属于别的测试集；
        # 不能因此放行重复解封。
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            candidate = Path(tmp) / "a"
            ledger.write_text(
                json.dumps({"out_dir": str(candidate)}) + "\n",
                encoding="utf-8",
            )
            key = {
                "data_sha256": "d",
                "test_split_sha256": "s",
                "test_fraction": 0.05,
            }
            check_same = experiment.check_test_access_ledger(
                ledger, key, candidate
            )
            self.assertEqual(check_same["matching_accesses"], 1)
            self.assertTrue(
                experiment._test_access_repeat_blocked(check_same)
            )
            check_other = experiment.check_test_access_ledger(
                ledger, key, Path(tmp) / "b"
            )
            self.assertTrue(check_other["repeat_access"])

    def test_test_ledger_repeat_detection_normalizes_path_spelling(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            key = {
                "data_sha256": "d",
                "test_split_sha256": "s",
                "test_fraction": 0.05,
            }
            candidate = Path(tmp) / "a"
            relative_candidate = os.path.relpath(
                candidate, Path.cwd()
            )
            experiment.append_test_access_ledger(
                ledger,
                {**key, "out_dir": relative_candidate},
            )
            check = experiment.check_test_access_ledger(
                ledger, key, candidate.resolve()
            )
            self.assertFalse(check["repeat_access"], check)
            check = experiment.check_test_access_ledger(
                ledger, key, Path(tmp) / "b"
            )
            self.assertTrue(check["repeat_access"], check)

    def test_entity_manifest_audit_runs_with_group_separator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entities.json"
            path.write_text(
                json.dumps(["doc0", "doc0", "doc1"]),
                encoding="utf-8",
            )
            report = experiment.audit_entity_manifest(
                b"a\nb\nc\n",
                b"\n",
                {
                    "train": (0, 2),
                    "val": (2, 8),
                    "test": (8, 10),
                },
                str(path),
            )
        self.assertTrue(report["available"])
        self.assertEqual(report["groups"], 3)
        self.assertEqual(report["entities"], 2)
        self.assertEqual(report["cross_split_entities"], 1)
        self.assertIn("doc0", dict(report["examples"]))

    def test_text_integrity_attributes_boundary_byte_to_owning_split(self):
        # val 末尾的非法 lead byte 不能因为跨 split 预热而被记到
        # 字节干净的 test 上，否则测试集标签完整性门会误拒。
        data = b"A" * 70 + b"B" * 19 + b"\xe2" + b"C" * 10
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.bin"
            path.write_bytes(data)
            dataset = experiment.ByteDataset(
                str(path),
                seq_len=4,
                val_fraction=0.2,
                test_fraction=0.1,
            )
            report = experiment.audit_data_splits(
                dataset,
                include_test=True,
            )
        splits = report["splits"]
        self.assertEqual(splits["train"]["utf8_replacements"], 0)
        self.assertEqual(splits["val"]["utf8_replacements"], 1)
        self.assertEqual(splits["test"]["utf8_replacements"], 0)
        self.assertEqual(
            experiment.test_label_integrity_violations(report),
            [],
        )

    def test_invalid_span_scanner_warmups_offset_boundary(self):
        # offset 落在一个合法 2 字节字符中间（d8 8e =
        # U+060E）：不能把 offset 处的续字节误报为非法。
        data = b"\x80)t\xb8\x83\xd8\x8e\x02M\x12"
        spans = experiment._invalid_utf8_nul_spans(
            memoryview(data),
            6,
            4,
            True,
        )
        self.assertEqual(spans, [])
        # 同一段里真正的非法续字节仍必须报告。
        self.assertEqual(
            experiment._invalid_utf8_nul_spans(
                memoryview(data), 3, 1, True
            ),
            [(3, 4, "invalid_utf8")],
        )
        # offset=0 的全局扫描不受预热影响。
        self.assertEqual(
            experiment._invalid_utf8_nul_spans(
                memoryview(data), 0, len(data), True
            ),
            [
                (0, 1, "invalid_utf8"),
                (3, 4, "invalid_utf8"),
                (4, 5, "invalid_utf8"),
            ],
        )

    def test_drop_corrupt_bytes_catches_trailing_lead_byte(self):
        # train 末尾的非法 lead byte 需要后继字节才能判定，
        # 但清洗范围仍必须把它删掉，不能留给训练窗口。
        data = b"A" * 69 + b"\xe2" + b"B" * 20 + b"C" * 10
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.bin"
            path.write_bytes(data)
            dataset = experiment.ByteDataset(
                str(path),
                seq_len=4,
                val_fraction=0.2,
                test_fraction=0.1,
                drop_train_corrupt_bytes=True,
            )
            report = experiment.audit_data_splits(
                dataset,
                include_test=True,
            )
        self.assertIn((69, 70), dataset.train_excluded_spans)
        self.assertEqual(
            report["splits"]["train"]["utf8_replacements"], 0
        )
        self.assertEqual(
            report["splits"]["val"]["utf8_replacements"], 0
        )

    def test_audit_reports_unseen_eval_label_coverage(self):
        # train 只有 A，val 只有 B，test 只有 C：B/C 都是训练集
        # 从未出现的字节类，评估 NLL 含不可约外推误差。
        data = b"A" * 60 + b"B" * 20 + b"C" * 20
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coverage.bin"
            path.write_bytes(data)
            dataset = experiment.ByteDataset(
                str(path),
                seq_len=4,
                val_fraction=0.2,
                test_fraction=0.2,
            )
            report = experiment.audit_data_splits(
                dataset,
                include_test=True,
            )
        coverage = report["label_coverage"]
        self.assertEqual(
            coverage["train"]["observed_byte_classes"], 1
        )
        for split, byte in (("val", ord("B")), ("test", ord("C"))):
            self.assertEqual(
                coverage[split]["unseen_label_classes"], 1
            )
            self.assertEqual(
                coverage[split]["unseen_label_token_share"], 1.0
            )
            self.assertEqual(
                report["splits"][split]["unseen_label_classes"],
                1,
            )
        self.assertTrue(
            any("从未出现" in warning for warning in report["warnings"])
        )

    def test_cleaning_records_deletion_bias(self):
        # train 中夹入一段非法字节，只清洗 train；删除内容与保留
        # 内容的字节/二元组分布完全不同，必须显式记录删除偏差。
        data = (
            b"A" * 25
            + b"\xff" * 5
            + b"A" * 30
            + b"B" * 20
            + b"C" * 20
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bias.bin"
            path.write_bytes(data)
            dataset = experiment.ByteDataset(
                str(path),
                seq_len=4,
                val_fraction=0.2,
                test_fraction=0.2,
                drop_train_corrupt_bytes=True,
            )
            report = experiment.audit_data_splits(
                dataset,
                include_test=True,
            )
        bias = dataset.train_cleaning["deletion_bias"]
        self.assertEqual(bias["dropped_bytes"], 5)
        self.assertEqual(bias["kept_bytes"], 55)
        self.assertEqual(bias["train_bytes"], 60)
        self.assertEqual(bias["drop_fraction"], round(5 / 60, 6))
        self.assertEqual(bias["byte_js"], 1.0)
        self.assertEqual(bias["bigram_js"], 1.0)
        self.assertEqual(
            bias["dropped_top_bytes"][0][0], 255
        )
        self.assertTrue(
            any(
                "删除与保留内容的字节分布" in warning
                for warning in report["warnings"]
            )
        )

    def test_interval_counts_advance_for_tiny_chunk_sizes(self):
        raw = memoryview(bytearray(b"abracadabra"))
        expected_bytes = 11
        expected_bigrams = 10
        for chunk_bytes in (0, -1, 1, 2, 3, 100):
            byte_counts = experiment._interval_byte_counts(
                raw,
                [(0, expected_bytes)],
                chunk_bytes=chunk_bytes,
            )
            bigram_counts = experiment._interval_bigram_counts(
                raw,
                [(0, expected_bytes)],
                chunk_bytes=chunk_bytes,
            )
            self.assertEqual(
                int(byte_counts.sum()), expected_bytes
            )
            self.assertEqual(
                int(bigram_counts.sum()), expected_bigrams
            )

    def test_interval_counts_accept_readonly_memoryview(self):
        raw = memoryview(b"abracadabra")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            byte_counts = experiment._interval_byte_counts(
                raw,
                [(0, 11)],
                chunk_bytes=3,
            )
            bigram_counts = experiment._interval_bigram_counts(
                raw,
                [(0, 11)],
                chunk_bytes=3,
            )
        self.assertEqual(int(byte_counts.sum()), 11)
        self.assertEqual(int(bigram_counts.sum()), 10)

    def test_include_test_in_cleaning_alias_is_effective(self):
        groups = [
            b"DUPAAAAA" if index == 0 else
            f"train{index:02d}".encode().ljust(8, b"X")
            for index in range(12)
        ]
        groups += [
            f"val{index:03d}".encode().ljust(8, b"V")
            for index in range(4)
        ]
        groups.append(b"DUPAAAAA")
        data = b"\n".join(groups) + b"\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groups.txt"
            path.write_bytes(data)
            dataset = experiment.ByteDataset(
                str(path),
                seq_len=3,
                val_fraction=0.3,
                test_fraction=0.2,
                group_separator=b"\n",
                deduplicate_train_groups=True,
                include_test_in_cleaning=True,
            )
        self.assertTrue(dataset.clean_train_against_test)
        self.assertTrue(
            dataset.train_cleaning["test_used_for_cleaning"]
        )
        self.assertTrue(
            dataset.eval_content_access["test_content_used"]
        )
        self.assertGreater(len(dataset.train_excluded_spans), 0)

    def test_time_order_manifest_accepts_plain_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "times.txt"
            path.write_text(
                "2020-01-01\n2020-01-02\n2020-01-03\n",
                encoding="utf-8",
            )
            evidence = experiment.read_time_order_manifest(
                str(path)
            )
            self.assertEqual(evidence["records"], 3)
            self.assertTrue(evidence["monotonic_non_decreasing"])
            self.assertEqual(
                evidence["min_timestamp"], "2020-01-01"
            )
            self.assertEqual(
                evidence["max_timestamp"], "2020-01-03"
            )

    def test_time_order_manifest_reports_actual_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "times.txt"
            path.write_text(
                "2020-01-03\n2020-01-01\n2020-01-02\n",
                encoding="utf-8",
            )
            evidence = experiment.read_time_order_manifest(
                str(path)
            )
            self.assertFalse(evidence["monotonic_non_decreasing"])
            self.assertEqual(evidence["order_violations"], [1])
            self.assertEqual(
                evidence["min_timestamp"], "2020-01-01"
            )
            self.assertEqual(
                evidence["max_timestamp"], "2020-01-03"
            )

    def test_time_order_manifest_rejects_mixed_timezones(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "times.txt"
            path.write_text(
                "2020-01-01\n2020-01-02T00:00:00Z\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                experiment.read_time_order_manifest(str(path))

    def test_explicit_null_protocol_values_are_checked(self):
        # cosine 调度下的 wsd_stable_steps、未启用分组/文档 mask
        # 时的 separator 都是显式 null，不是「缺记录」。
        keys = (
            "wsd_stable_steps",
            "document_separator",
            "group_separator",
        )
        rows = []
        for name in ("a", "b"):
            rows.append({
                "run": name,
                "protocol": {
                    "comparison_family": "neural",
                    **{key: None for key in keys},
                },
                "protocol_recorded_keys": list(keys),
            })
        report = experiment.check_controlled_variables(rows, [])
        for key in keys:
            self.assertNotIn(key, report["unchecked_keys"])
            self.assertIn(key, report["checked_keys"])
        self.assertTrue(report["ok"], report["violations"])

    def test_unrecorded_null_protocol_values_stay_unchecked(self):
        keys = ("wsd_stable_steps", "document_separator")
        rows = [
            {
                "run": name,
                "protocol": {
                    "comparison_family": "neural",
                    **{key: None for key in keys},
                },
                "protocol_recorded_keys": [],
            }
            for name in ("a", "b")
        ]
        report = experiment.check_controlled_variables(rows, [])
        for key in keys:
            self.assertIn(key, report["unchecked_keys"])

    def test_architecture_and_leakage_protocol_keys_are_controlled(self):
        # 这些键都会改变模型/数据口径，却曾经只写进 config.json
        # 而未进入受控变量列表：同 mode 的两条运行可以带着不同
        # 的 residual scaling、GQA 或泄露豁免被聚合成同一个
        # 实验单元，所谓「单因子」对比就不再是单因子。
        defaults = {
            "n_kv_heads": 4,
            "residual_scaling": "paper",
            "split_policy": "contiguous",
            "allow_eval_content_for_cleaning": False,
            "allow_entity_overlap": False,
        }
        self.assertEqual(
            set(defaults),
            set(experiment.SINGLE_FACTOR_DESIGN_KEYS),
        )
        for key in defaults:
            self.assertIn(key, experiment.COMPARISON_PROTOCOL_KEYS)
            # 也必须允许显式声明为实验因子，才能做真正的
            # 单因子对比。
            self.assertIn(key, experiment.parse_factor_spec(key))

        changed = {
            "n_kv_heads": 2,
            "residual_scaling": "global",
            "split_policy": "chronological",
            "allow_eval_content_for_cleaning": True,
            "allow_entity_overlap": True,
        }
        for key, variant in changed.items():
            rows = []
            for name, value in (("a", defaults[key]), ("b", variant)):
                protocol = {"comparison_family": "neural", **defaults}
                protocol[key] = value
                rows.append({
                    "run": name,
                    "protocol": protocol,
                    "protocol_recorded_keys": list(defaults),
                })
            report = experiment.check_controlled_variables(rows, [])
            self.assertIn(key, report["checked_keys"])
            self.assertTrue(
                any(
                    violation["key"] == key
                    for violation in report["violations"]
                ),
                report["violations"],
            )
            self.assertFalse(report["ok"])

        same_rows = [
            {
                "run": name,
                "protocol": {
                    "comparison_family": "neural",
                    **defaults,
                },
                "protocol_recorded_keys": list(defaults),
            }
            for name in ("a", "b")
        ]
        same_report = experiment.check_controlled_variables(
            same_rows, []
        )
        self.assertTrue(same_report["ok"], same_report["violations"])

    def test_multiple_changed_axes_are_not_single_factor(self):
        def cell(mode, residual_scaling):
            return {
                "label": (
                    f"mode={mode}, residual_scaling="
                    f"{residual_scaling}"
                ),
                "keys": {
                    "mode": mode,
                    "residual_scaling": residual_scaling,
                },
                "samples": {},
                "n_runs": 0,
            }

        aggregates = [
            cell("A", "paper"),
            cell("A", "global"),
            cell("B", "paper"),
        ]
        contrasts = experiment.factor_contrasts(
            aggregates,
            ["mode", "residual_scaling"],
            "val_loss",
        )
        pairs = {
            frozenset(contrast["cells"]): contrast
            for contrast in contrasts
        }
        single = pairs[frozenset((
            cell("A", "paper")["label"],
            cell("A", "global")["label"],
        ))]
        self.assertFalse(single["confounded"])
        self.assertEqual(
            single["differing_factors"], ["residual_scaling"]
        )
        multi = pairs[frozenset((
            cell("A", "global")["label"],
            cell("B", "paper")["label"],
        ))]
        self.assertTrue(multi["confounded"])
        self.assertEqual(
            set(multi["differing_factors"]),
            {"mode", "residual_scaling"},
        )

    def test_single_factor_design_audit_reports_structural_mismatch(self):
        rows = [
            {
                "run": name,
                "mode": "A",
                "seed": 42,
                "val_loss": 3.0,
                "protocol": {
                    "comparison_family": "neural",
                    "n_kv_heads": kv_heads,
                },
                "protocol_recorded_keys": ["n_kv_heads"],
            }
            for name, kv_heads in (("a", 4), ("b", 2))
        ]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report = experiment.print_comparison_analysis(
                rows,
                factors=["mode"],
                metric="val_loss",
            )
        audit = report["single_factor_design"]
        self.assertFalse(audit["ok"])
        self.assertIn("n_kv_heads", audit["violations"])
        self.assertIn("[single-factor]", output.getvalue())

    def test_ngram_bootstrap_honours_block_size(self):
        data = torch.tensor(
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            dtype=torch.uint8,
        )
        model = experiment.NgramBaseline(data, max_order=2)
        windows = torch.stack(
            [data[index:index + 5] for index in range(6)]
        )
        metrics = experiment._ngram_extended_metrics(
            model,
            [windows],
            alpha=1.0,
            order=2,
            split="val",
            include_auc=False,
            bootstrap_replicates=20,
            bootstrap_seed=7,
            bootstrap_block=3,
        )
        self.assertEqual(
            metrics["val_bootstrap_unit"], "moving_block"
        )
        self.assertEqual(metrics["val_bootstrap_block_size"], 3)

    def test_same_seed_reports_partially_missing_fingerprints(self):
        init = {
            "scheme": "name_hash",
            "shared_sha256": "shared",
            "moe_sha256": "moe",
        }
        trace = {"sha256": "batch"}
        rows = [
            {
                "run": "new-a",
                "mode": "A",
                "seed": 42,
                "init_fingerprints": dict(init),
                "training_trace": dict(trace),
            },
            {
                "run": "new-b",
                "mode": "B",
                "seed": 42,
                "init_fingerprints": dict(init),
                "training_trace": dict(trace),
            },
            {
                # 旧运行没有指纹：其余两条运行一致并不能证明
                # 这一条也与它们同起点。
                "run": "old",
                "mode": "A",
                "seed": 42,
                "init_fingerprints": None,
                "training_trace": None,
            },
        ]
        report = experiment.same_seed_randomness_check(rows)
        self.assertTrue(report["ok"])
        self.assertTrue(
            any("old" in message for message in report["unverified"]),
            report["unverified"],
        )
        self.assertTrue(
            any(
                "初始化指纹" in message
                for message in report["unverified"]
            )
        )
        self.assertTrue(
            any(
                "首批训练数据指纹" in message
                for message in report["unverified"]
            )
        )

    def test_cross_family_explicit_null_is_compared(self):
        recorded = ["group_separator"]
        rows = [
            {
                "run": name,
                "protocol": {
                    "comparison_family": family,
                    "group_separator": None,
                },
                "protocol_recorded_keys": recorded,
            }
            for name, family in (
                ("neural", "neural"),
                ("ngram", "statistical"),
            )
        ]
        report = experiment.check_controlled_variables(rows, [])
        self.assertNotIn(
            "group_separator",
            report["cross_family_unchecked_keys"],
        )
        self.assertIn(
            "group_separator",
            report["cross_family_checked_keys"],
        )


class PreregistrationTests(unittest.TestCase):
    """外部预注册文件：结构校验、命令行冲突与数据指纹核对。"""

    def _payload(self):
        return {
            "study_id": "smelt-isr-test",
            "frozen_commit": "a" * 40,
            "primary_metric": "val_loss",
            "common_step": 100,
            "min_effect": 0.01,
            "direction": "decrease",
            "baseline_mode": "smelt",
            "comparisons": [
                ["smelt", "A"],
                ["smelt", "B"],
            ],
            "seeds": [42, 43, 44],
            "budget_match": "smelt",
            "steps": 200,
            "data": {
                "sha256": "b" * 64,
                "split_policy": "contiguous",
                "val_fraction": 0.1,
                "test_fraction": 0.1,
            },
        }

    def _args(self, **overrides):
        values = dict(
            primary_metric=None,
            hypothesis_direction=None,
            hypothesis_min_effect=None,
            hypothesis_baseline=None,
            common_step=None,
            budget_match=None,
            equivalence_margin=0.01,
            steps=200,
            seeds=None,
            seed=42,
            mode="A",
            statistical_baseline=False,
        )
        values.update(overrides)
        return experiment.argparse.Namespace(**values)

    def _dataset(self):
        return experiment.argparse.Namespace(
            data_sha256="b" * 64,
            val_fraction=0.1,
            test_fraction=0.1,
        )

    def test_schema_requires_full_protocol(self):
        payload = self._payload()
        payload.pop("common_step")
        self.assertIn(
            "common_step",
            experiment.preregistration_schema_error(payload),
        )
        payload = self._payload()
        payload["seeds"] = [42, 43]
        self.assertIn(
            "至少需要",
            experiment.preregistration_schema_error(payload),
        )
        payload = self._payload()
        payload["comparisons"] = []
        self.assertIn(
            "comparisons",
            experiment.preregistration_schema_error(payload),
        )
        payload = self._payload()
        payload["seed"] = [42, 43, 44]
        self.assertIn(
            "未知字段",
            experiment.preregistration_schema_error(payload),
        )

    def test_loader_rejects_duplicate_keys(self):
        payload = self._payload()
        text = json.dumps(payload, ensure_ascii=False)
        text = text.replace(
            '"study_id": "smelt-isr-test"',
            '"study_id": "smelt-isr-test", '
            '"study_id": "duplicate"',
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prereg.json"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError) as context:
                experiment.load_preregistration(path)
        self.assertIn("重复键", str(context.exception))

    def test_loader_accepts_valid_file(self):
        payload = self._payload()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prereg.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            loaded, digest = experiment.load_preregistration(path)
        self.assertEqual(loaded["study_id"], "smelt-isr-test")
        self.assertEqual(len(digest), 64)

    def test_apply_fills_omitted_cli_values(self):
        args = self._args()
        applied, error = experiment.preregistration_apply(
            args,
            self._payload(),
            current_commit="a" * 40,
            working_tree_dirty=False,
            current_script_sha256="c" * 64,
        )
        self.assertIsNone(error)
        self.assertIn("primary_metric", applied)
        self.assertIn("seeds", applied)
        self.assertEqual(args.primary_metric, "val_loss")
        self.assertEqual(args.hypothesis_direction, "decrease")
        self.assertEqual(args.common_step, 100)
        self.assertEqual(args.seeds, [42, 43, 44])

    def test_apply_rejects_explicit_conflict(self):
        args = self._args(primary_metric="val_macro_f1")
        applied, error = experiment.preregistration_apply(
            args,
            self._payload(),
            current_commit="a" * 40,
            working_tree_dirty=False,
        )
        self.assertIsNotNone(error)
        self.assertIn("primary_metric", error)

    def test_apply_numeric_conflict_keeps_large_integer_precision(self):
        payload = self._payload()
        payload["common_step"] = 2 ** 53
        payload["steps"] = 2 ** 53 + 2
        args = self._args(
            common_step=2 ** 53 + 1,
            steps=2 ** 53 + 2,
        )
        _, error = experiment.preregistration_apply(
            args,
            payload,
            current_commit="a" * 40,
            working_tree_dirty=False,
        )
        self.assertIsNotNone(error)
        self.assertIn("common_step", error)

    def test_apply_rejects_commit_dirty_and_script_mismatch(self):
        args = self._args()
        _, error = experiment.preregistration_apply(
            args,
            self._payload(),
            current_commit="c" * 40,
            working_tree_dirty=False,
        )
        self.assertIn("frozen_commit", error)

        args = self._args()
        _, error = experiment.preregistration_apply(
            args,
            self._payload(),
            current_commit="a" * 40,
            working_tree_dirty=True,
        )
        self.assertIn("未提交修改", error)

        payload = self._payload()
        payload["frozen_script_sha256"] = "d" * 64
        args = self._args()
        _, error = experiment.preregistration_apply(
            args,
            payload,
            current_commit=None,
            working_tree_dirty=False,
            current_script_sha256="e" * 64,
        )
        self.assertIn("SHA-256", error)

    def test_apply_rejects_unverifiable_frozen_script(self):
        payload = self._payload()
        payload["frozen_script_sha256"] = "d" * 64
        args = self._args()
        _, error = experiment.preregistration_apply(
            args,
            payload,
            current_commit="a" * 40,
            working_tree_dirty=False,
            current_script_sha256=None,
        )
        self.assertIn("无法读取", error)
        self.assertIn("SHA-256", error)

    def test_apply_accepts_uppercase_frozen_script_sha256(self):
        payload = self._payload()
        payload["frozen_script_sha256"] = "D" * 64
        args = self._args()
        _, error = experiment.preregistration_apply(
            args,
            payload,
            current_commit="a" * 40,
            working_tree_dirty=False,
            current_script_sha256="d" * 64,
        )
        self.assertIsNone(error)

    def test_apply_allows_frozen_commit_ancestor(self):
        args = self._args()
        _, error = experiment.preregistration_apply(
            args,
            self._payload(),
            current_commit="c" * 40,
            working_tree_dirty=False,
            frozen_code_matches=True,
        )
        self.assertIsNone(error)

    def test_apply_fills_steps_only_when_not_explicit(self):
        payload = self._payload()
        payload["steps"] = 10
        payload["common_step"] = 5
        args = self._args(steps=2000)
        previous = sys.argv
        try:
            sys.argv = ["experiment.py"]
            applied, error = experiment.preregistration_apply(
                args,
                payload,
                current_commit="a" * 40,
                working_tree_dirty=False,
            )
            self.assertIsNone(error)
            self.assertIn("steps", applied)
            self.assertEqual(args.steps, 10)

            args = self._args(steps=2000)
            sys.argv = ["experiment.py", "--steps", "2000"]
            applied, error = experiment.preregistration_apply(
                args,
                payload,
                current_commit="a" * 40,
                working_tree_dirty=False,
            )
            self.assertIn("steps", error)
            self.assertEqual(args.steps, 2000)
        finally:
            sys.argv = previous

    def test_apply_rejects_common_step_beyond_steps(self):
        payload = self._payload()
        payload.pop("steps")
        args = self._args(steps=50)
        _, error = experiment.preregistration_apply(
            args,
            payload,
            current_commit="a" * 40,
            working_tree_dirty=False,
        )
        self.assertIn("common_step", error)

    def test_apply_checks_planned_suite_modes(self):
        payload = self._payload()
        payload["comparisons"] = [["smelt", "A"]]
        args = self._args(mode="B", suite_modes=["A"])
        _, error = experiment.preregistration_apply(
            args,
            payload,
            current_commit="a" * 40,
            working_tree_dirty=False,
        )
        self.assertIsNone(error)

        args = self._args(mode="B", suite_modes=["A", "B"])
        _, error = experiment.preregistration_apply(
            args,
            payload,
            current_commit="a" * 40,
            working_tree_dirty=False,
        )
        self.assertIn("B", error)

    def test_apply_defers_resume_mode_check(self):
        payload = self._payload()
        payload["comparisons"] = [["base", "smelt"]]
        args = self._args(
            mode="B",
            resume="checkpoint.pt",
        )
        previous = sys.argv
        try:
            sys.argv = ["experiment.py", "--resume", "checkpoint.pt"]
            _, error = experiment.preregistration_apply(
                args,
                payload,
                current_commit="a" * 40,
                working_tree_dirty=False,
            )
        finally:
            sys.argv = previous
        self.assertIsNone(error)
        args.preregistration_record = {
            "comparisons": [["base", "smelt"]],
        }
        args.mode = "base"
        self.assertIsNone(
            experiment._preregistration_restored_mode_error(args)
        )
        args.mode = "A"
        self.assertIn(
            "不在预注册",
            experiment._preregistration_restored_mode_error(args),
        )

    def test_apply_defers_resume_seed_check(self):
        payload = self._payload()
        args = self._args(
            resume="checkpoint.pt",
            seed=42,
        )
        previous = sys.argv
        try:
            sys.argv = ["experiment.py", "--resume", "checkpoint.pt"]
            applied, error = experiment.preregistration_apply(
                args,
                payload,
                current_commit="a" * 40,
                working_tree_dirty=False,
            )
        finally:
            sys.argv = previous
        self.assertIsNone(error)
        self.assertIsNone(args.seeds)
        # 恢复后的 seed 才按预注册集合核对。
        args.preregistration_record = {"seeds": [1, 2, 3]}
        args.seed = 42
        self.assertIn(
            "不在预注册",
            experiment._preregistration_restored_seed_error(args),
        )
        args.seed = 2
        self.assertIsNone(
            experiment._preregistration_restored_seed_error(args)
        )

    def test_apply_seed_sets_are_fixed(self):
        args = self._args(seeds=[42, 43])
        _, error = experiment.preregistration_apply(
            args,
            self._payload(),
            current_commit="a" * 40,
            working_tree_dirty=False,
        )
        self.assertIn("seeds", error)

        args = self._args(seed=99)
        original_argv = sys.argv
        try:
            sys.argv = ["experiment.py", "--seed", "99"]
            _, error = experiment.preregistration_apply(
                args,
                self._payload(),
                current_commit="a" * 40,
                working_tree_dirty=False,
            )
        finally:
            sys.argv = original_argv
        self.assertIn("不在预注册", error)

    def test_protocol_fallback_includes_cleaning_exemptions(self):
        protocol = experiment.experiment_protocol_from_payload({
            "config": {"mode": "A"},
            "args": {
                "lr": 1e-3,
                "clean_train_against_test": False,
                "allow_eval_content_for_cleaning": True,
                "allow_entity_overlap": True,
            },
        })
        self.assertTrue(
            protocol["allow_eval_content_for_cleaning"]
        )
        self.assertTrue(protocol["allow_entity_overlap"])
        # 旧配置完全没有 comparison_protocol 时，这两个键也
        # 必须出现在回填协议里；否则受控变量检查只能报「缺
        # 记录」，无法识别同口径。
        self.assertIn(
            "allow_eval_content_for_cleaning",
            experiment.COMPARISON_PROTOCOL_KEYS,
        )
        for key in (
            "allow_eval_content_for_cleaning",
            "allow_entity_overlap",
        ):
            self.assertIn(key, protocol)

    def test_protocol_fallback_preserves_recorded_lr_schedule(self):
        payload = {
            "config": {"mode": "A"},
            "args": {
                "lr": 1e-3,
                "lr_schedule": "wsd",
                "steps": 100,
            },
        }
        protocol = experiment.experiment_protocol_from_payload(
            payload
        )
        self.assertEqual(protocol["lr_schedule_name"], "wsd")
        self.assertEqual(protocol["lr_schedule"], "wsd")

        payload["args"]["lr_schedule"] = "cosine"
        protocol = experiment.experiment_protocol_from_payload(
            payload
        )
        self.assertEqual(
            protocol["lr_schedule"],
            "linear_warmup_cosine_to_10pct",
        )
        self.assertEqual(protocol["lr_schedule_name"], "cosine")

    def test_saved_cleaning_alias_uses_effective_or(self):
        # 旧检查点可能只把 allow_eval_content_for_cleaning 记为
        # True，而 clean_train_against_test 仍是 argparse 默认
        # False；有效口径应为 OR，不能按单键取 False。
        self.assertTrue(
            experiment._saved_clean_against_test_effective({
                "clean_train_against_test": False,
                "allow_eval_content_for_cleaning": True,
            })
        )
        self.assertTrue(
            experiment._saved_clean_against_test_effective({
                "clean_train_against_test": True,
                "allow_eval_content_for_cleaning": False,
            })
        )
        # 两个键都缺的最老检查点按历史行为回退。
        self.assertTrue(
            experiment._saved_clean_against_test_effective({
                "deduplicate_train_groups": True,
            })
        )
        self.assertFalse(
            experiment._saved_clean_against_test_effective({
                "clean_train_against_test": False,
                "allow_eval_content_for_cleaning": False,
                "deduplicate_train_groups": True,
            })
        )

    def test_resume_restores_cleaning_justification(self):
        args = self._args(domain_justification=None)
        saved = {
            "domain_justification": "synthetic records are corrupt",
        }
        restored = experiment._restore_resume_cleaning_justification(
            args, saved
        )
        self.assertTrue(restored)
        self.assertEqual(
            args.domain_justification,
            "synthetic records are corrupt",
        )
        # 显式给出的新依据不能被检查点静默覆盖。
        explicit = self._args(
            domain_justification="new justification"
        )
        restored = experiment._restore_resume_cleaning_justification(
            explicit, saved
        )
        self.assertFalse(restored)
        self.assertEqual(
            explicit.domain_justification,
            "new justification",
        )
        # 旧检查点可能只记录 config 字段名。
        legacy = self._args(domain_justification=None)
        restored = experiment._restore_resume_cleaning_justification(
            legacy,
            {"cleaning_domain_justification": "legacy justification"},
        )
        self.assertTrue(restored)
        self.assertEqual(
            legacy.domain_justification,
            "legacy justification",
        )

    def test_sweep_detects_preregistration_file_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "seed42"
            second = root / "seed43"
            first.mkdir()
            second.mkdir()
            (first / "config.json").write_text(
                json.dumps({
                    "preregistration": {
                        "file_sha256": "a" * 64,
                    }
                }),
                encoding="utf-8",
            )
            (second / "config.json").write_text(
                json.dumps({
                    "preregistration": {
                        "file_sha256": "b" * 64,
                    }
                }),
                encoding="utf-8",
            )
            error = experiment.preregistration_sweep_error(
                [str(first), str(second)],
                "a" * 64,
            )
        self.assertIn("seed43", error)
        self.assertIn("中途", error)

    def test_data_checks_sha_splits_cleaning_and_separator(self):
        payload = self._payload()
        payload["data"]["group_separator"] = "\\n\\n"
        payload["data"]["cleaning"] = {
            "deduplicate_train_groups": False,
        }
        args = self._args(
            split_policy="contiguous",
            group_separator="\\n\\n",
            deduplicate_train_groups=False,
        )
        self.assertIsNone(
            experiment.preregistration_data_error(
                args, payload, self._dataset()
            )
        )

        wrong_sha = self._dataset()
        wrong_sha.data_sha256 = "c" * 64
        self.assertIn(
            "data.sha256",
            experiment.preregistration_data_error(
                args, payload, wrong_sha
            ),
        )

        args.deduplicate_train_groups = True
        self.assertIn(
            "data.cleaning",
            experiment.preregistration_data_error(
                args, payload, self._dataset()
            ),
        )

        args.deduplicate_train_groups = False
        args.group_separator = "<|doc|>"
        self.assertIn(
            "group_separator",
            experiment.preregistration_data_error(
                args, payload, self._dataset()
            ),
        )


    def test_cleaning_aliases_are_equivalent_for_data_check(self):
        # `--clean-train-against-test` 与
        # `--allow-eval-content-for-cleaning` 是同义入口：任一
        # 为真都表示训练集清洗读取 test。预注册文件写哪个别名
        # 都不应改变核对结果。
        payload = self._payload()
        payload["data"]["cleaning"] = {
            "allow_eval_content_for_cleaning": True,
            "deduplicate_train_groups": True,
        }
        args = self._args(
            split_policy="contiguous",
            group_separator=None,
            clean_train_against_test=True,
            allow_eval_content_for_cleaning=False,
            deduplicate_train_groups=True,
        )
        self.assertIsNone(
            experiment.preregistration_data_error(
                args, payload, self._dataset()
            )
        )

        payload["data"]["cleaning"] = {
            "clean_train_against_test": True,
            "deduplicate_train_groups": True,
        }
        args = self._args(
            split_policy="contiguous",
            group_separator=None,
            clean_train_against_test=False,
            allow_eval_content_for_cleaning=True,
            deduplicate_train_groups=True,
        )
        self.assertIsNone(
            experiment.preregistration_data_error(
                args, payload, self._dataset()
            )
        )

    def test_data_rejects_cleaning_not_declared_in_preregistration(self):
        # 未声明 cleaning 等于声明全部关闭；否则看到验证结果后
        # 临时启用清洗就能改变训练样本集合而绕过预注册核对。
        payload = self._payload()
        args = self._args(
            split_policy="contiguous",
            group_separator=None,
            deduplicate_train_groups=True,
        )
        error = experiment.preregistration_data_error(
            args, payload, self._dataset()
        )
        self.assertIn("data.cleaning", error)

        # 只声明一个关闭开关也不能让未声明的清洗项静默通过。
        payload["data"]["cleaning"] = {
            "deduplicate_train_groups": False,
        }
        args = self._args(
            split_policy="contiguous",
            group_separator=None,
            deduplicate_train_groups=False,
            drop_train_shared_chunks=True,
        )
        error = experiment.preregistration_data_error(
            args, payload, self._dataset()
        )
        self.assertIn("drop_train_shared_chunks", error)

        # 全部关闭时，省略 cleaning 的旧预注册文件仍可核对通过。
        args = self._args(
            split_policy="contiguous",
            group_separator=None,
            deduplicate_train_groups=False,
            drop_train_shared_chunks=False,
            clean_train_against_test=False,
            allow_eval_content_for_cleaning=False,
            drop_train_corrupt_groups=False,
            drop_train_corrupt_bytes=False,
            drop_train_degenerate_groups=False,
            drop_train_degenerate_runs=False,
        )
        self.assertIsNone(
            experiment.preregistration_data_error(
                args, self._payload(), self._dataset()
            )
        )

    def test_schema_rejects_invalid_near_duplicate_cleaning_values(self):
        payload = self._payload()
        payload["data"]["cleaning"] = {
            "near_duplicate_threshold": 1.5,
        }
        error = experiment.preregistration_schema_error(payload)
        self.assertIn("near_duplicate_threshold", error)

        payload["data"]["cleaning"] = {
            "near_duplicate_shingle_bytes": 0,
        }
        error = experiment.preregistration_schema_error(payload)
        self.assertIn("near_duplicate_shingle_bytes", error)

        payload["data"]["cleaning"] = {
            "near_duplicate_threshold": 0.8,
            "near_duplicate_shingle_bytes": 8,
        }
        self.assertIsNone(
            experiment.preregistration_schema_error(payload)
        )

    def test_data_rejects_changed_near_duplicate_threshold(self):
        # 阈值决定删除哪些训练群组，看到结果后再调阈值等于
        # 事后改变数据；预注册显式写了就必须逐项核对。
        payload = self._payload()
        payload["data"]["cleaning"] = {
            "drop_train_near_duplicate_groups": True,
            "near_duplicate_threshold": 0.7,
            "near_duplicate_shingle_bytes": 8,
        }
        args = self._args(
            split_policy="contiguous",
            group_separator=None,
            drop_train_near_duplicate_groups=True,
            near_duplicate_threshold=0.8,
            near_duplicate_shingle_bytes=8,
        )
        error = experiment.preregistration_data_error(
            args, payload, self._dataset()
        )
        self.assertIn("near_duplicate_threshold", error)

        payload["data"]["cleaning"][
            "near_duplicate_threshold"
        ] = 0.8
        self.assertIsNone(
            experiment.preregistration_data_error(
                args, payload, self._dataset()
            )
        )

    def test_data_rejects_undeclared_group_separator(self):
        # 未声明 group_separator 等于不分群组；看到结果后再加
        # 分隔符会改变吸附后的切分边界，必须拒绝。
        payload = self._payload()
        args = self._args(
            split_policy="contiguous",
            group_separator="\\n\\n",
        )
        error = experiment.preregistration_data_error(
            args, payload, self._dataset()
        )
        self.assertIn("group_separator", error)

        # 声明 null 与不声明同义；显式声明相同分隔符则通过。
        payload["data"]["group_separator"] = None
        self.assertIn(
            "group_separator",
            experiment.preregistration_data_error(
                args, payload, self._dataset()
            ),
        )
        payload["data"]["group_separator"] = "\\n\\n"
        self.assertIsNone(
            experiment.preregistration_data_error(
                args, payload, self._dataset()
            )
        )

    def test_schema_rejects_conflicting_cleaning_aliases(self):
        payload = self._payload()
        payload["data"]["cleaning"] = {
            "clean_train_against_test": True,
            "allow_eval_content_for_cleaning": False,
        }
        error = experiment.preregistration_schema_error(payload)
        self.assertIn("同义开关", error)

    def test_normalize_cleaning_aliases_sets_both_fields(self):
        args = self._args(
            clean_train_against_test=False,
            allow_eval_content_for_cleaning=True,
        )
        effective = experiment._normalize_clean_against_test(args)
        self.assertTrue(effective)
        self.assertTrue(args.clean_train_against_test)
        self.assertTrue(args.allow_eval_content_for_cleaning)

    def test_optional_data_null_matches_data_checker(self):
        payload = self._payload()
        payload["data"]["cleaning"] = None
        payload["data"]["entity_manifest_sha256"] = None
        payload["data"]["time_order_manifest_sha256"] = None
        self.assertIsNone(
            experiment.preregistration_schema_error(payload)
        )

    def test_schema_rejects_out_of_range_seed(self):
        payload = self._payload()
        payload["seeds"] = [
            42,
            43,
            experiment.MAX_SEED + 1,
        ]
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn(str(experiment.MAX_SEED), error)


class ArtifactManifestTests(unittest.TestCase):
    """归档清单：内容可复核、路径不可逃逸、新增文件可披露。"""

    def _make_run(self, root):
        root.mkdir()
        config = {
            "config": {"mode": "A"},
            "run_metadata": {
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "script_sha256": "script-sha",
                "git_commit": "abc123",
                "git_dirty": False,
            },
            "data": {
                "sha256": "data-sha",
                "split_sha256": {"train": "train-sha"},
            },
            "comparison_protocol_sha256": "protocol-sha",
            "preregistration": {
                "study_id": "study-1",
                "file_sha256": "prereg-sha",
                "frozen_commit": "abc123",
            },
            "hypothesis": "delayed merge reduces val loss",
            "research_question": "RQ1",
        }
        (root / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        (root / "metrics.jsonl").write_text(
            '{"step": 1, "val_loss": 3.0}\n', encoding="utf-8"
        )
        nested = root / "logs" / "train.log"
        nested.parent.mkdir()
        nested.write_text("epoch 1\n", encoding="utf-8")
        return root

    def test_manifest_roundtrip_and_run_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            manifest_path = experiment.write_artifact_manifest(root)
            self.assertEqual(
                manifest_path.name,
                "artifact_manifest.json",
            )

            payload = json.loads(manifest_path.read_text())
            self.assertEqual(
                payload["schema"],
                "smelt-isr/artifact-manifest/v1",
            )
            paths = {entry["path"] for entry in payload["files"]}
            self.assertEqual(
                paths,
                {"config.json", "metrics.jsonl", "logs/train.log"},
            )
            self.assertNotIn("artifact_manifest.json", paths)
            self.assertEqual(payload["run"]["study_id"], "study-1")
            self.assertEqual(
                payload["run"]["data_sha256"], "data-sha"
            )
            self.assertEqual(
                payload["run"]["comparison_protocol_sha256"],
                "protocol-sha",
            )

            report = experiment.verify_artifact_manifest(root)
            self.assertTrue(report["ok"], report)
            self.assertTrue(report["integrity_ok"])
            self.assertEqual(report["checked"], 3)
            self.assertEqual(report["mismatched"], [])
            self.assertEqual(report["untracked"], [])
            self.assertEqual(len(report["manifest_sha256"]), 64)

    def test_manifest_detects_tampering_deletion_and_extra_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            manifest_path = experiment.write_artifact_manifest(root)
            original = (root / "metrics.jsonl").read_text()

            # 同长度替换只改内容：必须由 SHA-256 而不是大小发现。
            tampered = original.replace("3.0", "3.5")
            self.assertEqual(len(tampered), len(original))
            (root / "metrics.jsonl").write_text(tampered)
            report = experiment.verify_artifact_manifest(root)
            self.assertFalse(report["ok"])
            self.assertIn(
                "sha256",
                {item["reason"] for item in report["mismatched"]},
            )

            # 删除清单内文件必须进入 missing。
            (root / "metrics.jsonl").write_text(original)
            (root / "logs" / "train.log").unlink()
            report = experiment.verify_artifact_manifest(root)
            self.assertFalse(report["integrity_ok"])
            self.assertEqual(report["missing"], ["logs/train.log"])

            # 恢复文件后，清单外新增文件只进入 untracked：
            # 默认不判损坏（可能是后续日志），strict 才失败。
            (root / "logs" / "train.log").write_text(
                "epoch 1\n"
            )
            (root / "extra.txt").write_text("late output")
            report = experiment.verify_artifact_manifest(root)
            self.assertTrue(report["integrity_ok"])
            self.assertEqual(report["untracked"], ["extra.txt"])
            self.assertTrue(report["ok"])

            strict = experiment.verify_artifact_manifest(
                root, strict=True
            )
            self.assertFalse(strict["ok"])
            self.assertTrue(strict["integrity_ok"])

            # 重新生成后清单应吸收新增文件，strict 也通过。
            experiment.write_artifact_manifest(root)
            report = experiment.verify_artifact_manifest(
                root, strict=True
            )
            self.assertTrue(report["ok"], report)

            # 清单自身不能被写入自身（否则哈希自指）。
            payload = json.loads(manifest_path.read_text())
            self.assertNotIn(
                "artifact_manifest.json",
                {entry["path"] for entry in payload["files"]},
            )

    def test_manifest_path_validation_rejects_noncanonical(self):
        for value in (
            "",
            "/absolute",
            "../escape",
            "a/../b",
            "a//b",
            "a/./b",
            "a\\b",
            "./a",
            "a/",
            # Windows 的盘符/UNC 前缀在 POSIX 下不是绝对路径，
            # 但换到 Windows 后会把 root / path 解析到目录外。
            "C:/outside",
            "c:outside",
            "//server/share",
        ):
            self.assertIsNone(
                experiment._manifest_relative_path(value),
                value,
            )
        for value in ("config.json", "logs/train.log", "a-b_c.1"):
            self.assertEqual(
                experiment._manifest_relative_path(value),
                value,
            )
        self.assertIsNone(experiment._manifest_relative_path(3))

    def test_manifest_writer_refuses_unverifiable_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            try:
                (root / "bad\\name.txt").write_text(
                    "x", encoding="utf-8"
                )
            except OSError:
                self.skipTest("当前平台不允许该文件名")
            # 写入端不能生成一份校验端必然拒绝的清单；反斜杠
            # 路径在跨平台归档中不安全，应明确失败。
            with self.assertRaises(ValueError):
                experiment.write_artifact_manifest(root)

    def test_manifest_rejects_unsafe_and_duplicate_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            manifest_path = experiment.write_artifact_manifest(root)
            payload = json.loads(manifest_path.read_text())
            payload["files"].extend([
                {
                    "path": "../outside.txt",
                    "type": "file",
                    "bytes": 0,
                    "sha256": "x",
                },
                {
                    "path": "/etc/passwd",
                    "type": "file",
                    "bytes": 0,
                    "sha256": "x",
                },
                {
                    "path": "config.json",
                    "type": "file",
                    "bytes": 0,
                    "sha256": "x",
                },
            ])
            manifest_path.write_text(
                json.dumps(payload), encoding="utf-8"
            )

            report = experiment.verify_artifact_manifest(root)
            self.assertFalse(report["ok"])
            reasons = {
                item["reason"] for item in report["mismatched"]
            }
            self.assertIn("unsafe_path", reasons)
            self.assertIn("duplicate_path", reasons)

    def test_manifest_symlink_target_is_recorded_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            link = root / "latest.pt"
            try:
                link.symlink_to("metrics.jsonl")
            except (OSError, NotImplementedError):
                self.skipTest("当前平台不支持符号链接")
            experiment.write_artifact_manifest(root)

            payload = json.loads(
                (root / "artifact_manifest.json").read_text()
            )
            link_entry = next(
                entry
                for entry in payload["files"]
                if entry["path"] == "latest.pt"
            )
            self.assertEqual(link_entry["type"], "symlink")
            self.assertEqual(link_entry["target"], "metrics.jsonl")
            self.assertNotIn("sha256", link_entry)

            self.assertTrue(
                experiment.verify_artifact_manifest(root)["ok"]
            )

            link.unlink()
            link.symlink_to("config.json")
            report = experiment.verify_artifact_manifest(root)
            self.assertFalse(report["ok"])
            self.assertIn(
                "symlink_target",
                {item["reason"] for item in report["mismatched"]},
            )

    def test_manifest_symlink_loop_path_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            try:
                (root / "a").symlink_to("b")
                (root / "b").symlink_to("a")
            except (OSError, NotImplementedError):
                self.skipTest("当前平台不支持符号链接")
            manifest_path = experiment.write_artifact_manifest(root)
            payload = json.loads(manifest_path.read_text())
            # 恶意/损坏清单可以声明一条穿过符号链接环的普通文件
            # 路径；Path.resolve 会抛 RuntimeError，校验器必须把
            # 它变成结构化失败而不是 traceback。
            payload["files"].append({
                "path": "a/x",
                "type": "file",
                "bytes": 1,
                "sha256": "0" * 64,
            })
            manifest_path.write_text(
                json.dumps(payload), encoding="utf-8"
            )
            report = experiment.verify_artifact_manifest(root)
        self.assertFalse(report["ok"])
        self.assertIn(
            "path_resolution_error",
            {item["reason"] for item in report["mismatched"]},
        )

    def test_manifest_external_symlink_is_allowed_but_contained_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            outside = Path(tmp) / "outside.txt"
            outside.write_text("external artifact")

            # 清单记录链接目标而不是跟随链接：运行目录之外的
            # 合法归档链接必须能通过校验，否则 writer 写出的
            # 清单无法验证自己的输出。
            external_link = root / "external.txt"
            try:
                external_link.symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("当前平台不支持符号链接")
            experiment.write_artifact_manifest(root)
            self.assertTrue(
                experiment.verify_artifact_manifest(root)["ok"]
            )

            # 逃逸检查仍然生效：通过目录链接把清单路径引到
            # 运行目录之外必须拒绝，即使条目的父目录字符串
            # 看起来在 root 下。
            escape_dir = root / "escape"
            try:
                escape_dir.symlink_to(Path(tmp), target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("当前平台不支持目录符号链接")
            manifest_path = root / "artifact_manifest.json"
            payload = json.loads(manifest_path.read_text())
            payload["files"].append({
                "path": "escape/outside.txt",
                "type": "file",
                "bytes": outside.stat().st_size,
                "sha256": experiment._sha256_file(outside),
            })
            manifest_path.write_text(
                json.dumps(payload), encoding="utf-8"
            )
            report = experiment.verify_artifact_manifest(root)
            self.assertFalse(report["ok"])
            self.assertIn(
                "escapes_root",
                {item["reason"] for item in report["mismatched"]},
            )

    def test_manifest_run_summary_prefers_original_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            config = json.loads(
                (root / "config.json").read_text()
            )
            config["original_run_metadata"] = {
                "timestamp_utc": "2025-01-01T00:00:00+00:00",
                "script_sha256": "original-script",
                "git_commit": "original-commit",
                "git_dirty": False,
            }
            config["run_metadata"] = {
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "script_sha256": "resumed-script",
                "git_commit": "resumed-commit",
                "git_dirty": True,
            }
            (root / "config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            payload = experiment.build_artifact_manifest(root)
            summary = payload["run"]
            self.assertEqual(
                summary["run_started_utc"],
                "2025-01-01T00:00:00+00:00",
            )
            self.assertEqual(
                summary["script_sha256"], "original-script"
            )
            self.assertEqual(
                summary["git_commit"], "original-commit"
            )

    def test_manifest_entry_limit_counts_across_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            # _make_run 本身有 3 个文件（config、metrics、logs/）。
            # 上限按整棵目录树计数，不能被每个子目录各自重置。
            previous = experiment._ARTIFACT_MANIFEST_MAX_ENTRIES
            experiment._ARTIFACT_MANIFEST_MAX_ENTRIES = 2
            try:
                with self.assertRaises(ValueError) as raised:
                    experiment.build_artifact_manifest(root)
            finally:
                experiment._ARTIFACT_MANIFEST_MAX_ENTRIES = previous
            self.assertIn("上限", str(raised.exception))

    def test_manifest_expected_sha_pin_detects_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            manifest_path = experiment.write_artifact_manifest(root)
            pinned = experiment._sha256_file(manifest_path)
            old_argv = sys.argv
            try:
                sys.argv = [
                    "experiment.py",
                    "--verify-manifest",
                    str(root),
                    "--expect-manifest-sha256",
                    pinned,
                ]
                with contextlib.redirect_stdout(
                    io.StringIO()
                ) as out:
                    experiment.main()
                self.assertIn(
                    '"manifest_sha256_matches_expected": true',
                    out.getvalue(),
                )

                # 清单被整体替换但内部仍然自洽：只有外部记录的
                # 指纹能发现替换，必须判失败。
                sys.argv = [
                    "experiment.py",
                    "--verify-manifest",
                    str(root),
                    "--expect-manifest-sha256",
                    "0" * 64,
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        experiment.main()
                self.assertEqual(raised.exception.code, 1)
            finally:
                sys.argv = old_argv

            error = experiment.manifest_mode_error(
                argparse.Namespace(
                    write_manifest=None,
                    verify_manifest=None,
                    verify_manifest_strict=False,
                    expect_manifest_sha256="0" * 64,
                )
            )
            self.assertIn("--verify-manifest", error)

    def test_manifest_mode_rejects_conflicting_entries(self):
        both = experiment.manifest_mode_error(
            argparse.Namespace(
                write_manifest="run",
                verify_manifest="run",
                verify_manifest_strict=False,
            )
        )
        self.assertIn("不能同时使用", both)

        strict_only = experiment.manifest_mode_error(
            argparse.Namespace(
                write_manifest=None,
                verify_manifest=None,
                verify_manifest_strict=True,
            )
        )
        self.assertIn("--verify-manifest", strict_only)

        conflict = experiment.manifest_mode_error(
            argparse.Namespace(
                write_manifest="run",
                verify_manifest=None,
                verify_manifest_strict=False,
                compare=["a"],
            )
        )
        self.assertIn("--compare", conflict)

        for name, value in (
            ("plan_design", True),
            ("suite_modes", "smelt,A"),
            ("suite_ablation", True),
            ("paper_cell", "smelt-200M"),
        ):
            with self.subTest(conflict=name):
                error = experiment.manifest_mode_error(
                    argparse.Namespace(
                        write_manifest="run",
                        verify_manifest=None,
                        verify_manifest_strict=False,
                        **{name: value},
                    )
                )
                self.assertIn("不能与", error)

        ok = experiment.manifest_mode_error(
            argparse.Namespace(
                write_manifest="run",
                verify_manifest=None,
                verify_manifest_strict=False,
                compare=None,
                final_test_comparison=None,
                sanity_only=False,
                statistical_baseline=False,
                suite_baseline_tuning=False,
                preregistration=None,
                data=None,
                out=None,
                seeds=None,
                resume=None,
                evaluate_test=False,
            )
        )
        self.assertIsNone(ok)

    def test_manifest_cli_writes_and_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            old_argv = sys.argv
            try:
                sys.argv = [
                    "experiment.py",
                    "--write-manifest",
                    str(root),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    experiment.main()
                self.assertTrue(
                    (root / "artifact_manifest.json").is_file()
                )

                sys.argv = [
                    "experiment.py",
                    "--verify-manifest",
                    str(root),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    experiment.main()

                (root / "metrics.jsonl").write_text("broken")
                sys.argv = [
                    "experiment.py",
                    "--verify-manifest",
                    str(root),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        experiment.main()
                self.assertEqual(raised.exception.code, 1)
            finally:
                sys.argv = old_argv


class EnvironmentSnapshotTests(unittest.TestCase):
    """环境身份：关键依赖/编译选项可核对，完整包清单可审计。"""

    def test_snapshot_is_stable_and_contains_critical_packages(self):
        first = experiment.build_environment_snapshot()
        second = experiment.build_environment_snapshot()
        self.assertEqual(first["schema"], "smelt-isr/environment/v1")
        self.assertEqual(
            first["environment_id"], second["environment_id"]
        )
        self.assertEqual(len(first["environment_id"]), 64)
        names = {
            item["name"]
            for item in first["packages"]["all"]
        }
        self.assertIn("torch", names)
        critical_names = {
            item["name"]
            for item in first["packages"]["critical"]
        }
        self.assertIn("torch", names)
        self.assertEqual(
            critical_names,
            {
                name
                for name in names
                if experiment._is_critical_distribution(name)
            },
        )
        self.assertEqual(
            first["packages"]["all"],
            sorted(
                first["packages"]["all"],
                key=lambda item: item["name"],
            ),
        )

    def test_per_run_hashseed_does_not_change_environment_identity(self):
        snapshot = experiment.build_environment_snapshot()
        altered = json.loads(json.dumps(snapshot))
        altered["determinism"]["pythonhashseed_at_start"] = "12345"
        # 套件子进程按种子固定不同的 PYTHONHASHSEED；它属于运行
        # 参数而不是安装环境，不能让同一环境的多种子套件被
        # --compare 误报为环境不同。
        recalculated = experiment._canonical_json_sha256(
            experiment._environment_identity_payload(altered)
        )
        self.assertEqual(
            recalculated, snapshot["environment_id"]
        )

    def test_full_package_drift_does_not_change_critical_identity(self):
        snapshot = experiment.build_environment_snapshot()
        altered = json.loads(json.dumps(snapshot))
        altered["packages"]["all"].append({
            "name": "unrelated-tool",
            "version": "1.0",
        })
        altered["packages"]["all_sha256"] = (
            experiment._canonical_json_sha256(
                altered["packages"]["all"]
            )
        )
        # 非关键包不参与 environment_id：默认核对仍通过，
        # 严格模式才把完整包清单漂移判为失败。
        altered["environment_id"] = (
            experiment._canonical_json_sha256(
                experiment._environment_identity_payload(altered)
            )
        )
        report = experiment.compare_environment_snapshots(
            snapshot, altered
        )
        self.assertTrue(report["ok"], report)
        self.assertFalse(report["strict_ok"])
        self.assertTrue(report["all_packages_match"] is False)
        self.assertEqual(
            [item["name"] for item in report[
                "package_differences"
            ]["added"]],
            ["unrelated-tool"],
        )

    def test_critical_package_change_breaks_identity(self):
        snapshot = experiment.build_environment_snapshot()
        altered = json.loads(json.dumps(snapshot))
        critical = altered["packages"]["critical"]
        self.assertTrue(critical)
        name = critical[0]["name"]
        critical[0]["version"] = "0.0.0-drift"
        altered["packages"]["all"] = [
            (
                {**item, "version": "0.0.0-drift"}
                if item["name"] == name
                else item
            )
            for item in altered["packages"]["all"]
        ]
        altered["packages"]["all_sha256"] = (
            experiment._canonical_json_sha256(
                altered["packages"]["all"]
            )
        )
        altered["packages"]["critical_sha256"] = (
            experiment._canonical_json_sha256(critical)
        )
        altered["environment_id"] = (
            experiment._canonical_json_sha256(
                experiment._environment_identity_payload(altered)
            )
        )
        report = experiment.compare_environment_snapshots(
            snapshot, altered
        )
        self.assertFalse(report["ok"])
        self.assertTrue(
            report["critical_package_differences"]
        )
        self.assertTrue(
            any(
                item["path"].startswith("packages")
                for item in report["runtime_differences"]
            ),
            report["runtime_differences"],
        )

    def test_check_environment_reads_file_and_config_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = experiment.build_environment_snapshot()
            environment_path = root / "environment.json"
            environment_path.write_text(
                json.dumps(snapshot), encoding="utf-8"
            )
            report = experiment.check_environment_snapshot(
                environment_path
            )
            self.assertTrue(report["ok"], report)
            self.assertTrue(report["strict_ok"])

            config_path = root / "config.json"
            config_path.write_text(
                json.dumps({
                    "environment": {
                        "schema": snapshot["schema"],
                        "environment_id": snapshot[
                            "environment_id"
                        ],
                        "file": "environment.json",
                        "file_sha256": experiment._sha256_file(
                            environment_path
                        ),
                    },
                }),
                encoding="utf-8",
            )
            report = experiment.check_environment_snapshot(
                config_path
            )
            self.assertTrue(report["ok"], report)

            # config 记录的摘要必须能发现 environment.json 被替换：
            # 否则 --check-environment 会拿一份被篡改的“期望环境”
            # 去比较，核对本身失去意义。
            environment_path.write_text(
                json.dumps({"schema": snapshot["schema"]}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                experiment.check_environment_snapshot(config_path)

    def test_write_environment_record_keeps_original_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = experiment.build_environment_snapshot()
            current = json.loads(json.dumps(original))
            current["environment_id"] = "1" * 64
            record = experiment.write_run_environment_record(
                root,
                {"environment": current},
                {"environment": original},
            )
            self.assertEqual(
                record["environment_id"],
                original["environment_id"],
            )
            self.assertTrue(
                record["environment_changed_since_start"]
            )
            self.assertEqual(
                record["resume_environment_id"], "1" * 64
            )
            on_disk = json.loads(
                (root / "environment.json").read_text()
            )
            self.assertEqual(
                on_disk["environment_id"],
                original["environment_id"],
            )
            self.assertEqual(
                record["file_sha256"],
                experiment._sha256_file(
                    root / "environment.json"
                ),
            )

    def test_environment_mode_rejects_training_arguments(self):
        strict_only = experiment.environment_mode_error(
            argparse.Namespace(
                check_environment=None,
                check_environment_strict=True,
            )
        )
        self.assertIn("--check-environment", strict_only)

        conflict = experiment.environment_mode_error(
            argparse.Namespace(
                check_environment="run",
                check_environment_strict=False,
                data="corpus.txt",
            )
        )
        self.assertIn("--data", conflict)

        strict_manifest = experiment.environment_mode_error(
            argparse.Namespace(
                check_environment="run",
                check_environment_strict=False,
                verify_manifest_strict=True,
            )
        )
        self.assertIn("--verify-manifest-strict", strict_manifest)

        ok = experiment.environment_mode_error(
            argparse.Namespace(
                check_environment="run",
                check_environment_strict=False,
            )
        )
        self.assertIsNone(ok)

    def test_environment_cli_checks_and_fails_on_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "environment.json"
            snapshot = experiment.build_environment_snapshot()
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            old_argv = sys.argv
            try:
                sys.argv = [
                    "experiment.py",
                    "--check-environment",
                    str(path),
                ]
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    experiment.main()
                self.assertIn("[env] OK", out.getvalue())

                snapshot["environment_id"] = "0" * 64
                path.write_text(
                    json.dumps(snapshot), encoding="utf-8"
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        experiment.main()
                self.assertEqual(raised.exception.code, 1)
            finally:
                sys.argv = old_argv


    def test_environment_id_must_match_snapshot_content(self):
        snapshot = experiment.build_environment_snapshot()
        altered = json.loads(json.dumps(snapshot))
        altered["packages"]["critical"][0]["version"] = (
            "0.0.0-drift"
        )
        # 只改内容、保留旧 id：不能因为两个快照声明的 id 相同就
        # 把被篡改/损坏的归档当成同一环境。
        report = experiment.compare_environment_snapshots(
            snapshot, altered
        )
        self.assertFalse(report["ok"], report)
        self.assertFalse(report["strict_ok"], report)
        self.assertFalse(
            report["actual_environment_id_consistent"]
        )
        self.assertTrue(
            report["expected_environment_id_consistent"]
        )

    def test_check_environment_restores_run_time_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "environment.json"
            snapshot = experiment.build_environment_snapshot()
            path.write_text(
                json.dumps(snapshot), encoding="utf-8"
            )
            determinism = snapshot["determinism"]
            old_deterministic = (
                torch.are_deterministic_algorithms_enabled()
            )
            old_warn = (
                torch.is_deterministic_algorithms_warn_only_enabled()
            )
            old_cudnn_deterministic = (
                torch.backends.cudnn.deterministic
            )
            old_cudnn_benchmark = torch.backends.cudnn.benchmark
            try:
                # 模拟全新核对进程的默认状态：训练入口设置的
                # 确定性开关尚未应用。
                torch.use_deterministic_algorithms(
                    not bool(
                        determinism["deterministic_algorithms"]
                    ),
                    warn_only=not bool(
                        determinism[
                            "deterministic_algorithms_warn_only"
                        ]
                    ),
                )
                torch.backends.cudnn.deterministic = not bool(
                    determinism["cudnn_deterministic"]
                )
                torch.backends.cudnn.benchmark = not bool(
                    determinism["cudnn_benchmark"]
                )
                report = experiment.check_environment_snapshot(path)
            finally:
                torch.use_deterministic_algorithms(
                    old_deterministic, warn_only=old_warn
                )
                torch.backends.cudnn.deterministic = (
                    old_cudnn_deterministic
                )
                torch.backends.cudnn.benchmark = old_cudnn_benchmark
        self.assertTrue(report["ok"], report)
        self.assertEqual(
            report["actual_environment_id"],
            snapshot["environment_id"],
        )
        self.assertIn(
            "deterministic_algorithms",
            report["applied_environment_settings"],
        )
        self.assertEqual(report["environment_setting_errors"], {})

    def test_config_summary_id_must_match_environment_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = experiment.build_environment_snapshot()
            environment_path = root / "environment.json"
            environment_path.write_text(
                json.dumps(snapshot), encoding="utf-8"
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps({
                    "environment": {
                        "schema": snapshot["schema"],
                        "environment_id": "0" * 64,
                        "file": "environment.json",
                        "file_sha256": experiment._sha256_file(
                            environment_path
                        ),
                    },
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                experiment.check_environment_snapshot(config_path)

    def test_snapshot_device_preserves_non_default_index(self):
        device = experiment._snapshot_device({
            "device": {"type": "cuda", "index": 1},
        })
        self.assertEqual(device.type, "cuda")
        self.assertEqual(device.index, 1)
        # CPU 没有可区分的设备号；非法/布尔 index 不能触发
        # torch.device 构造错误。
        self.assertIsNone(
            experiment._snapshot_device({
                "device": {"type": "cpu", "index": 1},
            }).index
        )
        self.assertEqual(
            experiment._snapshot_device({
                "device": {"type": "cuda", "index": True},
            }).type,
            "cuda",
        )


class EdgeCaseRegressionTests(unittest.TestCase):
    """本轮审计修复的边界条件回归。"""

    def test_group_split_without_test_ends_at_eof(self):
        # 旧实现会把 cut_val 落在 19，ByteDataset 随即把
        # [19,25) 当成 test，尽管 test_fraction=0。
        buf = bytearray(b"a" * 25)
        for position in (5, 18, 19, 24):
            buf[position] = 10
        cut_train, cut_val = experiment._snap_group_split(
            buf,
            b"\n",
            nominal_train=5,
            nominal_val_end=5,
            min_len=2,
            has_test=False,
        )
        self.assertEqual(cut_train, 6)
        self.assertEqual(cut_val, 25)

    def test_empty_group_separator_is_rejected(self):
        with self.assertRaises(ValueError):
            list(experiment._iter_group_boundaries(
                bytearray(b"abc"), b"", 0
            ))
        with self.assertRaises(ValueError):
            list(experiment._iter_group_spans(
                bytearray(b"abc"), b"", 0, 3
            ))

    def test_overlapping_separator_does_not_create_extra_document(self):
        ids = torch.tensor([[65, 65, 65]])
        segments = experiment.document_segment_ids(ids, [65, 65])
        self.assertEqual(segments.tolist(), [[0, 0, 0]])

    def test_split_overlap_with_gap_is_not_cancelled(self):
        record = experiment._split_isolation_record(
            {
                "train": (0, 30),
                "val": (20, 50),
                "test": (60, 70),
            },
            70,
        )
        self.assertEqual(record["overlap_bytes"], 10)
        self.assertFalse(record["covers_all_bytes"])

    def test_metrics_truncation_drops_partial_trailing_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            path.write_text(
                '{"step":1,"val_loss":3.0}\n'
                '{"step":2,"val_lo',
                encoding="utf-8",
            )
            experiment._truncate_metrics_after_step(path, 1)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write('{"step":2,"val_loss":2.0}\n')
            records = experiment.read_metrics_records(path)
        self.assertEqual([row["step"] for row in records], [1, 2])

    def test_metrics_truncation_terminates_valid_last_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            path.write_text('{"step":1}', encoding="utf-8")
            experiment._truncate_metrics_after_step(path, 1)
            self.assertTrue(
                path.read_text(encoding="utf-8").endswith("\n")
            )
            path.write_text('{"step":2}', encoding="utf-8")
            experiment._truncate_metrics_after_step(path, 1)
            self.assertEqual(path.read_text(encoding="utf-8"), "")

    def test_update_run_config_refuses_to_overwrite_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            config = run_dir / "config.json"
            config.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                experiment.update_run_config(
                    run_dir, "final", {"val_loss": 1.0}
                )
            self.assertEqual(
                config.read_text(encoding="utf-8"), "{not json"
            )

    def test_local_test_ledger_counts_as_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self.assertFalse(
                experiment.test_evaluation_sealed(run_dir)
            )
            (run_dir / "test_access_ledger.jsonl").write_text(
                "", encoding="utf-8"
            )
            self.assertTrue(
                experiment.test_evaluation_sealed(run_dir)
            )

    def test_global_test_ledger_counts_as_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            ledger = Path(tmp) / "ledger.jsonl"
            ledger.write_text(
                json.dumps({"out_dir": str(run_dir)}) + "\n",
                encoding="utf-8",
            )
            self.assertFalse(
                experiment.test_evaluation_sealed(run_dir)
            )
            self.assertTrue(
                experiment.test_evaluation_sealed(
                    run_dir, ledger
                )
            )
            self.assertFalse(
                experiment.test_evaluation_sealed(
                    run_dir, Path(tmp) / "other.jsonl"
                )
            )

    def test_resume_split_error_detects_missing_current_split(self):
        error = experiment.resume_split_change_error(
            {"train": "a", "test": "b"},
            {"train": "a", "val": "c"},
        )
        self.assertIsNotNone(error)
        self.assertIn("test", error)

    def test_build_protocol_normalizes_cleaning_alias(self):
        cfg = _tiny_config()
        args = argparse.Namespace(
            lr=1e-3,
            warmup_steps=1,
            steps=2,
            weight_decay=0.0,
            grad_clip=1.0,
            batch_size=1,
            clean_train_against_test=False,
            allow_eval_content_for_cleaning=True,
        )
        protocol = experiment.build_comparison_protocol(
            cfg, args
        )
        self.assertTrue(protocol["clean_train_against_test"])
        self.assertTrue(
            protocol["allow_eval_content_for_cleaning"]
        )

    def test_protocol_fallback_reads_document_separator_from_config(self):
        protocol = experiment.experiment_protocol_from_payload({
            "config": {
                "mode": "smelt",
                "document_masks": True,
                "document_separator": [10],
            },
            "args": {"lr": 1e-3},
        })
        self.assertEqual(protocol["document_separator"], [10])
        self.assertTrue(protocol["document_masks"])

    def test_fragility_not_evaluable_is_not_reported_as_stable(self):
        contrast = {
            "baseline": "base",
            "variant": "A",
            "factor": "mode",
            "p_value": 0.003,
            "paired_differences": [1.0, 1.1, 0.9],
            "paired_seeds": [1, 2, 3],
            "significant": True,
        }
        experiment._apply_multiple_comparison_correction([contrast])
        self.assertEqual(
            contrast["fragility"]["verdict"], "not_evaluable"
        )
        summary = experiment._fragility_summary([contrast])
        self.assertEqual(
            summary["verdict_counts"], {"not_evaluable": 1}
        )
        self.assertIn("无法评估", summary["advisories"][0])

    def test_fragility_deduplicates_reversed_comparison(self):
        first = {
            "baseline": "local_base",
            "variant": "A",
            "factor": "mode",
            "significant": True,
            "fragility": {
                "available": True,
                "verdict": "stable",
                "n": 5,
                "n_positive": 5,
                "n_negative": 0,
                "alpha": 0.05,
            },
        }
        reversed_entry = dict(first)
        reversed_entry["baseline"] = "A"
        reversed_entry["variant"] = "local_base"
        summary = experiment._fragility_summary(
            [first, reversed_entry]
        )
        self.assertEqual(len(summary["entries"]), 1)

    def test_fragility_summary_accepts_factorial_effect_labels(self):
        effect = {
            "design": "A vs local_base",
            "effect_name": "merge_main",
            "significant": True,
            "fragility": {
                "available": True,
                "verdict": "stable",
                "n": 5,
                "n_positive": 5,
                "n_negative": 0,
                "alpha": 0.05,
            },
        }
        summary = experiment._fragility_summary([effect])
        self.assertEqual(len(summary["entries"]), 1)
        self.assertEqual(
            summary["entries"][0]["baseline"],
            "A vs local_base",
        )
        self.assertEqual(
            summary["entries"][0]["variant"], "merge_main"
        )

    def test_fragility_does_not_claim_nonsignificant_holm_result(self):
        first = {
            "baseline": "b",
            "variant": "A",
            "factor": "mode",
            "p_value": 0.028008456010145746,
            "paired_differences": [0.2, 1, 1, 1],
            "paired_seeds": [1, 2, 3, 4],
            "significant": True,
        }
        second = {
            "baseline": "b",
            "variant": "B",
            "factor": "mode",
            "p_value": 0.041102615679308056,
            "paired_differences": [0.2, 2, 2, 2],
            "paired_seeds": [1, 2, 3, 4],
            "significant": True,
        }
        experiment._apply_multiple_comparison_correction(
            [first, second]
        )
        self.assertFalse(second["significant"])
        self.assertEqual(
            second["fragility"]["verdict"], "not_significant"
        )
        self.assertNotIn(
            "fragile",
            experiment._fragility_summary(
                [first, second]
            )["verdict_counts"],
        )

    def test_fragility_reports_p_value_of_minimal_deletion_set(self):
        report = experiment._paired_result_fragility(
            [2.0, 2.0, 2.0, 2.0, 2.0, 0.5],
            alpha=0.05,
        )
        self.assertEqual(
            report["min_seed_deletions_to_lose_significance"], 3
        )
        self.assertGreater(
            report["min_seed_deletions_p_value"], 0.05
        )
        self.assertNotEqual(
            report["min_seed_deletions_p_value"],
            report["max_leave_one_out_p_value"],
        )

    def test_fragility_marks_untested_deletion_levels(self):
        report = experiment._paired_result_fragility(
            [1.0] * 59 + [0.5],
            alpha=0.05,
        )
        self.assertFalse(report["deletions_tested_exhaustively"])
        self.assertEqual(report["max_deletions_tested"], 2)
        self.assertEqual(report["configured_max_deletions"], 3)

    def test_exact_sign_test_all_zero_is_resolution_limited(self):
        report = experiment._exact_sign_test([0.0, 0.0])
        self.assertEqual(report["n_nonzero"], 0)
        self.assertTrue(report["resolution_limited"])

    def test_factorial_duplicate_seeds_are_raw_and_deduplicated(self):
        def cell():
            return {
                "samples": {
                    "val_loss": [
                        {"seed": 42, "value": 1.0},
                        {"seed": 42, "value": 1.2},
                    ]
                }
            }
        _mapping, _complete, duplicates = (
            experiment._cell_seed_values(cell(), "val_loss")
        )
        self.assertEqual(duplicates, [42])
        effect = experiment._paired_linear_effect(
            {"a": cell(), "b": cell()},
            {"a": 1.0, "b": -1.0},
            "val_loss",
        )
        self.assertEqual(effect["duplicate_seeds"], [42])

    def test_manifest_strict_requires_verify_mode(self):
        args = argparse.Namespace(
            write_manifest="run",
            verify_manifest=None,
            verify_manifest_strict=True,
        )
        error = experiment.manifest_mode_error(args)
        self.assertIsNotNone(error)
        self.assertIn("--verify-manifest", error)

    def test_manifest_rejects_compare_report(self):
        args = argparse.Namespace(
            write_manifest="run",
            verify_manifest=None,
            verify_manifest_strict=False,
            compare_report="report.json",
        )
        self.assertIsNotNone(experiment.manifest_mode_error(args))

    def test_load_selected_model_rejects_bad_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            torch.save(
                {"model": {}, "step": None, "val_loss": 1.0},
                path,
            )
            self.assertIsNone(
                experiment.load_selected_model(
                    path, torch.device("cpu")
                )
            )

    def test_baseline_tuning_rejects_boolean_or_nonfinite_lr(self):
        base = {
            "comparison_family": "baseline_tuning",
            "mode": "dense",
            "metric": "val_loss",
            "winner": {"lr": True, "metric_mean": 1.0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning.json"
            path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaises(ValueError):
                experiment.load_baseline_tuning(path)
            base["winner"] = {
                "lr": float("nan"),
                "metric_mean": 1.0,
            }
            path.write_text(
                json.dumps(base), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                experiment.load_baseline_tuning(path)

    def test_baseline_tuning_rejects_out_of_float_range_lr(self):
        base = {
            "comparison_family": "baseline_tuning",
            "mode": "dense",
            "metric": "val_loss",
            # JSON 允许任意大整数；float(10**400) 会溢出。
            "winner": {"lr": 10**400, "metric_mean": 1.0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning.json"
            path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaises(ValueError):
                experiment.load_baseline_tuning(path)
            base["winner"] = {
                "lr": 1e-3,
                "metric_mean": 10**400,
            }
            path.write_text(
                json.dumps(base), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                experiment.load_baseline_tuning(path)

    def test_analysis_helpers_tolerate_out_of_float_range_numbers(self):
        huge = 10**400
        self.assertEqual(
            experiment._metric_samples(
                [{"seed": 1, "run": "r", "val_loss": huge}],
                "val_loss",
            ),
            [],
        )
        self.assertEqual(
            experiment._collapse_seed_samples(
                [{"seed": 1, "run": "r", "value": huge}]
            ),
            [],
        )
        mapping, _complete, duplicates = (
            experiment._cell_seed_values(
                {
                    "samples": {
                        "val_loss": [
                            {"seed": 1, "value": huge}
                        ]
                    }
                },
                "val_loss",
            )
        )
        self.assertEqual(mapping, {})
        self.assertEqual(duplicates, [])
        self.assertIsNone(experiment._mean([huge]))
        self.assertIsNone(experiment._median([huge, 1.0]))
        self.assertIsNone(
            experiment._sample_std([huge, 1.0, 2.0])
        )
        self.assertIsNone(
            experiment._window_per_token_loss({
                "tokens": huge,
                "loss_sum": 1.0,
            })
        )
        self.assertIsNone(
            experiment._window_accuracy({
                "tokens": huge,
                "correct": 1.0,
            })
        )
        self.assertEqual(
            experiment._clean_window_differences({
                1: [huge, 1.0],
                "bad-key": [1.0],
            }),
            {},
        )
        # 排序/比较不能因为无法转成 float 而崩溃；相同大整数
        # 仍必须判为相等。
        experiment._factor_sort_rank(huge)
        self.assertTrue(
            experiment._axis_value_equal(huge, huge)
        )
        self.assertFalse(
            experiment._axis_value_equal(huge, huge + 1)
        )
        experiment._axis_value_sort_key(huge)
        self.assertEqual(
            experiment._paired_t_test([huge, 1.0, 2.0]),
            {"test": "invalid_values"},
        )
        self.assertEqual(
            experiment._paired_sign_flip_test(
                [huge, 1.0, 2.0, 3.0]
            ),
            {"test": "invalid_values"},
        )
        self.assertEqual(
            experiment._welch_t_test(
                [huge, 1.0, 2.0],
                [1.0, 2.0, 3.0],
            ),
            {"test": "invalid_values"},
        )
        self.assertEqual(
            experiment._paired_result_fragility(
                [huge, 1.0, 2.0]
            )["reason"],
            "non_finite_differences",
        )
        # 效率均值/标准差与无共同种子时的聚合均值也不能因
        # 超大整数在 float() 处崩溃。
        self.assertIsNone(
            experiment._cell_cost_mean(
                {"costs": {"n_params": {"mean": huge}}},
                "n_params",
            )
        )
        self.assertIsNone(
            experiment._cell_cost_std(
                {"costs": {"n_params": {"std": huge}}},
                "n_params",
            )
        )
        factorial = experiment._paired_linear_effect(
            {
                "a": {
                    "samples": {
                        "val_loss": [{"seed": 1, "value": 1.0}]
                    },
                    "val_loss_mean": huge,
                },
                "b": {
                    "samples": {
                        "val_loss": [{"seed": 2, "value": 1.0}]
                    },
                    "val_loss_mean": huge,
                },
            },
            {"a": 1.0, "b": -1.0},
            "val_loss",
        )
        self.assertEqual(factorial["test"], "missing_metric")
        # n>1023 时 2**n 不能再转 float；精确符号检验仍需
        # 给出可判定的 p 值与最小可达 p。
        sign = experiment._exact_sign_test([1.0] * 2000)
        self.assertTrue(sign["available"])
        self.assertEqual(sign["p_value"], 0.0)
        self.assertEqual(sign["min_attainable_p"], 0.0)

    def test_design_power_fallback_does_not_keep_stale_adequacy(self):
        # 旧/手工 power 记录缺重算字段时，不能沿用按更小家族
        # 算出的 adequate=True 放行新增的交互对比。
        power = {
            "expected_seed_sd_relative": 0.01,
            "target_power": 0.8,
            "alpha": 0.05,
            "planned_comparisons": 3,
            "planned_seeds": 7,
            "adequate": True,
            # 这些是旧家族/旧对比类型下的派生字段，重算失败后
            # 不能再作为有效结论展示。
            "required_seeds": 99,
            "required_seeds_with_attrition": 120,
            "achieved_power_at_min_effect": 0.99,
            "minimum_detectable_effect_relative": 0.001,
            "contrast_power": {"pairwise": {"adequate": True}},
        }
        adjusted = experiment._design_power_for_family(
            power,
            6,
            ["pairwise", "main", "interaction", "simple"],
        )
        self.assertIsNone(adjusted["adequate"])
        self.assertFalse(adjusted["adequacy_recomputed"])
        self.assertEqual(
            adjusted["stale_derived_fields"]["required_seeds"],
            99,
        )
        for key in (
            "required_seeds",
            "required_seeds_with_attrition",
            "achieved_power_at_min_effect",
            "minimum_detectable_effect_relative",
            "contrast_power",
        ):
            self.assertNotIn(key, adjusted)
        design = {"power": adjusted}
        self.assertIsNotNone(
            experiment.design_power_gate_error(design)
        )
        # 家族规模本身损坏（inf/NaN）也不能在 int() 处抛异常。
        guarded = experiment._design_power_for_family(
            {
                "expected_seed_sd_relative": 0.01,
                "min_effect_relative": 0.01,
                "target_power": 0.8,
                "alpha": 0.05,
                "planned_seeds": 6,
                "planned_comparisons": 1,
                "planned_contrasts": ["pairwise"],
            },
            float("inf"),
            ["pairwise"],
            force=True,
        )
        self.assertIsInstance(guarded, dict)

    def test_design_gate_handles_uncomputable_required_seeds(self):
        design = {
            "power": {
                "adequate": False,
                "binding_contrast": "interaction",
                "binding_contrast_label": "交互",
                "planned_comparisons": 6,
                "planned_seeds": 4,
                "target_power": 0.8,
            }
        }
        gate = experiment.design_power_gate_error(design)
        self.assertIsNotNone(gate)
        self.assertNotIn("None 个", gate)
        self.assertIn("无法计算", gate)
        # 损坏的 equivalence 标量不能触发 AttributeError，也不能
        # 把缺失的目标功效/家族规模写成 None。
        malformed = {
            "power": {
                "adequate": False,
                "plan_equivalence": True,
                "equivalence_adequate": False,
                "equivalence": 1.0,
            }
        }
        malformed_gate = experiment.design_power_gate_error(
            malformed
        )
        self.assertIsNotNone(malformed_gate)
        self.assertNotIn("None", malformed_gate)

    def test_plan_treatment_keys_may_be_non_strings(self):
        # 手工/旧计划可能用非字符串键；不能因 str(key) 后查不到
        # 值就把不同处理单元静默合并。
        plan = [
            {"label": "a", "mode": "A", 1: 0, "seed": 1},
            {"label": "b", "mode": "A", 1: 1, "seed": 2},
            {"label": "c", "mode": "A", 1: 2, "seed": 3},
        ]
        self.assertEqual(
            experiment.planned_comparisons_for_plan(plan), 3
        )
        self.assertEqual(
            experiment.planned_contrast_types_for_plan(plan),
            ["pairwise"],
        )

    def test_cost_comparison_tolerates_zero_and_huge_variants(self):
        def cell(mean, std=1.0, n=3):
            return {
                "costs": {
                    "n_params": {
                        "mean": mean,
                        "std": std,
                        "n": n,
                    }
                }
            }

        # 变体成本为 0 时不能在做变异系数时除零。
        _, verdict = experiment._cost_comparison(
            cell(1.0), cell(0.0), "improvement", True
        )
        self.assertIsNotNone(verdict)
        # 成本倍率极大时 exp/平方不能抛 OverflowError；无法给出
        # 有限区间时应明确标为不可分辨。
        cost, _ = experiment._cost_comparison(
            cell(1e-100), cell(1e100), "improvement", True
        )
        detail = cost["uncertainty"]["n_params"]
        self.assertFalse(detail["resolvable"])
        self.assertIsNone(detail["ratio_ci95"])

    def test_contrasts_tolerate_out_of_range_means_and_counts(self):
        huge = 10**400

        def cell(mode, mean, std=1.0):
            return {
                "label": mode,
                "keys": {"mode": mode},
                "samples": {"val_loss": []},
                "val_loss_mean": mean,
                "val_loss_std": std,
                "n_runs": 3,
                "effective_seed_count": 3,
                "costs": {},
            }

        contrasts = experiment.factor_contrasts(
            [cell("A", 1.0), cell("B", huge)],
            ["mode"],
            "val_loss",
        )
        self.assertEqual(len(contrasts), 1)
        self.assertIsNone(contrasts[0]["delta"])
        # 巨大 std 同样不能直接平方；无效值按缺失处理。
        contrasts = experiment.factor_contrasts(
            [cell("A", 1.0, huge), cell("B", 2.0, huge)],
            ["mode"],
            "val_loss",
        )
        self.assertIsNone(contrasts[0]["delta_over_noise"])
        # 损坏的运行数不能阻止对比生成或被当成负数。
        broken_a = dict(cell("A", 1.0), n_runs=[])
        broken_b = dict(cell("B", 2.0), n_runs={})
        contrasts = experiment.factor_contrasts(
            [broken_a, broken_b], ["mode"], "val_loss"
        )
        self.assertEqual(
            contrasts[0]["n_expected_baseline"], 0
        )
        baseline = experiment.compare_cells_to_baseline(
            [
                dict(cell("A", 1.0), effective_seed_count=[]),
                cell("B", 2.0),
            ],
            "A",
            "val_loss",
        )
        self.assertEqual(len(baseline), 1)
        self.assertEqual(baseline[0]["delta"], 1.0)

    def test_factor_keys_preserve_large_integer_precision(self):
        left = 2 ** 53
        right = left + 1
        self.assertNotEqual(
            experiment._stable_factor_key(left),
            experiment._stable_factor_key(right),
        )
        # int/float 的同一数值仍必须合并为同一水平。
        self.assertEqual(
            experiment._stable_factor_key(7),
            experiment._stable_factor_key(7.0),
        )
        self.assertLess(
            experiment._factor_sort_rank(left),
            experiment._factor_sort_rank(right),
        )

    def test_factor_keys_accept_unhashable_values(self):
        # 手工编辑/损坏的计划可能把 set 放进处理轴；稳定键必须
        # 仍可哈希，否则 planned_comparisons_for_plan 等会把键
        # 放进 set/dict 时崩溃。
        key = experiment._stable_factor_key({1, 2})
        self.assertEqual(hash(key), hash(key))
        self.assertEqual(
            key,
            experiment._stable_factor_key(frozenset({2, 1})),
        )
        plan = [
            {
                "label": "a",
                "mode": "A",
                "balance_coef": {0.0},
                "seed": 1,
            },
            {
                "label": "b",
                "mode": "A",
                "balance_coef": {0.0},
                "seed": 2,
            },
        ]
        self.assertEqual(
            experiment.planned_comparisons_for_plan(plan), 1
        )
        self.assertEqual(
            experiment.planned_contrast_types_for_plan(plan), []
        )
        report = experiment.run_order_balance_report(
            experiment.order_sweep_plan(plan, "randomized", 3)
        )
        self.assertTrue(report["available"])
        self.assertEqual(
            experiment._stable_factor_key({"a": 1, 2: 3}),
            experiment._stable_factor_key({"a": 1, 2: 3}),
        )

    def test_resume_checks_tolerate_corrupt_fingerprint_records(self):
        # 非字符串数据指纹不能被当成「未记录」而静默放行续跑。
        self.assertIn(
            "格式损坏",
            experiment.resume_data_change_error(123, "abc"),
        )
        self.assertIsNone(
            experiment.resume_data_change_error("", "abc")
        )
        # split 记录的键可能是损坏的混合类型；排序/拼接不能
        # 抛 TypeError，损坏的哈希值也不能静默放行。
        mixed = experiment.resume_split_change_error(
            {1: "a", "val": "b"}, {}
        )
        self.assertIn("1", mixed)
        self.assertIn("val", mixed)
        corrupt = experiment.resume_split_change_error(
            {"train": 1}, {"train": "a"}
        )
        self.assertIn("损坏", corrupt)
        self.assertIsNone(
            experiment.resume_split_change_error({}, {})
        )
        # 时序隔离记录的混合键同样必须可比较，且键顺序不影响
        # 判定。
        self.assertIsNone(
            experiment.resume_split_isolation_change_error(
                {1: "a", "b": 2},
                {"b": 2, 1: "a"},
            )
        )
        self.assertIsNotNone(
            experiment.resume_split_isolation_change_error(
                {1: "a"}, {1: "b"}
            )
        )

    def test_statistics_tolerate_large_finite_floats(self):
        huge = 1.0e308
        self.assertEqual(experiment._mean([huge, huge]), huge)
        self.assertEqual(experiment._mean([huge, -huge]), 0.0)
        self.assertEqual(experiment._median([huge, huge]), huge)
        self.assertEqual(experiment._median([huge, -huge]), 0.0)
        self.assertEqual(
            experiment._sample_std([huge, huge]), 0.0
        )
        # 两个相反的大数，标准差仍在 float 范围内：直接对
        # 1e308 ** 2 会抛 OverflowError。
        self.assertIsNotNone(
            experiment._sample_std([huge, -huge])
        )
        self.assertEqual(
            experiment._welch_t_test(
                [huge, huge, huge],
                [huge, huge, huge],
            )["note"],
            "zero_variance",
        )
        identical_large = experiment._welch_t_test(
            [0.0, 1e200, 2e200],
            [0.0, 1e200, 2e200],
        )
        self.assertEqual(identical_large["test"], "welch_t")
        self.assertEqual(identical_large["t"], 0.0)
        self.assertEqual(identical_large["p_value"], 1.0)
        self.assertEqual(identical_large["effect"], 0.0)
        self.assertIsNotNone(identical_large["se"])
        json.dumps(identical_large, allow_nan=False)
        # 符号翻转统计量对尺度不变：同一组大到 1e308 的差值
        # 不能因求和溢出而被误判成退化的 p=1。
        sign = experiment._paired_sign_flip_test(
            [huge, huge, huge]
        )
        self.assertEqual(sign["method"], "exact")
        self.assertEqual(sign["p_value"], 0.25)
        # 两向方差分解同样必须先在归一化尺度上计算，不能对
        # 1e200 的离差直接平方。
        components = experiment._window_variance_components(
            {1: [1e200, 0.0], 2: [1e200, 0.0]}
        )
        self.assertTrue(components["available"])
        self.assertEqual(
            components["between_seed_sd"], 0.0
        )
        self.assertIsNotNone(components["std_error_of_mean"])
        # 截断前的原始方差分量可为负，缩放实现不能把它丢掉。
        components = experiment._window_variance_components(
            {1: [0.0, 10.0], 2: [5.0, 5.0]}
        )
        raw = components[
            "variance_components_single_observation"
        ]
        self.assertLess(raw["raw_seed"], 0.0)
        self.assertTrue(raw["truncated_at_zero"])
        # 计时摘要换算单位后不能把 inf 写进报告。
        timing = experiment._timing_summary([huge, huge])
        self.assertIsNone(timing["mean"])
        self.assertEqual(timing["std"], 0.0)

    def test_paired_stats_keep_tiny_nonzero_scale(self):
        # 极小但非零的差值在平方时下溢为 0；标准差不能被报告成
        # 0，配对 t 也不能因此变成「零方差/精确分离」。
        tiny = [1.0e-200, 2.0e-200, 3.0e-200]
        self.assertTrue(
            math.isclose(
                experiment._sample_std(tiny),
                1.0e-200,
                rel_tol=1e-12,
            )
        )
        test = experiment._paired_t_test(tiny)
        self.assertEqual(test["test"], "paired_t")
        self.assertTrue(math.isfinite(test["t"]))
        self.assertAlmostEqual(
            test["t"],
            experiment._paired_t_test([1.0, 2.0, 3.0])["t"],
        )
        # 符号翻转统计量本身对公共尺度不变；固定绝对容差会让
        # 1e-13 量级的差值被误判成退化（p=1）。
        for scale in (1.0e-13, 1.0e-200, 1.0e-300):
            sign = experiment._paired_sign_flip_test(
                [scale, 2.0 * scale, 3.0 * scale]
            )
            self.assertEqual(sign["method"], "exact")
            self.assertEqual(sign["p_value"], 0.25)
            json.dumps(sign, allow_nan=False)

    def test_counterbalanced_empty_blocks_do_not_divide_by_zero(self):
        self.assertEqual(
            experiment._counterbalanced_order({1: []}, 0), []
        )
        self.assertEqual(
            experiment._counterbalanced_order({1: [], 2: []}, 0),
            [],
        )

    def test_sparkline_handles_out_of_range_and_nonfinite_values(self):
        huge = 10 ** 400
        self.assertEqual(experiment._sparkline([huge]), "")
        self.assertEqual(
            experiment._sparkline([1.0, float("nan")]), ""
        )
        self.assertEqual(
            experiment._sparkline([1.0, float("inf")]), ""
        )
        self.assertEqual(len(experiment._sparkline([1, 2, 3])), 3)
        # 有限但符号相反的极值：high-low 会溢出成 inf，走势条
        # 仍必须能按相对大小分块，而不是 int(NaN) 崩溃。
        spark = experiment._sparkline([-1.0e308, 0.0, 1.0e308])
        self.assertEqual(len(spark), 3)
        self.assertNotEqual(spark, "")

    def test_bad_case_comparison_skips_out_of_range_shares(self):
        huge = 10 ** 400
        category = sorted(experiment._BAD_CASE_CATEGORIES)[0]
        report = experiment.bad_case_comparison_report([
            {
                "bad_case_available": True,
                "bad_case_tokens_evaluated": 10,
                "bad_case_wrong_tokens": 2,
                "bad_case_error_category_counts": {
                    category: 2,
                },
                "bad_case_error_category_shares": {
                    category: huge,
                },
            },
            {
                "bad_case_available": True,
                "bad_case_tokens_evaluated": 10,
                "bad_case_wrong_tokens": 2,
                "bad_case_error_category_counts": {
                    category: 2,
                },
                "bad_case_error_category_shares": {
                    category: 0.2,
                },
            },
        ])
        self.assertEqual(report["n_runs_with_bad_cases"], 2)
        self.assertEqual(
            report["error_category_counts"][category], 4
        )
        self.assertNotIn(category, report["category_share_ranges"])

    def test_imbalance_advisory_tolerates_out_of_range_rates(self):
        huge = 10 ** 400
        report = experiment.imbalance_metric_advisory(
            [
                {
                    "run": "huge_majority",
                    "val_majority_accuracy": huge,
                },
                {
                    "run": "huge_chance",
                    "val_pr_auc_macro_chance": huge,
                },
            ],
            "val_loss",
        )
        self.assertIsNone(report["max_majority_accuracy"])
        self.assertIsNone(report["most_imbalanced_run"])
        self.assertIsNone(report["pr_auc_chance_mean"])

    def test_imbalance_advisory_omits_overflowing_pr_auc_lift(self):
        report = experiment.imbalance_metric_advisory(
            [
                {
                    "run": "huge_lift",
                    "val_majority_accuracy": 0.9995,
                    "val_pr_auc_macro": 1.0e308,
                    "val_pr_auc_macro_chance": -1.0e308,
                }
            ],
            "val_pr_auc_macro",
        )
        self.assertIsNone(report["pr_auc_over_chance"])
        self.assertIn("无法表示", report["advisories"][-1])
        json.dumps(report, allow_nan=False)

    def test_baseline_ngram_delta_overflow_is_omitted(self):
        rows = [
            {
                "run": "baseline",
                "mode": "dense",
                "seed": 1,
                "val_loss": 1.0e308,
                "protocol": {
                    "data_sha256": "d",
                    "eval_batches": 1,
                },
            },
            {
                "run": "ngram",
                "mode": "ngram",
                "seed": 2,
                "val_loss": -1.0e308,
                "protocol": {
                    "data_sha256": "d",
                    "eval_batches": 1,
                },
            },
        ]
        report = experiment.baseline_adequacy(
            rows, "dense", "val_loss"
        )
        comparison = report["ngram_comparison"]
        self.assertIsNone(comparison["delta"])
        self.assertIsNone(comparison["seeds_to_detect"])
        json.dumps(report, allow_nan=False)

    def test_baseline_adequacy_tolerates_out_of_range_costs(self):
        huge = 10 ** 400
        rows = [
            {
                "run": "baseline",
                "mode": "dense",
                "seed": 1,
                "val_loss": 1.0,
                "n_params": 100.0,
                "flops_per_forward": 100.0,
            },
            {
                "run": "huge_metric",
                "mode": "B",
                "seed": 2,
                "val_loss": huge,
                "n_params": huge,
                "flops_per_forward": huge,
            },
            {
                "run": "huge_cost",
                "mode": "B",
                "seed": 3,
                "val_loss": 1.1,
                "n_params": huge,
                "flops_per_forward": huge,
            },
        ]
        report = experiment.baseline_adequacy(
            rows, "dense", "val_loss"
        )
        self.assertEqual(report["capacity_confounds"], [])
        self.assertIsNone(report["baseline_capacity"])

    def test_curve_analysis_treats_out_of_range_numbers_as_nonfinite(self):
        huge = 10 ** 400
        records = [
            {
                "step": 1,
                "train_loss": huge,
                "val_loss": huge,
                "grad_norm": huge,
                "lr": huge,
            },
            {
                "step": 2,
                "train_loss": 1.0,
                "val_loss": 1.0,
                "grad_norm": 1.0,
                "lr": 1e-3,
            },
        ]
        analysis = experiment.analyze_training_curve(records)
        self.assertEqual(analysis["n_nonfinite_events"], 1)
        self.assertEqual(analysis["n_eval_points"], 1)
        self.assertEqual(analysis["n_train_steps"], 1)
        self.assertEqual(analysis["peak_lr"], 1e-3)
        self.assertIn(1, analysis["nonfinite_steps"])

    def test_timing_summary_handles_out_of_range_integers(self):
        huge_integer = 10 ** 400
        timing = experiment._timing_summary([huge_integer])
        self.assertEqual(timing["n"], 1)
        self.assertIsNone(timing["mean"])
        self.assertIsNone(timing["median"])
        # 混入一个不可用值不能只丢掉它后继续算，否则会把单次
        # 观测伪装成完整分布。
        timing = experiment._timing_summary([1.0, huge_integer])
        self.assertEqual(timing["n"], 2)
        self.assertIsNone(timing["mean"])
        self.assertIsNone(timing["std"])

    def test_timing_summary_cv_does_not_overflow(self):
        # 有限极值相互抵消后均值可以合法地接近 0，而标准差仍
        # 极大；std/mean 越过 float 上限时 CV 必须是 None。
        timing = experiment._timing_summary(
            [1.0e-308, 1.0e308, -1.0e308]
        )
        self.assertIsNone(timing["coefficient_of_variation"])
        json.dumps(timing, allow_nan=False)

    def test_analysis_and_resume_tolerate_out_of_range_numbers(self):
        huge = 10 ** 400

        # 参考基线来自 config.json：超范围整数按不可用处理。
        self.assertIsNone(
            experiment._trivial_reference_value(
                "val_loss", {"unigram_loss": huge}
            )
        )

        # resume 读取历史指标时不能因单个超范围整数崩溃。
        self.assertEqual(
            experiment.early_stop_state_from_records(
                [{"step": 1, "val_loss": huge}],
                0.0,
            ),
            (None, None, 0),
        )

        # 预注册等价界损坏时标记为 invalid，而不是抛异常。
        margin = experiment.equivalence_margin_preregistration(
            [
                {
                    "run": "bad_margin",
                    "hypothesis_equivalence_margin": huge,
                }
            ],
            0.01,
        )
        self.assertEqual(margin["invalid_runs"], ["bad_margin"])
        self.assertEqual(margin["declared_margins"], {})

        # 最小效应预注册值损坏时必须给出可读错误。
        args = argparse.Namespace(
            hypothesis_direction="decrease",
            hypothesis_min_effect=huge,
            primary_metric="val_loss",
            hypothesis_baseline=None,
            compare=None,
            final_test_comparison=None,
            sanity_only=False,
            statistical_baseline=False,
            hypothesis="验证损失降低 2%",
        )
        error = experiment.hypothesis_preregistration_error(args)
        self.assertIn("有限正数", error or "")

        # 对比阶段不能因坏的最小效应值中断；该运行仍可登记，
        # 但没有可用的绝对阈值。
        alignment = experiment.hypothesis_alignment(
            [
                {
                    "run": "bad_effect",
                    "hypothesis_direction": "decrease",
                    "primary_metric": "val_loss",
                    "hypothesis_min_effect": huge,
                }
            ],
            [{"label": "mode=B, seed=1", "runs": ["bad_effect"]}],
            [
                {
                    "variant": "mode=B, seed=1",
                    "baseline": "mode=A, seed=1",
                    "factor": "mode",
                    "delta": -1.0,
                    "mean_baseline": 1.0,
                    "p_value_holm": 0.01,
                    "significant": True,
                    "confounded": False,
                }
            ],
            "val_loss",
        )
        self.assertEqual(len(alignment["checks"]), 1)
        self.assertIsNone(alignment["checks"][0]["min_effect"])

    def test_print_comparison_tolerates_out_of_range_numbers(self):
        huge = 10 ** 400
        self.assertEqual(experiment._format_finite(huge), "-")
        self.assertEqual(
            experiment._format_finite(float("nan")), "-"
        )
        self.assertEqual(
            experiment._format_finite(1.25), "1.2500"
        )
        # 乘上展示缩放后溢出也必须按缺失值处理。
        self.assertEqual(
            experiment._format_finite(1e308, ".1f", 100.0), "-"
        )
        rows = [
            {
                "run": "huge",
                "mode": "B",
                "n_params": huge,
                "val_loss": huge,
                "val_macro_nll": huge,
                "val_bits_per_byte": huge,
                "val_token_accuracy": huge,
                "val_macro_f1": huge,
                "latency_ms_per_forward": huge,
                "training_ms_per_step": huge,
                "flops_per_forward": huge,
                "peak_memory_mb": huge,
                "routing_available": True,
                "routing_jaccard_branch": huge,
                "routing_slot_reuse_branch": huge,
                "routing_branch_self_reuse": huge,
                "routing_r1_load_entropy": huge,
                "routing_r2_load_entropy": huge,
                "routing_r2_dead_experts": huge,
                "routing_r2_router_entropy": huge,
                "routing_branch_cosine": huge,
            }
        ]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            experiment.print_comparison(rows)
        text = output.getvalue()
        self.assertIn("huge", text)
        self.assertIn("val_loss", text)
        # 正常有限值仍然正常格式化。
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            experiment.print_comparison([
                {
                    "run": "ok",
                    "mode": "B",
                    "val_loss": 1.25,
                    "flops_per_forward": 2e9,
                }
            ])
        self.assertIn("1.2500", output.getvalue())
        self.assertIn("2.000", output.getvalue())

    def test_paired_stats_keep_large_finite_effects(self):
        huge = 1.0e308
        test = experiment._paired_t_test([huge, huge, huge])
        self.assertEqual(test["note"], "noise_free_separation")
        self.assertTrue(math.isfinite(test["effect"]))
        self.assertEqual(test["effect"], huge)
        fragility = experiment._paired_result_fragility(
            [huge, huge, huge]
        )
        self.assertEqual(fragility["mean_difference"], huge)
        self.assertTrue(
            math.isfinite(fragility["mean_difference"])
        )
        stability = experiment._leave_one_out_effect_stability(
            [huge, huge, huge]
        )
        self.assertEqual(
            stability["leave_one_out_effect_min"], huge
        )
        self.assertEqual(
            stability["leave_one_out_effect_max"], huge
        )
        self.assertTrue(
            stability["leave_one_out_sign_stable"]
        )

    def test_welch_seed_plan_uses_two_sample_power(self):
        samples_a = [
            {"seed": 1, "value": 10.0},
            {"seed": 2, "value": 10.2},
            {"seed": 3, "value": 9.8},
        ]
        samples_b = [
            {"seed": 4, "value": 10.2},
            {"seed": 5, "value": 10.4},
            {"seed": 6, "value": 10.0},
        ]
        test = experiment._contrast_test(samples_a, samples_b)
        self.assertEqual(test["test"], "welch_t")
        contrast = {
            "test": test["test"],
            "se": test["se"],
            "df": test["df"],
            "delta": test["effect"],
            "mean_baseline": 10.0,
            "n_baseline": 3,
            "n_variant": 3,
        }
        experiment._attach_power_equivalence(
            contrast, test["effect"], test, 0.01
        )
        self.assertEqual(
            contrast["seeds_to_detect_design"], "welch"
        )
        # 每臂 n 个、se=sd·sqrt(2/n)、df=2(n-1)；n=17 才达到
        # 80% 功效，配对公式给出的 10 会系统性低估。
        self.assertEqual(contrast["seeds_to_detect"], 17)

    def test_zero_effect_power_is_exactly_alpha(self):
        self.assertEqual(
            experiment._t_power(0.0, 1.0, 2.0, 1e-5),
            1e-5,
        )
        # 极小 alpha + df=2 时固定 u 网格会漏掉尖峰；换到
        # w=scale·u 后应接近非中心 t 的参考值（scipy nct）。
        self.assertAlmostEqual(
            experiment._t_power(1.0, 1.0, 2.0, 1e-5),
            1.9999800001667738e-05,
            delta=1e-9,
        )
        self.assertGreater(
            experiment._t_power(2.0, 1.0, 2.0, 1e-5),
            experiment._t_power(1.0, 1.0, 2.0, 1e-5),
        )

    def test_t_power_matches_noncentral_t_reference_near_df_one(self):
        # df<3 时 chi 密度在 0 附近有弱奇异项 u^(df-1)；固定 u
        # 网格的 Simpson 会少积概率质量。以下参考值由高精度
        # 非中心 t 积分得到，修复前 df=1.1 的误差约 4.4%。
        for effect, df, expected in (
            (0.05, 1.01, 0.050062837107652636),
            (0.05, 1.1, 0.050068256192849915),
            (0.05, 1.5, 0.050091063051014614),
            (0.2, 2.0, 0.051850694985949765),
            (2.0, 2.5, 0.256924795068195),
        ):
            self.assertAlmostEqual(
                experiment._t_power(effect, 1.0, df, 0.05),
                expected,
                delta=1e-8,
            )

    def test_t_power_handles_small_df_without_overflow(self):
        u_values, weights = experiment._t_power_nodes(0.01)
        self.assertEqual(len(u_values), len(weights))
        self.assertTrue(
            all(math.isfinite(weight) for weight in weights)
        )
        self.assertAlmostEqual(sum(weights), 1.0, delta=5e-5)
        for effect, expected in (
            (1.0e-6, 0.05),
            (1.0, 0.05021351447305985),
        ):
            power = experiment._t_power(effect, 1.0, 0.01, 0.05)
            self.assertIsNotNone(power)
            self.assertAlmostEqual(power, expected, delta=5e-5)

    def test_t_power_resolves_tiny_alpha_for_small_df(self):
        # alpha 极小时 critical/√df 可达 1e19 以上，固定 u 网格
        # 会漏掉 u<1/scale 的全部概率质量并返回 0。
        for df, alpha, expected in (
            (0.5, 1.0e-10, 1.222065580725563e-10),
            (0.5, 1.0e-20, 1.2220655807292234e-20),
            (1.0, 1.0e-10, 1.4621550516047826e-10),
            (1.0, 1.0e-20, 1.4621550516047845e-20),
            (2.9, 1.0e-20, 2.5549713688518368e-20),
        ):
            power = experiment._t_power(1.0, 1.0, df, alpha)
            self.assertIsNotNone(power)
            self.assertGreater(power, alpha)
            self.assertAlmostEqual(
                power,
                expected,
                delta=expected * 1.0e-6,
            )

    def test_t_p_value_handles_infinite_and_invalid_df(self):
        # df→+inf 时 t 分布退化为标准正态；原实现得到
        # inf/inf=NaN。NaN/负自由度没有定义，必须返回 None
        # 而不是让调用方把 NaN 当成数值继续传播。
        for t in (0.0, 1.0, 1.959963984540054, 10.0):
            self.assertAlmostEqual(
                experiment._two_sided_t_p_value(t, float("inf")),
                math.erfc(t / math.sqrt(2.0)),
                delta=1e-12,
            )
        self.assertIsNone(
            experiment._two_sided_t_p_value(1.0, float("nan"))
        )
        self.assertIsNone(
            experiment._two_sided_t_p_value(1.0, float("-inf"))
        )
        self.assertIsNone(
            experiment._two_sided_t_p_value(float("inf"), float("nan"))
        )

    def test_t_p_value_stays_stable_for_large_finite_df(self):
        # 大而有限的自由度会让 df/(df+t²) 舍入成 1，
        # 原实现因此把 p 值退化成 1.0 或接近 0；应平滑趋近
        # 标准正态尾概率。
        for t in (1.0, 1.96, 3.0, 10.0):
            normal = math.erfc(t / math.sqrt(2.0))
            previous = None
            for df in (1.0e6, 1.0e9, 1.0e16, 1.0e100):
                p_value = experiment._two_sided_t_p_value(t, df)
                self.assertIsNotNone(p_value)
                self.assertAlmostEqual(
                    p_value,
                    normal,
                    delta=1e-6,
                )
                self.assertGreaterEqual(p_value, normal)
                if previous is not None:
                    self.assertLessEqual(p_value, previous)
                previous = p_value

    def test_tost_margin_for_power_preserves_tiny_scale(self):
        normal = experiment._tost_margin_for_power(1.0, 10.0)
        self.assertIsNotNone(normal)
        for scale in (1.0e-300, 1.0e-200, 1.0e200):
            tiny = experiment._tost_margin_for_power(
                scale, 10.0
            )
            self.assertIsNotNone(tiny)
            self.assertAlmostEqual(
                tiny / scale, normal, places=12
            )
        self.assertIsNone(
            experiment._tost_margin_for_power(1.0e308, 2.0)
        )

    def test_welch_test_preserves_tiny_scale(self):
        base_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        base_b = [1.1, 2.1, 3.1, 4.1, 5.1]
        normal = experiment._welch_t_test(base_a, base_b)
        tiny_scale = 1.0e-300
        tiny = experiment._welch_t_test(
            [value * tiny_scale for value in base_a],
            [value * tiny_scale for value in base_b],
        )
        self.assertEqual(tiny["test"], "welch_t")
        self.assertAlmostEqual(tiny["t"], normal["t"], places=12)
        self.assertAlmostEqual(
            tiny["p_value"], normal["p_value"], places=12
        )
        self.assertAlmostEqual(tiny["df"], normal["df"], places=12)
        self.assertAlmostEqual(
            tiny["effect"] / tiny_scale,
            normal["effect"],
            places=12,
        )
        self.assertAlmostEqual(
            tiny["se"] / tiny_scale,
            normal["se"],
            places=12,
        )

    def test_welch_test_preserves_large_scale(self):
        base_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        base_b = [1.1, 2.1, 3.1, 4.1, 5.1]
        normal = experiment._welch_t_test(base_a, base_b)
        for scale in (1.0e154, 1.0e200, 1.0e300):
            large = experiment._welch_t_test(
                [value * scale for value in base_a],
                [value * scale for value in base_b],
            )
            self.assertEqual(large["test"], "welch_t")
            self.assertAlmostEqual(
                large["t"], normal["t"], places=12
            )
            self.assertAlmostEqual(
                large["p_value"], normal["p_value"], places=12
            )
            self.assertAlmostEqual(
                large["df"], normal["df"], places=12
            )
            self.assertAlmostEqual(
                large["effect"] / scale,
                normal["effect"],
                places=12,
            )
            self.assertAlmostEqual(
                large["se"] / scale,
                normal["se"],
                places=12,
            )

    def test_welch_test_omits_unrepresentable_effect(self):
        result = experiment._welch_t_test(
            [-1.0e308, -1.0e308, -1.0e308],
            [1.0e308, 1.0e308, 1.0e308],
        )
        self.assertEqual(result["test"], "welch_t")
        self.assertEqual(
            result["note"], "noise_free_separation"
        )
        self.assertIsNone(result["effect"])
        self.assertEqual(result["se"], 0.0)
        json.dumps(result, allow_nan=False)

    def test_bad_case_ignores_invalid_confusion_pairs(self):
        report = experiment.bad_case_comparison_report([{
            "run": "damaged",
            "mode": "A",
            "seed": 1,
            "bad_case_available": True,
            "bad_case_tokens_evaluated": 10,
            "bad_case_wrong_tokens": 4,
            "bad_case_top_confusions": [
                ["bad", 66, 2],
                [65, 66, -3],
                [65, 66, 0],
                [70, 71, 4],
            ],
        }])
        run = report["runs"][0]
        self.assertEqual(run["top_confusion"], [70, 71, 4])
        self.assertEqual(
            report["persistent_confusions"],
            [{
                "target": 70,
                "predicted": 71,
                "runs_present": 1,
                "top_count_sum": 4,
            }],
        )

    def test_t_power_uses_normal_limit_for_large_df(self):
        expected = 0.17007504575308752
        for df in (1.0e6, 1.0e9, 1.0e16, float("inf")):
            self.assertAlmostEqual(
                experiment._t_power(1.0, 1.0, df),
                expected,
                delta=1e-12,
            )
        mde = experiment._minimum_detectable_effect(
            1.0, float("inf")
        )
        self.assertIsNotNone(mde)
        self.assertAlmostEqual(
            experiment._t_power(mde, 1.0, float("inf")),
            experiment._TARGET_POWER,
            delta=1e-9,
        )
        self.assertIsNone(
            experiment._t_power(1.0, 1.0, float("nan"))
        )
        self.assertIsNone(
            experiment._t_power(1.0, 1.0, float("-inf"))
        )

    def test_t_power_keeps_extreme_normal_tails(self):
        # 1-CDF 在大 z 处会抵消成 0，导致极小 alpha 下的功效
        # 被严重低估；负半轴必须直接用 erfc 计算。
        self.assertAlmostEqual(
            experiment._normal_cdf(-10.0),
            7.619853024160593e-24,
            delta=1e-36,
        )
        expected = 1.692381826523976e-30
        self.assertAlmostEqual(
            experiment._t_power(
                0.1, 1.0, 1000.0, 1.0e-30
            ),
            expected,
            delta=expected * 1e-9,
        )
        expected = 1.7443821172130487e-30
        self.assertAlmostEqual(
            experiment._t_power(
                0.1, 1.0, float("inf"), 1.0e-30
            ),
            expected,
            delta=expected * 1e-9,
        )
        expected = 1.2244731336713435e-28
        self.assertAlmostEqual(
            experiment._t_power(
                0.5, 1.0, 1000.0, 1.0e-30
            ),
            expected,
            delta=expected * 1e-9,
        )

    def test_minimum_detectable_effect_handles_tiny_alpha_and_large_se(
        self,
    ):
        # 家族校正后 alpha 可能很小，MDE 的非中心参数会超过
        # 固定的 100 上界；必须在临界值之上继续扩展。
        tiny_alpha = experiment._minimum_detectable_effect(
            1.0e-6, 1.0, alpha=1.0e-6
        )
        self.assertIsNotNone(tiny_alpha)
        self.assertAlmostEqual(
            experiment._t_power(
                tiny_alpha, 1.0e-6, 1.0, 1.0e-6
            ),
            experiment._TARGET_POWER,
            delta=1e-6,
        )
        # se 很大时 effect=mid*se 会先溢出，但 MDE 本身仍可表示；
        # 功效按 δ 的尺度计算即可避免假 None。
        large_se = experiment._minimum_detectable_effect(
            1.0e307, 4.0
        )
        self.assertIsNotNone(large_se)
        self.assertAlmostEqual(
            experiment._t_power(large_se, 1.0e307, 4.0),
            experiment._TARGET_POWER,
            delta=1e-9,
        )
        # 真正的 MDE 超过 float 上限时才返回 None。
        self.assertIsNone(
            experiment._minimum_detectable_effect(1.0e308, 4.0)
        )

    def test_sign_flip_required_seeds_is_global_minimum(self):
        # 6 是 2/2^n <= 0.05 的最小 n；当前样本量更大时不能
        # 把「所需种子数」误报成当前 n。
        for n in (6, 7, 12):
            result = experiment._paired_sign_flip_test(
                [float(index + 1) for index in range(n)]
            )
            self.assertEqual(
                result["required_seeds_for_alpha"], 6
            )

    def test_curve_diagnostics_sanitizes_nonfinite_metrics(self):
        report = experiment.curve_diagnostics_report([{
            "run": "damaged",
            "mode": "A",
            "seed": 1,
            "curve_available": True,
            "curve_status": "overfitting",
            "curve_best_val_loss": float("inf"),
            "curve_recent_train_drop": float("nan"),
            "curve_n_eval_points": 4,
            "curve_gap_widening_warning": True,
        }])
        run = report["runs"][0]
        self.assertIsNone(run["best_val_loss"])
        self.assertIsNone(run["recent_train_drop"])
        self.assertEqual(run["n_eval_points"], 4)
        self.assertTrue(run["gap_widening_warning"])
        json.dumps(report, allow_nan=False)

    def test_t_critical_supports_normal_limit(self):
        normal_critical = 1.959963984540054
        self.assertAlmostEqual(
            experiment._t_critical(float("inf")),
            normal_critical,
            delta=1e-12,
        )
        self.assertAlmostEqual(
            experiment._t_critical(1.0e16),
            normal_critical,
            delta=1e-12,
        )
        self.assertIsNone(experiment._t_critical(float("nan")))

    def test_t_critical_handles_small_df_and_alpha(self):
        # df=1 是 Cauchy 分布：双侧临界值有解析解
        # t = 1/tan(pi*alpha/2)=636619.77...；旧实现在
        # doubling 到 1048576 后、算 p 值前就撞上 1e6 上限，
        # 把合法的有限临界值误报为 None。
        alpha = 1.0e-6
        expected = 1.0 / math.tan(math.pi * alpha / 2.0)
        critical = experiment._t_critical(1.0, alpha)
        self.assertIsNotNone(critical)
        self.assertAlmostEqual(
            critical / expected, 1.0, places=12
        )
        # df=2 的双侧 p=1-t/sqrt(t^2+2)，反解得到
        # t=(1-alpha)*sqrt(2/(alpha*(2-alpha)))≈1e6。
        alpha = 1.0e-12
        expected = (1.0 - alpha) * math.sqrt(
            2.0 / (alpha * (2.0 - alpha))
        )
        critical = experiment._t_critical(2.0, alpha)
        self.assertIsNotNone(critical)
        self.assertAlmostEqual(
            critical / expected, 1.0, places=12
        )
        # df=0.5、alpha=1e-100 的临界值约 4e199，远超旧上限；
        # 用双侧 p 值自洽验证二分结果。
        critical = experiment._t_critical(0.5, 1.0e-100)
        self.assertIsNotNone(critical)
        self.assertGreater(critical, 1.0e40)
        p_value = experiment._two_sided_t_p_value(
            critical, 0.5
        )
        self.assertAlmostEqual(
            p_value / 1.0e-100, 1.0, places=10
        )

    def test_seed_plan_and_aggregation_collapse_pseudoreplication(self):
        plan = experiment._build_seed_plan(
            [1, 1, 1, 1, 1, 1], False
        )
        self.assertEqual(plan["n_seeds"], 1)
        self.assertEqual(plan["seeds"], [1])
        self.assertEqual(plan["duplicate_seeds"], [1] * 5)
        self.assertFalse(plan["robust_claim_possible"])

        rows = [
            {"run": "a", "mode": "A", "seed": 1, "val_loss": 1.0},
            {"run": "b", "mode": "A", "seed": 1, "val_loss": 1.1},
            {"run": "c", "mode": "A", "seed": "1", "val_loss": 1.2},
            {"run": "d", "mode": "A", "seed": "1", "val_loss": 1.3},
        ]
        aggregate = experiment.aggregate_runs(rows, ["mode"])[0]
        self.assertEqual(
            aggregate["duplicate_seeds"], [1, "1"]
        )

    def test_contrast_test_tolerates_unhashable_seed_records(self):
        samples_a = [
            {"seed": [1], "value": 1.0},
            {"seed": [2], "value": 1.1},
            {"seed": [3], "value": 0.9},
        ]
        samples_b = [
            {"seed": [1], "value": 1.2},
            {"seed": [2], "value": 1.3},
            {"seed": [3], "value": 1.1},
        ]
        result = experiment._contrast_test(
            samples_a, samples_b
        )
        self.assertEqual(result["test"], "paired_t")
        self.assertEqual(result["n_pairs"], 3)
        self.assertEqual(
            result["paired_seeds"], [[1], [2], [3]]
        )

    def test_time_order_manifest_accepts_all_digit_plain_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "times.txt"
            # 全数字 ISO-8601 basic 日期会被 json.loads 解析成
            # 整数，必须回退到纯文本而不是报缺少 timestamp。
            path.write_text(
                "20200101\n20200102\n", encoding="utf-8"
            )
            report = experiment.read_time_order_manifest(
                str(path)
            )
        self.assertEqual(report["records"], 2)
        self.assertTrue(report["monotonic_non_decreasing"])

    def test_metrics_reader_skips_invalid_utf8_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            path.write_bytes(
                b'{"step": 1, "val_loss": 1.0}\n\xff\n'
            )
            records = experiment.read_metrics_records(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["step"], 1)

    def test_metrics_reader_tolerates_unreadable_path(self):
        # 损坏/手工编辑的运行记录可能把 path 写成超长字符串；
        # 分析层应把它当成没有曲线，而不是让 Path.exists 抛
        # ENAMETOOLONG 中断整份对比报告。
        self.assertEqual(
            experiment.read_metrics_records(Path("x" * 10000)),
            [],
        )

    def test_json_safe_converts_sets_deterministically(self):
        payload = {"a": {3, 1, 2}, "b": frozenset([2, 1])}
        first = experiment._json_safe(payload)
        second = experiment._json_safe(payload)
        self.assertEqual(first["a"], [1, 2, 3])
        self.assertEqual(first["b"], [1, 2])
        self.assertEqual(first, second)
        json.dumps(first, allow_nan=False)

    def test_artifact_readers_tolerate_unreadable_path(self):
        long_path = Path("x" * 10000)
        self.assertFalse(experiment._path_exists(long_path))
        # 不能判定是否存在 metrics.jsonl 时，尘封检查取保守值，
        # 避免把不可读路径误判成「没有 test 记录」。
        self.assertTrue(experiment.test_record_exists(long_path))
        record = experiment._read_test_records(long_path)
        self.assertFalse(record["available"])
        self.assertEqual(
            experiment.read_test_access_ledger(long_path), []
        )

    def test_test_record_reader_skips_invalid_utf8_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            metrics = run_dir / "metrics.jsonl"
            metrics.write_bytes(
                json.dumps(
                    {"step": "test", "test_loss": 1.0}
                ).encode("utf-8")
                + b"\n\xff\xfe\n"
            )
            record = experiment._read_test_records(run_dir)
            self.assertTrue(record["available"])
            self.assertEqual(record["test_loss"], 1.0)

    def test_test_access_ledger_tolerates_unreadable_file(self):
        # 账本存在但读不出来（权限/IO）时读取器不能整份崩溃；
        # 但门控必须把「读不出来」当作无法核对，而不是当作
        # 空账本放行重复解封。
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            ledger.write_text(
                json.dumps({"out_dir": str(Path(tmp) / "a")})
                + "\n",
                encoding="utf-8",
            )
            with mock.patch(
                "builtins.open",
                side_effect=PermissionError("denied"),
            ):
                entries, error = (
                    experiment._read_test_access_ledger_entries(
                        ledger
                    )
                )
                self.assertEqual(entries, [])
                self.assertIsNotNone(error)
                self.assertEqual(
                    experiment.read_test_access_ledger(ledger),
                    [],
                )
                key = {
                    "data_sha256": "d",
                    "test_split_sha256": "s",
                    "test_fraction": 0.05,
                }
                check = experiment.check_test_access_ledger(
                    ledger, key, Path(tmp) / "out"
                )
            self.assertTrue(check["repeat_access"])
            self.assertIsNotNone(check["ledger_unreadable"])
            self.assertEqual(check["other_candidates"], [])
            self.assertIn("无法读取", check["note"])
            message = experiment._repeat_test_access_error(check)
            self.assertIn("访问账本不可读", message)

    def test_compare_and_pilot_tolerate_unreadable_run_path(self):
        # 与 metrics 读取一致：损坏/超长运行路径不能让
        # --compare 或 --design-pilot-runs 整份崩溃。
        long_path = "x" * 10000
        self.assertFalse(experiment._path_is_dir(long_path))
        rows = experiment.compare_runs([long_path])
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["val_loss"])
        evidence = experiment.build_pilot_power_evidence(
            [long_path], "val_loss"
        )
        self.assertEqual(evidence["source_run_count"], 1)
        self.assertTrue(
            evidence["attrition_check"]["unusable"]
        )
        json.dumps(evidence, allow_nan=False)

    def test_resume_truncation_drops_stale_terminal_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            path.write_text(
                json.dumps({"step": 10, "val_loss": 1.0})
                + "\n"
                + json.dumps({"step": "final", "val_loss": 9.9})
                + "\n",
                encoding="utf-8",
            )
            experiment._truncate_metrics_after_step(path, 10)
            steps = [
                record.get("step")
                for record in experiment.read_metrics_records(path)
            ]
        self.assertEqual(steps, [10])

    def test_resume_truncation_tolerates_unhashable_step(self):
        # 手工编辑/损坏的 metrics 可能把 step 写成 list/dict；
        # resume 裁剪不能因为无法哈希就直接抛 TypeError。
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            path.write_text(
                json.dumps({"step": 10, "val_loss": 1.0})
                + "\n"
                + json.dumps({"step": [], "val_loss": 9.9})
                + "\n"
                + json.dumps({"step": {"bad": True}, "val_loss": 9.8})
                + "\n"
                + json.dumps({"step": "final", "val_loss": 9.7})
                + "\n",
                encoding="utf-8",
            )
            experiment._truncate_metrics_after_step(path, 10)
            steps = [
                record.get("step")
                for record in experiment.read_metrics_records(path)
            ]
        self.assertEqual(steps, [10, [], {"bad": True}])

    def test_resume_cleaning_justification_tolerates_bad_args(self):
        args = argparse.Namespace(domain_justification=None)
        self.assertFalse(
            experiment._restore_resume_cleaning_justification(
                args, None
            )
        )
        self.assertFalse(
            experiment._restore_resume_cleaning_justification(
                args, ["not-a-dict"]
            )
        )

    def test_paper_protocol_keeps_undisclosed_warmup_defaults(self):
        previous = sys.argv
        sys.argv = ["experiment.py", "--paper-protocol"]
        try:
            args = argparse.Namespace(
                paper_cell=None,
                paper_arch=None,
                paper_protocol=True,
                mode="B",
                lr_schedule="cosine",
                wsd_stable_steps=None,
                wsd_final_lr_ratio=0.1,
                save_at_steps=None,
                document_masks=False,
                group_separator=None,
                seq_len=128,
                steps=2000,
                batch_size=32,
                warmup_steps=100,
                eval_interval=200,
            )
            experiment.apply_paper_cli(args)
        finally:
            sys.argv = previous
        self.assertEqual(args.warmup_steps, 100)
        self.assertEqual(args.eval_interval, 200)

    def test_paper_cell_rejects_explicit_residual_scale(self):
        class _Parser:
            def error(self, message):
                raise SystemExit(message)

        previous = sys.argv
        sys.argv = [
            "experiment.py",
            "--paper-cell",
            "200M:95",
            "--residual-scale",
            "0.5",
        ]
        try:
            args = argparse.Namespace(
                mode="B",
                paper_cell="200M:95",
                paper_arch=None,
                paper_protocol=False,
                residual_scaling="paper",
                residual_scale=0.5,
                lr_schedule="cosine",
                wsd_stable_steps=None,
                wsd_final_lr_ratio=0.1,
                save_at_steps=None,
                document_masks=False,
                group_separator=None,
                seq_len=128,
                steps=2000,
                batch_size=32,
                warmup_steps=100,
                eval_interval=200,
            )
            with self.assertRaises(SystemExit):
                experiment.apply_paper_cli(args, _Parser())
        finally:
            sys.argv = previous


    def test_bad_case_comparison_skips_corrupt_payload(self):
        # metrics.jsonl 可能被手工编辑或由崩溃进程留下半条记录：
        # 计数/混淆对/分层错误率的损坏值不能中断整份对比报告。
        rows = [
            {
                "run": "corrupt",
                "mode": "A",
                "seed": 1,
                "bad_case_available": True,
                "bad_case_tokens_evaluated": float("nan"),
                "bad_case_wrong_tokens": "bad",
                "bad_case_error_category_counts": {
                    "ambiguous": float("inf"),
                },
                "bad_case_error_category_shares": {
                    "ambiguous": float("nan"),
                },
                "bad_case_top_confusions": float("nan"),
                "bad_case_error_rate_by_position": {"x": 1},
                "bad_case_error_rate_by_train_frequency": [
                    {"name": "b", "tokens": float("inf"), "wrong": []},
                    "bad",
                ],
                "bad_case_error_frequency_basis": ["x"],
                "bad_case_token_accuracy": float("nan"),
            },
            {
                "run": "valid",
                "mode": "A",
                "seed": 2,
                "bad_case_available": True,
                "bad_case_tokens_evaluated": 10,
                "bad_case_wrong_tokens": 4,
                "bad_case_error_category_counts": {
                    "ambiguous": 3,
                    "systematic_confusion": 1,
                },
                "bad_case_error_category_shares": {
                    "ambiguous": 0.75,
                    "systematic_confusion": 0.25,
                },
                "bad_case_top_confusions": [[65, 66, 2]],
                "bad_case_error_rate_by_position": [
                    {"name": "position_0", "tokens": 10, "wrong": 4},
                ],
                "bad_case_error_rate_by_train_frequency": [
                    {"name": "ge_5pct", "tokens": 10, "wrong": 4},
                ],
                "bad_case_error_frequency_basis": "train_share",
                "bad_case_token_accuracy": 0.6,
            },
        ]
        report = experiment.bad_case_comparison_report(rows)
        self.assertEqual(report["tokens_evaluated"], 10)
        self.assertEqual(report["wrong_tokens"], 4)
        self.assertEqual(
            report["error_category_counts"]["ambiguous"], 3
        )
        self.assertAlmostEqual(
            report["error_category_shares"]["ambiguous"], 0.75
        )
        self.assertIsNone(report["runs"][0]["top_confusion"])
        self.assertEqual(
            report["runs"][1]["top_confusion"], [65, 66, 2]
        )
        self.assertEqual(
            report["runs"][0]["error_category_counts"]["ambiguous"],
            0,
        )
        self.assertEqual(
            report["error_rate_by_position"][0]["error_rate"], 0.4
        )

    def test_protocol_mode_fallback_tolerates_unhashable_mode(self):
        # 损坏的 config.mode 不应让 loop_count 的集合成员判断
        # 抛 TypeError；无法确认时保持 None（未核对）。
        protocol = experiment.experiment_protocol_from_payload({
            "config": {"mode": [1, 2]},
            "args": {"lr": 3e-4, "steps": 1, "batch_size": 2},
        })
        self.assertIsNone(protocol["loop_count"])

    def test_periodic_auc_tolerates_unhashable_selection_metric(self):
        self.assertFalse(
            experiment._periodic_auc_enabled(False, [1, 2])
        )
        self.assertTrue(
            experiment._periodic_auc_enabled(
                False, "val_pr_auc_macro"
            )
        )

    def test_dense_fnn_hidden_type_is_validated_before_use(self):
        # 非整数会被 effective_dense_fnn_hidden 直接返回，随后
        # 在比较时抛 TypeError；必须先做类型校验并给出明确的
        # ValueError。bool 是 int 的子类，但 True/False 不是
        # 合法的宽度配置。
        for value in ("32", True, False, 0, -1, 3.5, []):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _tiny_config(
                        mode="dense",
                        dense_fnn_hidden=value,
                    ).validate()
                with self.assertRaises(ValueError):
                    _tiny_config(
                        mode="smelt",
                        dense_fnn_hidden=value,
                    ).validate()
        _tiny_config(
            mode="dense",
            dense_fnn_hidden=None,
        ).validate()
        _tiny_config(
            mode="dense",
            dense_fnn_hidden=32,
        ).validate()

    def test_group_top_byte_share_avoids_readonly_buffer_warning(self):
        # torch.frombuffer 不接受只读缓冲区：原实现先 bytes(...)
        # 再 frombuffer，在 -W error 下会因 UserWarning 失败。
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            top, share = experiment._group_top_byte_share(
                memoryview(b"aab"),
                0,
                3,
            )
        self.assertEqual(top, ord("a"))
        self.assertAlmostEqual(share, 2.0 / 3.0)


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


class GroupClusterRegressionTests(unittest.TestCase):
    """文档/群组 cluster bootstrap 的边界条件回归。"""

    def _stats(
        self,
        groups,
        separator="separator-sha",
        unit="document_group",
    ):
        return {
            "schema": experiment._GROUP_STATS_SCHEMA,
            "assignment": experiment._GROUP_ASSIGNMENT,
            "unit": unit,
            "group_separator_sha256": separator,
            "groups": groups,
        }

    def test_group_stats_reject_negative_loss(self):
        stats = self._stats([
            [0, 10.0, -1.0, 5.0],
            [1, 10.0, 20.0, 5.0],
        ])
        self.assertIsNone(
            experiment._parse_group_stats({"group_stats": stats})
        )
        report = experiment._group_cluster_eval_intervals(
            stats, replicates=50, seed=1
        )
        self.assertFalse(report["available"])

    def test_group_cluster_ci_level_is_applied(self):
        stats = self._stats([
            [0, 10.0, 10.0, 5.0],
            [1, 10.0, 20.0, 5.0],
            [2, 10.0, 100.0, 5.0],
        ])
        narrow = experiment._group_cluster_eval_intervals(
            stats,
            replicates=500,
            seed=7,
            ci_level=0.80,
        )
        wide = experiment._group_cluster_eval_intervals(
            stats,
            replicates=500,
            seed=7,
            ci_level=0.95,
        )
        self.assertTrue(narrow["available"], narrow)
        self.assertTrue(wide["available"], wide)
        self.assertEqual(narrow["bootstrap_ci_level"], 0.80)
        self.assertEqual(wide["bootstrap_ci_level"], 0.95)
        self.assertLess(
            narrow["loss_ci95_high"] - narrow["loss_ci95_low"],
            wide["loss_ci95_high"] - wide["loss_ci95_low"],
        )
        for level in (0.0, 1.0, float("nan"), "0.95", True, None):
            with self.subTest(ci_level=level):
                report = experiment._group_cluster_eval_intervals(
                    stats,
                    replicates=50,
                    seed=1,
                    ci_level=level,
                )
                self.assertFalse(report["available"])
                self.assertEqual(
                    report["reason"], "invalid_ci_level"
                )

    def test_fixed_eval_group_ids_handles_invalid_batch_counts(self):
        separator = b"<|doc|>"
        payload = (b"aaaa" + separator) * 20 + b"bbbb"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.bin"
            path.write_bytes(payload)
            dataset = experiment.ByteDataset(
                str(path),
                seq_len=4,
                val_fraction=0.3,
                test_fraction=0.3,
                group_separator=separator,
            )
            for batches, batch_size in (
                (float("nan"), 1),
                (float("inf"), 1),
                ("x", 1),
                (1, None),
                (0, float("inf")),
            ):
                self.assertIsNone(
                    dataset.fixed_eval_group_ids(
                        "val", batches, batch_size
                    )
                )

    def test_group_eval_bootstrap_handles_extreme_seeds(self):
        stats = {
            "schema": experiment._GROUP_STATS_SCHEMA,
            "assignment": experiment._GROUP_ASSIGNMENT,
            "unit": "document_group",
            "group_separator_sha256": "x",
            "groups": [
                [0, 10.0, 1.0, 10.0],
                [1, 10.0, 2.0, 9.0],
            ],
        }
        huge = experiment._group_cluster_eval_intervals(
            stats, replicates=50, seed=10 ** 400
        )
        self.assertTrue(huge["available"])
        json.dumps(huge, allow_nan=False)
        invalid = experiment._group_cluster_eval_intervals(
            stats, replicates=50, seed=float("inf")
        )
        self.assertFalse(invalid["available"])
        self.assertEqual(
            invalid["reason"], "invalid_bootstrap_seed"
        )
        json.dumps(invalid, allow_nan=False)

    def test_fixed_starts_cache_tracks_start_count_and_clamps(self):
        dataset = types.SimpleNamespace(_fixed_starts={})
        first = experiment.ByteDataset._ensure_fixed_starts(
            dataset, "val", 3, 10
        )
        repeated = experiment.ByteDataset._ensure_fixed_starts(
            dataset, "val", 3, 10
        )
        self.assertIs(first, repeated)
        resized = experiment.ByteDataset._ensure_fixed_starts(
            dataset, "val", 3, 20
        )
        self.assertFalse(torch.equal(first, resized))
        self.assertLess(int(resized.max().item()), 20)
        for invalid in (float("nan"), float("inf"), None, "bad"):
            empty = experiment.ByteDataset._ensure_fixed_starts(
                dataset, "val", invalid, 10
            )
            self.assertEqual(empty.numel(), 0)
        clamped = experiment.ByteDataset._ensure_fixed_starts(
            dataset, "val", 8, 4
        )
        self.assertEqual(clamped.numel(), 4)
        self.assertEqual(
            clamped.tolist(),
            sorted(set(clamped.tolist())),
        )

    def test_accuracy_intervals_are_bounded_near_one(self):
        stats = self._stats([
            [0, 10.0, 5.0, 10.0 + 5.0e-10],
            [1, 10.0, 5.0, 10.0 + 5.0e-10],
        ])
        intervals = experiment._group_cluster_eval_intervals(
            stats, replicates=50, seed=1
        )
        self.assertTrue(intervals["available"])
        self.assertLessEqual(
            intervals["token_accuracy_ci95_low"],
            intervals["token_accuracy_ci95_high"],
        )
        self.assertLessEqual(
            intervals["token_accuracy_ci95_high"],
            1.0,
        )
        before = self._stats([
            [0, 10.0, 5.0, 0.0],
            [1, 10.0, 5.0, 0.0],
        ])
        paired = experiment._group_cluster_paired_comparison(
            {"group_stats": before},
            {"group_stats": stats},
        )
        self.assertTrue(paired["available"])
        accuracy = paired["accuracy_bootstrap_ci95"]
        self.assertLessEqual(
            accuracy["low"], accuracy["high"]
        )
        self.assertLessEqual(accuracy["high"], 1.0)

    def test_paired_accuracy_interval_keeps_negative_differences(self):
        before = self._stats([
            [0, 10.0, 20.0, 9.0],
            [1, 10.0, 20.0, 9.0],
            [2, 10.0, 20.0, 9.0],
        ])
        after = self._stats([
            [0, 10.0, 20.0, 0.0],
            [1, 10.0, 20.0, 0.0],
            [2, 10.0, 20.0, 0.0],
        ])
        report = experiment._group_cluster_paired_comparison(
            {"group_stats": before},
            {"group_stats": after},
        )
        self.assertTrue(report["available"])
        self.assertEqual(report["mean_accuracy_diff"], -0.9)
        interval = report["accuracy_bootstrap_ci95"]
        self.assertEqual(interval["low"], -0.9)
        self.assertEqual(interval["high"], -0.9)
        self.assertLessEqual(interval["low"], interval["high"])

    def test_paired_group_means_do_not_overflow(self):
        before = self._stats([
            [0, 1.0, 0.0, 0.0],
            [1, 1.0, 0.0, 0.0],
        ])
        after = self._stats([
            [0, 1.0, 1.0e308, 0.0],
            [1, 1.0, 1.0e308, 0.0],
        ])
        report = experiment._group_cluster_paired_comparison(
            {"group_stats": before},
            {"group_stats": after},
        )
        self.assertTrue(report["available"])
        self.assertTrue(math.isfinite(report["mean_loss_diff"]))
        self.assertTrue(
            math.isfinite(report["unweighted_mean_loss_diff"])
        )
        self.assertEqual(report["mean_loss_diff"], 1.0e308)
        json.dumps(report, allow_nan=False)

    def test_hierarchical_group_bootstrap_handles_bad_replicates(self):
        differences = {
            1: {0: (1.0, 10.0), 1: (2.0, 10.0)},
            2: {0: (2.0, 10.0), 1: (3.0, 10.0)},
        }
        for replicates in (0, 1, -5):
            report = experiment._hierarchical_group_bootstrap_ci(
                differences, replicates=replicates
            )
            self.assertFalse(report["available"])
        self.assertFalse(
            experiment._hierarchical_group_bootstrap_ci(
                [], replicates=10
            )["available"]
        )
        stats = self._stats([
            [0, 10.0, 20.0, 5.0],
            [1, 10.0, 30.0, 5.0],
        ])
        for replicates in (0, 1, "bad", None):
            report = experiment._group_cluster_eval_intervals(
                stats, replicates=replicates, seed=1
            )
            self.assertFalse(report["available"])

    def test_group_reports_stay_json_safe_with_extreme_tokens(self):
        token_count = 1.0e308
        stats = self._stats([
            [0, token_count, 1.0, 0.0],
            [1, token_count, 1.0, 0.0],
        ])
        intervals = experiment._group_cluster_eval_intervals(
            stats, replicates=50, seed=1
        )
        self.assertTrue(intervals["available"])
        self.assertIsNone(intervals["tokens"])
        json.dumps(intervals, allow_nan=False)
        paired = experiment._group_cluster_paired_comparison(
            {"group_stats": stats},
            {
                "group_stats": self._stats([
                    [0, token_count, 2.0, 0.0],
                    [1, token_count, 2.0, 0.0],
                ])
            },
        )
        self.assertTrue(paired["available"])
        self.assertIsNone(paired["tokens"])
        json.dumps(paired, allow_nan=False)

    def test_hierarchical_group_bootstrap_normalizes_extreme_tokens(self):
        for token_count in (1.0e308, 1.0e-308):
            differences = {
                1: {
                    0: (1.0, token_count),
                    1: (2.0, token_count),
                },
                2: {
                    0: (2.0, token_count),
                    1: (3.0, token_count),
                },
            }
            report = experiment._hierarchical_group_bootstrap_ci(
                differences, replicates=100
            )
            self.assertTrue(report["available"])
            self.assertTrue(math.isfinite(report["low"]))
            self.assertTrue(math.isfinite(report["high"]))
            json.dumps(report, allow_nan=False)
        mixed = {
            1: {0: (1.0, 1.0e308), 1: (2.0, 1.0e-308)},
            2: {0: (2.0, 1.0e308), 1: (3.0, 1.0e-308)},
        }
        report = experiment._hierarchical_group_bootstrap_ci(
            mixed, replicates=100
        )
        self.assertTrue(report["available"])
        json.dumps(report, allow_nan=False)

    def test_invalid_bootstrap_replicates_stay_json_safe(self):
        group_stats = {
            "schema": experiment._GROUP_STATS_SCHEMA,
            "assignment": experiment._GROUP_ASSIGNMENT,
            "unit": "document_group",
            "group_separator_sha256": "x",
            "groups": [
                [0, 10.0, 1.0, 10.0],
                [1, 10.0, 2.0, 9.0],
            ],
        }
        for result in (
            experiment._group_cluster_eval_intervals(
                group_stats,
                float("inf"),
                0,
            ),
            experiment._hierarchical_group_bootstrap_ci(
                {1: {0: (1.0, 10.0), 1: (2.0, 10.0)}},
                float("inf"),
            ),
            experiment._hierarchical_window_bootstrap_ci(
                {1: [1.0, 2.0], 2: [2.0, 3.0]},
                float("inf"),
            ),
            experiment._group_cluster_eval_intervals(
                group_stats,
                10 ** 400,
                0,
            ),
            experiment._hierarchical_group_bootstrap_ci(
                {1: {0: (1.0, 10.0), 1: (2.0, 10.0)}},
                10 ** 400,
            ),
            experiment._hierarchical_window_bootstrap_ci(
                {1: [1.0, 2.0], 2: [2.0, 3.0]},
                10 ** 400,
            ),
        ):
            self.assertEqual(
                result["reason"],
                "invalid_bootstrap_replicates",
            )
            json.dumps(result, allow_nan=False)

    def test_group_bootstrap_is_scale_stable(self):
        weights = torch.ones(3, dtype=torch.float64)
        small = torch.tensor(
            [1.0, 2.0, 3.0],
            dtype=torch.float64,
        )
        large = small * 1.0e300
        small_interval = experiment._group_stats_bootstrap_ci(
            small,
            weights,
            64,
            torch.Generator().manual_seed(7),
        )
        large_interval = experiment._group_stats_bootstrap_ci(
            large,
            weights,
            64,
            torch.Generator().manual_seed(7),
        )
        self.assertIsNotNone(small_interval)
        self.assertIsNotNone(large_interval)
        self.assertAlmostEqual(
            large_interval[0] / 1.0e300,
            small_interval[0],
            places=12,
        )
        self.assertAlmostEqual(
            large_interval[1] / 1.0e300,
            small_interval[1],
            places=12,
        )

    def test_eval_bootstrap_intervals_avoid_overflowing_ar_bounds(self):
        # 极小 token 数使逐窗口 loss 合法但接近 float 上限。
        # AR(1) 膨胀若用 (low + high) / 2 会先溢出，把区间写成
        # 非标准 JSON 的 inf；端点必须保持有限。
        n = 10
        tokens = torch.full(
            (n,), 1.0e-300, dtype=torch.float64
        )
        per_window_loss = torch.full(
            (n,), 9.0e307, dtype=torch.float64
        ) * (
            1.0
            + torch.arange(n, dtype=torch.float64) * 1.0e-3
        )
        loss_sums = per_window_loss * tokens
        zeros = torch.zeros((n, 3), dtype=torch.float64)
        report = experiment._bootstrap_eval_intervals(
            loss_sums,
            tokens,
            zeros,
            zeros,
            zeros,
            100,
            seed=1,
        )
        self.assertGreater(report["window_ci_inflation"], 1.0)
        self.assertIn("loss_ci95_low", report)
        self.assertTrue(
            math.isfinite(report["loss_ci95_low"])
        )
        self.assertTrue(
            math.isfinite(report["loss_ci95_high"])
        )
        self.assertLessEqual(
            report["loss_ci95_low"],
            report["loss_ci95_high"],
        )
        for value in report.values():
            if isinstance(value, float):
                self.assertTrue(math.isfinite(value), value)

    def test_eval_bootstrap_rejects_invalid_ci_and_seed(self):
        values = torch.ones(4, dtype=torch.float64)
        zeros = torch.zeros((4, 2), dtype=torch.float64)
        for level in (
            float("nan"),
            float("inf"),
            None,
            "0.95",
            0.0,
            1.0,
        ):
            with self.subTest(level=level):
                self.assertEqual(
                    experiment._bootstrap_eval_intervals(
                        values,
                        values,
                        zeros,
                        zeros,
                        zeros,
                        10,
                        seed=0,
                        ci_level=level,
                    ),
                    {},
                )
        for seed in (-1, 2 ** 64, 2 ** 65 + 7):
            with self.subTest(seed=seed):
                report = experiment._bootstrap_eval_intervals(
                    values,
                    values,
                    zeros,
                    zeros,
                    zeros,
                    10,
                    seed=seed,
                )
                self.assertTrue(report)

    def test_bootstrap_block_size_is_clamped_to_window_count(self):
        # 块长大于窗口数时只有前 n 个循环偏移参与抽样；不先
        # 截断会对合法的超大 --eval-bootstrap-block 分配巨量
        # 内存，但结果本应与 block=n 相同。
        small = experiment._bootstrap_window_indices(
            5,
            8,
            torch.Generator().manual_seed(11),
            block_size=5,
        )
        huge = experiment._bootstrap_window_indices(
            5,
            8,
            torch.Generator().manual_seed(11),
            block_size=10 ** 18,
        )
        self.assertTrue(torch.equal(small, huge))

    def test_eval_bootstrap_rejects_excessive_replicates(self):
        values = torch.ones(4, dtype=torch.float64)
        zeros = torch.zeros((4, 2), dtype=torch.float64)
        self.assertEqual(
            experiment._bootstrap_eval_intervals(
                values,
                values,
                zeros,
                zeros,
                zeros,
                experiment._MAX_BOOTSTRAP_REPLICATES + 1,
                seed=0,
            ),
            {},
        )

    def test_group_bootstrap_rejects_extreme_ci_levels(self):
        values = torch.tensor(
            [1.0, 2.0, 3.0],
            dtype=torch.float64,
        )
        weights = torch.ones(3, dtype=torch.float64)
        for ci_level in (
            10 ** 400,
            float("inf"),
            float("nan"),
            0.0,
            1.0,
            True,
            "bad",
        ):
            self.assertIsNone(
                experiment._group_stats_bootstrap_ci(
                    values,
                    weights,
                    32,
                    torch.Generator().manual_seed(0),
                    ci_level,
                )
            )

    def test_group_bootstrap_rejects_negative_or_2d_weights(self):
        generator = torch.Generator().manual_seed(0)
        values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        negative = torch.tensor(
            [1.0, -1.0, 1.0],
            dtype=torch.float64,
        )
        self.assertIsNone(
            experiment._group_stats_bootstrap_ci(
                values,
                negative,
                32,
                generator,
            )
        )
        two_dimensional = torch.ones(
            (3, 1),
            dtype=torch.float64,
        )
        self.assertIsNone(
            experiment._group_stats_bootstrap_ci(
                values,
                two_dimensional,
                32,
                generator,
            )
        )

    def test_hierarchical_group_bootstrap_is_scale_stable(self):
        small = {
            1: {
                0: (1.0, 10.0),
                1: (2.0, 10.0),
                2: (3.0, 10.0),
            },
            2: {
                0: (2.0, 10.0),
                1: (3.0, 10.0),
                2: (4.0, 10.0),
            },
        }
        large = {
            seed: {
                group: (
                    difference * 1.0e300,
                    tokens * 1.0e200,
                )
                for group, (difference, tokens) in mapping.items()
            }
            for seed, mapping in small.items()
        }
        small_ci = experiment._hierarchical_group_bootstrap_ci(
            small
        )
        large_ci = experiment._hierarchical_group_bootstrap_ci(
            large
        )
        self.assertTrue(small_ci["available"])
        self.assertTrue(large_ci["available"])
        self.assertAlmostEqual(
            large_ci["low"] / 1.0e300,
            small_ci["low"],
            places=12,
        )
        self.assertAlmostEqual(
            large_ci["high"] / 1.0e300,
            small_ci["high"],
            places=12,
        )
        before = self._stats([
            [0, 10.0, 0.0, 0.0],
            [1, 10.0, 1.0, 0.0],
            [2, 10.0, 3.0, 0.0],
        ])
        after = self._stats([
            [0, 10.0, 1.0, 0.0],
            [1, 10.0, 3.0, 0.0],
            [2, 10.0, 4.0, 0.0],
        ])
        scaled_before = self._stats([
            [0, 10.0, 0.0, 0.0],
            [1, 10.0, 1.0e300, 0.0],
            [2, 10.0, 3.0e300, 0.0],
        ])
        scaled_after = self._stats([
            [0, 10.0, 1.0e300, 0.0],
            [1, 10.0, 3.0e300, 0.0],
            [2, 10.0, 4.0e300, 0.0],
        ])
        small_pair = experiment._group_cluster_paired_comparison(
            {"group_stats": before},
            {"group_stats": after},
        )
        large_pair = experiment._group_cluster_paired_comparison(
            {"group_stats": scaled_before},
            {"group_stats": scaled_after},
        )
        self.assertTrue(small_pair["available"])
        self.assertTrue(large_pair["available"])
        self.assertAlmostEqual(
            large_pair["mean_loss_diff"] / 1.0e300,
            small_pair["mean_loss_diff"],
            places=12,
        )
        self.assertAlmostEqual(
            large_pair["bootstrap_ci95"]["low"] / 1.0e300,
            small_pair["bootstrap_ci95"]["low"],
            places=12,
        )
        self.assertAlmostEqual(
            large_pair["bootstrap_ci95"]["high"] / 1.0e300,
            small_pair["bootstrap_ci95"]["high"],
            places=12,
        )

    def test_hierarchical_window_bootstrap_handles_bad_replicates(self):
        differences = {
            1: [1.0, 2.0],
            2: [2.0, 3.0],
        }
        for replicates in (0, 1, -5, "bad", None):
            report = experiment._hierarchical_window_bootstrap_ci(
                differences, replicates=replicates
            )
            self.assertFalse(report["available"])
        self.assertFalse(
            experiment._hierarchical_window_bootstrap_ci(
                [], replicates=10
            )["available"]
        )

    def test_group_comparison_rejects_unit_mismatch(self):
        groups = [
            [0, 10.0, 20.0, 5.0],
            [1, 10.0, 30.0, 5.0],
        ]
        report = experiment._group_cluster_paired_comparison(
            {"group_stats": self._stats(groups)},
            {
                "group_stats": self._stats(
                    groups, unit="window"
                )
            },
        )
        self.assertFalse(report["available"])
        self.assertEqual(report["reason"], "group_unit_mismatch")
        for unit in ("window", "token"):
            stats = self._stats(groups, unit=unit)
            self.assertFalse(
                experiment._group_cluster_eval_intervals(
                    stats, replicates=50, seed=1
                )["available"]
            )
            self.assertFalse(
                experiment._group_cluster_paired_comparison(
                    {"group_stats": stats},
                    {"group_stats": stats},
                )["available"]
            )
        invalid = self._stats(groups)
        invalid["unit"] = {"name": "document_group"}
        self.assertIsNone(
            experiment._parse_group_stats(
                {"group_stats": invalid}
            )
        )


if __name__ == "__main__":
    unittest.main()
