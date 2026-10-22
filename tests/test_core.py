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
import unittest
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


class RoutingInterventionReproducibilityTests(unittest.TestCase):
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
        # n>1023 时 2**n 不能再转 float；精确符号检验仍需
        # 给出可判定的 p 值与最小可达 p。
        sign = experiment._exact_sign_test([1.0] * 2000)
        self.assertTrue(sign["available"])
        self.assertEqual(sign["p_value"], 0.0)
        self.assertEqual(sign["min_attainable_p"], 0.0)

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


if __name__ == "__main__":
    unittest.main()
