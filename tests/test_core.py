"""Fast regression tests for the methodological fixes.

Run with: python -m unittest discover -s tests -v
"""

import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
