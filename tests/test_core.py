"""Fast regression tests for the methodological fixes.

Run with: python -m unittest discover -s tests -v
"""

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
