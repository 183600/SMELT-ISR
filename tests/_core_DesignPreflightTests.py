"""设计阶段可行性预检（语料指纹/epoch 预算、试点墙钟估计、算力门）。"""

from _core_support import *


class DesignPreflightTests(unittest.TestCase):
    """--plan-design 阶段的语料与算力预检。"""

    def _write_corpus(self, root: Path, content: bytes) -> Path:
        corpus = root / "corpus.txt"
        corpus.write_bytes(content)
        return corpus

    def _prereg_record(
        self,
        corpus_sha: str,
        *,
        compute_budget: dict = None,
    ) -> dict:
        data = {
            "sha256": corpus_sha,
            "split_policy": "contiguous",
            "val_fraction": 0.1,
            "test_fraction": 0.1,
        }
        return {
            "study_id": "smelt-isr-preflight-test",
            "file_sha256": "d" * 64,
            "data": data,
            "compute_budget": compute_budget,
        }

    def _plan_args(
        self,
        corpus: Path = None,
        prereg: dict = None,
        *,
        steps: int = 10,
        pilot_runs: list = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            seeds=[1, 2, 3],
            allow_underpowered=False,
            allow_over_compute_budget=False,
            steps=steps,
            batch_size=2,
            seq_len=8,
            preregistration_record=prereg,
            run_order="randomized",
            run_order_seed=5,
            data=str(corpus) if corpus is not None else None,
            val_fraction=0.1,
            test_fraction=0.1,
            design_pilot_runs=pilot_runs,
        )

    def _ordered_plan(self):
        plan = experiment.build_sweep_plan(
            seeds=[1, 2, 3],
            modes=["A", "B"],
            ablation=False,
            balance_coef=0.01,
            default_mode="B",
        )
        return experiment.order_sweep_plan(plan, "randomized", 5)

    def test_data_preflight_matches_prereg_and_estimates_epochs(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._write_corpus(root, b"xy" * 10000)
            sha = hashlib.sha256(corpus.read_bytes()).hexdigest()
            prereg = self._prereg_record(sha)
            args = self._plan_args(corpus, prereg, steps=10)
            design = experiment.write_design_plan(
                root / "plan",
                self._ordered_plan(),
                "randomized",
                5,
                args,
            )
            data_file = design["training_budget"]["data_file"]
            self.assertTrue(data_file["available"])
            self.assertTrue(data_file["matches_preregistration"])
            self.assertEqual(data_file["sha256"], sha)
            self.assertEqual(data_file["size_bytes"], 20000)
            # train 切分 ≈ 80% → 16000 字节；每运行
            # 10×2×8=160 tokens → 0.01 epoch。
            self.assertEqual(data_file["estimated_train_bytes"], 16000)
            self.assertAlmostEqual(
                data_file["estimated_epochs_per_run"], 0.01
            )
            # 0.01 epoch 属 prototype 规模：应给出且仅给出
            # 「token 预算不足 1 个 epoch」的口径提示。
            self.assertEqual(len(data_file["warnings"]), 1)
            self.assertIn("不足", data_file["warnings"][0])
            self.assertIn("1 个 epoch", data_file["warnings"][0])

    def test_data_preflight_binds_corpus_identity_into_plan_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._write_corpus(root, b"ab" * 1000)
            sha = hashlib.sha256(corpus.read_bytes()).hexdigest()
            prereg = self._prereg_record(sha)
            design_first = experiment.write_design_plan(
                root / "plan1",
                self._ordered_plan(),
                "randomized",
                5,
                self._plan_args(corpus, prereg),
            )
            # 同一语料、同一输入 → 同一 plan_id。
            design_again = experiment.write_design_plan(
                root / "plan2",
                self._ordered_plan(),
                "randomized",
                5,
                self._plan_args(corpus, prereg),
            )
            self.assertEqual(
                design_first["plan_id"], design_again["plan_id"]
            )
            # 换语料 → 设计指纹必须改变（设计绑定语料身份）。
            corpus.write_bytes(b"cd" * 1000)
            prereg_new = self._prereg_record(
                hashlib.sha256(corpus.read_bytes()).hexdigest()
            )
            design_other = experiment.write_design_plan(
                root / "plan3",
                self._ordered_plan(),
                "randomized",
                5,
                self._plan_args(corpus, prereg_new),
            )
            self.assertNotEqual(
                design_first["plan_id"], design_other["plan_id"]
            )

    def test_data_preflight_rejects_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._write_corpus(root, b"ab" * 100)
            prereg = self._prereg_record("e" * 64)
            args = self._plan_args(corpus, prereg)
            with self.assertRaises(ValueError) as caught:
                experiment.write_design_plan(
                    root / "plan",
                    self._ordered_plan(),
                    "randomized",
                    5,
                    args,
                )
            self.assertIn("data.sha256", str(caught.exception))

    def test_data_preflight_rejects_missing_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prereg = self._prereg_record("e" * 64)
            args = self._plan_args(root / "missing.txt", prereg)
            with self.assertRaises(ValueError) as caught:
                experiment.write_design_plan(
                    root / "plan",
                    self._ordered_plan(),
                    "randomized",
                    5,
                    args,
                )
            self.assertIn("不存在", str(caught.exception))

    def test_data_preflight_degrades_gracefully_without_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._plan_args(None, None)
            design = experiment.write_design_plan(
                root / "plan",
                self._ordered_plan(),
                "randomized",
                5,
                args,
            )
            data_file = design["training_budget"]["data_file"]
            self.assertFalse(data_file["available"])
            self.assertEqual(
                data_file["reason"], "data_path_not_provided"
            )

    def test_data_preflight_warns_on_multiple_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 1000 字节语料；80% train = 800 字节；每运行
            # 100×2×8=1600 tokens → 2 epoch。
            corpus = self._write_corpus(root, b"z" * 1000)
            sha = hashlib.sha256(corpus.read_bytes()).hexdigest()
            prereg = self._prereg_record(sha)
            args = self._plan_args(corpus, prereg, steps=100)
            design = experiment.write_design_plan(
                root / "plan",
                self._ordered_plan(),
                "randomized",
                5,
                args,
            )
            data_file = design["training_budget"]["data_file"]
            self.assertGreater(
                data_file["estimated_epochs_per_run"], 1.0
            )
            self.assertTrue(
                any("epoch" in warning for warning in
                    data_file["warnings"])
            )

    def _write_pilot_with_efficiency(
        self,
        root: Path,
        mode: str,
        seed: int,
        ms_per_step: float,
    ) -> Path:
        run_dir = root / f"{mode}_seed{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "config": {"mode": mode, "balance_coef": 0.01},
                    "args": {"seed": seed},
                    "efficiency": {
                        "training_ms_per_step": ms_per_step,
                    },
                }
            ),
            encoding="utf-8",
        )
        return run_dir

    def test_wall_clock_estimate_uses_pilot_efficiency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._write_corpus(root, b"ab" * 1000)
            sha = hashlib.sha256(corpus.read_bytes()).hexdigest()
            prereg = self._prereg_record(sha)
            pilots = [
                self._write_pilot_with_efficiency(
                    root / "pilots", "A", 11, 100.0
                ),
                self._write_pilot_with_efficiency(
                    root / "pilots", "A", 12, 200.0
                ),
                self._write_pilot_with_efficiency(
                    root / "pilots", "B", 11, 400.0
                ),
                self._write_pilot_with_efficiency(
                    root / "pilots", "B", 12, 400.0
                ),
            ]
            args = self._plan_args(
                corpus,
                prereg,
                steps=1000,
                pilot_runs=[str(path) for path in pilots],
            )
            design = experiment.write_design_plan(
                root / "plan",
                self._ordered_plan(),
                "randomized",
                5,
                args,
            )
            wall_clock = design["training_budget"]["wall_clock"]
            self.assertTrue(wall_clock["available"])
            self.assertEqual(wall_clock["covered_runs"], 6)
            self.assertEqual(wall_clock["uncovered_runs"], 0)
            per_mode = wall_clock["per_mode"]
            # 保守口径：每 mode 取试点观测的最大步耗时。
            self.assertEqual(
                per_mode["A"]["max_training_ms_per_step"], 200.0
            )
            self.assertEqual(
                per_mode["B"]["max_training_ms_per_step"], 400.0
            )
            self.assertEqual(per_mode["A"]["planned_runs"], 3)
            self.assertEqual(per_mode["B"]["planned_runs"], 3)
            # 1000 步 × 0.2s = 200s/运行（A）；×0.4s = 400s（B）。
            self.assertAlmostEqual(
                per_mode["A"]["estimated_training_seconds_per_run"],
                200.0,
            )
            expected_total_hours = (200.0 * 3 + 400.0 * 3) / 3600.0
            self.assertAlmostEqual(
                wall_clock["estimated_total_training_hours"],
                expected_total_hours,
            )

    def test_wall_clock_unavailable_without_pilot_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._plan_args(None, None)
            design = experiment.write_design_plan(
                root / "plan",
                self._ordered_plan(),
                "randomized",
                5,
                args,
            )
            wall_clock = design["training_budget"]["wall_clock"]
            self.assertFalse(wall_clock["available"])
            self.assertEqual(wall_clock["reason"], "no_pilot_efficiency_records")

    def test_compute_budget_gate_blocks_over_budget(self):
        design = {
            "training_budget": {
                "compute_budget": {
                    "declared_max_total_training_hours": 1.0,
                    "estimated_total_training_hours": 5.0,
                    "estimate_available": True,
                    "within_budget": False,
                }
            }
        }
        error = experiment.design_budget_gate_error(design)
        self.assertIsNotNone(error)
        self.assertIn("compute_budget", error)
        self.assertIn("--allow-over-compute-budget", error)
        # 显式豁免：放行并把承认写回设计对象。
        self.assertIsNone(
            experiment.design_budget_gate_error(
                design, allow_over_budget=True
            )
        )
        self.assertTrue(
            design["training_budget"]["compute_budget"][
                "acknowledged_over_budget"
            ]
        )

    def test_compute_budget_gate_passes_within_budget(self):
        design = {
            "training_budget": {
                "compute_budget": {
                    "declared_max_total_training_hours": 10.0,
                    "estimated_total_training_hours": 5.0,
                    "estimate_available": True,
                    "within_budget": True,
                }
            }
        }
        self.assertIsNone(
            experiment.design_budget_gate_error(design)
        )

    def test_compute_budget_gate_skips_missing_estimate(self):
        design = {
            "training_budget": {
                "compute_budget": {
                    "declared_max_total_training_hours": 10.0,
                    "estimated_total_training_hours": None,
                    "estimate_available": False,
                    "within_budget": None,
                }
            }
        }
        self.assertIsNone(
            experiment.design_budget_gate_error(design)
        )
        self.assertIsNone(
            experiment.design_budget_gate_error({})
        )

    def _prereg_payload(self, **overrides):
        payload = {
            "study_id": "smelt-isr-budget-test",
            "frozen_commit": "a" * 40,
            "primary_metric": "val_loss",
            "common_step": 100,
            "min_effect": 0.01,
            "direction": "decrease",
            "baseline_mode": "smelt",
            "comparisons": [["smelt", "A"]],
            "seeds": [42, 43, 44],
            "budget_match": "smelt",
            "data": {
                "sha256": "b" * 64,
                "split_policy": "contiguous",
                "val_fraction": 0.1,
                "test_fraction": 0.1,
            },
        }
        payload.update(overrides)
        return payload

    def test_preregistration_schema_accepts_compute_budget(self):
        payload = self._prereg_payload(
            compute_budget={"max_total_training_hours": 72.0}
        )
        self.assertIsNone(
            experiment.preregistration_schema_error(payload)
        )

    def test_preregistration_schema_rejects_bad_compute_budget(self):
        base_error = experiment.preregistration_schema_error
        cases = [
            ({"max_total_training_hours": 0.0}, "正数"),
            ({"max_total_training_hours": -1.0}, "正数"),
            ({"max_total_training_hours": "72"}, "有限数"),
            ({"unknown_key": 1}, "未知字段"),
            ({}, "缺少必需字段"),
        ]
        for budget, fragment in cases:
            payload = self._prereg_payload(compute_budget=budget)
            error = base_error(payload)
            self.assertIsNotNone(error)
            self.assertIn(fragment, error)
        payload = self._prereg_payload(compute_budget=72.0)
        error = base_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("必须是对象", error)

    def test_preregistration_record_carries_compute_budget(self):
        payload = self._prereg_payload(
            compute_budget={"max_total_training_hours": 48.0}
        )
        args = argparse.Namespace(mode="A", seed=42, seeds=None)
        record = experiment.preregistration_record(
            payload, "preregistration.json", "c" * 64, [], args
        )
        self.assertEqual(
            record["compute_budget"],
            {"max_total_training_hours": 48.0},
        )
