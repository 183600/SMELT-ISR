from _core_support import *
from _core_support import _load_experiment, _tiny_config
from _core_DesignPreparationTests_base import _DesignPreparationTestsBase

class DesignPreparationTests(_DesignPreparationTestsBase):
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
        for scheme in (
            "grouped",
            "blocked",
            "randomized",
            "counterbalanced",
            "williams",
        ):
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
    def test_preregistration_accepts_williams_run_order(self):
        payload = self._payload()
        payload["run_order"] = {
            "scheme": "williams",
            "seed": 7,
        }
        self.assertIsNone(
            experiment.preregistration_schema_error(payload)
        )
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

        # 损坏的 sd/target/alpha/min_effect 也不能在直接调用
        # 时抛 OverflowError 或把 inf/nan 写进报告。
        for power in (
            {
                "expected_seed_sd_relative": 10**400,
                "target_power": 0.8,
                "alpha": 0.05,
                "planned_comparisons": 1,
            },
            {
                "expected_seed_sd_relative": 0.01,
                "target_power": 10**400,
                "alpha": 10**400,
                "planned_comparisons": 1,
            },
            {
                "expected_seed_sd_relative": -1,
                "target_power": float("nan"),
                "alpha": float("inf"),
                "planned_comparisons": 1,
            },
        ):
            plan = experiment.preregistration_power_plan(
                power, 0.01, 6, 1
            )
            self.assertTrue(
                math.isfinite(plan["expected_seed_sd_relative"])
            )
            self.assertTrue(math.isfinite(plan["target_power"]))
            self.assertTrue(math.isfinite(plan["alpha"]))
            self.assertFalse(plan["adequate"])
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
    def test_williams_order_balances_carryover_even(self):
        plan = experiment.build_sweep_plan(
            seeds=[1, 2, 3, 4, 5, 6],
            modes=["A", "B", "C", "D", "E", "F"],
            ablation=False,
            balance_coef=0.01,
            default_mode="B",
        )
        ordered = experiment.order_sweep_plan(
            plan, "williams", 13
        )
        repeated = experiment.order_sweep_plan(
            plan, "williams", 13
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
        carryover = report["carryover"]
        self.assertTrue(carryover["available"])
        self.assertTrue(carryover["balanced"])
        self.assertEqual(carryover["pairs_possible"], 30)
        self.assertEqual(carryover["pairs_observed"], 30)
        self.assertEqual(carryover["transitions"], 30)
        self.assertEqual(carryover["max_pair_imbalance"], 0)
        self.assertEqual(
            carryover["full_balance_block_multiple"], 6
        )
        pair_counts = {}
        for seed in range(1, 7):
            block = sorted(
                [
                    item
                    for item in ordered
                    if item["block"] == seed
                ],
                key=lambda item: item["position_in_block"],
            )
            modes = [item["mode"] for item in block]
            for left, right in zip(modes, modes[1:]):
                pair_counts[(left, right)] = (
                    pair_counts.get((left, right), 0) + 1
                )
        self.assertEqual(len(pair_counts), 30)
        self.assertEqual(set(pair_counts.values()), {1})
    def test_williams_order_balances_carryover_odd(self):
        plan = experiment.build_sweep_plan(
            seeds=list(range(1, 11)),
            modes=["A", "B", "C", "D", "E"],
            ablation=False,
            balance_coef=0.01,
            default_mode="B",
        )
        ordered = experiment.order_sweep_plan(
            plan, "williams", 17
        )
        report = experiment.run_order_balance_report(ordered)
        self.assertTrue(report["complete_blocks"])
        self.assertTrue(report["balanced"])
        self.assertEqual(report["max_position_imbalance"], 0)
        carryover = report["carryover"]
        self.assertTrue(carryover["available"])
        self.assertTrue(carryover["balanced"])
        self.assertEqual(carryover["pairs_possible"], 20)
        self.assertEqual(carryover["pairs_observed"], 20)
        self.assertEqual(carryover["transitions"], 40)
        self.assertEqual(carryover["min_pair_count"], 2)
        self.assertEqual(carryover["max_pair_count"], 2)
        self.assertEqual(carryover["max_pair_imbalance"], 0)
        self.assertEqual(
            carryover["full_balance_block_multiple"], 10
        )
    def test_williams_order_reports_partial_carryover(self):
        plan = experiment.build_sweep_plan(
            seeds=[1, 2, 3, 4, 5],
            modes=["A", "B", "C", "D", "E"],
            ablation=False,
            balance_coef=0.01,
            default_mode="B",
        )
        ordered = experiment.order_sweep_plan(
            plan, "williams", 19
        )
        report = experiment.run_order_balance_report(ordered)
        self.assertTrue(report["complete_blocks"])
        carryover = report["carryover"]
        self.assertTrue(carryover["available"])
        self.assertFalse(carryover["balanced"])
        self.assertLess(
            carryover["pairs_observed"],
            carryover["pairs_possible"],
        )
        self.assertEqual(
            carryover["full_balance_block_multiple"], 10
        )
    def test_print_design_summary_warns_partial_williams(self):
        plan = experiment.build_sweep_plan(
            seeds=[1, 2, 3, 4, 5],
            modes=["A", "B", "C", "D", "E"],
            ablation=False,
            balance_coef=0.01,
            default_mode="B",
        )
        ordered = experiment.order_sweep_plan(
            plan, "williams", 19
        )
        design = {
            "planned_runs": len(ordered),
            "order_scheme": "williams",
            "order_seed": 19,
            "plan_id": "x" * 20,
            "path": "design.json",
            "order_balance": experiment.run_order_balance_report(
                ordered
            ),
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            experiment.print_design_summary(design)
        text = output.getvalue()
        self.assertIn("[warn] williams", text)
        self.assertIn("10", text)
    def test_williams_order_requires_exact_carryover_counts(self):
        plan = experiment.build_sweep_plan(
            seeds=list(range(1, 8)),
            modes=["A", "B", "C", "D", "E", "F"],
            ablation=False,
            balance_coef=0.01,
            default_mode="B",
        )
        ordered = experiment.order_sweep_plan(
            plan, "williams", 23
        )
        report = experiment.run_order_balance_report(ordered)
        carryover = report["carryover"]
        self.assertTrue(carryover["available"])
        self.assertEqual(carryover["pairs_observed"], 30)
        self.assertEqual(carryover["max_pair_imbalance"], 1)
        self.assertFalse(carryover["balanced"])
    def test_williams_rejects_unequal_blocks(self):
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
                plan, "williams", 0
            )
        self.assertIn("williams", str(context.exception))
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
    def test_order_balance_report_rejects_malformed_block_positions(self):
        # 位置必须是 1..n 连续唯一；重复（全 1）或跳号会让
        # 位置计数极差看起来为 0，但不能算作完整区组证据。
        for positions in ([1, 1, 1, 1], [1, 2, 3, 5]):
            plan = experiment.build_sweep_plan(
                seeds=[1, 2, 3, 4],
                modes=["A", "B", "C", "D"],
                ablation=False,
                balance_coef=0.01,
                default_mode="B",
            )
            ordered = experiment.order_sweep_plan(
                plan, "williams", 13
            )
            blocks = {}
            for item in ordered:
                blocks.setdefault(
                    item["block"], []
                ).append(item)
            for block in blocks.values():
                for item, position in zip(block, positions):
                    item["position_in_block"] = position
            report = experiment.run_order_balance_report(ordered)
            self.assertTrue(report["available"])
            self.assertFalse(report["complete_blocks"])
            self.assertFalse(report["balanced"])
            carryover = report["carryover"]
            self.assertFalse(carryover["available"])
            self.assertFalse(carryover["balanced"])
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
