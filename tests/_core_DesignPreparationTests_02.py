from _core_support import *
from _core_support import _load_experiment, _tiny_config
from _core_DesignPreparationTests_base import _DesignPreparationTestsBase

REPO_ROOT = Path(__file__).resolve().parents[1]


def _governance_payload(self):
    payload = self._payload()
    payload["seeds"] = list(range(42, 62))
    payload["steps"] = 200
    return payload


class DesignPreparationTestsPart3(_DesignPreparationTestsBase):
    """试点规模规划、停止规则与排除标准（实验设计与准备）。"""

    # --------------------------------------------------
    # 试点规模规划 plan_pilot_size
    # --------------------------------------------------
    def _pilot_power(self, **overrides):
        power = {
            "expected_seed_sd_relative": 0.005,
            "target_power": 0.8,
            "alpha": 0.05,
            "planned_comparisons": 3,
            "expected_attrition_rate": 0.1,
            "pilot_calibration": "upper_confidence_bound",
        }
        power.update(overrides)
        return power

    def test_plan_pilot_size_meets_ucb_tolerance(self):
        plan = experiment.plan_pilot_size(
            self._pilot_power(), 0.01, 12, 3, n_cells=2
        )
        self.assertEqual(
            plan["schema"], "smelt-isr/pilot-size-plan/v1"
        )
        self.assertEqual(
            plan["calibration_policy"], "upper_confidence_bound"
        )
        sd_planning = plan["sd_planning"]
        self.assertTrue(sd_planning["binding_for_calibration"])
        self.assertIsNotNone(sd_planning["min_pilot_pairs"])
        # 判据必须被满足：解对应的乘数 ≤ 容差，且它是网格中
        # 第一个满足者（之前所有乘数都更大）。
        tolerance = sd_planning["tolerance"]
        table = sd_planning["table"]
        multipliers = [row["sd_ucb_multiplier"] for row in table]
        self.assertEqual(
            multipliers, sorted(multipliers, reverse=True)
        )
        first = table[
            multipliers.index(
                next(
                    value
                    for value in multipliers
                    if value <= tolerance
                )
            )
        ]
        self.assertEqual(
            first["pilot_pairs"], sd_planning["min_pilot_pairs"]
        )
        self.assertLessEqual(
            first["sd_ucb_multiplier"], tolerance
        )
        # 试点必须至少覆盖可评估单元对的最小共同种子数。
        self.assertGreaterEqual(
            plan["recommended_pilot_runs_per_cell"],
            experiment._DESIGN_PILOT_MIN_PAIRS,
        )
        self.assertEqual(
            plan["recommended_total_pilot_runs"],
            plan["recommended_pilot_runs_per_cell"]
            * plan["n_cells"],
        )
        json.dumps(plan, allow_nan=False)

    def test_plan_pilot_size_tighter_tolerance_needs_more_pairs(self):
        loose = experiment.plan_pilot_size(
            self._pilot_power(),
            0.01,
            12,
            3,
            n_cells=2,
            sd_ucb_tolerance=1.5,
        )
        tight = experiment.plan_pilot_size(
            self._pilot_power(),
            0.01,
            12,
            3,
            n_cells=2,
            sd_ucb_tolerance=1.2,
        )
        self.assertIsNotNone(loose["sd_planning"]["min_pilot_pairs"])
        self.assertIsNotNone(tight["sd_planning"]["min_pilot_pairs"])
        self.assertGreaterEqual(
            tight["sd_planning"]["min_pilot_pairs"],
            loose["sd_planning"]["min_pilot_pairs"],
        )

    def test_plan_pilot_size_implied_required_seeds_grow_with_ucb(self):
        plan = experiment.plan_pilot_size(
            self._pilot_power(), 0.01, 12, 3, n_cells=2
        )
        rows = plan["sd_planning"]["table"]
        implied = [
            row["implied_required_seeds_with_attrition"]
            for row in rows
            if row["implied_required_seeds_with_attrition"]
            is not None
        ]
        self.assertEqual(implied, sorted(implied, reverse=True))
        # 小试点（n=2）的 UCB 会把所需种子数推到不可达（None
        # 或远大于计划），大试点则应回到声明口径附近。
        self.assertTrue(
            rows[0]["implied_required_seeds_with_attrition"] is None
            or rows[0]["implied_required_seeds_with_attrition"]
            > rows[-1]["implied_required_seeds_with_attrition"]
        )

    def test_plan_pilot_size_point_policy_is_informational(self):
        plan = experiment.plan_pilot_size(
            self._pilot_power(pilot_calibration="point"),
            0.01,
            12,
            3,
            n_cells=2,
        )
        self.assertFalse(
            plan["sd_planning"]["binding_for_calibration"]
        )
        self.assertFalse(
            plan["attrition_planning"]["binding_for_calibration"]
        )
        self.assertTrue(
            any("point" in warning for warning in plan["warnings"])
        )

    def test_plan_pilot_size_attrition_slack(self):
        # 计划有余量（required < planned）时，脱落率判据可满足。
        power = self._pilot_power()
        declared = experiment.preregistration_power_plan(
            power, 0.01, 16, 3
        )
        self.assertLess(
            declared["required_seeds_with_attrition"], 16
        )
        plan = experiment.plan_pilot_size(
            power, 0.01, 16, 3, n_cells=2
        )
        self.assertIsNotNone(
            plan["attrition_planning"]["min_pilot_runs_per_cell"]
        )
        # 计划刚好等于所需种子数（零余量）时，任何有限试点的
        # 脱落率上界都会吃掉余量：判据不可满足，必须如实报告。
        required = declared["required_seeds_with_attrition"]
        tight = experiment.plan_pilot_size(
            power, 0.01, int(required), 3, n_cells=2
        )
        self.assertIsNone(
            tight["attrition_planning"]["min_pilot_runs_per_cell"]
        )
        self.assertFalse(
            tight["attrition_planning"]["achievable_within_grid"]
        )
        self.assertTrue(tight["warnings"])

    def test_plan_pilot_size_requires_declared_sd(self):
        self.assertIsNone(experiment.plan_pilot_size(None, 0.01, 12, 3))
        self.assertIsNone(
            experiment.plan_pilot_size(
                {"target_power": 0.8}, 0.01, 12, 3
            )
        )
        self.assertIsNone(
            experiment.plan_pilot_size(
                self._pilot_power(expected_seed_sd_relative=0.0),
                0.01,
                12,
                3,
            )
        )

    def test_design_plan_writes_pilot_size_plan_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(
                seeds=[1, 2, 3],
                allow_underpowered=False,
                steps=10,
                batch_size=2,
                seq_len=8,
                preregistration_record={
                    "study_id": "smelt-isr-design-test",
                    "power": {
                        "expected_seed_sd_relative": 0.005,
                        "min_effect_relative": 0.01,
                        "target_power": 0.8,
                        "alpha": 0.05,
                        "planned_comparisons": 3,
                        "planned_seeds": 3,
                        "expected_attrition_rate": 0.1,
                        "pilot_calibration": (
                            "upper_confidence_bound"
                        ),
                    },
                },
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
            pilot_size = design["pilot_size_plan"]
            self.assertIsNotNone(pilot_size)
            self.assertEqual(
                pilot_size["schema"],
                "smelt-isr/pilot-size-plan/v1",
            )
            # n_cells 来自计划的处理单元数（A/B 两个单元）。
            self.assertEqual(pilot_size["n_cells"], 2)
            self.assertGreaterEqual(
                pilot_size["recommended_pilot_runs_per_cell"],
                experiment._DESIGN_PILOT_MIN_PAIRS,
            )
            # 落盘的 design_plan.json 必须包含该块（随 plan_id
            # 冻结），且 JSON 安全。
            payload = json.loads(
                Path(design["path"]).read_text(encoding="utf-8")
            )
            self.assertIn("pilot_size_plan", payload)
            self.assertEqual(
                payload["pilot_size_plan"]["n_cells"], 2
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                experiment.print_design_summary(design)
            text = output.getvalue()
            self.assertIn("[design][pilot-size]", text)

    def test_design_plan_omits_pilot_size_plan_without_power(self):
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
            self.assertIsNone(design["pilot_size_plan"])
            payload = json.loads(
                Path(design["path"]).read_text(encoding="utf-8")
            )
            self.assertNotIn("pilot_size_plan", payload)

    # --------------------------------------------------
    # 停止规则 stopping_rules
    # --------------------------------------------------
    def test_design_plan_enforces_stopping_rules_max_total_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = dict(
                seeds=[1, 2, 3],
                allow_underpowered=False,
                steps=10,
                batch_size=2,
                seq_len=8,
                preregistration_record={
                    "study_id": "smelt-isr-design-test",
                    "stopping_rules": {"max_total_runs": 5},
                },
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
            # 3 个种子 × 2 个臂 = 6 次运行，超过声明的 5。
            with self.assertRaises(ValueError) as context:
                experiment.write_design_plan(
                    root, ordered, "randomized", 5,
                    argparse.Namespace(**base),
                )
            self.assertIn("max_total_runs", str(context.exception))
            # 上限足够时冻结成功，并随 plan_id 落盘。
            base["preregistration_record"] = {
                "study_id": "smelt-isr-design-test",
                "stopping_rules": {
                    "max_total_runs": 6,
                    "extension_requires_new_preregistration": True,
                },
            }
            design = experiment.write_design_plan(
                root, ordered, "randomized", 5,
                argparse.Namespace(**base),
            )
            self.assertTrue(
                design["stopping_rules"]["within_max_total_runs"]
            )
            payload = json.loads(
                Path(design["path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["stopping_rules"]["max_total_runs"], 6
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                experiment.print_design_summary(design)
            self.assertIn(
                "[design][stopping]", output.getvalue()
            )

    def test_preregistration_schema_accepts_governance_fields(self):
        payload = _governance_payload(self)
        payload["stopping_rules"] = {
            "max_total_runs": 60,
            "interim_look_steps": [100, 200],
            "extension_requires_new_preregistration": True,
        }
        payload["exclusion_criteria"] = {
            "attrition_definition": (
                "NaN/Inf、数值停止或崩溃的运行不可用"
            ),
            "exclusion_rules": [
                "不因结果方向排除运行",
                "失败目录保留并计入脱落率",
            ],
        }
        payload["min_effect_justification"] = (
            "先前研究的同规模效应量下界"
        )
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
        self.assertEqual(
            record["stopping_rules"]["max_total_runs"], 60
        )
        self.assertEqual(
            record["exclusion_criteria"]["attrition_definition"],
            "NaN/Inf、数值停止或崩溃的运行不可用",
        )
        self.assertEqual(
            record["min_effect_justification"],
            "先前研究的同规模效应量下界",
        )

    def test_preregistration_schema_rejects_malformed_stopping_rules(
        self,
    ):
        payload = _governance_payload(self)
        base_rules = {"max_total_runs": 60}
        payload["stopping_rules"] = dict(
            base_rules, max_total_runs=3
        )
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("max_total_runs", error)
        payload["stopping_rules"] = dict(
            base_rules, max_total_runs=True
        )
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("max_total_runs", error)
        payload["stopping_rules"] = dict(
            base_rules, bogus_key=1
        )
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("bogus_key", error)
        payload["stopping_rules"] = dict(
            base_rules, interim_look_steps=[100, 100]
        )
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("interim_look_steps", error)
        payload["stopping_rules"] = dict(
            base_rules, interim_look_steps=[999]
        )
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("interim_look_steps", error)
        payload["stopping_rules"] = dict(
            base_rules,
            extension_requires_new_preregistration="yes",
        )
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn(
            "extension_requires_new_preregistration", error
        )

    def test_preregistration_schema_rejects_malformed_exclusion_criteria(
        self,
    ):
        payload = _governance_payload(self)
        payload["exclusion_criteria"] = {"note": "only a note"}
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("attrition_definition", error)
        payload["exclusion_criteria"] = {
            "attrition_definition": "NaN/Inf 运行不可用",
            "exclusion_rules": ["", "不因结果方向排除"],
        }
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("exclusion_rules", error)
        payload["exclusion_criteria"] = {
            "attrition_definition": "NaN/Inf 运行不可用",
            "unknown": 1,
        }
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("unknown", error)

    def test_preregistration_schema_rejects_blank_min_effect_justification(
        self,
    ):
        payload = _governance_payload(self)
        payload["min_effect_justification"] = "   "
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("min_effect_justification", error)
        payload["min_effect_justification"] = 0.01
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("min_effect_justification", error)

    # --------------------------------------------------
    # Schema JSON 与示例文件保持交叉一致
    # --------------------------------------------------
    def test_schema_json_declares_governance_fields(self):
        schema = json.loads(
            (REPO_ROOT / "schemas" / "preregistration.v1.json")
            .read_text(encoding="utf-8")
        )
        properties = schema["properties"]
        for key in (
            "stopping_rules",
            "exclusion_criteria",
            "min_effect_justification",
        ):
            self.assertIn(key, properties)
        self.assertEqual(
            properties["stopping_rules"]["required"],
            ["max_total_runs"],
        )
        self.assertFalse(
            properties["stopping_rules"].get(
                "additionalProperties", True
            )
        )
        self.assertEqual(
            properties["exclusion_criteria"]["required"],
            ["attrition_definition"],
        )
        self.assertEqual(
            properties["min_effect_justification"]["type"],
            "string",
        )
        # 可选字段不得进入 required 列表：旧预注册文件不受影响。
        for key in (
            "stopping_rules",
            "exclusion_criteria",
            "min_effect_justification",
        ):
            self.assertNotIn(key, schema["required"])

    def test_example_file_covers_governance_fields(self):
        example = json.loads(
            (REPO_ROOT / "preregistration.example.json")
            .read_text(encoding="utf-8")
        )
        stopping = example["stopping_rules"]
        self.assertIsInstance(stopping["max_total_runs"], int)
        # 示例的运行上限必须覆盖其计划（种子数 × 对照臂数）。
        arms = {
            arm for pair in example["comparisons"] for arm in pair
        }
        planned_runs = len(example["seeds"]) * len(arms)
        self.assertGreaterEqual(
            stopping["max_total_runs"], planned_runs
        )
        exclusion = example["exclusion_criteria"]
        self.assertTrue(
            exclusion["attrition_definition"].strip()
        )
        self.assertIsInstance(
            example["min_effect_justification"], str
        )
        self.assertTrue(
            example["min_effect_justification"].strip()
        )
