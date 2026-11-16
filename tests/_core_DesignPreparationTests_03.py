from _core_support import *
from _core_support import _load_experiment, _tiny_config
from _core_DesignPreparationTests_base import _DesignPreparationTestsBase

REPO_ROOT = Path(__file__).resolve().parents[1]


class DesignPreparationTestsPart4(_DesignPreparationTestsBase):
    """停止规则的序贯 alpha 分配（alpha spending，实验设计与准备）。

    预注册 interim_look_steps 只声明「允许在哪里看」时，中期
    汇总仍是未校正的序贯检验；alpha_spending 把家族 alpha 按
    Lan-DeMets 消耗函数预先分配到各观察点。本组测试覆盖：
    边界计算的数学性质、入口 schema 校验、设计计划冻结
    （参与 plan_id）、按最终名义 alpha 重算先验功效与启动门。
    """

    def _power(self, **overrides):
        power = {
            "expected_seed_sd_relative": 0.005,
            "target_power": 0.8,
            "alpha": 0.05,
            "planned_comparisons": 3,
            "expected_attrition_rate": 0.0,
        }
        power.update(overrides)
        return power

    def _stopping(self, **overrides):
        stopping = {
            "max_total_runs": 12,
            "interim_look_steps": [1000],
            "extension_requires_new_preregistration": True,
            "alpha_spending": "pocock",
        }
        stopping.update(overrides)
        return stopping

    # --------------------------------------------------
    # Lan-DeMets 边界计算的数学性质
    # --------------------------------------------------
    def test_alpha_spending_pocock_boundaries(self):
        plan = experiment.interim_alpha_spending_plan(
            0.05, 4000, [1000, 2000], "pocock"
        )
        self.assertEqual(
            plan["schema"], "smelt-isr/alpha-spending/v1"
        )
        self.assertEqual(plan["scheme"], "pocock")
        self.assertEqual(plan["n_looks"], 3)
        looks = plan["looks"]
        self.assertEqual(
            [row["step"] for row in looks], [1000, 2000, 4000]
        )
        self.assertEqual(
            [row["information_fraction"] for row in looks],
            [0.25, 0.5, 1.0],
        )
        cumulative = [row["cumulative_alpha"] for row in looks]
        for earlier, later in zip(cumulative, cumulative[1:]):
            self.assertLess(earlier, later)
        self.assertAlmostEqual(cumulative[-1], 0.05, places=12)
        # 各观察点名义 alpha 之和等于家族 alpha；最终观察点
        # 的名义 alpha 必然小于原始 alpha（Pocock 均分口径）。
        nominal = [row["nominal_alpha"] for row in looks]
        self.assertAlmostEqual(sum(nominal), 0.05, places=12)
        self.assertLess(nominal[-1], 0.05)
        self.assertEqual(
            plan["effective_final_alpha"], nominal[-1]
        )
        self.assertEqual(
            plan["binding_look_alpha"], min(nominal)
        )
        # z 边界与名义 alpha 一一对应（双侧）。
        from statistics import NormalDist

        for row in looks:
            self.assertAlmostEqual(
                row["z_boundary"],
                NormalDist().inv_cdf(
                    1.0 - row["nominal_alpha"] / 2.0
                ),
                places=10,
            )

    def test_alpha_spending_obrien_fleming_spends_little_early(self):
        pocock = experiment.interim_alpha_spending_plan(
            0.05, 4000, [1000], "pocock"
        )
        obrien = experiment.interim_alpha_spending_plan(
            0.05, 4000, [1000], "obrien_fleming"
        )
        # OF 早期消耗极少，把 alpha 留给最终分析；Pocock 各
        # 观察点接近均分。因此 OF 的最终名义 alpha 更接近
        # 声明 alpha。
        self.assertLess(
            obrien["looks"][0]["cumulative_alpha"],
            pocock["looks"][0]["cumulative_alpha"],
        )
        self.assertGreater(
            obrien["effective_final_alpha"],
            pocock["effective_final_alpha"],
        )
        for plan in (pocock, obrien):
            self.assertLess(
                plan["effective_final_alpha"], 0.05
            )
        # OF 最终 z 边界比固定样本 z_{alpha/2} 只紧一点，
        # Pocock 则紧得多。
        from statistics import NormalDist

        fixed_z = NormalDist().inv_cdf(0.975)
        self.assertGreater(
            obrien["looks"][-1]["z_boundary"], fixed_z
        )
        self.assertLess(
            obrien["looks"][-1]["z_boundary"],
            pocock["looks"][-1]["z_boundary"],
        )

    def test_alpha_spending_plan_rejects_invalid_inputs(self):
        from statistics import NormalDist

        NormalDist()
        with self.assertRaises(ValueError):
            experiment.interim_alpha_spending_plan(
                0.05, 1000, [500], "lan_demets"
            )
        with self.assertRaises(ValueError):
            experiment.interim_alpha_spending_plan(
                0.05, 1000, [], "pocock"
            )
        with self.assertRaises(ValueError):
            experiment.interim_alpha_spending_plan(
                0.05, 1000, [600, 400], "pocock"
            )
        with self.assertRaises(ValueError):
            experiment.interim_alpha_spending_plan(
                0.05, 1000, [500, 500], "pocock"
            )
        # 等于总步数的观察点就是最终分析，重复计入会破坏
        # 信息分数定义。
        with self.assertRaises(ValueError):
            experiment.interim_alpha_spending_plan(
                0.05, 1000, [1000], "pocock"
            )
        with self.assertRaises(ValueError):
            experiment.interim_alpha_spending_plan(
                0.5, 1000, [500], "pocock"
            )
        with self.assertRaises(ValueError):
            experiment.interim_alpha_spending_plan(
                0.05, 0, [500], "pocock"
            )

    # --------------------------------------------------
    # 入口 schema 校验
    # --------------------------------------------------
    def test_schema_accepts_valid_alpha_spending(self):
        payload = self._payload()
        # 先验功效按声明的种子数判定：3 个种子在 sd=0.005、
        # 最小效应 0.01 下不足，会被入口先拒绝；这里用 10 个
        # 种子让功效判定通过，专注核对 alpha_spending 校验。
        payload["seeds"] = list(range(42, 52))
        payload["steps"] = 2000
        payload["power"] = self._power()
        payload["stopping_rules"] = self._stopping()
        self.assertIsNone(
            experiment.preregistration_schema_error(payload)
        )
        payload["stopping_rules"]["alpha_spending"] = (
            "obrien_fleming"
        )
        self.assertIsNone(
            experiment.preregistration_schema_error(payload)
        )
        payload["stopping_rules"]["alpha_spending"] = "none"
        self.assertIsNone(
            experiment.preregistration_schema_error(payload)
        )

    def test_schema_alpha_spending_requires_power_and_steps(self):
        payload = self._payload()
        payload["seeds"] = list(range(42, 52))
        payload["steps"] = 2000
        payload["stopping_rules"] = self._stopping()
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("alpha_spending", error)
        self.assertIn("power", error)

        payload["power"] = self._power()
        del payload["steps"]
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("steps", error)

    def test_schema_alpha_spending_requires_interim_looks(self):
        payload = self._payload()
        payload["seeds"] = list(range(42, 52))
        payload["steps"] = 2000
        payload["power"] = self._power()
        payload["stopping_rules"] = self._stopping(
            interim_look_steps=None
        )
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("interim_look_steps", error)

        payload["stopping_rules"] = self._stopping(
            interim_look_steps=[]
        )
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("interim_look_steps", error)

    def test_schema_alpha_spending_rejects_unordered_and_final_looks(self):
        payload = self._payload()
        payload["seeds"] = list(range(42, 52))
        payload["steps"] = 2000
        payload["power"] = self._power()
        # 观察步必须严格递增：消耗按观察顺序累积。
        payload["stopping_rules"] = self._stopping(
            interim_look_steps=[1500, 500]
        )
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("递增", error)
        # 等于总步数的观察点就是最终分析。
        payload["stopping_rules"] = self._stopping(
            interim_look_steps=[2000]
        )
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("steps", error)
        # 未知方案被枚举拒绝。
        payload["stopping_rules"] = self._stopping(
            alpha_spending="lan_demets"
        )
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn("alpha_spending", error)

    # --------------------------------------------------
    # 设计计划冻结序贯边界（参与 plan_id）
    # --------------------------------------------------
    def _design_args(self, record):
        return argparse.Namespace(
            seeds=[1, 2, 3],
            allow_underpowered=False,
            steps=10,
            batch_size=2,
            seq_len=8,
            preregistration_record=record,
            run_order="randomized",
            run_order_seed=5,
        )

    def _ordered_plan(self):
        plan = experiment.build_sweep_plan(
            seeds=[1, 2, 3],
            modes=["A", "B"],
            ablation=False,
            balance_coef=0.01,
            default_mode="B",
        )
        return experiment.order_sweep_plan(
            plan, "randomized", 5
        )

    def test_design_plan_freezes_alpha_spending_boundaries(self):
        record = {
            "power": experiment.preregistration_power_plan(
                self._power(), 0.01, 3, 3
            ),
            "stopping_rules": self._stopping(
                interim_look_steps=[5]
            ),
            "steps": 10,
        }
        with tempfile.TemporaryDirectory() as tmp:
            design = experiment.write_design_plan(
                Path(tmp),
                self._ordered_plan(),
                "randomized",
                5,
                self._design_args(record),
            )
            stopping = design["stopping_rules"]
            self.assertEqual(stopping["max_total_runs"], 12)
            spending = stopping["alpha_spending"]
            self.assertEqual(spending["scheme"], "pocock")
            self.assertEqual(
                [row["step"] for row in spending["looks"]],
                [5, 10],
            )
            self.assertAlmostEqual(
                spending["alpha"], 0.05, places=12
            )
            self.assertLess(
                spending["effective_final_alpha"], 0.05
            )
            # 序贯边界随设计计划落盘，子进程/审计可以核对。
            written = json.loads(
                Path(design["path"]).read_text(encoding="utf-8")
            )
        self.assertEqual(
            written["stopping_rules"]["alpha_spending"][
                "scheme"
            ],
            "pocock",
        )

    def test_design_plan_alpha_spending_changes_plan_id(self):
        base_record = {
            "power": experiment.preregistration_power_plan(
                self._power(), 0.01, 3, 3
            ),
            "stopping_rules": self._stopping(
                interim_look_steps=[5]
            ),
            "steps": 10,
        }
        of_record = json.loads(json.dumps(base_record))
        of_record["stopping_rules"]["alpha_spending"] = (
            "obrien_fleming"
        )
        none_record = json.loads(json.dumps(base_record))
        none_record["stopping_rules"].pop("alpha_spending")
        ids = []
        echoes = []
        for record in (
            base_record,
            of_record,
            none_record,
        ):
            with tempfile.TemporaryDirectory() as tmp:
                design = experiment.write_design_plan(
                    Path(tmp),
                    self._ordered_plan(),
                    "randomized",
                    5,
                    self._design_args(record),
                )
            ids.append(design["plan_id"])
            echoes.append(
                design["stopping_rules"].get("alpha_spending")
            )
        # 换序贯方案（或去掉声明）都是不同的设计，不能伪装
        # 成同一份计划。
        self.assertEqual(len(set(ids)), 3)
        self.assertEqual(
            [echo["scheme"] for echo in echoes[:2]],
            ["pocock", "obrien_fleming"],
        )
        self.assertIsNone(echoes[2])

    def test_design_plan_without_alpha_spending_has_no_echo(self):
        record = {
            "power": experiment.preregistration_power_plan(
                self._power(), 0.01, 3, 3
            ),
            "stopping_rules": {
                "max_total_runs": 12,
                "interim_look_steps": [5],
                "extension_requires_new_preregistration": True,
            },
            "steps": 10,
        }
        with tempfile.TemporaryDirectory() as tmp:
            design = experiment.write_design_plan(
                Path(tmp),
                self._ordered_plan(),
                "randomized",
                5,
                self._design_args(record),
            )
        self.assertNotIn(
            "alpha_spending", design["stopping_rules"]
        )
        self.assertEqual(
            design["stopping_rules"]["max_total_runs"], 12
        )

    # --------------------------------------------------
    # 按最终名义 alpha 重算先验功效 + 启动门
    # --------------------------------------------------
    def test_design_plan_recomputes_power_at_final_alpha(self):
        record = {
            "power": experiment.preregistration_power_plan(
                self._power(), 0.01, 3, 3
            ),
            "stopping_rules": self._stopping(
                interim_look_steps=[5]
            ),
            "steps": 10,
        }
        with tempfile.TemporaryDirectory() as tmp:
            design = experiment.write_design_plan(
                Path(tmp),
                self._ordered_plan(),
                "randomized",
                5,
                self._design_args(record),
            )
        spending = design["stopping_rules"]["alpha_spending"]
        final_power = spending["power_at_final_look"]
        self.assertEqual(
            final_power["alpha"],
            spending["effective_final_alpha"],
        )
        self.assertEqual(
            final_power["alpha_basis"],
            "alpha_spending_final_look_nominal",
        )
        self.assertEqual(
            final_power["declared_alpha"],
            spending["alpha"],
        )
        # 家族规模与对比类型保持与固定样本口径一致，只收紧
        # alpha：所需种子数不低于固定样本口径。
        self.assertEqual(
            final_power["planned_comparisons"],
            design["power"]["planned_comparisons"],
        )
        self.assertEqual(
            final_power["planned_contrasts"],
            design["power"]["planned_contrasts"],
        )
        self.assertGreaterEqual(
            final_power["required_seeds"],
            design["power"]["required_seeds"],
        )

    def _sequential_design(self, sd, n_seeds, scheme="pocock"):
        fixed_power = experiment.preregistration_power_plan(
            self._power(expected_seed_sd_relative=sd),
            0.01,
            n_seeds,
            1,
        )
        spending = experiment.interim_alpha_spending_plan(
            0.05, 9, [3, 6], scheme
        )
        sequential_power = experiment.preregistration_power_plan(
            dict(
                self._power(expected_seed_sd_relative=sd),
                alpha=spending["effective_final_alpha"],
            ),
            0.01,
            n_seeds,
            1,
        )
        return {
            "power": fixed_power,
            "stopping_rules": {
                "max_total_runs": 12,
                "alpha_spending": dict(
                    spending,
                    power_at_final_look=sequential_power,
                    declared_alpha=0.05,
                ),
            },
        }

    def test_power_gate_blocks_underpowered_sequential_design(self):
        # 噪声大：固定样本口径下计划种子就不足；声明序贯
        # alpha 分配后更不足，启动门必须拦截，且优先报告
        # 序贯口径（不能让固定样本错误掩盖真正的约束）。
        design = self._sequential_design(0.05, 8)
        self.assertFalse(design["power"]["adequate"])
        gate = experiment.design_power_gate_error(design)
        self.assertIsNotNone(gate)
        self.assertIn("序贯", gate)
        self.assertIn("alpha_spending", gate)
        # 豁免路径与固定样本口径一致：显式承认或
        # --allow-underpowered 都放行（豁免随报告落盘）。
        self.assertIsNone(
            experiment.design_power_gate_error(
                design, allow_underpowered=True
            )
        )
        design["power"]["underpowered_acknowledged"] = True
        self.assertIsNone(
            experiment.design_power_gate_error(design)
        )

    def test_power_gate_passes_adequate_sequential_design(self):
        # 噪声小 + OF 消耗函数把大部分 alpha 留给最终分析：
        # 固定样本与序贯口径都充足，启动门不拦截。
        design = self._sequential_design(
            0.002, 8, scheme="obrien_fleming"
        )
        self.assertTrue(design["power"]["adequate"])
        sequential_power = design["stopping_rules"][
            "alpha_spending"
        ]["power_at_final_look"]
        self.assertTrue(sequential_power["adequate"])
        self.assertIsNone(
            experiment.design_power_gate_error(design)
        )

    def test_power_gate_ignores_unavailable_spending_echo(self):
        # 手工编辑/旧版记录可能把序贯回显标记为不可用；
        # 此时启动门退回固定样本口径，不能把「无法核对」
        # 当成「序贯不足」，也不能崩溃。
        design = self._sequential_design(
            0.002, 8, scheme="obrien_fleming"
        )
        design["stopping_rules"]["alpha_spending"] = {
            "scheme": "pocock",
            "available": False,
            "reason": "total_steps_missing",
        }
        self.assertIsNone(
            experiment.design_power_gate_error(design)
        )

    # --------------------------------------------------
    # 设计摘要打印
    # --------------------------------------------------
    def test_design_summary_prints_alpha_spending(self):
        design = self._sequential_design(
            0.002, 8, scheme="obrien_fleming"
        )
        design.update({
            "planned_runs": 6,
            "order_scheme": "randomized",
            "order_seed": 5,
            "plan_id": "a" * 64,
            "path": "design_plan.json",
            "training_budget": {},
        })
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            experiment.print_design_summary(design)
        output = buffer.getvalue()
        self.assertIn("[design][alpha-spending]", output)
        self.assertIn("obrien_fleming", output)
        self.assertIn("名义 alpha", output)

    def test_design_summary_warns_on_unavailable_spending(self):
        design = self._sequential_design(
            0.002, 8, scheme="obrien_fleming"
        )
        design.update({
            "planned_runs": 6,
            "order_scheme": "randomized",
            "order_seed": 5,
            "plan_id": "a" * 64,
            "path": "design_plan.json",
            "training_budget": {},
        })
        design["stopping_rules"]["alpha_spending"] = {
            "scheme": "pocock",
            "available": False,
            "reason": "total_steps_missing",
        }
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            experiment.print_design_summary(design)
        output = buffer.getvalue()
        self.assertIn(
            "[design][alpha-spending][warn]", output
        )
        self.assertIn("total_steps_missing", output)
