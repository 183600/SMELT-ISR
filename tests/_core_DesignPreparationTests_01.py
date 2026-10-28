from _core_support import *
from _core_support import _load_experiment, _tiny_config
from _core_DesignPreparationTests_base import _DesignPreparationTestsBase

class DesignPreparationTestsPart2(_DesignPreparationTestsBase):
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
    def test_chi_square_large_df_handles_transition_and_overflow(self):
        a = 5.0e11
        expected = 0.5 + 1.0 / (
            3.0 * math.sqrt(2.0 * math.pi * a)
        )
        cdf = experiment._chi_square_cdf(1.0e12, 1.0e12)
        survival = experiment._chi_square_survival(
            1.0e12, 1.0e12
        )
        self.assertIsNotNone(cdf)
        self.assertIsNotNone(survival)
        self.assertAlmostEqual(cdf, expected, places=14)
        self.assertAlmostEqual(cdf + survival, 1.0, places=15)

        huge_cdf = experiment._chi_square_cdf(1.0e20, 1.0e20)
        huge_survival = experiment._chi_square_survival(
            1.0e20, 1.0e20
        )
        self.assertIsNotNone(huge_cdf)
        self.assertIsNotNone(huge_survival)
        self.assertAlmostEqual(
            huge_cdf + huge_survival, 1.0, places=15
        )
        self.assertIsNotNone(
            experiment._chi_square_cdf(1.0e308, 1.0e308)
        )
        self.assertIsNotNone(
            experiment._chi_square_quantile(0.5, 1.0e12)
        )
    def test_chi_square_and_gamma_reject_huge_integer_inputs(self):
        huge = 10 ** 400
        self.assertIsNone(
            experiment._regularized_incomplete_gamma_parts(
                huge, 1.0
            )
        )
        self.assertIsNone(
            experiment._regularized_incomplete_gamma_parts(
                1.0, huge
            )
        )
        self.assertIsNone(
            experiment._chi_square_cdf(1.0, huge)
        )
        self.assertIsNone(
            experiment._chi_square_survival(huge, 1.0)
        )
        self.assertIsNone(
            experiment._chi_square_quantile(0.5, huge)
        )
        self.assertIsNone(
            experiment._sd_upper_confidence_bound(
                1.0, huge, 0.95
            )
        )
        self.assertIsNone(
            experiment._binomial_upper_confidence_bound(
                1, huge, 0.95
            )
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
