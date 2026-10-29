from _core_support import *
from _core_support import _load_experiment, _tiny_config
class EdgeCaseRegressionTestsPart2(unittest.TestCase):
    def test_tost_power_large_noncentrality_small_df(self):
        # df<3、δ 很大时，非中心 t CDF 的固定 u 网格会错过
        # 条件正态尾的窄转折；改为对正态位移积分卡方尾概率。
        for df, expected in (
            (2.0, 0.3678579568413044),
            (0.5, 0.25619706631310957),
        ):
            power = experiment._tost_power(
                1.0, 1.0e-5, df, 1.0, 0.05
            )
            self.assertIsNotNone(power)
            self.assertAlmostEqual(
                power,
                expected,
                delta=expected * 1.0e-9,
            )
        # df>=2 时两个非中心 t CDF 都接近 1，直接相减会把
        # 极小的 TOST 功效抵消成 0；正态混合积分保留该尾概率。
        expected = 9.814390514141384e-23
        power = experiment._tost_power(
            -1.0e-3, 1.0e-4, 10.0, 1.0e-3, 1.0e-6
        )
        self.assertIsNotNone(power)
        self.assertAlmostEqual(
            power, expected, delta=expected * 1.0e-9
        )
    def test_tost_power_large_df_matches_normal_limit(self):
        # df 很大时 chi 密度宽度仍约为 1，而 u 轴长度约为
        # sqrt(df)；_tost_power_normal_mix 的固定节点上限会跨过
        # 整段密度，把很低的功效误报为 1。参考值由 SciPy 的
        # t 临界值与正态 CDF 差独立计算。
        for df, margin, effect, expected in (
            (1.0e6, 5.0, 1.0, 0.9907356077143294),
            (1.0e6, 2.0, 0.5, 0.2461771369824705),
            (1.0e9, 5.0, 1.0, 0.9907356456869842),
            (1.0e9, 2.0, 0.5, 0.24617815924233505),
        ):
            power = experiment._tost_power(
                effect, 1.0, df, margin, 0.05
            )
            self.assertIsNotNone(power)
            self.assertAlmostEqual(
                power, expected, delta=1.0e-8
            )
    def test_tost_power_treats_infinite_df_as_normal_limit(self):
        # df=inf 与 _t_power 的大 df 分支一致，应退化为正态。
        for effect, margin, expected in (
            (0.0, 5.0, 0.9992067699909475),
            (0.5, 2.0, 0.246178160265617),
            (1.0, 5.0, 0.9907356457249948),
        ):
            power = experiment._tost_power(
                effect,
                1.0,
                float("inf"),
                margin,
                0.05,
            )
            self.assertIsNotNone(power)
            self.assertAlmostEqual(power, expected, delta=1.0e-14)
        margin = experiment._tost_margin_for_power(
            1.0,
            float("inf"),
            0.8,
            0.05,
        )
        self.assertIsNotNone(margin)
        self.assertAlmostEqual(
            margin,
            2.9264051924960726,
            delta=1.0e-12,
        )
        self.assertIsNone(
            experiment._tost_power(
                0.0, 1.0, float("nan"), 5.0, 0.05
            )
        )
        self.assertIsNone(
            experiment._tost_power(
                0.0, 1.0, float("-inf"), 5.0, 0.05
            )
        )
        self.assertIsNone(
            experiment._tost_margin_for_power(
                1.0, float("nan"), 0.8, 0.05
            )
        )
    def test_tost_test_treats_infinite_df_as_normal_limit(self):
        # _tost_test 与 _tost_power/_t_cdf 同一口径：df=inf
        # 应退化为正态极限，而不是返回缺失结果。
        result = experiment._tost_test(
            0.0, 1.0, float("inf"), 2.0, 0.05
        )
        expected = 0.5 * math.erfc(2.0 / math.sqrt(2.0))
        self.assertAlmostEqual(
            float(result["tost_p_upper"]),
            expected,
            delta=1.0e-15,
        )
        self.assertAlmostEqual(
            float(result["tost_p_lower"]),
            expected,
            delta=1.0e-15,
        )
        self.assertAlmostEqual(
            float(result["tost_p_value"]),
            expected,
            delta=1.0e-15,
        )
        self.assertEqual(
            experiment._tost_test(
                0.0, 1.0, float("nan"), 2.0, 0.05
            ),
            {},
        )
        self.assertEqual(
            experiment._tost_test(
                0.0, 1.0, float("-inf"), 2.0, 0.05
            ),
            {},
        )
    def test_power_helpers_reject_negative_standard_error(self):
        # 负标准误不是“零方差”的另一种写法；不能把它当成
        # 精确已知的零噪声，从而给出功效 1 或所需种子数 1。
        self.assertIsNone(
            experiment._t_power(1.0, -1.0, 10.0)
        )
        self.assertIsNone(
            experiment._tost_power(0.0, -1.0, 10.0, 2.0)
        )
        self.assertEqual(
            experiment._tost_test(
                0.0, -1.0, 10.0, 2.0
            ),
            {},
        )
        self.assertIsNone(
            experiment._minimum_detectable_effect(-1.0, 10.0)
        )
        self.assertIsNone(
            experiment._seeds_to_detect(1.0, -1.0)
        )
        self.assertIsNone(
            experiment._seeds_for_tost(-1.0, 2.0)
        )
        self.assertIsNone(
            experiment._tost_margin_for_power(-1.0, 10.0)
        )
    def test_tost_power_handles_overflowing_noncentrality(self):
        # se 小到 effect/se 溢出为 inf 时，δ→∞ 的极限仍可
        # 由卡方尾概率精确表示，不能误报为 0 功效。
        tiny_se = 5.0e-324
        self.assertAlmostEqual(
            experiment._tost_power(
                0.005, tiny_se, 10.0, 0.01, 0.05
            ),
            0.9908757207816047,
            places=12,
        )
        self.assertAlmostEqual(
            experiment._tost_power(
                0.02, tiny_se, 10.0, 0.01, 0.05
            ),
            1.694474393006736e-05,
            delta=1e-17,
        )
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
        # alpha 极小所需界远超 2^64；必须继续扩展且返回的界
        # 确实达到目标功效。
        for alpha in (1.0e-30, 1.0e-100, 1.0e-300):
            margin = experiment._tost_margin_for_power(
                1.0, 5.0, 0.8, alpha
            )
            self.assertIsNotNone(margin)
            self.assertGreaterEqual(
                experiment._tost_power(
                    0.0, 1.0, 5.0, margin, alpha
                ),
                0.8,
            )
    def test_welch_test_preserves_tiny_scale(self):
        base_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        base_b = [1.1, 2.1, 3.1, 4.1, 5.1]
        normal = experiment._welch_t_test(base_a, base_b)
        # 1e-80 落在旧缩放阈值（1e-150）之上，但 vari/n 的平方
        # 已进入次正规数范围；必须仍与正常量级给出同一 df。
        for tiny_scale in (1.0e-80, 1.0e-100, 1.0e-300):
            tiny = experiment._welch_t_test(
                [value * tiny_scale for value in base_a],
                [value * tiny_scale for value in base_b],
            )
            self.assertEqual(tiny["test"], "welch_t")
            self.assertAlmostEqual(
                tiny["t"], normal["t"], places=12
            )
            self.assertAlmostEqual(
                tiny["p_value"], normal["p_value"], places=12
            )
            self.assertAlmostEqual(
                tiny["df"], normal["df"], places=12
            )
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
    def test_t_power_handles_overflowing_noncentrality(self):
        # se 小到 effect/se 溢出为 inf 时，非中心参数趋于无穷，
        # 功效应为 1；不能把 inf 带进正态混合积分而得到 0。
        for df in (0.5, 2.0, 10.0, float("inf")):
            self.assertEqual(
                experiment._t_power(1.0, 5.0e-324, df),
                1.0,
            )
    def test_t_power_handles_finite_huge_noncentrality(self):
        # effect/se 仍可表示为有限 float，但在正态混合积分中
        # log_x=2*log_u-log(2) 已超出 math.exp 的范围；这里
        # 必须直接取 chi-square CDF 的极限 1，不能抛
        # OverflowError 中断功效/等价性报告。
        for df in (0.1, 0.5, 1.0):
            for se in (1.0e-200, 1.0e-300):
                self.assertAlmostEqual(
                    experiment._t_power(1.0, se, df),
                    1.0,
                    delta=1.0e-12,
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
    def test_structural_config_integers_are_type_checked(self):
        for field in (
            "vocab_size",
            "seq_len",
            "d_model",
            "n_heads",
            "n_layers",
            "n_experts",
            "expert_hidden",
            "top_k",
            "k1",
            "k2",
            "n_kv_heads",
        ):
            for value in (True, 3.5, "4"):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        _tiny_config(**{field: value}).validate()
        _tiny_config(n_kv_heads=None).validate()
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
