from _core_support import *
from _core_support import _load_experiment, _tiny_config

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

    def test_evaluate_emits_group_stats_with_explicit_separator(self):
        separator = b"<|doc|>"
        payload = (b"abcd" + separator) * 20 + b"wxyz"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.bin"
            path.write_bytes(payload)
            dataset = experiment.ByteDataset(
                str(path),
                seq_len=4,
                val_fraction=0.25,
                test_fraction=0.25,
                group_separator=separator,
            )
            model = experiment.LanguageModel(
                _tiny_config(seq_len=4)
            ).eval()
            group_sink = []
            metrics = experiment.evaluate(
                model,
                dataset,
                8,
                1,
                torch.device("cpu"),
                bootstrap_replicates=20,
                group_sink=group_sink,
            )
        self.assertEqual(len(group_sink), 1, group_sink)
        parsed = experiment._parse_group_stats(
            {"group_stats": group_sink[0]}
        )
        self.assertIsNotNone(parsed, group_sink[0])
        self.assertEqual(
            parsed["unit"], "document_group"
        )
        self.assertEqual(
            metrics["val_group_cluster_available"], 1.0
        )
        self.assertGreaterEqual(
            metrics["val_group_cluster_n_groups"], 2.0
        )

    def test_evaluate_skips_group_stats_without_separator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.bin"
            path.write_bytes(b"abcdefgh" * 16)
            dataset = experiment.ByteDataset(
                str(path),
                seq_len=4,
                val_fraction=0.25,
                test_fraction=0.25,
            )
            model = experiment.LanguageModel(
                _tiny_config(seq_len=4)
            ).eval()
            group_sink = []
            experiment.evaluate(
                model,
                dataset,
                4,
                1,
                torch.device("cpu"),
                group_sink=group_sink,
            )
        self.assertEqual(group_sink, [])

    def test_cell_window_analysis_pairs_group_stats_across_seeds(self):
        def payload(offset):
            windows = [
                {
                    "loss_sum": 40.0 + offset * 10.0,
                    "tokens": 10,
                    "correct": 5,
                },
                {
                    "loss_sum": 50.0 + offset * 10.0,
                    "tokens": 10,
                    "correct": 5,
                },
            ]
            groups = [
                [0, 10.0, 40.0 + offset * 10.0, 5.0],
                [1, 10.0, 50.0 + offset * 10.0, 5.0],
                [2, 10.0, 60.0 + offset * 10.0, 5.0],
            ]
            return {
                "split": "val",
                "data_sha256": "deadbeef",
                "split_sha256": "cafebabe",
                "seq_len": 8,
                "batch_size": 2,
                "batches": 1,
                "n_windows": 2,
                "overlap_fraction": 0.0,
                "windows": windows,
                "group_stats": self._stats(groups),
            }

        cell_a = {
            "window_payloads": [
                {"seed": seed, "payload": payload(0.0)}
                for seed in (1, 2, 3)
            ]
        }
        cell_b = {
            "window_payloads": [
                {"seed": seed, "payload": payload(-0.5)}
                for seed in (1, 2, 3)
            ]
        }
        report = experiment._cell_window_paired_analysis(
            cell_a, cell_b
        )
        self.assertTrue(report["available"], report)
        self.assertEqual(report["n_group_paired_seeds"], 3)
        hierarchical = report[
            "hierarchical_group_bootstrap_ci95"
        ]
        self.assertTrue(hierarchical["available"], hierarchical)
        self.assertEqual(hierarchical["n_seeds"], 3)
        self.assertEqual(hierarchical["n_groups"], 3)
        self.assertAlmostEqual(
            hierarchical["low"], -0.5, places=12
        )
        self.assertAlmostEqual(
            hierarchical["high"], -0.5, places=12
        )
        for entry in report["per_seed"]:
            group_cluster = entry.get("group_cluster")
            self.assertIsInstance(group_cluster, dict)
            self.assertTrue(group_cluster["available"])
            self.assertNotIn(
                "group_differences", group_cluster
            )
        display = experiment._format_window_paired(
            {"window_paired": report}
        )
        self.assertIn("种子×文档群组 bootstrap", display)
