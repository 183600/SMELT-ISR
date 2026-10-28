from _core_support import *
from _core_support import _load_experiment, _tiny_config

class NearDuplicateDataTests(unittest.TestCase):
    """近重复数据审计/清洗：候选生成、精确复核与 train-only 边界。"""

    @staticmethod
    def _printable(tag: str, size: int) -> bytes:
        raw = hashlib.sha256(tag.encode("utf-8")).digest()
        out = bytearray()
        counter = 0
        while len(out) < size:
            raw = hashlib.sha256(
                raw + str(counter).encode("ascii")
            ).digest()
            out.extend(32 + (value % 95) for value in raw)
            counter += 1
        return bytes(out[:size])

    @classmethod
    def _near_pair(cls, size: int = 640):
        core = (
            b"alpha beta gamma delta epsilon zeta eta theta "
            * 12
        )[: size - 64]
        filler = cls._printable("near-filler", size - len(core))
        base = core + filler
        variant = base.replace(b"alpha", b"omega", 1)
        return base, variant

    def test_bottom_k_sketch_uses_distinct_shingle_hashes(self):
        # 重复 shingle 的哈希不能占满 bottom-k 草图：MinHash
        # 的集合口径要求每个值只计一次，否则高度重复内容的
        # 草图会丢掉其他真实值并造成近重复候选漏检。
        self.assertEqual(
            experiment._near_duplicate_bottom_k(
                [5] * 20, 8
            ),
            (5,),
        )
        self.assertEqual(
            experiment._near_duplicate_bottom_k(
                [9, 3, 9, 3, 7, 7, 1], 3
            ),
            (1, 3, 7),
        )

    def test_planner_drops_near_duplicate_train_group(self):
        base, variant = self._near_pair()
        separator = b"\n\n"
        unrelated = [
            self._printable(f"doc-{index}", 640)
            for index in range(2)
        ]
        raw = bytearray(
            separator.join([base, *unrelated, variant])
        )
        train_end = len(base) + len(unrelated[0]) + 2 * len(
            separator
        )
        eval_ranges = {"val": (train_end, len(raw))}

        excluded, stats = (
            experiment._plan_train_near_duplicate_cleaning(
                raw,
                separator,
                train_end,
                eval_ranges,
                [],
                threshold=0.8,
                shingle_bytes=8,
                apply_cleaning=True,
            )
        )

        self.assertEqual(excluded, [(0, len(base))])
        self.assertEqual(stats["groups_dropped"], 1)
        self.assertEqual(stats["groups_flagged"], 1)
        self.assertEqual(
            stats["flagged_duplicate_of"],
            {"val": 1, "test": 0, "train": 0},
        )
        self.assertGreaterEqual(
            stats["examples"][0][3], 0.8
        )
        self.assertLessEqual(
            stats["examples"][0][3], 1.0
        )

        # 只读审计报告同一匹配，但不返回删除区间。
        audit_excluded, audit_stats = (
            experiment._plan_train_near_duplicate_cleaning(
                raw,
                separator,
                train_end,
                eval_ranges,
                [],
                threshold=0.8,
                shingle_bytes=8,
                apply_cleaning=False,
            )
        )
        self.assertEqual(audit_excluded, [])
        self.assertEqual(audit_stats["groups_dropped"], 0)
        self.assertEqual(audit_stats["groups_flagged"], 1)
        self.assertGreater(
            audit_stats["projected_train_bytes_if_flagged_dropped"],
            0,
        )

    def test_planner_rejects_invalid_parameters(self):
        raw = bytearray(b"abc" * 100)
        for kwargs in (
            {"threshold": 0.0},
            {"threshold": 1.5},
            {"shingle_bytes": 0},
            {"shingle_bytes": 65},
            {"sketch_size": 1},
        ):
            with self.assertRaises(ValueError):
                experiment._plan_train_near_duplicate_cleaning(
                    raw,
                    b"\n\n",
                    len(raw),
                    {"val": (0, 0)},
                    [],
                    **kwargs,
                )

    def test_dataset_cleaning_only_changes_train(self):
        base, variant = self._near_pair()
        groups = [
            self._printable("u1", 640),
            base,
            self._printable("u2", 640),
            self._printable("u3", 640),
            variant,
            self._printable("v2", 640),
            self._printable("t1", 640),
            self._printable("t2", 640),
        ]
        payload = b"\n\n".join(groups)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "near.txt"
            path.write_bytes(payload)

            plain = experiment.ByteDataset(
                str(path),
                seq_len=8,
                val_fraction=0.25,
                test_fraction=0.25,
                group_separator=b"\n\n",
            )
            cleaned = experiment.ByteDataset(
                str(path),
                seq_len=8,
                val_fraction=0.25,
                test_fraction=0.25,
                group_separator=b"\n\n",
                drop_train_near_duplicate_groups=True,
                near_duplicate_threshold=0.8,
                near_duplicate_shingle_bytes=8,
            )

        self.assertEqual(
            plain.split_bounds, cleaned.split_bounds
        )
        self.assertEqual(
            plain.split_sha256, cleaned.split_sha256
        )
        self.assertEqual(
            cleaned.train_excluded_spans, [(642, 1282)]
        )
        self.assertEqual(
            cleaned.train_cleaning["cleaning_kinds"],
            ["near_duplicate"],
        )
        self.assertEqual(
            cleaned.train_cleaning["near_duplicates"][
                "flagged_duplicate_of"
            ],
            {"val": 1, "test": 0, "train": 0},
        )
        self.assertEqual(
            cleaned.effective_train_bytes,
            plain.effective_train_bytes - 640,
        )

    def test_audit_does_not_delete_and_is_deterministic(self):
        base, variant = self._near_pair()
        groups = [
            self._printable("u1", 640),
            base,
            self._printable("u2", 640),
            self._printable("u3", 640),
            variant,
            self._printable("v2", 640),
            self._printable("t1", 640),
            self._printable("t2", 640),
        ]
        payload = b"\n\n".join(groups)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.txt"
            path.write_bytes(payload)
            dataset = experiment.ByteDataset(
                str(path),
                seq_len=8,
                val_fraction=0.25,
                test_fraction=0.25,
                group_separator=b"\n\n",
                near_duplicate_audit=True,
            )
            audit_one = experiment.audit_data_splits(dataset)
            audit_two = experiment.audit_data_splits(dataset)

        self.assertEqual(dataset.train_excluded_spans, [])
        self.assertEqual(
            audit_one["near_duplicates"]["groups_flagged"], 1
        )
        self.assertEqual(
            audit_one["near_duplicates"],
            audit_two["near_duplicates"],
        )
        self.assertTrue(
            any(
                "近重复" in warning
                for warning in audit_one["warnings"]
            )
        )

    def test_test_content_is_not_used_for_near_dedup_by_default(self):
        base, variant = self._near_pair()
        groups = [
            base,
            self._printable("u1", 640),
            self._printable("u2", 640),
            self._printable("u3", 640),
            self._printable("v1", 640),
            self._printable("v2", 640),
            variant,
            self._printable("t2", 640),
        ]
        payload = b"\n\n".join(groups)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sealed.txt"
            path.write_bytes(payload)
            sealed = experiment.ByteDataset(
                str(path),
                seq_len=8,
                val_fraction=0.25,
                test_fraction=0.25,
                group_separator=b"\n\n",
                drop_train_near_duplicate_groups=True,
            )
            leaky = experiment.ByteDataset(
                str(path),
                seq_len=8,
                val_fraction=0.25,
                test_fraction=0.25,
                group_separator=b"\n\n",
                drop_train_near_duplicate_groups=True,
                clean_train_against_test=True,
            )

        self.assertEqual(sealed.train_excluded_spans, [])
        self.assertFalse(
            sealed.train_cleaning["test_used_for_cleaning"]
        )
        self.assertEqual(
            leaky.train_cleaning["cleaning_eval_splits"],
            ["test", "val"],
        )
        self.assertTrue(
            leaky.train_cleaning["test_used_for_cleaning"]
        )
        self.assertEqual(
            leaky.train_excluded_spans, [(0, 640)]
        )

    @staticmethod
    def _mutate_byte(content: bytes, position: int) -> bytes:
        """把一个可打印字节改成另一个可打印字节，制造近重复。"""
        value = content[position]
        replacement = 32 + ((value - 32 + 1) % 95)
        return (
            content[:position]
            + bytes([replacement])
            + content[position + 1:]
        )

    def test_eval_near_duplicate_audit_reports_effective_groups(self):
        separator = b"\n\n"
        val_base = self._printable("eval-val-base", 640)
        val_variant = self._mutate_byte(val_base, 300)
        val_other = self._printable("eval-val-other", 640)
        test_base = self._printable("eval-test-base", 640)
        test_variant = self._mutate_byte(test_base, 320)
        test_other = self._printable("eval-test-other", 640)

        val_raw = separator.join(
            [val_base, val_variant, val_other]
        )
        test_raw = separator.join(
            [test_base, test_variant, test_other]
        )
        raw = bytearray(val_raw + separator + test_raw)
        val_range = (0, len(val_raw))
        test_range = (
            len(val_raw) + len(separator),
            len(raw),
        )

        report = experiment._eval_near_duplicate_audit(
            raw,
            separator,
            {"val": val_range, "test": test_range},
            threshold=0.8,
            shingle_bytes=8,
        )

        self.assertEqual(
            report["schema"],
            "smelt-isr/eval-near-duplicate/v1",
        )
        self.assertEqual(report["splits"], ["val", "test"])
        self.assertTrue(report["test_content_used"])
        self.assertTrue(report["exact_verification"])

        val_info = report["per_split"]["val"]
        self.assertEqual(val_info["groups_total"], 3)
        self.assertEqual(val_info["groups_scanned"], 3)
        self.assertEqual(val_info["within_pairs"], 1)
        self.assertEqual(val_info["clusters"], 2)
        self.assertEqual(val_info["effective_groups"], 2)
        self.assertEqual(val_info["redundant_groups"], 1)
        self.assertEqual(val_info["largest_cluster_groups"], 2)
        self.assertEqual(val_info["effective_fraction"], 2 / 3)
        self.assertTrue(
            val_info["examples"][0][4] >= 0.8
        )

        test_info = report["per_split"]["test"]
        self.assertEqual(test_info["groups_scanned"], 3)
        self.assertEqual(test_info["within_pairs"], 1)
        self.assertEqual(test_info["effective_groups"], 2)

        # 两组近重复内容分别来自不同随机基串，跨 split 不应
        # 被误报为近重复。
        self.assertEqual(
            report["cross_split"]["val_test"]["pairs"], 0
        )

    def test_eval_near_duplicate_audit_detects_cross_split_only_with_test(self):
        separator = b"\n\n"
        val_base = self._printable("cross-val-base", 640)
        test_variant = self._mutate_byte(val_base, 200)
        raw = bytearray(
            val_base + separator + test_variant
        )
        val_range = (0, len(val_base))
        test_range = (
            len(val_base) + len(separator),
            len(raw),
        )

        val_only = experiment._eval_near_duplicate_audit(
            raw,
            separator,
            {"val": val_range},
            threshold=0.8,
            shingle_bytes=8,
        )
        self.assertFalse(val_only["test_content_used"])
        self.assertEqual(val_only["splits"], ["val"])
        self.assertEqual(
            val_only["cross_split"], {}
        )

        with_test = experiment._eval_near_duplicate_audit(
            raw,
            separator,
            {"val": val_range, "test": test_range},
            threshold=0.8,
            shingle_bytes=8,
        )
        self.assertTrue(with_test["test_content_used"])
        self.assertGreaterEqual(
            with_test["cross_split"]["val_test"]["pairs"], 1
        )
        self.assertEqual(
            with_test["per_split"]["val"]["within_pairs"], 0
        )
        self.assertEqual(
            with_test["per_split"]["test"]["within_pairs"], 0
        )

    def test_eval_near_duplicate_audit_is_read_only_and_deterministic(self):
        separator = b"\n\n"
        base = self._printable("audit-base", 640)
        variant = self._mutate_byte(base, 100)
        groups = [
            *[
                self._printable(f"audit-train-{index}", 640)
                for index in range(6)
            ],
            base,
            variant,
            self._printable("audit-val-other", 640),
            self._mutate_byte(base, 300),
            self._mutate_byte(variant, 400),
            self._printable("audit-test-other", 640),
        ]
        payload = separator.join(groups)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval-near.txt"
            path.write_bytes(payload)
            dataset = experiment.ByteDataset(
                str(path),
                seq_len=8,
                val_fraction=0.25,
                test_fraction=0.25,
                group_separator=separator,
                near_duplicate_audit=True,
                near_duplicate_threshold=0.8,
                near_duplicate_shingle_bytes=8,
            )
            sealed_one = experiment.audit_data_splits(dataset)
            sealed_two = experiment.audit_data_splits(dataset)
            unsealed = experiment.audit_data_splits(
                dataset,
                include_test=True,
            )
            excluded_after = list(dataset.train_excluded_spans)

        self.assertEqual(excluded_after, [])
        self.assertEqual(sealed_one, sealed_two)
        self.assertIsNotNone(
            sealed_one["evaluation_near_duplicates"]
        )
        self.assertEqual(
            sealed_one["evaluation_near_duplicates"]["splits"],
            ["val"],
        )
        self.assertFalse(
            sealed_one["evaluation_near_duplicates"][
                "test_content_used"
            ]
        )
        self.assertNotIn(
            "test",
            sealed_one["evaluation_near_duplicates"][
                "per_split"
            ],
        )
        self.assertEqual(
            sealed_one["evaluation_near_duplicates"][
                "cross_split"
            ],
            {},
        )
        self.assertTrue(
            any(
                "评估集内部" in warning
                for warning in sealed_one["warnings"]
            )
        )

        self.assertEqual(
            unsealed["evaluation_near_duplicates"]["splits"],
            ["val", "test"],
        )
        self.assertTrue(
            unsealed["evaluation_near_duplicates"][
                "test_content_used"
            ]
        )
        self.assertIn(
            "test",
            unsealed["evaluation_near_duplicates"][
                "per_split"
            ],
        )
        self.assertGreaterEqual(
            unsealed["evaluation_near_duplicates"][
                "cross_split"
            ]["val_test"]["pairs"],
            1,
        )
        self.assertTrue(
            any(
                "跨 split 近重复评估群组" in warning
                for warning in unsealed["warnings"]
            )
        )

    def test_eval_near_duplicate_audit_rejects_invalid_parameters(self):
        raw = bytearray(b"abc" * 100)
        for kwargs in (
            {"threshold": 0.0},
            {"threshold": 1.5},
            {"shingle_bytes": 0},
            {"shingle_bytes": 65},
            {"sketch_size": 1},
        ):
            with self.assertRaises(ValueError):
                experiment._eval_near_duplicate_audit(
                    raw,
                    b"\n\n",
                    {"val": (0, len(raw))},
                    **kwargs,
                )
        with self.assertRaises(ValueError):
            experiment._eval_near_duplicate_audit(
                raw,
                b"",
                {"val": (0, len(raw))},
            )
