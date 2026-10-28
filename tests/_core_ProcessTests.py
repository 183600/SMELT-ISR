from _core_support import *
from _core_support import _load_experiment, _tiny_config

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

    def test_require_pythonhashseed_rejects_non_fixed_values(self):
        previous = experiment._PYTHONHASHSEED_AT_START
        try:
            for value in (None, "", "random", "abc", "-1", "4294967296"):
                experiment._PYTHONHASHSEED_AT_START = value
                self.assertIsNotNone(
                    experiment.pythonhashseed_error(True),
                    value,
                )
            for value in ("0", "42", " 7 "):
                experiment._PYTHONHASHSEED_AT_START = value
                self.assertIsNone(
                    experiment.pythonhashseed_error(True),
                    value,
                )
            experiment._PYTHONHASHSEED_AT_START = ""
            self.assertIsNone(
                experiment.pythonhashseed_error(False)
            )
        finally:
            experiment._PYTHONHASHSEED_AT_START = previous

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
            self.assertEqual(check_same["matching_accesses"], 1)
            self.assertEqual(
                len(check_same["same_candidates"]), 1
            )
            self.assertTrue(
                experiment._test_access_repeat_blocked(check_same)
            )
            self.assertIn(
                "本目录已登记过",
                experiment._repeat_test_access_error(check_same),
            )

    def test_test_ledger_missing_key_is_conservative(self):
        # 账本条目缺少测试集指纹时无法证明它属于别的测试集；
        # 不能因此放行重复解封。
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            candidate = Path(tmp) / "a"
            ledger.write_text(
                json.dumps({"out_dir": str(candidate)}) + "\n",
                encoding="utf-8",
            )
            key = {
                "data_sha256": "d",
                "test_split_sha256": "s",
                "test_fraction": 0.05,
            }
            check_same = experiment.check_test_access_ledger(
                ledger, key, candidate
            )
            self.assertEqual(check_same["matching_accesses"], 1)
            self.assertTrue(
                experiment._test_access_repeat_blocked(check_same)
            )
            check_other = experiment.check_test_access_ledger(
                ledger, key, Path(tmp) / "b"
            )
            self.assertTrue(check_other["repeat_access"])

    def test_test_ledger_repeat_detection_normalizes_path_spelling(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            key = {
                "data_sha256": "d",
                "test_split_sha256": "s",
                "test_fraction": 0.05,
            }
            candidate = Path(tmp) / "a"
            relative_candidate = os.path.relpath(
                candidate, Path.cwd()
            )
            experiment.append_test_access_ledger(
                ledger,
                {**key, "out_dir": relative_candidate},
            )
            check = experiment.check_test_access_ledger(
                ledger, key, candidate.resolve()
            )
            self.assertFalse(check["repeat_access"], check)
            check = experiment.check_test_access_ledger(
                ledger, key, Path(tmp) / "b"
            )
            self.assertTrue(check["repeat_access"], check)

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

    def test_audit_reports_unseen_eval_label_coverage(self):
        # train 只有 A，val 只有 B，test 只有 C：B/C 都是训练集
        # 从未出现的字节类，评估 NLL 含不可约外推误差。
        data = b"A" * 60 + b"B" * 20 + b"C" * 20
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coverage.bin"
            path.write_bytes(data)
            dataset = experiment.ByteDataset(
                str(path),
                seq_len=4,
                val_fraction=0.2,
                test_fraction=0.2,
            )
            report = experiment.audit_data_splits(
                dataset,
                include_test=True,
            )
        coverage = report["label_coverage"]
        self.assertEqual(
            coverage["train"]["observed_byte_classes"], 1
        )
        for split, byte in (("val", ord("B")), ("test", ord("C"))):
            self.assertEqual(
                coverage[split]["unseen_label_classes"], 1
            )
            self.assertEqual(
                coverage[split]["unseen_label_token_share"], 1.0
            )
            self.assertEqual(
                report["splits"][split]["unseen_label_classes"],
                1,
            )
        self.assertTrue(
            any("从未出现" in warning for warning in report["warnings"])
        )

    def test_cleaning_records_deletion_bias(self):
        # train 中夹入一段非法字节，只清洗 train；删除内容与保留
        # 内容的字节/二元组分布完全不同，必须显式记录删除偏差。
        data = (
            b"A" * 25
            + b"\xff" * 5
            + b"A" * 30
            + b"B" * 20
            + b"C" * 20
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bias.bin"
            path.write_bytes(data)
            dataset = experiment.ByteDataset(
                str(path),
                seq_len=4,
                val_fraction=0.2,
                test_fraction=0.2,
                drop_train_corrupt_bytes=True,
            )
            report = experiment.audit_data_splits(
                dataset,
                include_test=True,
            )
        bias = dataset.train_cleaning["deletion_bias"]
        self.assertEqual(bias["dropped_bytes"], 5)
        self.assertEqual(bias["kept_bytes"], 55)
        self.assertEqual(bias["train_bytes"], 60)
        self.assertEqual(bias["drop_fraction"], round(5 / 60, 6))
        self.assertEqual(bias["byte_js"], 1.0)
        self.assertEqual(bias["bigram_js"], 1.0)
        self.assertEqual(
            bias["dropped_top_bytes"][0][0], 255
        )
        self.assertTrue(
            any(
                "删除与保留内容的字节分布" in warning
                for warning in report["warnings"]
            )
        )

    def test_interval_counts_advance_for_tiny_chunk_sizes(self):
        raw = memoryview(bytearray(b"abracadabra"))
        expected_bytes = 11
        expected_bigrams = 10
        for chunk_bytes in (0, -1, 1, 2, 3, 100):
            byte_counts = experiment._interval_byte_counts(
                raw,
                [(0, expected_bytes)],
                chunk_bytes=chunk_bytes,
            )
            bigram_counts = experiment._interval_bigram_counts(
                raw,
                [(0, expected_bytes)],
                chunk_bytes=chunk_bytes,
            )
            self.assertEqual(
                int(byte_counts.sum()), expected_bytes
            )
            self.assertEqual(
                int(bigram_counts.sum()), expected_bigrams
            )

    def test_interval_counts_accept_readonly_memoryview(self):
        raw = memoryview(b"abracadabra")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            byte_counts = experiment._interval_byte_counts(
                raw,
                [(0, 11)],
                chunk_bytes=3,
            )
            bigram_counts = experiment._interval_bigram_counts(
                raw,
                [(0, 11)],
                chunk_bytes=3,
            )
        self.assertEqual(int(byte_counts.sum()), 11)
        self.assertEqual(int(bigram_counts.sum()), 10)

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
