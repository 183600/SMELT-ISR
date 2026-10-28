from _core_support import *
from _core_support import _load_experiment, _tiny_config

class RunAuditTests(unittest.TestCase):
    """运行级复现审计：跨文件身份必须一致，证据不足要显式披露。"""

    def _build_run(
        self,
        root,
        *,
        metadata_overrides=None,
        config_overrides=None,
    ):
        root.mkdir()
        snapshot = experiment.build_environment_snapshot()
        environment_bytes = json.dumps(
            snapshot, ensure_ascii=False, indent=2
        ).encode("utf-8")
        (root / "environment.json").write_bytes(environment_bytes)
        environment_sha = hashlib.sha256(
            environment_bytes
        ).hexdigest()
        identity = {
            "seed": 7,
            "data_sha256": "b" * 64,
            "data_split_sha256": {
                "train": "1" * 64,
                "val": "2" * 64,
                "test": "3" * 64,
            },
            "script_sha256": "d" * 64,
            "environment_id": snapshot["environment_id"],
            "comparison_protocol_sha256": "c" * 64,
        }
        metadata = {
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "command": (
                "python experiment.py --data corpus.txt --out run"
            ),
            "cwd": "/project",
            "git_commit": "a" * 40,
            "git_dirty": False,
            "git_diff_sha256": None,
            "script_sha256": identity["script_sha256"],
            "environment_id": identity["environment_id"],
            "pythonhashseed_at_start": "7",
            "rng_seed_manifest": {"seed": 7},
            "deterministic_algorithms": True,
            "deterministic_algorithms_warn_only": False,
            "cudnn_deterministic": True,
            "allow_tf32_matmul": False,
            "allow_tf32_cudnn": False,
        }
        if metadata_overrides:
            metadata.update(metadata_overrides)
        config = {
            "config_schema": (
                experiment._ENVIRONMENT_CONFIG_SCHEMA
            ),
            "args": {"seed": 7, "mode": "smelt"},
            "data": {
                "sha256": identity["data_sha256"],
                "split_sha256": identity["data_split_sha256"],
                "effective_train_sha256": "4" * 64,
            },
            "comparison_protocol_sha256": (
                identity["comparison_protocol_sha256"]
            ),
            "run_metadata": metadata,
            "environment": {
                "schema": experiment._ENVIRONMENT_SCHEMA,
                "environment_id": identity["environment_id"],
                "file": "environment.json",
                "file_sha256": environment_sha,
            },
            "sanity_check": {
                "reproducibility": {
                    "repeats": 2,
                    "identical": True,
                    "mismatches": [],
                }
            },
        }
        if config_overrides:
            config.update(config_overrides)
        (root / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        records = [
            {"step": "reference_baselines", **identity},
            {"step": 1, "val_loss": 2.0},
            {
                "step": "final",
                **identity,
                "final_loss": 1.0,
            },
        ]
        (root / "metrics.jsonl").write_text(
            "".join(
                json.dumps(record) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        return config, identity, metadata

    def _codes(self, report, key):
        return {
            item["code"]
            for item in report[key]
            if isinstance(item, dict)
        }

    def test_audit_accepts_consistent_run_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(root)
            report = experiment.audit_run_directory(root)
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["errors"], [])
            self.assertIn(
                "manifest_missing",
                self._codes(report, "warnings"),
            )

            experiment.write_artifact_manifest(root)
            strict = experiment.audit_run_directory(
                root, strict=True
            )
            self.assertTrue(strict["ok"], strict)
            self.assertEqual(strict["errors"], [])
            self.assertEqual(strict["warnings"], [])

    def test_audit_accepts_complete_final_line_without_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(root)
            metrics_path = root / "metrics.jsonl"
            raw = metrics_path.read_bytes()
            self.assertTrue(raw.endswith(b"\n"))
            metrics_path.write_bytes(raw[:-1])
            experiment.write_artifact_manifest(root)
            report = experiment.audit_run_directory(
                root, strict=True
            )
            self.assertTrue(report["ok"], report)
            self.assertNotIn(
                "metrics_trailing_partial",
                self._codes(report, "warnings"),
            )

    def test_audit_still_flags_incomplete_final_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            path.write_bytes(b'{"step": 1}\n{"step": 2')
            audit = experiment._audit_metrics_log(path)
            self.assertTrue(audit["trailing_partial"])
            self.assertEqual(audit["malformed_lines"], [2])

    def test_audit_does_not_require_full_identity_on_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(root)
            metrics_path = root / "metrics.jsonl"
            records = [
                json.loads(line)
                for line in metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            records[-1].pop("environment_id")
            records[-1].pop("comparison_protocol_sha256")
            metrics_path.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            experiment.write_artifact_manifest(root)
            report = experiment.audit_run_directory(
                root, strict=True
            )
            self.assertTrue(report["ok"], report)

    def test_audit_detects_metrics_from_another_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(root)
            metrics_path = root / "metrics.jsonl"
            records = [
                json.loads(line)
                for line in metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            records[0]["script_sha256"] = "e" * 64
            metrics_path.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            self.assertFalse(report["ok"])
            self.assertIn(
                "record_identity_mismatch",
                self._codes(report, "errors"),
            )

    def test_audit_detects_duplicate_or_misplaced_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(root)
            metrics_path = root / "metrics.jsonl"
            records = [
                json.loads(line)
                for line in metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            metrics_path.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in records + [records[0]]
                ),
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            self.assertFalse(report["ok"])
            self.assertIn(
                "multiple_reference_baselines",
                self._codes(report, "errors"),
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(root)
            metrics_path = root / "metrics.jsonl"
            records = [
                json.loads(line)
                for line in metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            metrics_path.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in [records[1], records[0]]
                ),
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            self.assertFalse(report["ok"])
            self.assertIn(
                "metrics_header_not_first",
                self._codes(report, "errors"),
            )

    def test_audit_discloses_missing_test_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            config, identity, _ = self._build_run(root)
            metrics_path = root / "metrics.jsonl"
            records = [
                json.loads(line)
                for line in metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            records.append(
                {"step": "test", **identity, "test_loss": 1.0}
            )
            metrics_path.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            warning_codes = self._codes(report, "warnings")
            self.assertIn(
                "test_ledger_missing", warning_codes
            )
            self.assertNotIn(
                "test_ledger_entry_missing", warning_codes
            )

            local_ledger = root / "test_access_ledger.jsonl"
            local_ledger.write_text(
                json.dumps(
                    {"out_dir": str(root.resolve())}
                )
                + "\n",
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            warning_codes = self._codes(report, "warnings")
            self.assertNotIn(
                "test_ledger_missing", warning_codes
            )
            self.assertNotIn(
                "test_ledger_entry_missing", warning_codes
            )

            local_ledger.unlink()
            global_ledger = Path(tmp) / "global_test_ledger.jsonl"
            global_ledger.write_text(
                json.dumps(
                    {"out_dir": str(root.resolve())}
                )
                + "\n",
                encoding="utf-8",
            )
            config["args"]["test_ledger"] = str(global_ledger)
            (root / "config.json").write_text(
                json.dumps(
                    config, ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            warning_codes = self._codes(report, "warnings")
            self.assertNotIn(
                "test_ledger_missing", warning_codes
            )
            self.assertNotIn(
                "test_ledger_entry_missing", warning_codes
            )

    def test_audit_matches_relative_test_ledger_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            root = base / "run"
            _, identity, _ = self._build_run(
                root,
                metadata_overrides={"cwd": str(base)},
            )
            metrics_path = root / "metrics.jsonl"
            records = [
                json.loads(line)
                for line in metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            records.append(
                {"step": "test", **identity, "test_loss": 1.0}
            )
            metrics_path.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            (root / "test_access_ledger.jsonl").write_text(
                json.dumps({"out_dir": "run"}) + "\n",
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            warning_codes = self._codes(report, "warnings")
            self.assertNotIn(
                "test_ledger_entry_missing", warning_codes
            )

    def test_final_comparison_normalizes_ledger_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "run"
            ledger = Path(tmp) / "ledger.jsonl"
            ledger.write_text(
                json.dumps({"out_dir": str(candidate)}) + "\n"
                + json.dumps({
                    "out_dir": os.path.relpath(
                        candidate, Path.cwd()
                    )
                })
                + "\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                report = experiment.print_final_test_comparison(
                    [], str(ledger)
                )
            counts = report["ledger"]["per_candidate"]
            self.assertEqual(list(counts.values()), [2])
            self.assertIn("重复解封", output.getvalue())

    def test_audit_detects_crash_between_ledger_and_seal(self):
        # 访问账本在 test_started/封条之前写入；进程若在这之间
        # 崩溃，只有账本文件能证明测试集已解封。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(root)
            (root / "test_access_ledger.jsonl").write_text(
                json.dumps(
                    {"out_dir": str(root.resolve())}
                )
                + "\n",
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            warning_codes = self._codes(report, "warnings")
            self.assertIn("test_seal_missing", warning_codes)
            self.assertIn(
                "test_started_without_summary",
                warning_codes,
            )
            self.assertNotIn(
                "test_ledger_missing", warning_codes
            )

    def test_audit_detects_crash_in_global_test_ledger(self):
        # 全局 --test-ledger 同样先于 test_started/封条写入；
        # 进程在这之间崩溃时运行目录里没有本地账本，审计必须
        # 从全局账本里找到本目录的条目，而不是当成从未解封。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            config, _, _ = self._build_run(root)
            global_ledger = Path(tmp) / "global_test_ledger.jsonl"
            global_ledger.write_text(
                json.dumps(
                    {"out_dir": str(root.resolve())}
                )
                + "\n",
                encoding="utf-8",
            )
            config["args"]["test_ledger"] = str(global_ledger)
            (root / "config.json").write_text(
                json.dumps(
                    config, ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            warning_codes = self._codes(report, "warnings")
            self.assertIn("test_seal_missing", warning_codes)
            self.assertIn(
                "test_started_without_summary",
                warning_codes,
            )
            self.assertNotIn(
                "test_ledger_missing", warning_codes
            )
            self.assertNotIn(
                "test_ledger_entry_missing", warning_codes
            )

            # 账本里只有其它候选的访问时，不能把本目录误判成
            # 已解封；但既然配置了全局账本且可读，仍应指出本
            # 目录没有对应访问记录。
            global_ledger.write_text(
                json.dumps(
                    {"out_dir": str(Path(tmp) / "other")}
                )
                + "\n",
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            warning_codes = self._codes(report, "warnings")
            self.assertNotIn("test_seal_missing", warning_codes)
            self.assertNotIn(
                "test_started_without_summary",
                warning_codes,
            )

    def test_audit_flags_unreadable_test_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _, identity, _ = self._build_run(root)
            metrics_path = root / "metrics.jsonl"
            records = [
                json.loads(line)
                for line in metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            records.append(
                {"step": "test", **identity, "test_loss": 1.0}
            )
            metrics_path.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            local_ledger = root / "test_access_ledger.jsonl"
            local_ledger.write_text(
                json.dumps(
                    {"out_dir": str(root.resolve())}
                )
                + "\n",
                encoding="utf-8",
            )
            real_open = open
            ledger_key = str(local_ledger)

            def _selective_open(path, *args, **kwargs):
                if str(path) == ledger_key:
                    raise PermissionError("denied")
                return real_open(path, *args, **kwargs)

            with mock.patch(
                "builtins.open", new=_selective_open
            ):
                report = experiment.audit_run_directory(root)
            warning_codes = self._codes(report, "warnings")
            self.assertIn(
                "test_ledger_unreadable", warning_codes
            )
            self.assertNotIn(
                "test_ledger_entry_missing", warning_codes
            )

    def test_audit_detects_started_without_summary(self):
        for step_record in (
            {"step": "test_started", "split": "test"},
            {
                "step": "bad_cases",
                "split": "test",
                "wrong_tokens": 1,
            },
        ):
            with self.subTest(step=step_record["step"]):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "run"
                    self._build_run(root)
                    metrics_path = root / "metrics.jsonl"
                    records = [
                        json.loads(line)
                        for line in metrics_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                    ]
                    records.append(step_record)
                    metrics_path.write_text(
                        "".join(
                            json.dumps(record) + "\n"
                            for record in records
                        ),
                        encoding="utf-8",
                    )
                    report = experiment.audit_run_directory(root)
                    warning_codes = self._codes(
                        report, "warnings"
                    )
                    self.assertIn(
                        "test_seal_missing", warning_codes
                    )
                    self.assertIn(
                        "test_started_without_summary",
                        warning_codes,
                    )
                    self.assertIn(
                        "test_ledger_missing", warning_codes
                    )

    def test_audit_detects_repeated_test_unsealing(self):
        for step_record, expected_code in (
            (
                {"step": "test_started", "split": "test"},
                "multiple_test_started",
            ),
            (
                {
                    "step": "bad_cases",
                    "split": "test",
                    "wrong_tokens": 1,
                },
                "multiple_test_bad_cases",
            ),
        ):
            with self.subTest(step=step_record["step"]):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "run"
                    self._build_run(root)
                    metrics_path = root / "metrics.jsonl"
                    records = [
                        json.loads(line)
                        for line in metrics_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                    ]
                    records.extend([step_record, step_record])
                    metrics_path.write_text(
                        "".join(
                            json.dumps(record) + "\n"
                            for record in records
                        ),
                        encoding="utf-8",
                    )
                    report = experiment.audit_run_directory(root)
                    self.assertFalse(report["ok"])
                    self.assertIn(
                        expected_code,
                        self._codes(report, "errors"),
                    )

    def test_audit_detects_duplicate_test_ledger_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(root)
            entries = (
                json.dumps({"out_dir": str(root.resolve())})
                + "\n"
            ) * 2
            (root / "test_access_ledger.jsonl").write_text(
                entries, encoding="utf-8"
            )
            report = experiment.audit_run_directory(root)
            self.assertFalse(report["ok"])
            self.assertIn(
                "multiple_test_ledger_entries",
                self._codes(report, "errors"),
            )

            config, _, _ = self._build_run(
                Path(tmp) / "allowed",
                config_overrides={
                    "args": {
                        "seed": 7,
                        "mode": "smelt",
                        "allow_repeat_test_access": True,
                    }
                },
            )
            allowed_root = Path(tmp) / "allowed"
            (allowed_root / "test_access_ledger.jsonl").write_text(
                (
                    json.dumps(
                        {"out_dir": str(allowed_root.resolve())}
                    )
                    + "\n"
                ) * 2,
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(allowed_root)
            self.assertTrue(report["ok"], report)
            self.assertIn(
                "test_ledger_repeated_access",
                self._codes(report, "warnings"),
            )

    def test_audit_ignores_ledger_entries_for_other_test_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _, identity, _ = self._build_run(root)
            metrics_path = root / "metrics.jsonl"
            records = [
                json.loads(line)
                for line in metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            records.append(
                {"step": "test", **identity, "test_loss": 1.0}
            )
            metrics_path.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            (root / "test_access_ledger.jsonl").write_text(
                json.dumps(
                    {
                        "out_dir": str(root.resolve()),
                        "data_sha256": "e" * 64,
                        "test_split_sha256": "3" * 64,
                        "test_fraction": 0.05,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(root)
            warning_codes = self._codes(report, "warnings")
            self.assertIn(
                "test_ledger_entry_missing", warning_codes
            )
            self.assertEqual(
                report["checks"]["test_seal"]["ledger_matches"], 0
            )

    def test_audit_detects_sanity_repeat_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(
                root,
                config_overrides={
                    "sanity_check": {
                        "reproducibility": {
                            "repeats": 2,
                            "identical": False,
                            "mismatches": ["final_loss"],
                        }
                    }
                },
            )
            report = experiment.audit_run_directory(root)
            self.assertFalse(report["ok"])
            self.assertIn(
                "sanity_repeat_mismatch",
                self._codes(report, "errors"),
            )

    def test_audit_strict_treats_evidence_gaps_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(
                root,
                metadata_overrides={
                    "pythonhashseed_at_start": None,
                    "deterministic_algorithms": False,
                },
            )
            report = experiment.audit_run_directory(root)
            self.assertTrue(report["ok"], report)
            warning_codes = self._codes(report, "warnings")
            self.assertIn(
                "pythonhashseed_unset", warning_codes
            )
            self.assertIn(
                "nondeterministic_algorithms", warning_codes
            )
            strict = experiment.audit_run_directory(
                root, strict=True
            )
            self.assertFalse(strict["ok"])

    def test_audit_checks_external_source_and_data_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._build_run(root)
            source = Path(tmp) / "experiment.py"
            source.write_text("# archived copy\n", encoding="utf-8")
            data = Path(tmp) / "corpus.txt"
            data.write_text("corpus\n", encoding="utf-8")
            report = experiment.audit_run_directory(
                root,
                source_path=source,
                data_path=data,
            )
            self.assertFalse(report["ok"])
            error_codes = self._codes(report, "errors")
            self.assertIn(
                "source_sha_mismatch", error_codes
            )
            self.assertIn("data_sha_mismatch", error_codes)

    def test_audit_detects_preregistration_edited_after_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            config, identity, _ = self._build_run(root)
            prereg = Path(tmp) / "preregistration.json"
            payload = {
                "study_id": "smelt-isr-audit-test",
                "frozen_commit": "a" * 40,
                "primary_metric": "val_loss",
                "common_step": 100,
                "min_effect": 0.01,
                "direction": "decrease",
                "baseline_mode": "smelt",
                "comparisons": [
                    ["smelt", "A"],
                    ["smelt", "B"],
                ],
                "seeds": [42, 43, 44],
                "budget_match": "smelt",
                "steps": 200,
                "frozen_script_sha256": identity[
                    "script_sha256"
                ],
                "data": {
                    "sha256": identity["data_sha256"],
                    "split_policy": "contiguous",
                    "val_fraction": 0.1,
                    "test_fraction": 0.1,
                },
            }
            prereg.write_text(
                json.dumps(payload), encoding="utf-8"
            )
            _, prereg_sha = experiment.load_preregistration(
                prereg
            )
            config["preregistration"] = {
                **{
                    key: payload[key]
                    for key in (
                        "study_id",
                        "frozen_commit",
                        "primary_metric",
                        "common_step",
                        "direction",
                        "min_effect",
                        "baseline_mode",
                        "comparisons",
                        "seeds",
                        "budget_match",
                    )
                },
                "file_sha256": prereg_sha,
            }
            (root / "config.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            report = experiment.audit_run_directory(
                root, preregistration_path=prereg
            )
            self.assertTrue(report["ok"], report)

            payload["min_effect"] = 0.5
            prereg.write_text(
                json.dumps(payload), encoding="utf-8"
            )
            report = experiment.audit_run_directory(
                root, preregistration_path=prereg
            )
            self.assertFalse(report["ok"])
            self.assertIn(
                "preregistration_sha_mismatch",
                self._codes(report, "errors"),
            )

    def test_audit_mode_requires_run_and_rejects_training(self):
        self.assertIsNotNone(
            experiment.audit_mode_error(
                argparse.Namespace(audit_report="report.json")
            )
        )
        self.assertIsNotNone(
            experiment.audit_mode_error(
                argparse.Namespace(
                    audit_run="run",
                    out="other",
                )
            )
        )
        for name, value in (
            ("plan_design", True),
            ("suite_modes", "smelt,A"),
            ("suite_ablation", True),
            ("paper_cell", "smelt-200M"),
        ):
            with self.subTest(conflict=name):
                self.assertIsNotNone(
                    experiment.audit_mode_error(
                        argparse.Namespace(
                            audit_run="run",
                            **{name: value},
                        )
                    )
                )
        self.assertIsNone(
            experiment.audit_mode_error(
                argparse.Namespace(audit_run="run")
            )
        )
