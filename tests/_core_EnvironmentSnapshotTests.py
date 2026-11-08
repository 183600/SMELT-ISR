from _core_support import *
from _core_support import _load_experiment, _tiny_config

class EnvironmentSnapshotTests(unittest.TestCase):
    """环境身份：关键依赖/编译选项可核对，完整包清单可审计。"""

    def test_snapshot_is_stable_and_contains_critical_packages(self):
        first = experiment.build_environment_snapshot()
        second = experiment.build_environment_snapshot()
        self.assertEqual(first["schema"], "smelt-isr/environment/v1")
        self.assertEqual(
            first["environment_id"], second["environment_id"]
        )
        self.assertEqual(len(first["environment_id"]), 64)
        names = {
            item["name"]
            for item in first["packages"]["all"]
        }
        self.assertIn("torch", names)
        critical_names = {
            item["name"]
            for item in first["packages"]["critical"]
        }
        self.assertIn("torch", names)
        self.assertEqual(
            critical_names,
            {
                name
                for name in names
                if experiment._is_critical_distribution(name)
            },
        )
        self.assertEqual(
            first["packages"]["all"],
            sorted(
                first["packages"]["all"],
                key=lambda item: item["name"],
            ),
        )

    def test_per_run_hashseed_does_not_change_environment_identity(self):
        snapshot = experiment.build_environment_snapshot()
        altered = json.loads(json.dumps(snapshot))
        altered["determinism"]["pythonhashseed_at_start"] = "12345"
        # 套件子进程按种子固定不同的 PYTHONHASHSEED；它属于运行
        # 参数而不是安装环境，不能让同一环境的多种子套件被
        # --compare 误报为环境不同。
        recalculated = experiment._canonical_json_sha256(
            experiment._environment_identity_payload(altered)
        )
        self.assertEqual(
            recalculated, snapshot["environment_id"]
        )

    def test_full_package_drift_does_not_change_critical_identity(self):
        snapshot = experiment.build_environment_snapshot()
        altered = json.loads(json.dumps(snapshot))
        altered["packages"]["all"].append({
            "name": "unrelated-tool",
            "version": "1.0",
        })
        altered["packages"]["all_sha256"] = (
            experiment._canonical_json_sha256(
                altered["packages"]["all"]
            )
        )
        # 非关键包不参与 environment_id：默认核对仍通过，
        # 严格模式才把完整包清单漂移判为失败。
        altered["environment_id"] = (
            experiment._canonical_json_sha256(
                experiment._environment_identity_payload(altered)
            )
        )
        report = experiment.compare_environment_snapshots(
            snapshot, altered
        )
        self.assertTrue(report["ok"], report)
        self.assertFalse(report["strict_ok"])
        self.assertTrue(report["all_packages_match"] is False)
        self.assertEqual(
            [item["name"] for item in report[
                "package_differences"
            ]["added"]],
            ["unrelated-tool"],
        )

    def test_critical_package_change_breaks_identity(self):
        snapshot = experiment.build_environment_snapshot()
        altered = json.loads(json.dumps(snapshot))
        critical = altered["packages"]["critical"]
        self.assertTrue(critical)
        name = critical[0]["name"]
        critical[0]["version"] = "0.0.0-drift"
        altered["packages"]["all"] = [
            (
                {**item, "version": "0.0.0-drift"}
                if item["name"] == name
                else item
            )
            for item in altered["packages"]["all"]
        ]
        altered["packages"]["all_sha256"] = (
            experiment._canonical_json_sha256(
                altered["packages"]["all"]
            )
        )
        altered["packages"]["critical_sha256"] = (
            experiment._canonical_json_sha256(critical)
        )
        altered["environment_id"] = (
            experiment._canonical_json_sha256(
                experiment._environment_identity_payload(altered)
            )
        )
        report = experiment.compare_environment_snapshots(
            snapshot, altered
        )
        self.assertFalse(report["ok"])
        self.assertTrue(
            report["critical_package_differences"]
        )
        self.assertTrue(
            any(
                item["path"].startswith("packages")
                for item in report["runtime_differences"]
            ),
            report["runtime_differences"],
        )

    def test_check_environment_reads_file_and_config_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = experiment.build_environment_snapshot()
            environment_path = root / "environment.json"
            environment_path.write_text(
                json.dumps(snapshot), encoding="utf-8"
            )
            report = experiment.check_environment_snapshot(
                environment_path
            )
            self.assertTrue(report["ok"], report)
            self.assertTrue(report["strict_ok"])

            config_path = root / "config.json"
            config_path.write_text(
                json.dumps({
                    "environment": {
                        "schema": snapshot["schema"],
                        "environment_id": snapshot[
                            "environment_id"
                        ],
                        "file": "environment.json",
                        "file_sha256": experiment._sha256_file(
                            environment_path
                        ),
                    },
                }),
                encoding="utf-8",
            )
            report = experiment.check_environment_snapshot(
                config_path
            )
            self.assertTrue(report["ok"], report)

            # config 记录的摘要必须能发现 environment.json 被替换：
            # 否则 --check-environment 会拿一份被篡改的“期望环境”
            # 去比较，核对本身失去意义。
            environment_path.write_text(
                json.dumps({"schema": snapshot["schema"]}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                experiment.check_environment_snapshot(config_path)

    def test_write_environment_record_keeps_original_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = experiment.build_environment_snapshot()
            current = json.loads(json.dumps(original))
            current["environment_id"] = "1" * 64
            record = experiment.write_run_environment_record(
                root,
                {"environment": current},
                {"environment": original},
            )
            self.assertEqual(
                record["environment_id"],
                original["environment_id"],
            )
            self.assertTrue(
                record["environment_changed_since_start"]
            )
            self.assertEqual(
                record["resume_environment_id"], "1" * 64
            )
            on_disk = json.loads(
                (root / "environment.json").read_text()
            )
            self.assertEqual(
                on_disk["environment_id"],
                original["environment_id"],
            )
            self.assertEqual(
                record["file_sha256"],
                experiment._sha256_file(
                    root / "environment.json"
                ),
            )

    def test_environment_end_snapshot_uses_current_run_device(self):
        start = {
            "environment_id": "start-environment",
            "device": {"type": "cuda", "index": 1},
        }
        original = {
            "environment_id": "original-environment",
            "device": {"type": "cuda", "index": 0},
        }
        current = {
            "schema": experiment._ENVIRONMENT_SCHEMA,
            "environment_id": "current-environment",
            "captured_utc": "2026-01-01T00:00:00+00:00",
            "device": {"type": "cuda", "index": 1},
        }
        calls = []

        def fake_build(device=None):
            calls.append(device)
            return json.loads(json.dumps(current))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(
                experiment,
                "build_environment_snapshot",
                side_effect=fake_build,
            ):
                record = (
                    experiment.write_run_environment_end_record(
                        root,
                        {"environment": start},
                        original,
                    )
                )
            self.assertEqual(
                calls, [torch.device("cuda", 1)]
            )
            self.assertEqual(
                record["environment_id"],
                "current-environment",
            )
            self.assertTrue(
                (root / "environment_end.json").is_file()
            )

    def test_environment_mode_rejects_training_arguments(self):
        strict_only = experiment.environment_mode_error(
            argparse.Namespace(
                check_environment=None,
                check_environment_strict=True,
            )
        )
        self.assertIn("--check-environment", strict_only)

        conflict = experiment.environment_mode_error(
            argparse.Namespace(
                check_environment="run",
                check_environment_strict=False,
                data="corpus.txt",
            )
        )
        self.assertIn("--data", conflict)

        strict_manifest = experiment.environment_mode_error(
            argparse.Namespace(
                check_environment="run",
                check_environment_strict=False,
                verify_manifest_strict=True,
            )
        )
        self.assertIn("--verify-manifest-strict", strict_manifest)

        ok = experiment.environment_mode_error(
            argparse.Namespace(
                check_environment="run",
                check_environment_strict=False,
            )
        )
        self.assertIsNone(ok)

    def test_environment_cli_checks_and_fails_on_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "environment.json"
            snapshot = experiment.build_environment_snapshot()
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            old_argv = sys.argv
            try:
                sys.argv = [
                    "experiment.py",
                    "--check-environment",
                    str(path),
                ]
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    experiment.main()
                self.assertIn("[env] OK", out.getvalue())

                snapshot["environment_id"] = "0" * 64
                path.write_text(
                    json.dumps(snapshot), encoding="utf-8"
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        experiment.main()
                self.assertEqual(raised.exception.code, 1)
            finally:
                sys.argv = old_argv


    def test_environment_id_must_match_snapshot_content(self):
        snapshot = experiment.build_environment_snapshot()
        altered = json.loads(json.dumps(snapshot))
        altered["packages"]["critical"][0]["version"] = (
            "0.0.0-drift"
        )
        # 只改内容、保留旧 id：不能因为两个快照声明的 id 相同就
        # 把被篡改/损坏的归档当成同一环境。
        report = experiment.compare_environment_snapshots(
            snapshot, altered
        )
        self.assertFalse(report["ok"], report)
        self.assertFalse(report["strict_ok"], report)
        self.assertFalse(
            report["actual_environment_id_consistent"]
        )
        self.assertTrue(
            report["expected_environment_id_consistent"]
        )

    def test_check_environment_restores_run_time_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "environment.json"
            snapshot = experiment.build_environment_snapshot()
            path.write_text(
                json.dumps(snapshot), encoding="utf-8"
            )
            determinism = snapshot["determinism"]
            old_deterministic = (
                torch.are_deterministic_algorithms_enabled()
            )
            old_warn = (
                torch.is_deterministic_algorithms_warn_only_enabled()
            )
            old_cudnn_deterministic = (
                torch.backends.cudnn.deterministic
            )
            old_cudnn_benchmark = torch.backends.cudnn.benchmark
            try:
                # 模拟全新核对进程的默认状态：训练入口设置的
                # 确定性开关尚未应用。
                torch.use_deterministic_algorithms(
                    not bool(
                        determinism["deterministic_algorithms"]
                    ),
                    warn_only=not bool(
                        determinism[
                            "deterministic_algorithms_warn_only"
                        ]
                    ),
                )
                torch.backends.cudnn.deterministic = not bool(
                    determinism["cudnn_deterministic"]
                )
                torch.backends.cudnn.benchmark = not bool(
                    determinism["cudnn_benchmark"]
                )
                report = experiment.check_environment_snapshot(path)
            finally:
                torch.use_deterministic_algorithms(
                    old_deterministic, warn_only=old_warn
                )
                torch.backends.cudnn.deterministic = (
                    old_cudnn_deterministic
                )
                torch.backends.cudnn.benchmark = old_cudnn_benchmark
        self.assertTrue(report["ok"], report)
        self.assertEqual(
            report["actual_environment_id"],
            snapshot["environment_id"],
        )
        self.assertIn(
            "deterministic_algorithms",
            report["applied_environment_settings"],
        )
        self.assertEqual(report["environment_setting_errors"], {})

    def test_config_summary_id_must_match_environment_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = experiment.build_environment_snapshot()
            environment_path = root / "environment.json"
            environment_path.write_text(
                json.dumps(snapshot), encoding="utf-8"
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps({
                    "environment": {
                        "schema": snapshot["schema"],
                        "environment_id": "0" * 64,
                        "file": "environment.json",
                        "file_sha256": experiment._sha256_file(
                            environment_path
                        ),
                    },
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                experiment.check_environment_snapshot(config_path)

    def test_snapshot_device_preserves_non_default_index(self):
        device = experiment._snapshot_device({
            "device": {"type": "cuda", "index": 1},
        })
        self.assertEqual(device.type, "cuda")
        self.assertEqual(device.index, 1)
        # CPU 没有可区分的设备号；非法/布尔 index 不能触发
        # torch.device 构造错误。
        self.assertIsNone(
            experiment._snapshot_device({
                "device": {"type": "cpu", "index": 1},
            }).index
        )
        self.assertEqual(
            experiment._snapshot_device({
                "device": {"type": "cuda", "index": True},
            }).type,
            "cuda",
        )
