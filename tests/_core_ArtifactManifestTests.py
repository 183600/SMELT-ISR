from _core_support import *
from _core_support import _load_experiment, _tiny_config

class ArtifactManifestTests(unittest.TestCase):
    """归档清单：内容可复核、路径不可逃逸、新增文件可披露。"""

    def _make_run(self, root):
        root.mkdir()
        config = {
            "config": {"mode": "A"},
            "run_metadata": {
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "script_sha256": "script-sha",
                "git_commit": "abc123",
                "git_dirty": False,
            },
            "data": {
                "sha256": "data-sha",
                "split_sha256": {"train": "train-sha"},
            },
            "comparison_protocol_sha256": "protocol-sha",
            "preregistration": {
                "study_id": "study-1",
                "file_sha256": "prereg-sha",
                "frozen_commit": "abc123",
            },
            "hypothesis": "delayed merge reduces val loss",
            "research_question": "RQ1",
        }
        (root / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        (root / "metrics.jsonl").write_text(
            '{"step": 1, "val_loss": 3.0}\n', encoding="utf-8"
        )
        nested = root / "logs" / "train.log"
        nested.parent.mkdir()
        nested.write_text("epoch 1\n", encoding="utf-8")
        return root

    def test_manifest_roundtrip_and_run_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            manifest_path = experiment.write_artifact_manifest(root)
            self.assertEqual(
                manifest_path.name,
                "artifact_manifest.json",
            )

            payload = json.loads(manifest_path.read_text())
            self.assertEqual(
                payload["schema"],
                "smelt-isr/artifact-manifest/v1",
            )
            paths = {entry["path"] for entry in payload["files"]}
            self.assertEqual(
                paths,
                {"config.json", "metrics.jsonl", "logs/train.log"},
            )
            self.assertNotIn("artifact_manifest.json", paths)
            self.assertEqual(payload["run"]["study_id"], "study-1")
            self.assertEqual(
                payload["run"]["data_sha256"], "data-sha"
            )
            self.assertEqual(
                payload["run"]["comparison_protocol_sha256"],
                "protocol-sha",
            )

            report = experiment.verify_artifact_manifest(root)
            self.assertTrue(report["ok"], report)
            self.assertTrue(report["integrity_ok"])
            self.assertEqual(report["checked"], 3)
            self.assertEqual(report["mismatched"], [])
            self.assertEqual(report["untracked"], [])
            self.assertEqual(len(report["manifest_sha256"]), 64)

    def test_manifest_detects_tampering_deletion_and_extra_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            manifest_path = experiment.write_artifact_manifest(root)
            original = (root / "metrics.jsonl").read_text()

            # 同长度替换只改内容：必须由 SHA-256 而不是大小发现。
            tampered = original.replace("3.0", "3.5")
            self.assertEqual(len(tampered), len(original))
            (root / "metrics.jsonl").write_text(tampered)
            report = experiment.verify_artifact_manifest(root)
            self.assertFalse(report["ok"])
            self.assertIn(
                "sha256",
                {item["reason"] for item in report["mismatched"]},
            )

            # 删除清单内文件必须进入 missing。
            (root / "metrics.jsonl").write_text(original)
            (root / "logs" / "train.log").unlink()
            report = experiment.verify_artifact_manifest(root)
            self.assertFalse(report["integrity_ok"])
            self.assertEqual(report["missing"], ["logs/train.log"])

            # 恢复文件后，清单外新增文件只进入 untracked：
            # 默认不判损坏（可能是后续日志），strict 才失败。
            (root / "logs" / "train.log").write_text(
                "epoch 1\n"
            )
            (root / "extra.txt").write_text("late output")
            report = experiment.verify_artifact_manifest(root)
            self.assertTrue(report["integrity_ok"])
            self.assertEqual(report["untracked"], ["extra.txt"])
            self.assertTrue(report["ok"])

            strict = experiment.verify_artifact_manifest(
                root, strict=True
            )
            self.assertFalse(strict["ok"])
            self.assertTrue(strict["integrity_ok"])

            # 重新生成后清单应吸收新增文件，strict 也通过。
            experiment.write_artifact_manifest(root)
            report = experiment.verify_artifact_manifest(
                root, strict=True
            )
            self.assertTrue(report["ok"], report)

            # 清单自身不能被写入自身（否则哈希自指）。
            payload = json.loads(manifest_path.read_text())
            self.assertNotIn(
                "artifact_manifest.json",
                {entry["path"] for entry in payload["files"]},
            )

    def test_manifest_path_validation_rejects_noncanonical(self):
        for value in (
            "",
            "/absolute",
            "../escape",
            "a/../b",
            "a//b",
            "a/./b",
            "a\\b",
            "./a",
            "a/",
            # Windows 的盘符/UNC 前缀在 POSIX 下不是绝对路径，
            # 但换到 Windows 后会把 root / path 解析到目录外。
            "C:/outside",
            "c:outside",
            "//server/share",
        ):
            self.assertIsNone(
                experiment._manifest_relative_path(value),
                value,
            )
        for value in ("config.json", "logs/train.log", "a-b_c.1"):
            self.assertEqual(
                experiment._manifest_relative_path(value),
                value,
            )
        self.assertIsNone(experiment._manifest_relative_path(3))

    def test_manifest_writer_refuses_unverifiable_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            try:
                (root / "bad\\name.txt").write_text(
                    "x", encoding="utf-8"
                )
            except OSError:
                self.skipTest("当前平台不允许该文件名")
            # 写入端不能生成一份校验端必然拒绝的清单；反斜杠
            # 路径在跨平台归档中不安全，应明确失败。
            with self.assertRaises(ValueError):
                experiment.write_artifact_manifest(root)

    def test_manifest_rejects_unsafe_and_duplicate_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            manifest_path = experiment.write_artifact_manifest(root)
            payload = json.loads(manifest_path.read_text())
            payload["files"].extend([
                {
                    "path": "../outside.txt",
                    "type": "file",
                    "bytes": 0,
                    "sha256": "x",
                },
                {
                    "path": "/etc/passwd",
                    "type": "file",
                    "bytes": 0,
                    "sha256": "x",
                },
                {
                    "path": "config.json",
                    "type": "file",
                    "bytes": 0,
                    "sha256": "x",
                },
            ])
            manifest_path.write_text(
                json.dumps(payload), encoding="utf-8"
            )

            report = experiment.verify_artifact_manifest(root)
            self.assertFalse(report["ok"])
            reasons = {
                item["reason"] for item in report["mismatched"]
            }
            self.assertIn("unsafe_path", reasons)
            self.assertIn("duplicate_path", reasons)

    def test_manifest_symlink_target_is_recorded_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            link = root / "latest.pt"
            try:
                link.symlink_to("metrics.jsonl")
            except (OSError, NotImplementedError):
                self.skipTest("当前平台不支持符号链接")
            experiment.write_artifact_manifest(root)

            payload = json.loads(
                (root / "artifact_manifest.json").read_text()
            )
            link_entry = next(
                entry
                for entry in payload["files"]
                if entry["path"] == "latest.pt"
            )
            self.assertEqual(link_entry["type"], "symlink")
            self.assertEqual(link_entry["target"], "metrics.jsonl")
            self.assertNotIn("sha256", link_entry)

            self.assertTrue(
                experiment.verify_artifact_manifest(root)["ok"]
            )

            link.unlink()
            link.symlink_to("config.json")
            report = experiment.verify_artifact_manifest(root)
            self.assertFalse(report["ok"])
            self.assertIn(
                "symlink_target",
                {item["reason"] for item in report["mismatched"]},
            )

    def test_manifest_symlink_loop_path_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            try:
                (root / "a").symlink_to("b")
                (root / "b").symlink_to("a")
            except (OSError, NotImplementedError):
                self.skipTest("当前平台不支持符号链接")
            manifest_path = experiment.write_artifact_manifest(root)
            payload = json.loads(manifest_path.read_text())
            # 恶意/损坏清单可以声明一条穿过符号链接环的普通文件
            # 路径；Path.resolve 会抛 RuntimeError，校验器必须把
            # 它变成结构化失败而不是 traceback。
            payload["files"].append({
                "path": "a/x",
                "type": "file",
                "bytes": 1,
                "sha256": "0" * 64,
            })
            manifest_path.write_text(
                json.dumps(payload), encoding="utf-8"
            )
            report = experiment.verify_artifact_manifest(root)
        self.assertFalse(report["ok"])
        self.assertIn(
            "path_resolution_error",
            {item["reason"] for item in report["mismatched"]},
        )

    def test_manifest_external_symlink_is_allowed_but_contained_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            outside = Path(tmp) / "outside.txt"
            outside.write_text("external artifact")

            # 清单记录链接目标而不是跟随链接：运行目录之外的
            # 合法归档链接必须能通过校验，否则 writer 写出的
            # 清单无法验证自己的输出。
            external_link = root / "external.txt"
            try:
                external_link.symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("当前平台不支持符号链接")
            experiment.write_artifact_manifest(root)
            self.assertTrue(
                experiment.verify_artifact_manifest(root)["ok"]
            )

            # 逃逸检查仍然生效：通过目录链接把清单路径引到
            # 运行目录之外必须拒绝，即使条目的父目录字符串
            # 看起来在 root 下。
            escape_dir = root / "escape"
            try:
                escape_dir.symlink_to(Path(tmp), target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("当前平台不支持目录符号链接")
            manifest_path = root / "artifact_manifest.json"
            payload = json.loads(manifest_path.read_text())
            payload["files"].append({
                "path": "escape/outside.txt",
                "type": "file",
                "bytes": outside.stat().st_size,
                "sha256": experiment._sha256_file(outside),
            })
            manifest_path.write_text(
                json.dumps(payload), encoding="utf-8"
            )
            report = experiment.verify_artifact_manifest(root)
            self.assertFalse(report["ok"])
            self.assertIn(
                "escapes_root",
                {item["reason"] for item in report["mismatched"]},
            )

    def test_manifest_run_summary_prefers_original_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            config = json.loads(
                (root / "config.json").read_text()
            )
            config["original_run_metadata"] = {
                "timestamp_utc": "2025-01-01T00:00:00+00:00",
                "script_sha256": "original-script",
                "git_commit": "original-commit",
                "git_dirty": False,
            }
            config["run_metadata"] = {
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "script_sha256": "resumed-script",
                "git_commit": "resumed-commit",
                "git_dirty": True,
            }
            (root / "config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            payload = experiment.build_artifact_manifest(root)
            summary = payload["run"]
            self.assertEqual(
                summary["run_started_utc"],
                "2025-01-01T00:00:00+00:00",
            )
            self.assertEqual(
                summary["script_sha256"], "original-script"
            )
            self.assertEqual(
                summary["git_commit"], "original-commit"
            )

    def test_manifest_entry_limit_counts_across_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            # _make_run 本身有 3 个文件（config、metrics、logs/）。
            # 上限按整棵目录树计数，不能被每个子目录各自重置。
            previous = experiment._ARTIFACT_MANIFEST_MAX_ENTRIES
            experiment._ARTIFACT_MANIFEST_MAX_ENTRIES = 2
            try:
                with self.assertRaises(ValueError) as raised:
                    experiment.build_artifact_manifest(root)
            finally:
                experiment._ARTIFACT_MANIFEST_MAX_ENTRIES = previous
            self.assertIn("上限", str(raised.exception))

    def test_manifest_expected_sha_pin_detects_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            manifest_path = experiment.write_artifact_manifest(root)
            pinned = experiment._sha256_file(manifest_path)
            old_argv = sys.argv
            try:
                sys.argv = [
                    "experiment.py",
                    "--verify-manifest",
                    str(root),
                    "--expect-manifest-sha256",
                    pinned,
                ]
                with contextlib.redirect_stdout(
                    io.StringIO()
                ) as out:
                    experiment.main()
                self.assertIn(
                    '"manifest_sha256_matches_expected": true',
                    out.getvalue(),
                )

                # 清单被整体替换但内部仍然自洽：只有外部记录的
                # 指纹能发现替换，必须判失败。
                sys.argv = [
                    "experiment.py",
                    "--verify-manifest",
                    str(root),
                    "--expect-manifest-sha256",
                    "0" * 64,
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        experiment.main()
                self.assertEqual(raised.exception.code, 1)
            finally:
                sys.argv = old_argv

            error = experiment.manifest_mode_error(
                argparse.Namespace(
                    write_manifest=None,
                    verify_manifest=None,
                    verify_manifest_strict=False,
                    expect_manifest_sha256="0" * 64,
                )
            )
            self.assertIn("--verify-manifest", error)

    def test_manifest_mode_rejects_conflicting_entries(self):
        both = experiment.manifest_mode_error(
            argparse.Namespace(
                write_manifest="run",
                verify_manifest="run",
                verify_manifest_strict=False,
            )
        )
        self.assertIn("不能同时使用", both)

        strict_only = experiment.manifest_mode_error(
            argparse.Namespace(
                write_manifest=None,
                verify_manifest=None,
                verify_manifest_strict=True,
            )
        )
        self.assertIn("--verify-manifest", strict_only)

        conflict = experiment.manifest_mode_error(
            argparse.Namespace(
                write_manifest="run",
                verify_manifest=None,
                verify_manifest_strict=False,
                compare=["a"],
            )
        )
        self.assertIn("--compare", conflict)

        for name, value in (
            ("plan_design", True),
            ("suite_modes", "smelt,A"),
            ("suite_ablation", True),
            ("paper_cell", "smelt-200M"),
        ):
            with self.subTest(conflict=name):
                error = experiment.manifest_mode_error(
                    argparse.Namespace(
                        write_manifest="run",
                        verify_manifest=None,
                        verify_manifest_strict=False,
                        **{name: value},
                    )
                )
                self.assertIn("不能与", error)

        ok = experiment.manifest_mode_error(
            argparse.Namespace(
                write_manifest="run",
                verify_manifest=None,
                verify_manifest_strict=False,
                compare=None,
                final_test_comparison=None,
                sanity_only=False,
                statistical_baseline=False,
                suite_baseline_tuning=False,
                preregistration=None,
                data=None,
                out=None,
                seeds=None,
                resume=None,
                evaluate_test=False,
            )
        )
        self.assertIsNone(ok)

    def test_manifest_cli_writes_and_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_run(Path(tmp) / "run")
            old_argv = sys.argv
            try:
                sys.argv = [
                    "experiment.py",
                    "--write-manifest",
                    str(root),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    experiment.main()
                self.assertTrue(
                    (root / "artifact_manifest.json").is_file()
                )

                sys.argv = [
                    "experiment.py",
                    "--verify-manifest",
                    str(root),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    experiment.main()

                (root / "metrics.jsonl").write_text("broken")
                sys.argv = [
                    "experiment.py",
                    "--verify-manifest",
                    str(root),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        experiment.main()
                self.assertEqual(raised.exception.code, 1)
            finally:
                sys.argv = old_argv
