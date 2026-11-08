from _core_support import *
from _core_support import _load_experiment, _tiny_config

class ReproducibilityTests(unittest.TestCase):
    def test_compare_report_file_writes_analysis_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reports" / "analysis.json"
            with contextlib.redirect_stdout(io.StringIO()):
                written = experiment.write_comparison_report_file(
                    path,
                    {
                        "metric": "val_loss",
                        "fragility_summary": {
                            "verdict_counts": {"stable": 1}
                        },
                    },
                )
            self.assertEqual(written, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["metric"], "val_loss")
            self.assertEqual(
                payload["fragility_summary"]["verdict_counts"],
                {"stable": 1},
            )
            self.assertIn("generated_at", payload)

    def test_set_seed_records_effective_manifest(self):
        experiment.set_seed(321)
        manifest = experiment._LAST_SEED_MANIFEST
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest["seed"], 321)
        self.assertEqual(manifest["torch_initial_seed"], 321)
        self.assertTrue(manifest["python_random"])
        if torch.cuda.is_available():
            self.assertTrue(manifest["cuda_initial_seeds"])
            self.assertTrue(
                all(
                    value == 321
                    for value in manifest[
                        "cuda_initial_seeds"
                    ]
                )
            )

    def test_optimizer_coverage_detects_missing_params(self):
        model = experiment.LanguageModel(_tiny_config())
        optimizer = torch.optim.AdamW(
            model.token_embedding.parameters(),
            lr=1e-3,
        )
        report = experiment._probe_optimizer_coverage(
            model, optimizer
        )
        self.assertFalse(report["ok"])
        self.assertTrue(report["missing_params"])

        full = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
        )
        self.assertTrue(
            experiment._probe_optimizer_coverage(
                model, full
            )["ok"]
        )

    def test_fresh_out_dir_guard_protects_existing_tracking(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            self.assertIsNone(
                experiment.fresh_out_dir_error(
                    out_dir,
                    resume_path=None,
                    allow_overwrite=False,
                )
            )
            (out_dir / "metrics.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            error = experiment.fresh_out_dir_error(
                out_dir,
                resume_path=None,
                allow_overwrite=False,
            )
            self.assertIsNotNone(error)
            self.assertIn("metrics.jsonl", error)
            self.assertIsNone(
                experiment.fresh_out_dir_error(
                    out_dir,
                    resume_path=None,
                    allow_overwrite=True,
                )
            )
            self.assertIsNone(
                experiment.fresh_out_dir_error(
                    out_dir,
                    resume_path="last.pt",
                    allow_overwrite=False,
                )
            )

    def test_fresh_out_dir_guard_refuses_unreadable_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with mock.patch.object(
                Path,
                "iterdir",
                side_effect=PermissionError("denied"),
            ):
                error = experiment.fresh_out_dir_error(
                    out_dir,
                    resume_path=None,
                    allow_overwrite=False,
                )
            self.assertIsNotNone(error)
            self.assertIn("无法读取", error)
            self.assertIsNone(
                experiment.fresh_out_dir_error(
                    out_dir,
                    resume_path=None,
                    allow_overwrite=True,
                )
            )

    def test_tracking_header_fields_are_self_contained(self):
        class _Dataset:
            data_sha256 = "data-sha"
            split_sha256 = {"train": "train-sha"}
            effective_train_sha256 = "effective-sha"

        class _Args:
            seed = 7

        header = experiment.tracking_header_fields(
            _Args(),
            _Dataset(),
            {
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "script_sha256": "script-sha",
                "git_commit": "commit-sha",
                "git_dirty": False,
                "pythonhashseed_at_start": "7",
                "rng_seed_manifest": {"seed": 7},
                "deterministic_algorithms": True,
            },
            comparison_protocol_sha256="protocol-sha",
        )
        self.assertEqual(header["seed"], 7)
        self.assertEqual(header["data_sha256"], "data-sha")
        self.assertEqual(
            header["data_split_sha256"]["train"],
            "train-sha",
        )
        self.assertEqual(
            header["data_effective_train_sha256"],
            "effective-sha",
        )
        self.assertEqual(header["script_sha256"], "script-sha")
        self.assertEqual(
            header["comparison_protocol_sha256"],
            "protocol-sha",
        )
        self.assertEqual(
            header["pythonhashseed_at_start"], "7"
        )
        self.assertEqual(
            header["rng_seed_manifest"], {"seed": 7}
        )

    def test_tracking_provenance_prefers_original_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "config.json").write_text(
                json.dumps({
                    "original_run_metadata": {
                        "timestamp_utc": "2026-01-01T00:00:00+00:00",
                        "script_sha256": "original-script",
                        "git_commit": "original-commit",
                        "git_dirty": False,
                    },
                    "run_metadata": {
                        "timestamp_utc": "2026-01-02T00:00:00+00:00",
                        "script_sha256": "resumed-script",
                        "git_commit": "resumed-commit",
                        "git_dirty": True,
                    },
                    "resume_count": 2,
                    "comparison_protocol_sha256": "protocol-sha",
                }),
                encoding="utf-8",
            )
            provenance = experiment.load_tracking_provenance(
                run_dir
            )
            self.assertEqual(
                provenance["run_started_utc"],
                "2026-01-01T00:00:00+00:00",
            )
            self.assertEqual(
                provenance["original_script_sha256"],
                "original-script",
            )
            self.assertEqual(
                provenance["original_git_commit"],
                "original-commit",
            )
            self.assertEqual(provenance["resume_count"], 2)
            self.assertEqual(
                provenance["comparison_protocol_sha256"],
                "protocol-sha",
            )

    def test_tracking_provenance_defaults_without_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            provenance = experiment.load_tracking_provenance(
                Path(tmp)
            )
            self.assertIsNone(provenance["run_started_utc"])
            self.assertEqual(provenance["resume_count"], 0)
            self.assertIsNone(
                provenance["original_source_tree_sha256"]
            )

    def test_source_tree_fingerprint_is_location_independent(self):
        def _write_tree(root, body):
            root.mkdir(parents=True)
            (root / "exp").mkdir()
            (root / "selfcheck").mkdir()
            (root / "experiment.py").write_text(
                "print('entry')\n", encoding="utf-8"
            )
            (root / "exp" / "part_000.py").write_text(
                body, encoding="utf-8"
            )
            (root / "selfcheck" / "checks_001.py").write_text(
                "VALUE = 1\n", encoding="utf-8"
            )

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            _write_tree(first, "VALUE = 1\n")
            _write_tree(second, "VALUE = 1\n")
            one = experiment.source_tree_fingerprint(first)
            two = experiment.source_tree_fingerprint(second)
            self.assertTrue(one["available"], one)
            self.assertEqual(one["sha256"], two["sha256"])
            self.assertEqual(one["file_count"], 3)
            self.assertEqual(
                [entry["path"] for entry in one["files"]],
                [
                    "exp/part_000.py",
                    "experiment.py",
                    "selfcheck/checks_001.py",
                ],
            )

            (second / "exp" / "part_000.py").write_text(
                "VALUE = 2\n", encoding="utf-8"
            )
            changed = experiment.source_tree_fingerprint(second)
            self.assertNotEqual(one["sha256"], changed["sha256"])
            differences = experiment.source_tree_differences(
                one, changed
            )
            self.assertEqual(
                differences["changed"], ["exp/part_000.py"]
            )
            self.assertEqual(differences["added"], [])
            self.assertEqual(differences["removed"], [])

    def test_source_tree_detects_added_and_removed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "exp").mkdir()
            (root / "experiment.py").write_text(
                "", encoding="utf-8"
            )
            (root / "exp" / "a.py").write_text(
                "A = 1\n", encoding="utf-8"
            )
            before = experiment.source_tree_fingerprint(root)
            (root / "exp" / "a.py").unlink()
            (root / "exp" / "b.py").write_text(
                "B = 1\n", encoding="utf-8"
            )
            after = experiment.source_tree_fingerprint(root)
            differences = experiment.source_tree_differences(
                before, after
            )
            self.assertEqual(differences["removed"], ["exp/a.py"])
            self.assertEqual(differences["added"], ["exp/b.py"])

    def test_source_tree_check_rejects_missing_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(
                json.dumps({
                    "run_metadata": {
                        "script_sha256": "d" * 64,
                    }
                }),
                encoding="utf-8",
            )
            report = experiment.check_source_tree_against_run(
                run_dir
            )
            self.assertFalse(report["ok"])
            self.assertIn("source_tree_sha256", report["error"])

    def test_source_tree_check_uses_full_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(
                json.dumps({
                    "run_metadata": {
                        "source_tree_sha256": "a" * 64,
                    }
                }),
                encoding="utf-8",
            )
            report = experiment.check_source_tree_against_run(
                run_dir
            )
            self.assertTrue(report["current_source_tree"]["available"])
            self.assertEqual(
                report["recorded_sha256"], "a" * 64
            )
            self.assertFalse(report["ok"])
            self.assertTrue(report["error"])

    def test_non_expert_digest_tracks_parameter_updates(self):
        model = experiment.LanguageModel(_tiny_config())
        before = experiment._non_expert_parameter_digest(
            model
        )
        self.assertEqual(
            before,
            experiment._non_expert_parameter_digest(model),
        )
        with torch.no_grad():
            model.token_embedding.weight.add_(1.0)
        self.assertNotEqual(
            before,
            experiment._non_expert_parameter_digest(model),
        )
