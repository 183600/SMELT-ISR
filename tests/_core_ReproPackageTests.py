"""可复现性包导出回归测试（不依赖 torch，纯标准库）。"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from exp.part_051_repro_package import (
    export_reproducibility_package,
    list_package_contents,
    _source_tree_fingerprint,
)

_CONFIG_TEMPLATE = {
    "hypothesis": "延迟合并降低验证损失",
    "research_question": "RQ1",
    "run_metadata": {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "script_sha256": "a" * 64,
        "source_tree_sha256": "b" * 64,
        "git_commit": "c" * 40,
        "command": "python experiment.py --mode B --seeds 42",
    },
    "environment": {
        "schema": "smelt-isr/environment/v1",
        "environment_id": "d" * 64,
        "file": "environment.json",
        "file_sha256": "e" * 64,
    },
    "data": {
        "sha256": "f" * 64,
        "split_policy": "contiguous",
        "val_fraction": 0.05,
        "test_fraction": 0.05,
    },
    "comparison_protocol_sha256": "g" * 64,
    "preregistration": {
        "study_id": "smelt-isr-rq1",
        "file_sha256": "h" * 64,
        "frozen_commit": "c" * 40,
    },
}


def _make_run_dir(root: Path, name: str, *, with_metrics: bool = True) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    config = dict(_CONFIG_TEMPLATE)
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False), encoding="utf-8"
    )
    if with_metrics:
        (run_dir / "metrics.jsonl").write_text(
            '{"step": 100, "val_loss": 1.2}\n{"step": 200, "val_loss": 1.1}\n',
            encoding="utf-8",
        )
    (run_dir / "environment.json").write_text(
        json.dumps({"schema": "smelt-isr/environment/v1", "environment_id": "d" * 64}),
        encoding="utf-8",
    )
    return run_dir


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReproPackageTests(unittest.TestCase):
    def test_export_zip_contains_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _make_run_dir(root, "smelt_seed42")
            second = _make_run_dir(root, "B_seed42")
            out_dir = root / "out"
            archive = export_reproducibility_package(
                [first, second], out_dir, format="zip"
            )
            self.assertTrue(archive.exists())
            self.assertEqual(archive.suffix, ".zip")
            names = list_package_contents(archive)
            self.assertIn("package_meta.json", names)
            self.assertIn("source_tree_fingerprint.json", names)
            self.assertIn("installed_packages.json", names)
            self.assertIn("reproduce_analysis.py", names)
            self.assertIn("runs/smelt_seed42/config.json", names)
            self.assertIn("runs/smelt_seed42/metrics.jsonl", names)
            self.assertIn("runs/smelt_seed42/summary.json", names)
            self.assertIn("runs/B_seed42/config.json", names)

            with zipfile.ZipFile(archive) as zipf:
                meta = json.loads(zipf.read("package_meta.json").decode("utf-8"))
            self.assertEqual(meta["format"], "zip")
            self.assertGreaterEqual(meta["source_tree_file_count"], 1)
            self.assertEqual(len(meta["runs"]), 2)
            run_names = {run["run_name"] for run in meta["runs"]}
            self.assertEqual(run_names, {"smelt_seed42", "B_seed42"})
            study_ids = {run["study_id"] for run in meta["runs"]}
            self.assertEqual(study_ids, {"smelt-isr-rq1"})

    def test_export_tar_gz_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _make_run_dir(root, "smelt_seed42")
            out_dir = root / "out"
            archive = export_reproducibility_package(
                [run_dir], out_dir, format="tar.gz"
            )
            self.assertTrue(archive.exists())
            self.assertTrue(archive.name.endswith(".tar.gz"))
            names = list_package_contents(archive)
            self.assertIn("package_meta.json", names)
            self.assertIn("runs/smelt_seed42/config.json", names)

    def test_source_tree_fingerprint_deterministic(self):
        first = _source_tree_fingerprint()
        second = _source_tree_fingerprint()
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertGreaterEqual(first["file_count"], 1)
        self.assertTrue(
            any(item["path"] == "experiment.py" for item in first["files"])
        )
        self.assertTrue(
            any(item["path"].startswith("exp/") for item in first["files"])
        )

    def test_source_tree_fingerprint_records_files(self):
        fingerprint = _source_tree_fingerprint()
        paths = {item["path"] for item in fingerprint["files"]}
        self.assertIn("experiment.py", paths)
        self.assertTrue(any(p.startswith("exp/") for p in paths))
        for item in fingerprint["files"]:
            self.assertIn("path", item)
            self.assertIn("bytes", item)
            self.assertIn("sha256", item)
            self.assertEqual(len(item["sha256"]), 64)

    def test_missing_run_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            with self.assertRaises(ValueError):
                export_reproducibility_package(
                    [Path(tmp) / "does_not_exist"], out_dir
                )

    def test_empty_run_dirs_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            with self.assertRaises(ValueError):
                export_reproducibility_package([], out_dir)

    def test_run_summary_preserves_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _make_run_dir(root, "smelt_seed42")
            out_dir = root / "out"
            archive = export_reproducibility_package(
                [run_dir], out_dir, format="zip"
            )
            with zipfile.ZipFile(archive) as zipf:
                summary = json.loads(
                    zipf.read("runs/smelt_seed42/summary.json").decode("utf-8")
                )
        self.assertEqual(summary["study_id"], "smelt-isr-rq1")
        self.assertEqual(summary["environment_id"], "d" * 64)
        self.assertEqual(summary["data_sha256"], "f" * 64)
        self.assertEqual(summary["script_sha256"], "a" * 64)
        self.assertIn("config.json", summary["files_archived"])
        self.assertIn("metrics.jsonl", summary["files_archived"])

    def test_export_does_not_modify_run_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _make_run_dir(root, "smelt_seed42")
            before = {
                name: _sha256_file(run_dir / name)
                for name in ("config.json", "metrics.jsonl", "environment.json")
                if (run_dir / name).is_file()
            }
            out_dir = root / "out"
            export_reproducibility_package([run_dir], out_dir, format="zip")
            after = {
                name: _sha256_file(run_dir / name)
                for name in before
                if (run_dir / name).is_file()
            }
        self.assertEqual(before, after)

    def test_invalid_format_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _make_run_dir(root, "smelt_seed42")
            out_dir = root / "out"
            with self.assertRaises(ValueError):
                export_reproducibility_package(
                    [run_dir], out_dir, format="7z"
                )


if __name__ == "__main__":
    unittest.main()
