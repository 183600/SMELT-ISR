from __future__ import annotations

import datetime
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

def _resolve_repo_root() -> Path:
    """
    定位仓库根目录（experiment.py 所在目录）。

    本文件有两种装载方式：随 exp 包被 exec 到共享命名空间时，
    exp/__init__.py 会把 ``__file__`` 重定向到 <repo>/experiment.py；
    作为独立子模块导入时 ``__file__`` 是 <repo>/exp/本文件。
    两种情况都要得到同一仓库根，否则源码树指纹会指向错误目录、
    静默产出空指纹（空串的 SHA-256），归档看起来成功实则无效。
    """
    entry = Path(__file__).resolve()
    if entry.name == "experiment.py":
        return entry.parent
    if entry.parent.name == "exp":
        return entry.parent.parent
    for parent in entry.parents:
        if (parent / "experiment.py").is_file():
            return parent
    return entry.parent


_REPO_ROOT = _resolve_repo_root()
_ENTRY_SCRIPT = _REPO_ROOT / "experiment.py"
_RUNTIME_DIRS = ("exp", "selfcheck")
_SKIP_NAMES = frozenset({".pyc", ".pyo"})

PACKAGE_META_NAME = "package_meta.json"
SOURCE_TREE_NAME = "source_tree_fingerprint.json"
INSTALLED_PACKAGES_NAME = "installed_packages.json"
REPRODUCE_NAME = "reproduce_analysis.py"
RUNS_DIR_NAME = "runs"

EVIDENCE_FILES = (
    "config.json",
    "metrics.jsonl",
    "environment.json",
    "comparison_analysis.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Optional[object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _safe_rel_name(path: Path, root: Path) -> Optional[str]:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    parts = relative.parts
    if any(part in ("", ".", "..") for part in parts):
        return None
    return relative.as_posix()


def _canonical_json_sha256_local(value: object) -> str:
    # 与 exp/part_037 的 _canonical_json_sha256 同一口径；独立
    # 子模块导入路径下的兜底实现，保证两种装载方式产出同一指纹。
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _source_tree_fingerprint() -> dict:
    """运行时代码指纹（experiment.py + exp/ + selfcheck/）。

    优先复用 exp/part_037 的官方 ``source_tree_fingerprint()``：
    指纹必须与 ``--print-source-fingerprint`` / ``--check-source``
    完全同口径，归档内的 source_tree_fingerprint.json 才能与
    当前进程核对。作为独立子模块导入（无共享命名空间）时退回
    自包含实现，并用与官方一致的 canonical JSON 摘要。
    """
    official = globals().get("source_tree_fingerprint")
    if callable(official):
        try:
            record = official()
        except Exception:  # pragma: no cover - 防御：官方实现异常时退回本地实现
            record = None
        if isinstance(record, dict) and record.get("available"):
            return {
                "schema": record.get("schema"),
                "root": str(_REPO_ROOT),
                "file_count": record.get("file_count"),
                "total_bytes": record.get("total_bytes"),
                "files": record.get("files"),
                "sha256": record.get("sha256"),
            }
    files = []
    if _ENTRY_SCRIPT.is_file():
        files.append(("experiment.py", _ENTRY_SCRIPT.stat().st_size, _sha256_file(_ENTRY_SCRIPT)))
    for dirname in _RUNTIME_DIRS:
        directory = _REPO_ROOT / dirname
        if not directory.is_dir():
            continue
        # 与官方口径一致：只计入 Python 源文件，跳过 __pycache__。
        for path in sorted(directory.rglob("*.py")):
            if "__pycache__" in path.parts or not path.is_file():
                continue
            relative = _safe_rel_name(path, _REPO_ROOT)
            if relative is None:
                continue
            files.append((relative, path.stat().st_size, _sha256_file(path)))
    files.sort(key=lambda item: item[0])
    entries = [
        {"path": rel, "bytes": size, "sha256": sha}
        for rel, size, sha in files
    ]
    digest = _canonical_json_sha256_local(
        {
            "schema": "smelt-isr/source-tree/v1",
            "files": entries,
        }
    )
    return {
        "schema": "smelt-isr/source-tree/v1",
        "root": str(_REPO_ROOT),
        "file_count": len(files),
        "files": entries,
        "sha256": digest,
    }


def _git_output(*args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_archive_into(dest: Path) -> bool:
    """把 git archive --format=tar HEAD 写入 dest，成功返回 True。"""
    try:
        with open(dest, "wb") as handle:
            result = subprocess.run(
                ["git", "archive", "--format=tar", "HEAD"],
                cwd=str(_REPO_ROOT),
                stdout=handle,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def _git_diff_text() -> Optional[str]:
    return _git_output("diff", "HEAD")


def _repro_installed_distributions() -> list:
    """复现包用的原始发行包清单（保留包名大小写、不去重）。

    必须与 part_037 的环境快照实现区分命名：后者按 PEP 503
    规范化、去重并按 name 排序，environment_id 与关键依赖
    (critical) 判定都依赖它。同名定义会在模块装配时覆盖那份
    实现，破坏环境快照口径。
    """
    try:
        import importlib.metadata as metadata
    except ImportError:  # pragma: no cover - Python < 3.10 not supported
        return []
    items = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        version = dist.version
        if name:
            items.append({"name": str(name), "version": str(version)})
    items.sort(key=lambda item: item["name"].lower())
    return items


def _environment_id_from_config(config: Optional[dict]) -> Optional[str]:
    if not isinstance(config, dict):
        return None
    environment = config.get("environment")
    if isinstance(environment, dict):
        value = environment.get("environment_id")
        if isinstance(value, str) and value:
            return value
    metadata = config.get("run_metadata")
    if isinstance(metadata, dict):
        env = metadata.get("environment")
        if isinstance(env, dict):
            value = env.get("environment_id")
            if isinstance(value, str) and value:
                return value
    return None


def _collect_run_evidence(run_dir: Path) -> dict:
    evidence: dict = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "files": {},
        "summary": {},
    }
    config = _read_json(run_dir / "config.json")
    if isinstance(config, dict):
        evidence["files"]["config.json"] = config
        preregistration = config.get("preregistration")
        if not isinstance(preregistration, dict):
            preregistration = {}
        data = config.get("data")
        if not isinstance(data, dict):
            data = {}
        metadata = config.get("run_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        environment = config.get("environment")
        if not isinstance(environment, dict):
            environment = {}
        evidence["summary"] = {
            "study_id": preregistration.get("study_id"),
            "hypothesis": config.get("hypothesis"),
            "research_question": config.get("research_question"),
            "run_started_utc": metadata.get("timestamp_utc"),
            "script_sha256": metadata.get("script_sha256"),
            "source_tree_sha256": metadata.get("source_tree_sha256"),
            "git_commit": metadata.get("git_commit"),
            "environment_id": environment.get("environment_id"),
            "environment_file_sha256": environment.get("file_sha256"),
            "data_sha256": data.get("sha256"),
            "split_policy": data.get("split_policy"),
            "comparison_protocol_sha256": config.get(
                "comparison_protocol_sha256"
            ),
            "command": metadata.get("command"),
            "preregistration_file_sha256": preregistration.get("file_sha256"),
        }
    for name in EVIDENCE_FILES:
        path = run_dir / name
        if path.is_file():
            evidence["files"][name] = path.read_bytes()
    for path in sorted(run_dir.glob("routing*.json")):
        evidence["files"][path.name] = path.read_bytes()
    return evidence


_REPRODUCE_SCRIPT = '''#!/usr/bin/env python3
"""由 export_reproducibility_package 生成的复现脚本样本。

读取归档内各运行目录的 config.json / metrics.jsonl，给出用于复核
`--compare` 结论的等价命令。此脚本只产出命令与文件清单，不自行
训练；真正的重跑需在拥有相同数据、代码与环境的机器上手工执行。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
# Reproducibility: Set random seeds
import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass


def main(archive_root: str) -> int:
    root = Path(archive_root)
    runs_dir = root / "runs"
    if not runs_dir.is_dir():
        print(f"归档内未发现 runs/ 目录: {runs_dir}", file=sys.stderr)
        return 2
    runs = sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "config.json").is_file())
    if not runs:
        print(f"未在 {runs_dir} 中发现含 config.json 的运行目录", file=sys.stderr)
        return 2
    print(f"归档内发现 {len(runs)} 个运行目录，可用于 --compare 复核：")
    for run in runs:
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        prereg = config.get("preregistration") or {}
        print(
            f"  {run.name}: study_id={prereg.get('study_id')!r} "
            f"command={config.get('run_metadata', {}).get('command')!r}"
        )
    print()
    print("建议在等价环境下执行（示例，先用 --check-environment 核对环境）：")
    print(
        "  python experiment.py --compare "
        + " ".join(str(r) for r in runs)
        + " --compare-report analysis/comparison_analysis.json"
    )
    print()
    print("环境核对示例：")
    print("  python experiment.py --check-environment <某一运行目录>")
    print("  python experiment.py --check-environment --check-environment-strict")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
'''


def repro_package_mode_error(args) -> Optional[str]:
    """
    校验 --export-repro-package 的互斥与搭配。

    复现包导出是只读的归档工具：不需要 --data/--out/种子，也
    不能与训练、预注册、对比等入口混用，否则用户很容易以为
    「只是打包」而实际又启动了一次训练，或反过来把训练参数
    静默忽略。
    """
    run_dirs = getattr(args, "export_repro_package", None)
    if run_dirs is None:
        if getattr(args, "repro_package_format", None) not in (
            None,
            "zip",
        ) or getattr(args, "repro_package_out", None) not in (
            None,
            "reproducibility_package",
        ):
            return (
                "--repro-package-format / --repro-package-out 只能配合 "
                "--export-repro-package 使用"
            )
        return None
    conflicts = [
        flag
        for flag, active in (
            ("--compare", getattr(args, "compare", None) is not None),
            (
                "--compare-report",
                getattr(args, "compare_report", None) is not None,
            ),
            (
                "--compare-csv",
                getattr(args, "compare_csv", None) is not None,
            ),
            (
                "--final-test-comparison",
                getattr(args, "final_test_comparison", None) is not None,
            ),
            ("--resume", getattr(args, "resume", None) is not None),
            (
                "--evaluate-test",
                bool(getattr(args, "evaluate_test", False)),
            ),
            ("--sanity-only", bool(getattr(args, "sanity_only", False))),
            (
                "--statistical-baseline",
                bool(getattr(args, "statistical_baseline", False)),
            ),
            (
                "--suite-baseline-tuning",
                bool(getattr(args, "suite_baseline_tuning", False)),
            ),
            ("--plan-design", bool(getattr(args, "plan_design", False))),
            (
                "--design-pilot-runs",
                getattr(args, "design_pilot_runs", None) is not None,
            ),
            (
                "--preregistration",
                getattr(args, "preregistration", None) is not None,
            ),
            (
                "--write-manifest",
                getattr(args, "write_manifest", None) is not None,
            ),
            (
                "--verify-manifest",
                getattr(args, "verify_manifest", None) is not None,
            ),
            (
                "--check-environment",
                getattr(args, "check_environment", None) is not None,
            ),
            (
                "--write-data-plan",
                getattr(args, "write_data_plan", None) is not None,
            ),
            (
                "--check-data-plan",
                getattr(args, "check_data_plan", None) is not None,
            ),
        )
        if active
    ]
    if conflicts:
        return (
            "--export-repro-package 是只读归档入口，不能与以下"
            "训练/评估/核对参数同时使用（否则会被静默忽略）: "
            + ", ".join(conflicts)
        )
    return None


def export_reproducibility_package(
    run_dirs,
    out_dir,
    *,
    format: str = "zip",
) -> Path:
    """把一个或多个运行目录的证据链打包成自包含可复现性归档。

    归档包含：
      - runs/<run_name>/  ：config.json、metrics.jsonl、environment.json、
        comparison_analysis.json、routing*.json 及 per-run summary.json；
      - source_tree_fingerprint.json：experiment.py + exp/ + selfcheck/ 的
        文件内容 SHA-256 指纹（不依赖 git，可跨主机核对代码身份）；
      - source_tree.tar：git archive HEAD（若 git 可用且仓库完整）；
      - installed_packages.json：importlib.metadata 列出的已安装发行包；
      - package_meta.json：导出元数据（时间、格式、运行列表、各运行
        摘要、环境 ID）；
      - reproduce_analysis.py：复核命令样本（不自行训练）。

    参数 ``format`` 接受 "zip" 或 "tar.gz"。返回归档文件路径。
    不修改任何运行目录，只读取。
    """
    run_dirs = [Path(d) for d in run_dirs]
    for directory in run_dirs:
        if not directory.is_dir():
            raise ValueError(f"运行目录不存在: {directory}")
    if not run_dirs:
        raise ValueError("run_dirs 不能为空：至少需要一个运行目录")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = format.lower()
    if fmt not in ("zip", "tar.gz", "tgz"):
        raise ValueError(f"不支持的归档格式: {format!r}，应为 zip 或 tar.gz")

    tmp = Path(tempfile.mkdtemp(prefix="smelt_repro_package_"))
    try:
        payload_dir = tmp / "payload"
        payload_dir.mkdir()

        # 代码快照
        source_tree = _source_tree_fingerprint()
        _write_json(payload_dir / SOURCE_TREE_NAME, source_tree)

        git_archive_path = tmp / "source_tree.tar"
        git_available = _git_archive_into(git_archive_path)
        git_diff = _git_diff_text()

        # 依赖 / 环境
        _write_json(
            payload_dir / INSTALLED_PACKAGES_NAME,
            _repro_installed_distributions(),
        )

        # 逐运行收集证据
        runs_payload_dir = payload_dir / RUNS_DIR_NAME
        runs_payload_dir.mkdir()
        run_summaries = []
        environment_ids = set()
        for directory in run_dirs:
            evidence = _collect_run_evidence(run_dir=directory)
            run_target = runs_payload_dir / directory.name
            run_target.mkdir()
            for name, content in evidence["files"].items():
                target_path = run_target / name
                if isinstance(content, dict):
                    target_path.write_text(
                        json.dumps(content, ensure_ascii=False, indent=2)
                        + "\n",
                        encoding="utf-8",
                    )
                else:
                    target_path.write_bytes(content)
            summary = dict(evidence["summary"])
            summary["run_name"] = evidence["run_name"]
            summary["files_archived"] = sorted(evidence["files"])
            _write_json(run_target / "summary.json", summary)
            run_summaries.append(summary)
            eid = summary.get("environment_id")
            if isinstance(eid, str) and eid:
                environment_ids.add(eid)

        package_meta = {
            "schema": "smelt-isr/reproducibility-package/v1",
            "generated_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "format": fmt,
            "repository_root": str(_REPO_ROOT),
            "source_tree_sha256": source_tree["sha256"],
            "source_tree_file_count": source_tree["file_count"],
            "git_available": bool(git_available),
            "git_dirty": bool(git_diff),
            "git_diff_sha256": (
                hashlib.sha256(git_diff.encode("utf-8")).hexdigest()
                if git_diff
                else None
            ),
            "environment_ids": sorted(environment_ids),
            "installed_package_count": len(_repro_installed_distributions()),
            "runs": run_summaries,
        }
        _write_json(payload_dir / PACKAGE_META_NAME, package_meta)
        (payload_dir / REPRODUCE_NAME).write_text(
            _REPRODUCE_SCRIPT, encoding="utf-8"
        )

        if fmt == "zip":
            archive_path = out_dir / "reproducibility_package.zip"
            if archive_path.exists():
                archive_path.unlink()
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for path in sorted(payload_dir.rglob("*")):
                    if path.is_file():
                        arcname = path.relative_to(payload_dir).as_posix()
                        zipf.write(path, arcname=arcname)
        else:
            suffix = ".tar.gz"
            archive_path = out_dir / ("reproducibility_package" + suffix)
            if archive_path.exists():
                archive_path.unlink()
            with tarfile.open(archive_path, "w:gz") as tarf:
                for path in sorted(payload_dir.rglob("*")):
                    if path.is_file():
                        arcname = path.relative_to(payload_dir).as_posix()
                        tarf.add(path, arcname=arcname)
        return archive_path
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def list_package_contents(archive_path: str) -> list:
    """列出归档内的相对路径（用于测试/审计只读复核）。"""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise ValueError(f"归档文件不存在: {archive_path}")
    names = []
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zipf:
            names = sorted(zipf.namelist())
    else:
        with tarfile.open(archive_path) as tarf:
            names = sorted(member.name for member in tarf.getmembers() if member.isfile())
    return names
