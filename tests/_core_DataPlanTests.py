from _core_support import *
from _core_support import _load_experiment, _tiny_config
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

class DataPlanTests(unittest.TestCase):
    """数据协议准备（data plan）：生成、冻结、核对与训练入口口径一致。"""

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
    def _write_corpus(
        cls,
        root: Path,
        *,
        groups: int = 12,
        group_bytes: int = 64,
        separator: bytes = b"<|doc|>",
    ) -> Path:
        path = root / "corpus.txt"
        docs = [
            cls._printable(f"plan-doc-{index}", group_bytes)
            for index in range(groups)
        ]
        path.write_bytes(separator.join(docs))
        return path

    def _args(
        self,
        corpus: Path,
        *,
        seq_len: int = 16,
        **overrides,
    ) -> argparse.Namespace:
        values = dict(
            data=str(corpus),
            seq_len=seq_len,
            split_policy="contiguous",
            val_fraction=0.2,
            test_fraction=0.2,
            group_separator="<|doc|>",
            time_order_manifest=None,
            entity_manifest=None,
            deduplicate_train_groups=False,
            drop_train_shared_chunks=False,
            drop_train_near_duplicate_groups=False,
            near_duplicate_audit=False,
            near_duplicate_threshold=0.8,
            near_duplicate_shingle_bytes=8,
            drop_train_corrupt_groups=False,
            drop_train_corrupt_bytes=False,
            drop_train_degenerate_groups=False,
            drop_train_degenerate_runs=False,
            clean_train_against_test=False,
            allow_eval_content_for_cleaning=False,
            domain_justification=None,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def _dataset(self, args: argparse.Namespace):
        separator = experiment.parse_group_separator(
            args.group_separator
        )
        return experiment.build_byte_dataset(
            args, int(args.seq_len), separator, device=None
        )

    def _plan(self, args: argparse.Namespace):
        separator, evidence = experiment.data_protocol_cli_setup(
            args
        )
        dataset = self._dataset(args)
        return experiment.build_data_plan(
            args, dataset, time_order_evidence=evidence
        )

    def test_plan_roundtrip_and_freeze_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._write_corpus(root)
            args = self._args(corpus)
            plan = self._plan(args)
            self.assertEqual(
                plan["corpus"]["sha256"],
                self._dataset(args).data_sha256,
            )
            block = plan["preregistration_data_block"]
            self.assertEqual(
                block["sha256"], plan["corpus"]["sha256"]
            )
            self.assertEqual(
                block["group_separator"], "<|doc|>"
            )
            self.assertFalse(block["cleaning"][
                "deduplicate_train_groups"
            ])
            self.assertEqual(
                block["split_policy"], "contiguous"
            )
            target = root / "data_plan.json"
            written = experiment.write_data_plan(target, plan)
            self.assertTrue(written["written"])
            self.assertEqual(
                written["plan_id"], plan["plan_id"]
            )
            reloaded = experiment.load_data_plan(target)
            self.assertEqual(
                reloaded["plan_id"], plan["plan_id"]
            )
            # 同一计划重复写入保留已冻结的文件字节。
            again = experiment.write_data_plan(target, plan)
            self.assertFalse(again["written"])
            self.assertEqual(
                again["file_sha256"], written["file_sha256"]
            )
            # 内容不同的计划不能覆盖冻结路径。
            mutated = json.loads(
                json.dumps(plan, ensure_ascii=False)
            )
            mutated["corpus"]["bytes"] = int(
                mutated["corpus"]["bytes"]
            ) + 1
            mutated["plan_id"] = experiment.data_plan_id(
                {
                    key: value
                    for key, value in mutated.items()
                    if key not in ("plan_id", "created_utc")
                }
            )
            with self.assertRaises(ValueError):
                experiment.write_data_plan(target, mutated)

    def test_plan_freezes_data_processing_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._write_corpus(root)
            args = self._args(corpus)
            plan = self._plan(args)
            processing = plan["processing"]
            protocol = processing["protocol"]
            self.assertEqual(
                protocol["statistical_fit_scope"], "train_only"
            )
            self.assertEqual(
                plan["preregistration_data_block"]["processing"],
                protocol,
            )
            self.assertEqual(
                plan["preregistration_data_block"]["processing"][
                    "effective_train_sha256"
                ],
                self._dataset(args).effective_train_sha256,
            )
            declared = json.loads(
                json.dumps(
                    plan["preregistration_data_block"],
                    ensure_ascii=False,
                )
            )
            declared["processing"][
                "effective_train_sha256"
            ] = "0" * 64
            report = experiment.check_data_plan(
                {"data": declared},
                plan,
                args=args,
                dataset=self._dataset(args),
            )
            self.assertFalse(report["ok"])
            fields = {
                difference["field"]
                for difference in report["differences"]
            }
            self.assertIn(
                "data.processing.effective_train_sha256",
                fields,
            )

    def test_load_data_plan_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._write_corpus(root)
            plan = self._plan(self._args(corpus))
            target = root / "data_plan.json"
            experiment.write_data_plan(target, plan)
            tampered = json.loads(
                target.read_text(encoding="utf-8")
            )
            tampered["split"]["val_fraction"] = 0.9
            target.write_text(
                json.dumps(tampered, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                experiment.load_data_plan(target)
            list_file = root / "list.json"
            list_file.write_text("[1, 2]", encoding="utf-8")
            with self.assertRaises(ValueError):
                experiment.load_data_plan(list_file)

    def test_check_data_plan_passes_for_matching_preregistration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._write_corpus(root)
            args = self._args(corpus)
            plan = self._plan(args)
            payload = {
                "data": plan["preregistration_data_block"],
                "metadata": {"data_plan_id": plan["plan_id"]},
            }
            report = experiment.check_data_plan(
                payload, plan, args=args, dataset=self._dataset(args)
            )
            self.assertTrue(report["ok"])
            self.assertEqual(report["differences"], [])
            self.assertIsNone(report["gate_error"])
            self.assertEqual(
                report["metadata_data_plan_id_status"], "matched"
            )

    def test_check_data_plan_detects_declared_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._write_corpus(root)
            args = self._args(corpus)
            plan = self._plan(args)
            declared = json.loads(
                json.dumps(
                    plan["preregistration_data_block"],
                    ensure_ascii=False,
                )
            )
            declared["sha256"] = "0" * 64
            declared["val_fraction"] = 0.5
            declared["cleaning"][
                "deduplicate_train_groups"
            ] = True
            payload = {"data": declared}
            report = experiment.check_data_plan(
                payload, plan, args=args, dataset=self._dataset(args)
            )
            self.assertFalse(report["ok"])
            fields = {
                difference["field"]
                for difference in report["differences"]
            }
            self.assertIn("data.sha256", fields)
            self.assertIn("data.val_fraction", fields)
            self.assertIn(
                "data.cleaning.deduplicate_train_groups", fields
            )
            self.assertIsNotNone(report["gate_error"])

    def test_check_data_plan_metadata_plan_id_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._write_corpus(root)
            args = self._args(corpus)
            plan = self._plan(args)
            payload = {
                "data": plan["preregistration_data_block"],
                "metadata": {"data_plan_id": "0" * 64},
            }
            report = experiment.check_data_plan(payload, plan)
            self.assertFalse(report["ok"])
            self.assertEqual(
                report["metadata_data_plan_id_status"],
                "mismatched",
            )
            fields = {
                difference["field"]
                for difference in report["differences"]
            }
            self.assertIn("metadata.data_plan_id", fields)

    def test_protocol_setup_enforces_cleaning_and_time_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._write_corpus(root)
            no_separator = self._args(
                corpus, group_separator=None,
                deduplicate_train_groups=True,
            )
            with self.assertRaises(ValueError):
                experiment.data_protocol_cli_setup(no_separator)
            no_justification = self._args(
                corpus,
                group_separator=None,
                drop_train_corrupt_bytes=True,
            )
            with self.assertRaises(ValueError):
                experiment.data_protocol_cli_setup(
                    no_justification
                )
            justified = self._args(
                corpus,
                group_separator=None,
                drop_train_corrupt_bytes=True,
                domain_justification="语料混入二进制附件，非任务文本",
            )
            separator, evidence = (
                experiment.data_protocol_cli_setup(justified)
            )
            self.assertIsNone(separator)
            self.assertIsNone(evidence)
            self.assertEqual(
                justified.cleaning_domain_justification,
                "语料混入二进制附件，非任务文本",
            )
            bad_threshold = self._args(
                corpus, near_duplicate_threshold=0.0
            )
            with self.assertRaises(ValueError):
                experiment.data_protocol_cli_setup(bad_threshold)
            chronological = self._args(
                corpus, split_policy="chronological"
            )
            with self.assertRaises(ValueError):
                experiment.data_protocol_cli_setup(chronological)
            manifest = root / "time.txt"
            manifest.write_text(
                "2026-01-01T00:00:00Z\n"
                "2026-01-02T00:00:00Z\n"
                "2026-01-03T00:00:00Z\n",
                encoding="utf-8",
            )
            ordered = self._args(
                corpus,
                split_policy="chronological",
                time_order_manifest=str(manifest),
            )
            separator, evidence = (
                experiment.data_protocol_cli_setup(ordered)
            )
            self.assertTrue(evidence["monotonic_non_decreasing"])
            self.assertEqual(
                ordered.time_order_evidence, evidence
            )
            unordered = root / "unordered.txt"
            unordered.write_text(
                "2026-01-03T00:00:00Z\n"
                "2026-01-01T00:00:00Z\n",
                encoding="utf-8",
            )
            violated = self._args(
                corpus,
                split_policy="chronological",
                time_order_manifest=str(unordered),
            )
            with self.assertRaises(ValueError):
                experiment.data_protocol_cli_setup(violated)

    def test_mode_error_guards_preparation_entries(self):
        args = self._args(Path("unused.txt"))
        self.assertIsNone(experiment.data_plan_mode_error(args))
        both = self._args(
            Path("unused.txt"),
            write_data_plan="a.json",
            check_data_plan="b.json",
        )
        message = experiment.data_plan_mode_error(both)
        self.assertIn("不能同时使用", message)
        write_conflict = self._args(
            Path("unused.txt"),
            write_data_plan="a.json",
            compare=["runs/A"],
        )
        message = experiment.data_plan_mode_error(write_conflict)
        self.assertIn("--write-data-plan", message)
        self.assertIn("--compare", message)
        write_with_prereg = self._args(
            Path("unused.txt"),
            write_data_plan="a.json",
            preregistration="prereg.json",
        )
        message = experiment.data_plan_mode_error(
            write_with_prereg
        )
        self.assertIn("--preregistration", message)
        check_missing_prereg = self._args(
            Path("unused.txt"), check_data_plan="plan.json"
        )
        message = experiment.data_plan_mode_error(
            check_missing_prereg
        )
        self.assertIn("--preregistration", message)
        check_ok = self._args(
            Path("unused.txt"),
            check_data_plan="plan.json",
            preregistration="prereg.json",
        )
        self.assertIsNone(experiment.data_plan_mode_error(check_ok))
        check_with_data_plan = self._args(
            Path("unused.txt"),
            check_data_plan="plan.json",
            preregistration="prereg.json",
            data_plan="plan.json",
        )
        message = experiment.data_plan_mode_error(
            check_with_data_plan
        )
        self.assertIn("--data-plan", message)
        train_missing_prereg = self._args(
            Path("unused.txt"), data_plan="plan.json"
        )
        message = experiment.data_plan_mode_error(
            train_missing_prereg
        )
        self.assertIn("--preregistration", message)
        train_ok = self._args(
            Path("unused.txt"),
            data_plan="plan.json",
            preregistration="prereg.json",
        )
        self.assertIsNone(experiment.data_plan_mode_error(train_ok))
        train_conflict = self._args(
            Path("unused.txt"),
            data_plan="plan.json",
            preregistration="prereg.json",
            compare=["runs/A"],
        )
        message = experiment.data_plan_mode_error(train_conflict)
        self.assertIn("--compare", message)

    def test_data_plan_differences_reports_actionable_fields(self):
        declared = {
            "sha256": "a" * 64,
            "split_policy": "contiguous",
            "val_fraction": 0.1,
            "test_fraction": 0.1,
            "group_separator": "<|doc|>",
            "cleaning": {
                "deduplicate_train_groups": False,
                "near_duplicate_threshold": 0.8,
            },
        }
        expected = {
            "sha256": "b" * 64,
            "split_policy": "chronological",
            "val_fraction": 0.2,
            "test_fraction": 0.1,
            "group_separator": "\n\n",
            "cleaning": {
                "deduplicate_train_groups": True,
                "near_duplicate_threshold": 0.7,
            },
        }
        differences = experiment.data_plan_differences(
            declared, expected
        )
        fields = [
            difference["field"] for difference in differences
        ]
        self.assertEqual(
            fields,
            [
                "data.sha256",
                "data.split_policy",
                "data.val_fraction",
                "data.group_separator",
                "data.cleaning.deduplicate_train_groups",
                "data.cleaning.near_duplicate_threshold",
            ],
        )
        for difference in differences:
            self.assertTrue(difference["reason"])
        self.assertEqual(
            experiment.data_plan_differences(
                expected, expected
            ),
            [],
        )

    def test_data_plan_includes_time_order_and_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self._write_corpus(root)
            manifest = root / "time.txt"
            manifest.write_text(
                "2026-01-01T00:00:00Z\n"
                "2026-01-02T00:00:00Z\n",
                encoding="utf-8",
            )
            args = self._args(
                corpus, time_order_manifest=str(manifest)
            )
            plan = self._plan(args)
            self.assertIn(
                "time_order_evidence", plan
            )
            self.assertTrue(
                plan["time_order_evidence"][
                    "monotonic_non_decreasing"
                ]
            )
            isolation = plan["split"]["isolation"]
            self.assertTrue(isolation)
            dataset = self._dataset(args)
            for split, digest in dataset.split_sha256.items():
                self.assertEqual(
                    plan["split"]["sha256"][split], digest
                )

