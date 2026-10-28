from _core_support import *
from _core_support import _load_experiment, _tiny_config

class PreregistrationTests(unittest.TestCase):
    """外部预注册文件：结构校验、命令行冲突与数据指纹核对。"""

    def _payload(self):
        return {
            "study_id": "smelt-isr-test",
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
            "data": {
                "sha256": "b" * 64,
                "split_policy": "contiguous",
                "val_fraction": 0.1,
                "test_fraction": 0.1,
            },
        }

    def _args(self, **overrides):
        values = dict(
            primary_metric=None,
            hypothesis_direction=None,
            hypothesis_min_effect=None,
            hypothesis_baseline=None,
            common_step=None,
            budget_match=None,
            equivalence_margin=0.01,
            steps=200,
            seeds=None,
            seed=42,
            mode="A",
            statistical_baseline=False,
        )
        values.update(overrides)
        return experiment.argparse.Namespace(**values)

    def _dataset(self):
        return experiment.argparse.Namespace(
            data_sha256="b" * 64,
            val_fraction=0.1,
            test_fraction=0.1,
        )

    def test_schema_requires_full_protocol(self):
        payload = self._payload()
        payload.pop("common_step")
        self.assertIn(
            "common_step",
            experiment.preregistration_schema_error(payload),
        )
        payload = self._payload()
        payload["seeds"] = [42, 43]
        self.assertIn(
            "至少需要",
            experiment.preregistration_schema_error(payload),
        )
        payload = self._payload()
        payload["comparisons"] = []
        self.assertIn(
            "comparisons",
            experiment.preregistration_schema_error(payload),
        )
        payload = self._payload()
        payload["seed"] = [42, 43, 44]
        self.assertIn(
            "未知字段",
            experiment.preregistration_schema_error(payload),
        )

    def test_loader_rejects_duplicate_keys(self):
        payload = self._payload()
        text = json.dumps(payload, ensure_ascii=False)
        text = text.replace(
            '"study_id": "smelt-isr-test"',
            '"study_id": "smelt-isr-test", '
            '"study_id": "duplicate"',
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prereg.json"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError) as context:
                experiment.load_preregistration(path)
        self.assertIn("重复键", str(context.exception))

    def test_loader_accepts_valid_file(self):
        payload = self._payload()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prereg.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            loaded, digest = experiment.load_preregistration(path)
        self.assertEqual(loaded["study_id"], "smelt-isr-test")
        self.assertEqual(len(digest), 64)

    def test_apply_fills_omitted_cli_values(self):
        args = self._args()
        applied, error = experiment.preregistration_apply(
            args,
            self._payload(),
            current_commit="a" * 40,
            working_tree_dirty=False,
            current_script_sha256="c" * 64,
        )
        self.assertIsNone(error)
        self.assertIn("primary_metric", applied)
        self.assertIn("seeds", applied)
        self.assertEqual(args.primary_metric, "val_loss")
        self.assertEqual(args.hypothesis_direction, "decrease")
        self.assertEqual(args.common_step, 100)
        self.assertEqual(args.seeds, [42, 43, 44])

    def test_apply_rejects_explicit_conflict(self):
        args = self._args(primary_metric="val_macro_f1")
        applied, error = experiment.preregistration_apply(
            args,
            self._payload(),
            current_commit="a" * 40,
            working_tree_dirty=False,
        )
        self.assertIsNotNone(error)
        self.assertIn("primary_metric", error)

    def test_apply_numeric_conflict_keeps_large_integer_precision(self):
        payload = self._payload()
        payload["common_step"] = 2 ** 53
        payload["steps"] = 2 ** 53 + 2
        args = self._args(
            common_step=2 ** 53 + 1,
            steps=2 ** 53 + 2,
        )
        _, error = experiment.preregistration_apply(
            args,
            payload,
            current_commit="a" * 40,
            working_tree_dirty=False,
        )
        self.assertIsNotNone(error)
        self.assertIn("common_step", error)

    def test_apply_rejects_commit_dirty_and_script_mismatch(self):
        args = self._args()
        _, error = experiment.preregistration_apply(
            args,
            self._payload(),
            current_commit="c" * 40,
            working_tree_dirty=False,
        )
        self.assertIn("frozen_commit", error)

        args = self._args()
        _, error = experiment.preregistration_apply(
            args,
            self._payload(),
            current_commit="a" * 40,
            working_tree_dirty=True,
        )
        self.assertIn("未提交修改", error)

        payload = self._payload()
        payload["frozen_script_sha256"] = "d" * 64
        args = self._args()
        _, error = experiment.preregistration_apply(
            args,
            payload,
            current_commit=None,
            working_tree_dirty=False,
            current_script_sha256="e" * 64,
        )
        self.assertIn("SHA-256", error)

    def test_apply_rejects_unverifiable_frozen_script(self):
        payload = self._payload()
        payload["frozen_script_sha256"] = "d" * 64
        args = self._args()
        _, error = experiment.preregistration_apply(
            args,
            payload,
            current_commit="a" * 40,
            working_tree_dirty=False,
            current_script_sha256=None,
        )
        self.assertIn("无法读取", error)
        self.assertIn("SHA-256", error)

    def test_apply_accepts_uppercase_frozen_script_sha256(self):
        payload = self._payload()
        payload["frozen_script_sha256"] = "D" * 64
        args = self._args()
        _, error = experiment.preregistration_apply(
            args,
            payload,
            current_commit="a" * 40,
            working_tree_dirty=False,
            current_script_sha256="d" * 64,
        )
        self.assertIsNone(error)

    def test_apply_allows_frozen_commit_ancestor(self):
        args = self._args()
        _, error = experiment.preregistration_apply(
            args,
            self._payload(),
            current_commit="c" * 40,
            working_tree_dirty=False,
            frozen_code_matches=True,
        )
        self.assertIsNone(error)

    def test_apply_fills_steps_only_when_not_explicit(self):
        payload = self._payload()
        payload["steps"] = 10
        payload["common_step"] = 5
        args = self._args(steps=2000)
        previous = sys.argv
        try:
            sys.argv = ["experiment.py"]
            applied, error = experiment.preregistration_apply(
                args,
                payload,
                current_commit="a" * 40,
                working_tree_dirty=False,
            )
            self.assertIsNone(error)
            self.assertIn("steps", applied)
            self.assertEqual(args.steps, 10)

            args = self._args(steps=2000)
            sys.argv = ["experiment.py", "--steps", "2000"]
            applied, error = experiment.preregistration_apply(
                args,
                payload,
                current_commit="a" * 40,
                working_tree_dirty=False,
            )
            self.assertIn("steps", error)
            self.assertEqual(args.steps, 2000)
        finally:
            sys.argv = previous

    def test_apply_rejects_common_step_beyond_steps(self):
        payload = self._payload()
        payload.pop("steps")
        args = self._args(steps=50)
        _, error = experiment.preregistration_apply(
            args,
            payload,
            current_commit="a" * 40,
            working_tree_dirty=False,
        )
        self.assertIn("common_step", error)

    def test_apply_checks_planned_suite_modes(self):
        payload = self._payload()
        payload["comparisons"] = [["smelt", "A"]]
        args = self._args(mode="B", suite_modes=["A"])
        _, error = experiment.preregistration_apply(
            args,
            payload,
            current_commit="a" * 40,
            working_tree_dirty=False,
        )
        self.assertIsNone(error)

        args = self._args(mode="B", suite_modes=["A", "B"])
        _, error = experiment.preregistration_apply(
            args,
            payload,
            current_commit="a" * 40,
            working_tree_dirty=False,
        )
        self.assertIn("B", error)

    def test_apply_defers_resume_mode_check(self):
        payload = self._payload()
        payload["comparisons"] = [["base", "smelt"]]
        args = self._args(
            mode="B",
            resume="checkpoint.pt",
        )
        previous = sys.argv
        try:
            sys.argv = ["experiment.py", "--resume", "checkpoint.pt"]
            _, error = experiment.preregistration_apply(
                args,
                payload,
                current_commit="a" * 40,
                working_tree_dirty=False,
            )
        finally:
            sys.argv = previous
        self.assertIsNone(error)
        args.preregistration_record = {
            "comparisons": [["base", "smelt"]],
        }
        args.mode = "base"
        self.assertIsNone(
            experiment._preregistration_restored_mode_error(args)
        )
        args.mode = "A"
        self.assertIn(
            "不在预注册",
            experiment._preregistration_restored_mode_error(args),
        )

    def test_apply_defers_resume_seed_check(self):
        payload = self._payload()
        args = self._args(
            resume="checkpoint.pt",
            seed=42,
        )
        previous = sys.argv
        try:
            sys.argv = ["experiment.py", "--resume", "checkpoint.pt"]
            applied, error = experiment.preregistration_apply(
                args,
                payload,
                current_commit="a" * 40,
                working_tree_dirty=False,
            )
        finally:
            sys.argv = previous
        self.assertIsNone(error)
        self.assertIsNone(args.seeds)
        # 恢复后的 seed 才按预注册集合核对。
        args.preregistration_record = {"seeds": [1, 2, 3]}
        args.seed = 42
        self.assertIn(
            "不在预注册",
            experiment._preregistration_restored_seed_error(args),
        )
        args.seed = 2
        self.assertIsNone(
            experiment._preregistration_restored_seed_error(args)
        )

    def test_apply_seed_sets_are_fixed(self):
        args = self._args(seeds=[42, 43])
        _, error = experiment.preregistration_apply(
            args,
            self._payload(),
            current_commit="a" * 40,
            working_tree_dirty=False,
        )
        self.assertIn("seeds", error)

        args = self._args(seed=99)
        original_argv = sys.argv
        try:
            sys.argv = ["experiment.py", "--seed", "99"]
            _, error = experiment.preregistration_apply(
                args,
                self._payload(),
                current_commit="a" * 40,
                working_tree_dirty=False,
            )
        finally:
            sys.argv = original_argv
        self.assertIn("不在预注册", error)

    def test_protocol_fallback_includes_cleaning_exemptions(self):
        protocol = experiment.experiment_protocol_from_payload({
            "config": {"mode": "A"},
            "args": {
                "lr": 1e-3,
                "clean_train_against_test": False,
                "allow_eval_content_for_cleaning": True,
                "allow_entity_overlap": True,
            },
        })
        self.assertTrue(
            protocol["allow_eval_content_for_cleaning"]
        )
        self.assertTrue(protocol["allow_entity_overlap"])
        # 旧配置完全没有 comparison_protocol 时，这两个键也
        # 必须出现在回填协议里；否则受控变量检查只能报「缺
        # 记录」，无法识别同口径。
        self.assertIn(
            "allow_eval_content_for_cleaning",
            experiment.COMPARISON_PROTOCOL_KEYS,
        )
        for key in (
            "allow_eval_content_for_cleaning",
            "allow_entity_overlap",
        ):
            self.assertIn(key, protocol)

    def test_protocol_fallback_preserves_recorded_lr_schedule(self):
        payload = {
            "config": {"mode": "A"},
            "args": {
                "lr": 1e-3,
                "lr_schedule": "wsd",
                "steps": 100,
            },
        }
        protocol = experiment.experiment_protocol_from_payload(
            payload
        )
        self.assertEqual(protocol["lr_schedule_name"], "wsd")
        self.assertEqual(protocol["lr_schedule"], "wsd")

        payload["args"]["lr_schedule"] = "cosine"
        protocol = experiment.experiment_protocol_from_payload(
            payload
        )
        self.assertEqual(
            protocol["lr_schedule"],
            "linear_warmup_cosine_to_10pct",
        )
        self.assertEqual(protocol["lr_schedule_name"], "cosine")

    def test_saved_cleaning_alias_uses_effective_or(self):
        # 旧检查点可能只把 allow_eval_content_for_cleaning 记为
        # True，而 clean_train_against_test 仍是 argparse 默认
        # False；有效口径应为 OR，不能按单键取 False。
        self.assertTrue(
            experiment._saved_clean_against_test_effective({
                "clean_train_against_test": False,
                "allow_eval_content_for_cleaning": True,
            })
        )
        self.assertTrue(
            experiment._saved_clean_against_test_effective({
                "clean_train_against_test": True,
                "allow_eval_content_for_cleaning": False,
            })
        )
        # 两个键都缺的最老检查点按历史行为回退。
        self.assertTrue(
            experiment._saved_clean_against_test_effective({
                "deduplicate_train_groups": True,
            })
        )
        self.assertFalse(
            experiment._saved_clean_against_test_effective({
                "clean_train_against_test": False,
                "allow_eval_content_for_cleaning": False,
                "deduplicate_train_groups": True,
            })
        )

    def test_resume_restores_cleaning_justification(self):
        args = self._args(domain_justification=None)
        saved = {
            "domain_justification": "synthetic records are corrupt",
        }
        restored = experiment._restore_resume_cleaning_justification(
            args, saved
        )
        self.assertTrue(restored)
        self.assertEqual(
            args.domain_justification,
            "synthetic records are corrupt",
        )
        # 显式给出的新依据不能被检查点静默覆盖。
        explicit = self._args(
            domain_justification="new justification"
        )
        restored = experiment._restore_resume_cleaning_justification(
            explicit, saved
        )
        self.assertFalse(restored)
        self.assertEqual(
            explicit.domain_justification,
            "new justification",
        )
        # 旧检查点可能只记录 config 字段名。
        legacy = self._args(domain_justification=None)
        restored = experiment._restore_resume_cleaning_justification(
            legacy,
            {"cleaning_domain_justification": "legacy justification"},
        )
        self.assertTrue(restored)
        self.assertEqual(
            legacy.domain_justification,
            "legacy justification",
        )

    def test_sweep_detects_preregistration_file_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "seed42"
            second = root / "seed43"
            first.mkdir()
            second.mkdir()
            (first / "config.json").write_text(
                json.dumps({
                    "preregistration": {
                        "file_sha256": "a" * 64,
                    }
                }),
                encoding="utf-8",
            )
            (second / "config.json").write_text(
                json.dumps({
                    "preregistration": {
                        "file_sha256": "b" * 64,
                    }
                }),
                encoding="utf-8",
            )
            error = experiment.preregistration_sweep_error(
                [str(first), str(second)],
                "a" * 64,
            )
        self.assertIn("seed43", error)
        self.assertIn("中途", error)

    def test_data_checks_sha_splits_cleaning_and_separator(self):
        payload = self._payload()
        payload["data"]["group_separator"] = "\\n\\n"
        payload["data"]["cleaning"] = {
            "deduplicate_train_groups": False,
        }
        args = self._args(
            split_policy="contiguous",
            group_separator="\\n\\n",
            deduplicate_train_groups=False,
        )
        self.assertIsNone(
            experiment.preregistration_data_error(
                args, payload, self._dataset()
            )
        )

        wrong_sha = self._dataset()
        wrong_sha.data_sha256 = "c" * 64
        self.assertIn(
            "data.sha256",
            experiment.preregistration_data_error(
                args, payload, wrong_sha
            ),
        )

        args.deduplicate_train_groups = True
        self.assertIn(
            "data.cleaning",
            experiment.preregistration_data_error(
                args, payload, self._dataset()
            ),
        )

        args.deduplicate_train_groups = False
        args.group_separator = "<|doc|>"
        self.assertIn(
            "group_separator",
            experiment.preregistration_data_error(
                args, payload, self._dataset()
            ),
        )


    def test_cleaning_aliases_are_equivalent_for_data_check(self):
        # `--clean-train-against-test` 与
        # `--allow-eval-content-for-cleaning` 是同义入口：任一
        # 为真都表示训练集清洗读取 test。预注册文件写哪个别名
        # 都不应改变核对结果。
        payload = self._payload()
        payload["data"]["cleaning"] = {
            "allow_eval_content_for_cleaning": True,
            "deduplicate_train_groups": True,
        }
        args = self._args(
            split_policy="contiguous",
            group_separator=None,
            clean_train_against_test=True,
            allow_eval_content_for_cleaning=False,
            deduplicate_train_groups=True,
        )
        self.assertIsNone(
            experiment.preregistration_data_error(
                args, payload, self._dataset()
            )
        )

        payload["data"]["cleaning"] = {
            "clean_train_against_test": True,
            "deduplicate_train_groups": True,
        }
        args = self._args(
            split_policy="contiguous",
            group_separator=None,
            clean_train_against_test=False,
            allow_eval_content_for_cleaning=True,
            deduplicate_train_groups=True,
        )
        self.assertIsNone(
            experiment.preregistration_data_error(
                args, payload, self._dataset()
            )
        )

    def test_data_rejects_cleaning_not_declared_in_preregistration(self):
        # 未声明 cleaning 等于声明全部关闭；否则看到验证结果后
        # 临时启用清洗就能改变训练样本集合而绕过预注册核对。
        payload = self._payload()
        args = self._args(
            split_policy="contiguous",
            group_separator=None,
            deduplicate_train_groups=True,
        )
        error = experiment.preregistration_data_error(
            args, payload, self._dataset()
        )
        self.assertIn("data.cleaning", error)

        # 只声明一个关闭开关也不能让未声明的清洗项静默通过。
        payload["data"]["cleaning"] = {
            "deduplicate_train_groups": False,
        }
        args = self._args(
            split_policy="contiguous",
            group_separator=None,
            deduplicate_train_groups=False,
            drop_train_shared_chunks=True,
        )
        error = experiment.preregistration_data_error(
            args, payload, self._dataset()
        )
        self.assertIn("drop_train_shared_chunks", error)

        # 全部关闭时，省略 cleaning 的旧预注册文件仍可核对通过。
        args = self._args(
            split_policy="contiguous",
            group_separator=None,
            deduplicate_train_groups=False,
            drop_train_shared_chunks=False,
            clean_train_against_test=False,
            allow_eval_content_for_cleaning=False,
            drop_train_corrupt_groups=False,
            drop_train_corrupt_bytes=False,
            drop_train_degenerate_groups=False,
            drop_train_degenerate_runs=False,
        )
        self.assertIsNone(
            experiment.preregistration_data_error(
                args, self._payload(), self._dataset()
            )
        )

    def test_schema_rejects_invalid_near_duplicate_cleaning_values(self):
        payload = self._payload()
        payload["data"]["cleaning"] = {
            "near_duplicate_threshold": 1.5,
        }
        error = experiment.preregistration_schema_error(payload)
        self.assertIn("near_duplicate_threshold", error)

        payload["data"]["cleaning"] = {
            "near_duplicate_shingle_bytes": 0,
        }
        error = experiment.preregistration_schema_error(payload)
        self.assertIn("near_duplicate_shingle_bytes", error)

        payload["data"]["cleaning"] = {
            "near_duplicate_threshold": 0.8,
            "near_duplicate_shingle_bytes": 8,
        }
        self.assertIsNone(
            experiment.preregistration_schema_error(payload)
        )

    def test_data_rejects_changed_near_duplicate_threshold(self):
        # 阈值决定删除哪些训练群组，看到结果后再调阈值等于
        # 事后改变数据；预注册显式写了就必须逐项核对。
        payload = self._payload()
        payload["data"]["cleaning"] = {
            "drop_train_near_duplicate_groups": True,
            "near_duplicate_threshold": 0.7,
            "near_duplicate_shingle_bytes": 8,
        }
        args = self._args(
            split_policy="contiguous",
            group_separator=None,
            drop_train_near_duplicate_groups=True,
            near_duplicate_threshold=0.8,
            near_duplicate_shingle_bytes=8,
        )
        error = experiment.preregistration_data_error(
            args, payload, self._dataset()
        )
        self.assertIn("near_duplicate_threshold", error)

        payload["data"]["cleaning"][
            "near_duplicate_threshold"
        ] = 0.8
        self.assertIsNone(
            experiment.preregistration_data_error(
                args, payload, self._dataset()
            )
        )

    def test_data_rejects_undeclared_group_separator(self):
        # 未声明 group_separator 等于不分群组；看到结果后再加
        # 分隔符会改变吸附后的切分边界，必须拒绝。
        payload = self._payload()
        args = self._args(
            split_policy="contiguous",
            group_separator="\\n\\n",
        )
        error = experiment.preregistration_data_error(
            args, payload, self._dataset()
        )
        self.assertIn("group_separator", error)

        # 声明 null 与不声明同义；显式声明相同分隔符则通过。
        payload["data"]["group_separator"] = None
        self.assertIn(
            "group_separator",
            experiment.preregistration_data_error(
                args, payload, self._dataset()
            ),
        )
        payload["data"]["group_separator"] = "\\n\\n"
        self.assertIsNone(
            experiment.preregistration_data_error(
                args, payload, self._dataset()
            )
        )

    def test_schema_rejects_conflicting_cleaning_aliases(self):
        payload = self._payload()
        payload["data"]["cleaning"] = {
            "clean_train_against_test": True,
            "allow_eval_content_for_cleaning": False,
        }
        error = experiment.preregistration_schema_error(payload)
        self.assertIn("同义开关", error)

    def test_normalize_cleaning_aliases_sets_both_fields(self):
        args = self._args(
            clean_train_against_test=False,
            allow_eval_content_for_cleaning=True,
        )
        effective = experiment._normalize_clean_against_test(args)
        self.assertTrue(effective)
        self.assertTrue(args.clean_train_against_test)
        self.assertTrue(args.allow_eval_content_for_cleaning)

    def test_optional_data_null_matches_data_checker(self):
        payload = self._payload()
        payload["data"]["cleaning"] = None
        payload["data"]["entity_manifest_sha256"] = None
        payload["data"]["time_order_manifest_sha256"] = None
        self.assertIsNone(
            experiment.preregistration_schema_error(payload)
        )

    def test_schema_rejects_out_of_range_seed(self):
        payload = self._payload()
        payload["seeds"] = [
            42,
            43,
            experiment.MAX_SEED + 1,
        ]
        error = experiment.preregistration_schema_error(payload)
        self.assertIsNotNone(error)
        self.assertIn(str(experiment.MAX_SEED), error)
