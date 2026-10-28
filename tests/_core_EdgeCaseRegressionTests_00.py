from _core_support import *
from _core_support import _load_experiment, _tiny_config
class EdgeCaseRegressionTests(unittest.TestCase):
    def test_group_split_without_test_ends_at_eof(self):
        # 旧实现会把 cut_val 落在 19，ByteDataset 随即把
        # [19,25) 当成 test，尽管 test_fraction=0。
        buf = bytearray(b"a" * 25)
        for position in (5, 18, 19, 24):
            buf[position] = 10
        cut_train, cut_val = experiment._snap_group_split(
            buf,
            b"\n",
            nominal_train=5,
            nominal_val_end=5,
            min_len=2,
            has_test=False,
        )
        self.assertEqual(cut_train, 6)
        self.assertEqual(cut_val, 25)
    def test_empty_group_separator_is_rejected(self):
        with self.assertRaises(ValueError):
            list(experiment._iter_group_boundaries(
                bytearray(b"abc"), b"", 0
            ))
        with self.assertRaises(ValueError):
            list(experiment._iter_group_spans(
                bytearray(b"abc"), b"", 0, 3
            ))
    def test_overlapping_separator_does_not_create_extra_document(self):
        ids = torch.tensor([[65, 65, 65]])
        segments = experiment.document_segment_ids(ids, [65, 65])
        self.assertEqual(segments.tolist(), [[0, 0, 0]])
    def test_split_overlap_with_gap_is_not_cancelled(self):
        record = experiment._split_isolation_record(
            {
                "train": (0, 30),
                "val": (20, 50),
                "test": (60, 70),
            },
            70,
        )
        self.assertEqual(record["overlap_bytes"], 10)
        self.assertFalse(record["covers_all_bytes"])
    def test_metrics_truncation_drops_partial_trailing_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            path.write_text(
                '{"step":1,"val_loss":3.0}\n'
                '{"step":2,"val_lo',
                encoding="utf-8",
            )
            experiment._truncate_metrics_after_step(path, 1)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write('{"step":2,"val_loss":2.0}\n')
            records = experiment.read_metrics_records(path)
        self.assertEqual([row["step"] for row in records], [1, 2])
    def test_metrics_truncation_terminates_valid_last_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            path.write_text('{"step":1}', encoding="utf-8")
            experiment._truncate_metrics_after_step(path, 1)
            self.assertTrue(
                path.read_text(encoding="utf-8").endswith("\n")
            )
            path.write_text('{"step":2}', encoding="utf-8")
            experiment._truncate_metrics_after_step(path, 1)
            self.assertEqual(path.read_text(encoding="utf-8"), "")
    def test_update_run_config_refuses_to_overwrite_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            config = run_dir / "config.json"
            config.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                experiment.update_run_config(
                    run_dir, "final", {"val_loss": 1.0}
                )
            self.assertEqual(
                config.read_text(encoding="utf-8"), "{not json"
            )
    def test_local_test_ledger_counts_as_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self.assertFalse(
                experiment.test_evaluation_sealed(run_dir)
            )
            (run_dir / "test_access_ledger.jsonl").write_text(
                "", encoding="utf-8"
            )
            self.assertTrue(
                experiment.test_evaluation_sealed(run_dir)
            )
    def test_global_test_ledger_counts_as_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            ledger = Path(tmp) / "ledger.jsonl"
            ledger.write_text(
                json.dumps({"out_dir": str(run_dir)}) + "\n",
                encoding="utf-8",
            )
            self.assertFalse(
                experiment.test_evaluation_sealed(run_dir)
            )
            self.assertTrue(
                experiment.test_evaluation_sealed(
                    run_dir, ledger
                )
            )
            self.assertFalse(
                experiment.test_evaluation_sealed(
                    run_dir, Path(tmp) / "other.jsonl"
                )
            )
    def test_resume_split_error_detects_missing_current_split(self):
        error = experiment.resume_split_change_error(
            {"train": "a", "test": "b"},
            {"train": "a", "val": "c"},
        )
        self.assertIsNotNone(error)
        self.assertIn("test", error)
    def test_build_protocol_normalizes_cleaning_alias(self):
        cfg = _tiny_config()
        args = argparse.Namespace(
            lr=1e-3,
            warmup_steps=1,
            steps=2,
            weight_decay=0.0,
            grad_clip=1.0,
            batch_size=1,
            clean_train_against_test=False,
            allow_eval_content_for_cleaning=True,
        )
        protocol = experiment.build_comparison_protocol(
            cfg, args
        )
        self.assertTrue(protocol["clean_train_against_test"])
        self.assertTrue(
            protocol["allow_eval_content_for_cleaning"]
        )
    def test_protocol_fallback_reads_document_separator_from_config(self):
        protocol = experiment.experiment_protocol_from_payload({
            "config": {
                "mode": "smelt",
                "document_masks": True,
                "document_separator": [10],
            },
            "args": {"lr": 1e-3},
        })
        self.assertEqual(protocol["document_separator"], [10])
        self.assertTrue(protocol["document_masks"])
    def test_fragility_not_evaluable_is_not_reported_as_stable(self):
        contrast = {
            "baseline": "base",
            "variant": "A",
            "factor": "mode",
            "p_value": 0.003,
            "paired_differences": [1.0, 1.1, 0.9],
            "paired_seeds": [1, 2, 3],
            "significant": True,
        }
        experiment._apply_multiple_comparison_correction([contrast])
        self.assertEqual(
            contrast["fragility"]["verdict"], "not_evaluable"
        )
        summary = experiment._fragility_summary([contrast])
        self.assertEqual(
            summary["verdict_counts"], {"not_evaluable": 1}
        )
        self.assertIn("无法评估", summary["advisories"][0])
    def test_fragility_deduplicates_reversed_comparison(self):
        first = {
            "baseline": "local_base",
            "variant": "A",
            "factor": "mode",
            "significant": True,
            "fragility": {
                "available": True,
                "verdict": "stable",
                "n": 5,
                "n_positive": 5,
                "n_negative": 0,
                "alpha": 0.05,
            },
        }
        reversed_entry = dict(first)
        reversed_entry["baseline"] = "A"
        reversed_entry["variant"] = "local_base"
        summary = experiment._fragility_summary(
            [first, reversed_entry]
        )
        self.assertEqual(len(summary["entries"]), 1)
    def test_fragility_summary_accepts_factorial_effect_labels(self):
        effect = {
            "design": "A vs local_base",
            "effect_name": "merge_main",
            "significant": True,
            "fragility": {
                "available": True,
                "verdict": "stable",
                "n": 5,
                "n_positive": 5,
                "n_negative": 0,
                "alpha": 0.05,
            },
        }
        summary = experiment._fragility_summary([effect])
        self.assertEqual(len(summary["entries"]), 1)
        self.assertEqual(
            summary["entries"][0]["baseline"],
            "A vs local_base",
        )
        self.assertEqual(
            summary["entries"][0]["variant"], "merge_main"
        )
    def test_fragility_does_not_claim_nonsignificant_holm_result(self):
        first = {
            "baseline": "b",
            "variant": "A",
            "factor": "mode",
            "p_value": 0.028008456010145746,
            "paired_differences": [0.2, 1, 1, 1],
            "paired_seeds": [1, 2, 3, 4],
            "significant": True,
        }
        second = {
            "baseline": "b",
            "variant": "B",
            "factor": "mode",
            "p_value": 0.041102615679308056,
            "paired_differences": [0.2, 2, 2, 2],
            "paired_seeds": [1, 2, 3, 4],
            "significant": True,
        }
        experiment._apply_multiple_comparison_correction(
            [first, second]
        )
        self.assertFalse(second["significant"])
        self.assertEqual(
            second["fragility"]["verdict"], "not_significant"
        )
        self.assertNotIn(
            "fragile",
            experiment._fragility_summary(
                [first, second]
            )["verdict_counts"],
        )
    def test_fragility_reports_p_value_of_minimal_deletion_set(self):
        report = experiment._paired_result_fragility(
            [2.0, 2.0, 2.0, 2.0, 2.0, 0.5],
            alpha=0.05,
        )
        self.assertEqual(
            report["min_seed_deletions_to_lose_significance"], 3
        )
        self.assertGreater(
            report["min_seed_deletions_p_value"], 0.05
        )
        self.assertNotEqual(
            report["min_seed_deletions_p_value"],
            report["max_leave_one_out_p_value"],
        )
    def test_fragility_marks_untested_deletion_levels(self):
        report = experiment._paired_result_fragility(
            [1.0] * 59 + [0.5],
            alpha=0.05,
        )
        self.assertFalse(report["deletions_tested_exhaustively"])
        self.assertEqual(report["max_deletions_tested"], 2)
        self.assertEqual(report["configured_max_deletions"], 3)
    def test_exact_sign_test_all_zero_is_resolution_limited(self):
        report = experiment._exact_sign_test([0.0, 0.0])
        self.assertEqual(report["n_nonzero"], 0)
        self.assertTrue(report["resolution_limited"])
    def test_factorial_duplicate_seeds_are_raw_and_deduplicated(self):
        def cell():
            return {
                "samples": {
                    "val_loss": [
                        {"seed": 42, "value": 1.0},
                        {"seed": 42, "value": 1.2},
                    ]
                }
            }
        _mapping, _complete, duplicates = (
            experiment._cell_seed_values(cell(), "val_loss")
        )
        self.assertEqual(duplicates, [42])
        effect = experiment._paired_linear_effect(
            {"a": cell(), "b": cell()},
            {"a": 1.0, "b": -1.0},
            "val_loss",
        )
        self.assertEqual(effect["duplicate_seeds"], [42])
    def test_manifest_strict_requires_verify_mode(self):
        args = argparse.Namespace(
            write_manifest="run",
            verify_manifest=None,
            verify_manifest_strict=True,
        )
        error = experiment.manifest_mode_error(args)
        self.assertIsNotNone(error)
        self.assertIn("--verify-manifest", error)
    def test_manifest_rejects_compare_report(self):
        args = argparse.Namespace(
            write_manifest="run",
            verify_manifest=None,
            verify_manifest_strict=False,
            compare_report="report.json",
        )
        self.assertIsNotNone(experiment.manifest_mode_error(args))
    def test_load_selected_model_rejects_bad_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            torch.save(
                {"model": {}, "step": None, "val_loss": 1.0},
                path,
            )
            self.assertIsNone(
                experiment.load_selected_model(
                    path, torch.device("cpu")
                )
            )
    def test_baseline_tuning_rejects_boolean_or_nonfinite_lr(self):
        base = {
            "comparison_family": "baseline_tuning",
            "mode": "dense",
            "metric": "val_loss",
            "winner": {"lr": True, "metric_mean": 1.0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning.json"
            path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaises(ValueError):
                experiment.load_baseline_tuning(path)
            base["winner"] = {
                "lr": float("nan"),
                "metric_mean": 1.0,
            }
            path.write_text(
                json.dumps(base), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                experiment.load_baseline_tuning(path)
    def test_baseline_tuning_rejects_out_of_float_range_lr(self):
        base = {
            "comparison_family": "baseline_tuning",
            "mode": "dense",
            "metric": "val_loss",
            # JSON 允许任意大整数；float(10**400) 会溢出。
            "winner": {"lr": 10**400, "metric_mean": 1.0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning.json"
            path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaises(ValueError):
                experiment.load_baseline_tuning(path)
            base["winner"] = {
                "lr": 1e-3,
                "metric_mean": 10**400,
            }
            path.write_text(
                json.dumps(base), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                experiment.load_baseline_tuning(path)
    def test_analysis_helpers_tolerate_out_of_float_range_numbers(self):
        huge = 10**400
        self.assertEqual(
            experiment._metric_samples(
                [{"seed": 1, "run": "r", "val_loss": huge}],
                "val_loss",
            ),
            [],
        )
        self.assertEqual(
            experiment._collapse_seed_samples(
                [{"seed": 1, "run": "r", "value": huge}]
            ),
            [],
        )
        mapping, _complete, duplicates = (
            experiment._cell_seed_values(
                {
                    "samples": {
                        "val_loss": [
                            {"seed": 1, "value": huge}
                        ]
                    }
                },
                "val_loss",
            )
        )
        self.assertEqual(mapping, {})
        self.assertEqual(duplicates, [])
        self.assertIsNone(experiment._mean([huge]))
        self.assertIsNone(experiment._median([huge, 1.0]))
        self.assertIsNone(
            experiment._sample_std([huge, 1.0, 2.0])
        )
        self.assertIsNone(
            experiment._window_per_token_loss({
                "tokens": huge,
                "loss_sum": 1.0,
            })
        )
        self.assertIsNone(
            experiment._window_accuracy({
                "tokens": huge,
                "correct": 1.0,
            })
        )
        self.assertEqual(
            experiment._clean_window_differences({
                1: [huge, 1.0],
                "bad-key": [1.0],
            }),
            {},
        )
        # 排序/比较不能因为无法转成 float 而崩溃；相同大整数
        # 仍必须判为相等。
        experiment._factor_sort_rank(huge)
        self.assertTrue(
            experiment._axis_value_equal(huge, huge)
        )
        self.assertFalse(
            experiment._axis_value_equal(huge, huge + 1)
        )
        experiment._axis_value_sort_key(huge)
        self.assertEqual(
            experiment._paired_t_test([huge, 1.0, 2.0]),
            {"test": "invalid_values"},
        )
        self.assertEqual(
            experiment._paired_sign_flip_test(
                [huge, 1.0, 2.0, 3.0]
            ),
            {"test": "invalid_values"},
        )
        self.assertEqual(
            experiment._welch_t_test(
                [huge, 1.0, 2.0],
                [1.0, 2.0, 3.0],
            ),
            {"test": "invalid_values"},
        )
        self.assertEqual(
            experiment._paired_result_fragility(
                [huge, 1.0, 2.0]
            )["reason"],
            "non_finite_differences",
        )
        # 效率均值/标准差与无共同种子时的聚合均值也不能因
        # 超大整数在 float() 处崩溃。
        self.assertIsNone(
            experiment._cell_cost_mean(
                {"costs": {"n_params": {"mean": huge}}},
                "n_params",
            )
        )
        self.assertIsNone(
            experiment._cell_cost_std(
                {"costs": {"n_params": {"std": huge}}},
                "n_params",
            )
        )
        factorial = experiment._paired_linear_effect(
            {
                "a": {
                    "samples": {
                        "val_loss": [{"seed": 1, "value": 1.0}]
                    },
                    "val_loss_mean": huge,
                },
                "b": {
                    "samples": {
                        "val_loss": [{"seed": 2, "value": 1.0}]
                    },
                    "val_loss_mean": huge,
                },
            },
            {"a": 1.0, "b": -1.0},
            "val_loss",
        )
        self.assertEqual(factorial["test"], "missing_metric")
        # n>1023 时 2**n 不能再转 float；精确符号检验仍需
        # 给出可判定的 p 值与最小可达 p。
        sign = experiment._exact_sign_test([1.0] * 2000)
        self.assertTrue(sign["available"])
        self.assertEqual(sign["p_value"], 0.0)
        self.assertEqual(sign["min_attainable_p"], 0.0)
    def test_design_power_fallback_does_not_keep_stale_adequacy(self):
        # 旧/手工 power 记录缺重算字段时，不能沿用按更小家族
        # 算出的 adequate=True 放行新增的交互对比。
        power = {
            "expected_seed_sd_relative": 0.01,
            "target_power": 0.8,
            "alpha": 0.05,
            "planned_comparisons": 3,
            "planned_seeds": 7,
            "adequate": True,
            # 这些是旧家族/旧对比类型下的派生字段，重算失败后
            # 不能再作为有效结论展示。
            "required_seeds": 99,
            "required_seeds_with_attrition": 120,
            "achieved_power_at_min_effect": 0.99,
            "minimum_detectable_effect_relative": 0.001,
            "contrast_power": {"pairwise": {"adequate": True}},
        }
        adjusted = experiment._design_power_for_family(
            power,
            6,
            ["pairwise", "main", "interaction", "simple"],
        )
        self.assertIsNone(adjusted["adequate"])
        self.assertFalse(adjusted["adequacy_recomputed"])
        self.assertEqual(
            adjusted["stale_derived_fields"]["required_seeds"],
            99,
        )
        for key in (
            "required_seeds",
            "required_seeds_with_attrition",
            "achieved_power_at_min_effect",
            "minimum_detectable_effect_relative",
            "contrast_power",
        ):
            self.assertNotIn(key, adjusted)
        design = {"power": adjusted}
        self.assertIsNotNone(
            experiment.design_power_gate_error(design)
        )
        # 家族规模本身损坏（inf/NaN）也不能在 int() 处抛异常。
        guarded = experiment._design_power_for_family(
            {
                "expected_seed_sd_relative": 0.01,
                "min_effect_relative": 0.01,
                "target_power": 0.8,
                "alpha": 0.05,
                "planned_seeds": 6,
                "planned_comparisons": 1,
                "planned_contrasts": ["pairwise"],
            },
            float("inf"),
            ["pairwise"],
            force=True,
        )
        self.assertIsInstance(guarded, dict)
    def test_design_gate_handles_uncomputable_required_seeds(self):
        design = {
            "power": {
                "adequate": False,
                "binding_contrast": "interaction",
                "binding_contrast_label": "交互",
                "planned_comparisons": 6,
                "planned_seeds": 4,
                "target_power": 0.8,
            }
        }
        gate = experiment.design_power_gate_error(design)
        self.assertIsNotNone(gate)
        self.assertNotIn("None 个", gate)
        self.assertIn("无法计算", gate)
        # 损坏的 equivalence 标量不能触发 AttributeError，也不能
        # 把缺失的目标功效/家族规模写成 None。
        malformed = {
            "power": {
                "adequate": False,
                "plan_equivalence": True,
                "equivalence_adequate": False,
                "equivalence": 1.0,
            }
        }
        malformed_gate = experiment.design_power_gate_error(
            malformed
        )
        self.assertIsNotNone(malformed_gate)
        self.assertNotIn("None", malformed_gate)
    def test_plan_treatment_keys_may_be_non_strings(self):
        # 手工/旧计划可能用非字符串键；不能因 str(key) 后查不到
        # 值就把不同处理单元静默合并。
        plan = [
            {"label": "a", "mode": "A", 1: 0, "seed": 1},
            {"label": "b", "mode": "A", 1: 1, "seed": 2},
            {"label": "c", "mode": "A", 1: 2, "seed": 3},
        ]
        self.assertEqual(
            experiment.planned_comparisons_for_plan(plan), 3
        )
        self.assertEqual(
            experiment.planned_contrast_types_for_plan(plan),
            ["pairwise"],
        )
    def test_cost_comparison_tolerates_zero_and_huge_variants(self):
        def cell(mean, std=1.0, n=3):
            return {
                "costs": {
                    "n_params": {
                        "mean": mean,
                        "std": std,
                        "n": n,
                    }
                }
            }

        # 变体成本为 0 时不能在做变异系数时除零。
        _, verdict = experiment._cost_comparison(
            cell(1.0), cell(0.0), "improvement", True
        )
        self.assertIsNotNone(verdict)
        # 成本倍率极大时 exp/平方不能抛 OverflowError；无法给出
        # 有限区间时应明确标为不可分辨。
        cost, _ = experiment._cost_comparison(
            cell(1e-100), cell(1e100), "improvement", True
        )
        detail = cost["uncertainty"]["n_params"]
        self.assertFalse(detail["resolvable"])
        self.assertIsNone(detail["ratio_ci95"])
    def test_contrasts_tolerate_out_of_range_means_and_counts(self):
        huge = 10**400

        def cell(mode, mean, std=1.0):
            return {
                "label": mode,
                "keys": {"mode": mode},
                "samples": {"val_loss": []},
                "val_loss_mean": mean,
                "val_loss_std": std,
                "n_runs": 3,
                "effective_seed_count": 3,
                "costs": {},
            }

        contrasts = experiment.factor_contrasts(
            [cell("A", 1.0), cell("B", huge)],
            ["mode"],
            "val_loss",
        )
        self.assertEqual(len(contrasts), 1)
        self.assertIsNone(contrasts[0]["delta"])
        # 巨大 std 同样不能直接平方；无效值按缺失处理。
        contrasts = experiment.factor_contrasts(
            [cell("A", 1.0, huge), cell("B", 2.0, huge)],
            ["mode"],
            "val_loss",
        )
        self.assertIsNone(contrasts[0]["delta_over_noise"])
        # 损坏的运行数不能阻止对比生成或被当成负数。
        broken_a = dict(cell("A", 1.0), n_runs=[])
        broken_b = dict(cell("B", 2.0), n_runs={})
        contrasts = experiment.factor_contrasts(
            [broken_a, broken_b], ["mode"], "val_loss"
        )
        self.assertEqual(
            contrasts[0]["n_expected_baseline"], 0
        )
        baseline = experiment.compare_cells_to_baseline(
            [
                dict(cell("A", 1.0), effective_seed_count=[]),
                cell("B", 2.0),
            ],
            "A",
            "val_loss",
        )
        self.assertEqual(len(baseline), 1)
        self.assertEqual(baseline[0]["delta"], 1.0)
    def test_factor_keys_preserve_large_integer_precision(self):
        left = 2 ** 53
        right = left + 1
        self.assertNotEqual(
            experiment._stable_factor_key(left),
            experiment._stable_factor_key(right),
        )
        # int/float 的同一数值仍必须合并为同一水平。
        self.assertEqual(
            experiment._stable_factor_key(7),
            experiment._stable_factor_key(7.0),
        )
        self.assertLess(
            experiment._factor_sort_rank(left),
            experiment._factor_sort_rank(right),
        )
    def test_factor_keys_accept_unhashable_values(self):
        # 手工编辑/损坏的计划可能把 set 放进处理轴；稳定键必须
        # 仍可哈希，否则 planned_comparisons_for_plan 等会把键
        # 放进 set/dict 时崩溃。
        key = experiment._stable_factor_key({1, 2})
        self.assertEqual(hash(key), hash(key))
        self.assertEqual(
            key,
            experiment._stable_factor_key(frozenset({2, 1})),
        )
        plan = [
            {
                "label": "a",
                "mode": "A",
                "balance_coef": {0.0},
                "seed": 1,
            },
            {
                "label": "b",
                "mode": "A",
                "balance_coef": {0.0},
                "seed": 2,
            },
        ]
        self.assertEqual(
            experiment.planned_comparisons_for_plan(plan), 1
        )
        self.assertEqual(
            experiment.planned_contrast_types_for_plan(plan), []
        )
        report = experiment.run_order_balance_report(
            experiment.order_sweep_plan(plan, "randomized", 3)
        )
        self.assertTrue(report["available"])
        self.assertEqual(
            experiment._stable_factor_key({"a": 1, 2: 3}),
            experiment._stable_factor_key({"a": 1, 2: 3}),
        )
    def test_resume_checks_tolerate_corrupt_fingerprint_records(self):
        # 非字符串数据指纹不能被当成「未记录」而静默放行续跑。
        self.assertIn(
            "格式损坏",
            experiment.resume_data_change_error(123, "abc"),
        )
        # 损坏的保存指纹不能因为当前指纹缺失/为空而绕过格式
        # 检查，否则无法核对时会被静默放行。
        self.assertIn(
            "格式损坏",
            experiment.resume_data_change_error(123, None),
        )
        self.assertIn(
            "格式损坏",
            experiment.resume_data_change_error(123, ""),
        )
        self.assertIsNone(
            experiment.resume_data_change_error("", "abc")
        )
        # split 记录的键可能是损坏的混合类型；排序/拼接不能
        # 抛 TypeError，损坏的哈希值也不能静默放行。
        mixed = experiment.resume_split_change_error(
            {1: "a", "val": "b"}, {}
        )
        self.assertIn("1", mixed)
        self.assertIn("val", mixed)
        corrupt = experiment.resume_split_change_error(
            {"train": 1}, {"train": "a"}
        )
        self.assertIn("损坏", corrupt)
        self.assertIsNone(
            experiment.resume_split_change_error({}, {})
        )
        # 时序隔离记录的混合键同样必须可比较，且键顺序不影响
        # 判定。
        self.assertIsNone(
            experiment.resume_split_isolation_change_error(
                {1: "a", "b": 2},
                {"b": 2, 1: "a"},
            )
        )
        self.assertIsNotNone(
            experiment.resume_split_isolation_change_error(
                {1: "a"}, {1: "b"}
            )
        )
    def test_resume_checks_reject_corrupt_top_level_fingerprints(self):
        # None 只代表旧检查点缺字段；列表/数字等损坏值不能被
        # 当成「没有记录」而静默放行续跑。
        self.assertIsNone(
            experiment.resume_split_change_error(None, {})
        )
        self.assertIsNone(
            experiment.resume_split_isolation_change_error(
                None, {}
            )
        )
        self.assertIsNone(
            experiment.resume_effective_train_change_error(
                None, "abc"
            )
        )
        self.assertIn(
            "格式损坏",
            experiment.resume_split_change_error([], {}),
        )
        self.assertIn(
            "格式损坏",
            experiment.resume_split_change_error("abc", {}),
        )
        self.assertIn(
            "格式损坏",
            experiment.resume_split_isolation_change_error(
                123, {}
            ),
        )
        self.assertIn(
            "格式损坏",
            experiment.resume_effective_train_change_error(
                123, "abc"
            ),
        )
        # 当前数据新加载，缺字段只会是 None；其余损坏值同样
        # 不能静默放行。
        self.assertIsNone(
            experiment.resume_split_change_error(
                {"train": "a"}, None
            )
        )
        self.assertIn(
            "格式损坏",
            experiment.resume_split_change_error(
                {"train": "a"}, []
            ),
        )
        self.assertIn(
            "格式损坏",
            experiment.resume_split_change_error(
                {"train": "a"}, 123
            ),
        )
        self.assertIsNone(
            experiment.resume_effective_train_change_error("abc", None)
        )
        self.assertIsNone(
            experiment.resume_effective_train_change_error("abc", "")
        )
        self.assertIn(
            "格式损坏",
            experiment.resume_effective_train_change_error("abc", 123),
        )
    def test_degenerate_run_cleaning_handles_partial_exclusions(self):
        # 一个被其他清洗剔除的字节把长退化段切成两段；两个
        # 剩余片段都仍达到阈值，不能因为原段与排除区间重叠就
        # 整段跳过，否则 post-clean 训练流仍含退化段。
        raw = bytearray(b"a" * 100)
        spans, stats = (
            experiment._plan_train_degenerate_run_cleaning(
                raw,
                len(raw),
                32,
                [(50, 51)],
            )
        )
        self.assertEqual(spans, [(0, 50), (51, 100)])
        self.assertEqual(stats["runs_dropped"], 2)
        self.assertEqual(stats["dropped_bytes"], 99)
        self.assertEqual(stats["max_run_bytes"], 50)
        self.assertEqual(stats["post_clean_train_bytes"], 0)
        # 排除区间若把剩余两侧都压到阈值以下，则不清洗。
        spans, stats = (
            experiment._plan_train_degenerate_run_cleaning(
                raw,
                len(raw),
                32,
                [(20, 80)],
            )
        )
        self.assertEqual(spans, [])
        self.assertEqual(stats["runs_dropped"], 0)
    def test_statistics_tolerate_large_finite_floats(self):
        huge = 1.0e308
        self.assertEqual(experiment._mean([huge, huge]), huge)
        self.assertEqual(experiment._mean([huge, -huge]), 0.0)
        self.assertEqual(experiment._median([huge, huge]), huge)
        self.assertEqual(experiment._median([huge, -huge]), 0.0)
        self.assertEqual(
            experiment._sample_std([huge, huge]), 0.0
        )
        # 两个相反的大数，标准差仍在 float 范围内：直接对
        # 1e308 ** 2 会抛 OverflowError。
        self.assertIsNotNone(
            experiment._sample_std([huge, -huge])
        )
        self.assertEqual(
            experiment._welch_t_test(
                [huge, huge, huge],
                [huge, huge, huge],
            )["note"],
            "zero_variance",
        )
        identical_large = experiment._welch_t_test(
            [0.0, 1e200, 2e200],
            [0.0, 1e200, 2e200],
        )
        self.assertEqual(identical_large["test"], "welch_t")
        self.assertEqual(identical_large["t"], 0.0)
        self.assertEqual(identical_large["p_value"], 1.0)
        self.assertEqual(identical_large["effect"], 0.0)
        self.assertIsNotNone(identical_large["se"])
        json.dumps(identical_large, allow_nan=False)
        # 符号翻转统计量对尺度不变：同一组大到 1e308 的差值
        # 不能因求和溢出而被误判成退化的 p=1。
        sign = experiment._paired_sign_flip_test(
            [huge, huge, huge]
        )
        self.assertEqual(sign["method"], "exact")
        self.assertEqual(sign["p_value"], 0.25)
        # 两向方差分解同样必须先在归一化尺度上计算，不能对
        # 1e200 的离差直接平方。
        components = experiment._window_variance_components(
            {1: [1e200, 0.0], 2: [1e200, 0.0]}
        )
        self.assertTrue(components["available"])
        self.assertEqual(
            components["between_seed_sd"], 0.0
        )
        self.assertIsNotNone(components["std_error_of_mean"])
        # 截断前的原始方差分量可为负，缩放实现不能把它丢掉。
        components = experiment._window_variance_components(
            {1: [0.0, 10.0], 2: [5.0, 5.0]}
        )
        raw = components[
            "variance_components_single_observation"
        ]
        self.assertLess(raw["raw_seed"], 0.0)
        self.assertTrue(raw["truncated_at_zero"])
        # 计时摘要换算单位后不能把 inf 写进报告。
        timing = experiment._timing_summary([huge, huge])
        self.assertIsNone(timing["mean"])
        self.assertEqual(timing["std"], 0.0)
    def test_paired_stats_keep_tiny_nonzero_scale(self):
        # 极小但非零的差值在平方时下溢为 0；标准差不能被报告成
        # 0，配对 t 也不能因此变成「零方差/精确分离」。
        tiny = [1.0e-200, 2.0e-200, 3.0e-200]
        self.assertTrue(
            math.isclose(
                experiment._sample_std(tiny),
                1.0e-200,
                rel_tol=1e-12,
            )
        )
        test = experiment._paired_t_test(tiny)
        self.assertEqual(test["test"], "paired_t")
        self.assertTrue(math.isfinite(test["t"]))
        self.assertAlmostEqual(
            test["t"],
            experiment._paired_t_test([1.0, 2.0, 3.0])["t"],
        )
        # 符号翻转统计量本身对公共尺度不变；固定绝对容差会让
        # 1e-13 量级的差值被误判成退化（p=1）。
        for scale in (1.0e-13, 1.0e-200, 1.0e-300):
            sign = experiment._paired_sign_flip_test(
                [scale, 2.0 * scale, 3.0 * scale]
            )
            self.assertEqual(sign["method"], "exact")
            self.assertEqual(sign["p_value"], 0.25)
            json.dumps(sign, allow_nan=False)
    def test_counterbalanced_empty_blocks_do_not_divide_by_zero(self):
        self.assertEqual(
            experiment._counterbalanced_order({1: []}, 0), []
        )
        self.assertEqual(
            experiment._counterbalanced_order({1: [], 2: []}, 0),
            [],
        )
    def test_sparkline_handles_out_of_range_and_nonfinite_values(self):
        huge = 10 ** 400
        self.assertEqual(experiment._sparkline([huge]), "")
        self.assertEqual(
            experiment._sparkline([1.0, float("nan")]), ""
        )
        self.assertEqual(
            experiment._sparkline([1.0, float("inf")]), ""
        )
        self.assertEqual(len(experiment._sparkline([1, 2, 3])), 3)
        # 有限但符号相反的极值：high-low 会溢出成 inf，走势条
        # 仍必须能按相对大小分块，而不是 int(NaN) 崩溃。
        spark = experiment._sparkline([-1.0e308, 0.0, 1.0e308])
        self.assertEqual(len(spark), 3)
        self.assertNotEqual(spark, "")
    def test_bad_case_comparison_skips_out_of_range_shares(self):
        huge = 10 ** 400
        category = sorted(experiment._BAD_CASE_CATEGORIES)[0]
        report = experiment.bad_case_comparison_report([
            {
                "bad_case_available": True,
                "bad_case_tokens_evaluated": 10,
                "bad_case_wrong_tokens": 2,
                "bad_case_error_category_counts": {
                    category: 2,
                },
                "bad_case_error_category_shares": {
                    category: huge,
                },
            },
            {
                "bad_case_available": True,
                "bad_case_tokens_evaluated": 10,
                "bad_case_wrong_tokens": 2,
                "bad_case_error_category_counts": {
                    category: 2,
                },
                "bad_case_error_category_shares": {
                    category: 0.2,
                },
            },
        ])
        self.assertEqual(report["n_runs_with_bad_cases"], 2)
        self.assertEqual(
            report["error_category_counts"][category], 4
        )
        self.assertNotIn(category, report["category_share_ranges"])
    def test_imbalance_advisory_tolerates_out_of_range_rates(self):
        huge = 10 ** 400
        report = experiment.imbalance_metric_advisory(
            [
                {
                    "run": "huge_majority",
                    "val_majority_accuracy": huge,
                },
                {
                    "run": "huge_chance",
                    "val_pr_auc_macro_chance": huge,
                },
            ],
            "val_loss",
        )
        self.assertIsNone(report["max_majority_accuracy"])
        self.assertIsNone(report["most_imbalanced_run"])
        self.assertIsNone(report["pr_auc_chance_mean"])
    def test_imbalance_advisory_omits_overflowing_pr_auc_lift(self):
        report = experiment.imbalance_metric_advisory(
            [
                {
                    "run": "huge_lift",
                    "val_majority_accuracy": 0.9995,
                    "val_pr_auc_macro": 1.0e308,
                    "val_pr_auc_macro_chance": -1.0e308,
                }
            ],
            "val_pr_auc_macro",
        )
        self.assertIsNone(report["pr_auc_over_chance"])
        self.assertIn("无法表示", report["advisories"][-1])
        json.dumps(report, allow_nan=False)
    def test_baseline_ngram_delta_overflow_is_omitted(self):
        rows = [
            {
                "run": "baseline",
                "mode": "dense",
                "seed": 1,
                "val_loss": 1.0e308,
                "protocol": {
                    "data_sha256": "d",
                    "eval_batches": 1,
                },
            },
            {
                "run": "ngram",
                "mode": "ngram",
                "seed": 2,
                "val_loss": -1.0e308,
                "protocol": {
                    "data_sha256": "d",
                    "eval_batches": 1,
                },
            },
        ]
        report = experiment.baseline_adequacy(
            rows, "dense", "val_loss"
        )
        comparison = report["ngram_comparison"]
        self.assertIsNone(comparison["delta"])
        self.assertIsNone(comparison["seeds_to_detect"])
        json.dumps(report, allow_nan=False)
    def test_baseline_adequacy_tolerates_out_of_range_costs(self):
        huge = 10 ** 400
        rows = [
            {
                "run": "baseline",
                "mode": "dense",
                "seed": 1,
                "val_loss": 1.0,
                "n_params": 100.0,
                "flops_per_forward": 100.0,
            },
            {
                "run": "huge_metric",
                "mode": "B",
                "seed": 2,
                "val_loss": huge,
                "n_params": huge,
                "flops_per_forward": huge,
            },
            {
                "run": "huge_cost",
                "mode": "B",
                "seed": 3,
                "val_loss": 1.1,
                "n_params": huge,
                "flops_per_forward": huge,
            },
        ]
        report = experiment.baseline_adequacy(
            rows, "dense", "val_loss"
        )
        self.assertEqual(report["capacity_confounds"], [])
        self.assertIsNone(report["baseline_capacity"])
    def test_curve_analysis_treats_out_of_range_numbers_as_nonfinite(self):
        huge = 10 ** 400
        records = [
            {
                "step": 1,
                "train_loss": huge,
                "val_loss": huge,
                "grad_norm": huge,
                "lr": huge,
            },
            {
                "step": 2,
                "train_loss": 1.0,
                "val_loss": 1.0,
                "grad_norm": 1.0,
                "lr": 1e-3,
            },
        ]
        analysis = experiment.analyze_training_curve(records)
        self.assertEqual(analysis["n_nonfinite_events"], 1)
        self.assertEqual(analysis["n_eval_points"], 1)
        self.assertEqual(analysis["n_train_steps"], 1)
        self.assertEqual(analysis["peak_lr"], 1e-3)
        self.assertIn(1, analysis["nonfinite_steps"])
    def test_timing_summary_handles_out_of_range_integers(self):
        huge_integer = 10 ** 400
        timing = experiment._timing_summary([huge_integer])
        self.assertEqual(timing["n"], 1)
        self.assertIsNone(timing["mean"])
        self.assertIsNone(timing["median"])
        # 混入一个不可用值不能只丢掉它后继续算，否则会把单次
        # 观测伪装成完整分布。
        timing = experiment._timing_summary([1.0, huge_integer])
        self.assertEqual(timing["n"], 2)
        self.assertIsNone(timing["mean"])
        self.assertIsNone(timing["std"])
    def test_timing_summary_cv_does_not_overflow(self):
        # 有限极值相互抵消后均值可以合法地接近 0，而标准差仍
        # 极大；std/mean 越过 float 上限时 CV 必须是 None。
        timing = experiment._timing_summary(
            [1.0e-308, 1.0e308, -1.0e308]
        )
        self.assertIsNone(timing["coefficient_of_variation"])
        json.dumps(timing, allow_nan=False)
    def test_analysis_and_resume_tolerate_out_of_range_numbers(self):
        huge = 10 ** 400

        # 参考基线来自 config.json：超范围整数按不可用处理。
        self.assertIsNone(
            experiment._trivial_reference_value(
                "val_loss", {"unigram_loss": huge}
            )
        )

        # resume 读取历史指标时不能因单个超范围整数崩溃。
        self.assertEqual(
            experiment.early_stop_state_from_records(
                [{"step": 1, "val_loss": huge}],
                0.0,
            ),
            (None, None, 0),
        )

        # 预注册等价界损坏时标记为 invalid，而不是抛异常。
        margin = experiment.equivalence_margin_preregistration(
            [
                {
                    "run": "bad_margin",
                    "hypothesis_equivalence_margin": huge,
                }
            ],
            0.01,
        )
        self.assertEqual(margin["invalid_runs"], ["bad_margin"])
        self.assertEqual(margin["declared_margins"], {})

        # 最小效应预注册值损坏时必须给出可读错误。
        args = argparse.Namespace(
            hypothesis_direction="decrease",
            hypothesis_min_effect=huge,
            primary_metric="val_loss",
            hypothesis_baseline=None,
            compare=None,
            final_test_comparison=None,
            sanity_only=False,
            statistical_baseline=False,
            hypothesis="验证损失降低 2%",
        )
        error = experiment.hypothesis_preregistration_error(args)
        self.assertIn("有限正数", error or "")

        # 对比阶段不能因坏的最小效应值中断；该运行仍可登记，
        # 但没有可用的绝对阈值。
        alignment = experiment.hypothesis_alignment(
            [
                {
                    "run": "bad_effect",
                    "hypothesis_direction": "decrease",
                    "primary_metric": "val_loss",
                    "hypothesis_min_effect": huge,
                }
            ],
            [{"label": "mode=B, seed=1", "runs": ["bad_effect"]}],
            [
                {
                    "variant": "mode=B, seed=1",
                    "baseline": "mode=A, seed=1",
                    "factor": "mode",
                    "delta": -1.0,
                    "mean_baseline": 1.0,
                    "p_value_holm": 0.01,
                    "significant": True,
                    "confounded": False,
                }
            ],
            "val_loss",
        )
        self.assertEqual(len(alignment["checks"]), 1)
        self.assertIsNone(alignment["checks"][0]["min_effect"])
    def test_print_comparison_tolerates_out_of_range_numbers(self):
        huge = 10 ** 400
        self.assertEqual(experiment._format_finite(huge), "-")
        self.assertEqual(
            experiment._format_finite(float("nan")), "-"
        )
        self.assertEqual(
            experiment._format_finite(1.25), "1.2500"
        )
        # 乘上展示缩放后溢出也必须按缺失值处理。
        self.assertEqual(
            experiment._format_finite(1e308, ".1f", 100.0), "-"
        )
        rows = [
            {
                "run": "huge",
                "mode": "B",
                "n_params": huge,
                "val_loss": huge,
                "val_macro_nll": huge,
                "val_bits_per_byte": huge,
                "val_token_accuracy": huge,
                "val_macro_f1": huge,
                "latency_ms_per_forward": huge,
                "training_ms_per_step": huge,
                "flops_per_forward": huge,
                "peak_memory_mb": huge,
                "routing_available": True,
                "routing_jaccard_branch": huge,
                "routing_slot_reuse_branch": huge,
                "routing_branch_self_reuse": huge,
                "routing_r1_load_entropy": huge,
                "routing_r2_load_entropy": huge,
                "routing_r2_dead_experts": huge,
                "routing_r2_router_entropy": huge,
                "routing_branch_cosine": huge,
            }
        ]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            experiment.print_comparison(rows)
        text = output.getvalue()
        self.assertIn("huge", text)
        self.assertIn("val_loss", text)
        # 正常有限值仍然正常格式化。
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            experiment.print_comparison([
                {
                    "run": "ok",
                    "mode": "B",
                    "val_loss": 1.25,
                    "flops_per_forward": 2e9,
                }
            ])
        self.assertIn("1.2500", output.getvalue())
        self.assertIn("2.000", output.getvalue())
    def test_paired_stats_keep_large_finite_effects(self):
        huge = 1.0e308
        test = experiment._paired_t_test([huge, huge, huge])
        self.assertEqual(test["note"], "noise_free_separation")
        self.assertTrue(math.isfinite(test["effect"]))
        self.assertEqual(test["effect"], huge)
        fragility = experiment._paired_result_fragility(
            [huge, huge, huge]
        )
        self.assertEqual(fragility["mean_difference"], huge)
        self.assertTrue(
            math.isfinite(fragility["mean_difference"])
        )
        stability = experiment._leave_one_out_effect_stability(
            [huge, huge, huge]
        )
        self.assertEqual(
            stability["leave_one_out_effect_min"], huge
        )
        self.assertEqual(
            stability["leave_one_out_effect_max"], huge
        )
        self.assertTrue(
            stability["leave_one_out_sign_stable"]
        )
    def test_welch_seed_plan_uses_two_sample_power(self):
        samples_a = [
            {"seed": 1, "value": 10.0},
            {"seed": 2, "value": 10.2},
            {"seed": 3, "value": 9.8},
        ]
        samples_b = [
            {"seed": 4, "value": 10.2},
            {"seed": 5, "value": 10.4},
            {"seed": 6, "value": 10.0},
        ]
        test = experiment._contrast_test(samples_a, samples_b)
        self.assertEqual(test["test"], "welch_t")
        contrast = {
            "test": test["test"],
            "se": test["se"],
            "df": test["df"],
            "delta": test["effect"],
            "mean_baseline": 10.0,
            "n_baseline": 3,
            "n_variant": 3,
        }
        experiment._attach_power_equivalence(
            contrast, test["effect"], test, 0.01
        )
        self.assertEqual(
            contrast["seeds_to_detect_design"], "welch"
        )
        # 每臂 n 个、se=sd·sqrt(2/n)、df=2(n-1)；n=17 才达到
        # 80% 功效，配对公式给出的 10 会系统性低估。
        self.assertEqual(contrast["seeds_to_detect"], 17)
    def test_zero_effect_power_is_exactly_alpha(self):
        self.assertEqual(
            experiment._t_power(0.0, 1.0, 2.0, 1e-5),
            1e-5,
        )
        # 极小 alpha + df=2 时固定 u 网格会漏掉尖峰；换到
        # w=scale·u 后应接近非中心 t 的参考值（scipy nct）。
        self.assertAlmostEqual(
            experiment._t_power(1.0, 1.0, 2.0, 1e-5),
            1.9999800001667738e-05,
            delta=1e-9,
        )
        self.assertGreater(
            experiment._t_power(2.0, 1.0, 2.0, 1e-5),
            experiment._t_power(1.0, 1.0, 2.0, 1e-5),
        )
    def test_t_power_matches_noncentral_t_reference_near_df_one(self):
        # df<3 时 chi 密度在 0 附近有弱奇异项 u^(df-1)；固定 u
        # 网格的 Simpson 会少积概率质量。以下参考值由高精度
        # 非中心 t 积分得到，修复前 df=1.1 的误差约 4.4%。
        for effect, df, expected in (
            (0.05, 1.01, 0.050062837107652636),
            (0.05, 1.1, 0.050068256192849915),
            (0.05, 1.5, 0.050091063051014614),
            (0.2, 2.0, 0.051850694985949765),
            (2.0, 2.5, 0.256924795068195),
        ):
            self.assertAlmostEqual(
                experiment._t_power(effect, 1.0, df, 0.05),
                expected,
                delta=1e-8,
            )
    def test_noncentral_t_cdf_keeps_extreme_negative_tail(self):
        # df>=3 的固定 u 网格在 |t| 很大时只有一两个节点落在
        # 正态尾的窄峰内，df=3、t=-100 的相对误差约 11%。
        # 参考值由 scipy.stats.nct.cdf 核对。
        for df, value, reference in (
            (3.0, -100.0, 4.0172758868251233e-07),
            (5.0, -100.0, 2.737192949747577e-10),
            (10.0, -30.0, 3.692727448008916e-12),
            (100.0, -30.0, 3.347625324326119e-54),
        ):
            cdf = experiment._noncentral_t_cdf(
                value, df, 0.5
            )
            self.assertIsNotNone(cdf)
            self.assertAlmostEqual(
                cdf,
                reference,
                delta=reference * 1.0e-6,
            )
    def test_noncentral_t_cdf_resolves_small_df_moderate_shift(self):
        # df=0.5、t=50、δ=100 时正态尾转折落在 u≈1.4，
        # t=u^(1/6) 网格上 800 个区间只有几个节点，CDF 会偏高
        # 约 4.8e-5；提高默认节点数后应达到双精度精度。
        reference = 0.06794650010173649
        cdf = experiment._noncentral_t_cdf(50.0, 0.5, 100.0)
        self.assertIsNotNone(cdf)
        self.assertAlmostEqual(
            cdf,
            reference,
            delta=reference * 1.0e-8,
        )
    def test_noncentral_t_cdf_keeps_extreme_positive_tail_small_df(self):
        # df<3 时默认 u 上界只到 sqrt(df-1)+12；δ 很大而
        # t 较小时正态尾转折落在截断之外，固定节点会漏掉
        # 整段仍可表示的正尾概率（如 df=1、δ=30、t=1 时
        # 修复前约少 34 个数量级）。参考值由高精度非中心 t
        # 积分/scipy.stats.nct 核对。
        for df, value, noncentrality, reference in (
            (
                1.0015698948012253,
                3.5197953216011717,
                58.69935129607124,
                5.4144106824582216e-58,
            ),
            (1.0, 1.0, 30.0, 7.212994172451246e-100),
            (2.0, 1.0, 30.0, 2.972314784251633e-131),
        ):
            cdf = experiment._noncentral_t_cdf(
                value, df, noncentrality
            )
            self.assertIsNotNone(cdf)
            self.assertAlmostEqual(
                cdf,
                reference,
                delta=reference * 1.0e-8,
            )
        self.assertAlmostEqual(
            experiment._noncentral_t_cdf(0.0, 1.0, 30.0),
            experiment._normal_cdf(-30.0),
            delta=0.0,
        )
    def test_noncentral_t_cdf_extends_chi_tail_for_large_shift(self):
        # df>=3 时默认 u 上界 sqrt(df-1)+12 只按 chi 众数附近
        # 取界；δ 很大时正态 CDF 的转折点会落在更远的 chi 尾部，
        # 例如 df≈766、δ≈97、t≈52.6 时主导区间在 u≈44，而默认
        # 上界仅 39.7，修复前会把 CDF 低估约 20 个数量级。
        # 参考值由 mpmath 分区间积分与 scipy.stats.nct 交叉核对。
        for df, value, noncentrality, reference in (
            (
                765.9433863028157,
                52.56226389840401,
                97.07556923763332,
                4.131299637815571e-142,
            ),
            (
                935.935103087207,
                26.934708243729347,
                63.62909882746632,
                1.2755489939161262e-206,
            ),
            (
                640.5084742335358,
                91.23876824784116,
                148.90803192290394,
                1.221547687472e-87,
            ),
        ):
            cdf = experiment._noncentral_t_cdf(
                value, df, noncentrality
            )
            self.assertIsNotNone(cdf)
            self.assertAlmostEqual(
                cdf,
                reference,
                delta=reference * 1.0e-6,
            )
    def test_noncentral_t_cdf_rejects_undefined_inputs(self):
        # NaN/负自由度没有定义；原实现先走「df 非有限 -> 正态
        # 极限」分支，把 df=NaN/-inf 误报成正态 CDF。NaN 的
        # 取值与非中心参数也必须返回 None，不能静默变成 0。
        for df in (float("nan"), float("-inf"), -1.0, 0.0):
            self.assertIsNone(
                experiment._noncentral_t_cdf(1.0, df, 0.5),
                df,
            )
        self.assertIsNone(
            experiment._noncentral_t_cdf(
                float("nan"), 5.0, 0.5
            )
        )
        self.assertIsNone(
            experiment._noncentral_t_cdf(
                1.0, 5.0, float("nan")
            )
        )
        # δ=±inf 的极限仍可用：CDF 分别趋于 1/0。
        self.assertEqual(
            experiment._noncentral_t_cdf(
                1.0, 5.0, float("inf")
            ),
            1.0,
        )
        self.assertEqual(
            experiment._noncentral_t_cdf(
                1.0, 5.0, float("-inf")
            ),
            0.0,
        )
        self.assertIsNone(
            experiment._noncentral_t_cdf(
                float("inf"), 5.0, float("inf")
            )
        )

    def test_t_power_handles_small_df_without_overflow(self):
        u_values, weights = experiment._t_power_nodes(0.01)
        self.assertEqual(len(u_values), len(weights))
        self.assertTrue(
            all(math.isfinite(weight) for weight in weights)
        )
        self.assertAlmostEqual(sum(weights), 1.0, delta=5e-5)
        for effect, expected in (
            (1.0e-6, 0.05),
            (1.0, 0.05021351447305985),
        ):
            power = experiment._t_power(effect, 1.0, 0.01, 0.05)
            self.assertIsNotNone(power)
            self.assertAlmostEqual(power, expected, delta=5e-5)
    def test_t_power_resolves_tiny_alpha_for_small_df(self):
        # alpha 极小时 critical/√df 可达 1e19 以上，固定 u 网格
        # 会漏掉 u<1/scale 的全部概率质量并返回 0。
        for df, alpha, expected in (
            (0.5, 1.0e-10, 1.222065580725563e-10),
            (0.5, 1.0e-20, 1.2220655807292234e-20),
            (1.0, 1.0e-10, 1.4621550516047826e-10),
            (1.0, 1.0e-20, 1.4621550516047845e-20),
            (2.9, 1.0e-20, 2.5549713688518368e-20),
        ):
            power = experiment._t_power(1.0, 1.0, df, alpha)
            self.assertIsNotNone(power)
            self.assertGreater(power, alpha)
            self.assertAlmostEqual(
                power,
                expected,
                delta=expected * 1.0e-6,
            )
    def test_t_power_large_noncentrality_small_df(self):
        # df<3 且 δ 很大时，w=scale·u 的转折宽度约为 1，而
        # 积分区间长度约为 δ；固定 20000 个区间会把转折稀释，
        # 例如 df=0.1、δ=1e5 时高估约 1.5%。参考值由高精度
        # chi-CDF 正态混合积分得到。
        for df, alpha, effect, expected in (
            (0.1, 0.01, 100000.0, 0.03350029910411294),
            (0.5, 1.0e-8, 100000.0, 3.846215750003292e-06),
            (4.0, 1.0e-20, 100000.0, 0.19714960547528733),
        ):
            power = experiment._t_power(
                effect, 1.0, df, alpha
            )
            self.assertIsNotNone(power)
            self.assertAlmostEqual(
                power,
                expected,
                delta=expected * 1.0e-8,
            )
    def test_t_power_uses_normal_mix_for_small_df_default_grid(self):
        # df=3–4 时默认 u 网格只有 800 个 Simpson 区间，
        # 对 alpha 很小、非中心参数接近 0 的功效可低估约
        # 2e-5；改用正态位移的 chi-CDF 混合积分。
        for effect, df, alpha, expected in (
            (
                0.14002715713472244,
                3.7321698113324584,
                0.00021201218199043572,
                0.00021967586829116905,
            ),
            (
                -0.003972075556613209,
                4.228971419082644,
                2.5110661961583918e-05,
                2.5111492140108046e-05,
            ),
        ):
            power = experiment._t_power(
                effect, 1.0, df, alpha
            )
            self.assertIsNotNone(power)
            self.assertAlmostEqual(
                power,
                expected,
                delta=expected * 1.0e-8,
            )
    def test_t_p_value_uses_exact_beta_before_normal_limit(self):
        # df=1e6、t=30 时一阶正态尾修正仍有约 2% 误差；
        # 不完全 Beta 在此自由度下精度约 5e-10，应优先使用。
        self.assertAlmostEqual(
            experiment._two_sided_t_p_value(30.0, 1.0e6)
            / 1.2020094233664118e-197,
            1.0,
            places=8,
        )
    def test_t_p_value_avoids_lgamma_cancellation_at_large_df(self):
        # df~1.5e8 时 lgamma(a+b)-lgamma(a) 的相消会让 p 值
        # 产生约 1e-6 的相对误差；x 接近 1 时也不能依赖 1-x。
        self.assertAlmostEqual(
            experiment._two_sided_t_p_value(1.7, 1.5e8)
            / 0.08913092759024125,
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            experiment._two_sided_t_p_value(
                30.0,
                2.0e8,
            )
            / 9.8233910423912479e-198,
            1.0,
            delta=1.0e-8,
        )
    def test_t_p_value_avoids_complement_rounding_for_small_tail(self):
        # df=0.5、t≈4.24e7 时 x=df/(df+t²)≈2.78e-16；若先算
        # complement_x 再取 1-complement_x，表示误差会让
        # 1-complement_x 变成 3.33e-16，上尾被高估约 5%。
        # 直接计算 I_x 应保留约 1e-16 的相对精度。
        p_value = experiment._two_sided_t_p_value(
            42443372.28529559,
            0.5,
        )
        self.assertIsNotNone(p_value)
        self.assertAlmostEqual(
            p_value / 9.845217954535378e-05,
            1.0,
            places=12,
        )
        critical = experiment._t_critical(0.5, 1.0e-4)
        self.assertIsNotNone(critical)
        self.assertAlmostEqual(
            experiment._two_sided_t_p_value(critical, 0.5)
            / 1.0e-4,
            1.0,
            places=12,
        )
    def test_incomplete_beta_converges_for_large_parameters(self):
        # 连分式步数上限为 200 时，a=b=1e6、x=0.5 只能算到
        # 0.499687；提高上限后应恢复对称值 0.5，并让大样本
        # 二项分布 Clopper-Pearson 上界与 beta 分位数一致。
        for a in (1.0e6, 1.0e7):
            self.assertAlmostEqual(
                experiment._regularized_incomplete_beta(
                    0.5,
                    a,
                    a,
                ),
                0.5,
                delta=5.0e-8,
            )
        bound = experiment._binomial_upper_confidence_bound(
            500000000,
            1000000000,
            0.95,
        )
        self.assertIsNotNone(bound)
        self.assertAlmostEqual(
            bound / 0.5000260079193697,
            1.0,
            places=9,
        )
    def test_binomial_upper_bound_resolves_confidence_near_one(self):
        # failures>4096 时若按 I_p(k+1,n-k)=confidence 二分，
        # confidence=1-1e-15 与计算值在双精度下无法区分，
        # 上界会提前停在偏小约 9e-5 的位置。改用互补尾
        # I_{1-p}(n-k,k+1)=alpha 后应满足精确定义。
        confidence = 1.0 - 1.0e-15
        alpha = 1.0 - confidence
        bound = experiment._binomial_upper_confidence_bound(
            5000,
            1000000,
            confidence,
        )
        self.assertIsNotNone(bound)
        cdf = experiment._binomial_lower_tail_probability(
            5000,
            1000000,
            bound,
        )
        self.assertAlmostEqual(cdf / alpha, 1.0, places=9)
    def test_t_p_value_handles_infinite_and_invalid_df(self):
        # df→+inf 时 t 分布退化为标准正态；原实现得到
        # inf/inf=NaN。NaN/负自由度没有定义，必须返回 None
        # 而不是让调用方把 NaN 当成数值继续传播。
        for t in (0.0, 1.0, 1.959963984540054, 10.0):
            self.assertAlmostEqual(
                experiment._two_sided_t_p_value(t, float("inf")),
                math.erfc(t / math.sqrt(2.0)),
                delta=1e-12,
            )
        self.assertIsNone(
            experiment._two_sided_t_p_value(1.0, float("nan"))
        )
        self.assertIsNone(
            experiment._two_sided_t_p_value(1.0, float("-inf"))
        )
        self.assertIsNone(
            experiment._two_sided_t_p_value(float("inf"), float("nan"))
        )
    def test_t_p_value_stays_stable_for_large_finite_df(self):
        # 大而有限的自由度会让 df/(df+t²) 舍入成 1，
        # 原实现因此把 p 值退化成 1.0 或接近 0；应平滑趋近
        # 标准正态尾概率。
        for t in (1.0, 1.96, 3.0, 10.0):
            normal = math.erfc(t / math.sqrt(2.0))
            previous = None
            for df in (1.0e6, 1.0e9, 1.0e16, 1.0e100):
                p_value = experiment._two_sided_t_p_value(t, df)
                self.assertIsNotNone(p_value)
                self.assertAlmostEqual(
                    p_value,
                    normal,
                    delta=1e-6,
                )
                self.assertGreaterEqual(p_value, normal)
                if previous is not None:
                    self.assertLessEqual(p_value, previous)
                previous = p_value
    def test_t_p_value_keeps_upper_tail_near_one(self):
        for df, t in (
            (1.0e7, 1.0e-5),
            (1.0e8, 1.0e-4),
            (2.0e8, 1.0e-4),
        ):
            p_value = experiment._two_sided_t_p_value(t, df)
            self.assertIsNotNone(p_value)
            self.assertAlmostEqual(
                p_value,
                math.erfc(t / math.sqrt(2.0)),
                delta=1.0e-9,
            )
            self.assertLess(p_value, 1.0)
    def test_t_critical_inverts_upper_tail_near_one(self):
        for df in (1.0e5, 1.0e8):
            critical = experiment._t_critical(df, 0.999999)
            self.assertIsNotNone(critical)
            self.assertAlmostEqual(
                experiment._two_sided_t_p_value(critical, df),
                0.999999,
                places=12,
            )
    def test_tost_test_keeps_small_upper_tail(self):
        result = experiment._tost_test(
            -5.0,
            1.0,
            100.0,
            15.0,
            alpha=1.0e-20,
        )
        expected = experiment._two_sided_t_p_value(10.0, 100.0)
        self.assertIsNotNone(expected)
        self.assertAlmostEqual(
            float(result["tost_p_lower"]) / (expected / 2.0),
            1.0,
            places=12,
        )
        self.assertGreater(float(result["tost_p_value"]), 1.0e-20)
