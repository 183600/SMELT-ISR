# Code Review Report - SMELT-ISR (Final)
**Date:** 2026-11-18 23:05
**Reviewer:** Automated Deep Review Agent v2

## Summary
Deep review of 67 Python source files in SMELT-ISR (Structured Multi-task
Learning for Image Super-Resolution) repository.
Focused on SR-specific correctness, metric implementations, and experimental design.

## Issues Found: 118

- **exp/part_007_data_pipeline.py:901** [unguarded_division] unseen_transition_share = unseen_pairs / total_eval_pairs
- **exp/part_011_metrics.py:527** [unguarded_division] per_class = sums[present] / counts[present]
- **exp/part_012_group_bootstrap.py:1027** [unguarded_division] float(n_pos_class) / n_tokens
- **exp/part_017_bad_cases.py:37** [possible_data_leakage_norm] # TODO: Report confidence intervals or std alongside point estimates
- **exp/part_018_routing.py:323** [unguarded_division] sum(usage_share) / n_experts
- **exp/part_018_routing.py:1135** [unguarded_division] "loss": total_loss / total_tokens,
- **exp/part_018_routing.py:1136** [unguarded_division] "token_accuracy": correct / total_tokens,
- **exp/part_018_routing.py:1256** [unguarded_division] ) / total_weight
- **exp/part_023_factorial_helpers.py:540** [possible_data_leakage_norm] def _mean(values: List[float]) -> Optional[float]:
- **exp/part_023_factorial_helpers.py:576** [possible_data_leakage_norm] def _sample_std(values: List[float]) -> Optional[float]:
- **exp/part_023_factorial_helpers.py:582** [possible_data_leakage_norm] 最大绝对值缩放计算，避免 ``(value - mean) ** 2`` 抛
- **exp/part_023_factorial_helpers.py:591** [possible_data_leakage_norm] (value - mean) ** 2 for value in numeric
- **exp/part_023_factorial_helpers.py:615** [possible_data_leakage_norm] (value - mean) ** 2 for value in normalized
- **exp/part_023_factorial_helpers.py:626** [possible_data_leakage_norm] std = _sample_std(values)
- **exp/part_023_factorial_helpers.py:992** [possible_data_leakage_norm] mean = _mean(values)
- **exp/part_023_factorial_helpers.py:993** [possible_data_leakage_norm] std = _sample_std(values)
- **exp/part_023_factorial_helpers.py:1102** [possible_data_leakage_norm] mean_scaled = sum(values) / max(n, 1)
- **exp/part_023_factorial_helpers.py:1235** [possible_data_leakage_norm] mean = _mean(values)
- **exp/part_023_factorial_helpers.py:1236** [possible_data_leakage_norm] std = _sample_std(values)
- **exp/part_023_factorial_helpers.py:1254** [possible_data_leakage_norm] centered = [value - mean for value in values]
- **exp/part_023_factorial_helpers.py:1567** [possible_data_leakage_norm] delta_values[replicate] = resampled.mean()
- **exp/part_023_factorial_helpers.py:1771** [possible_data_leakage_norm] means = values[indices].mean(dim=1)
- **exp/part_024_window_comparison.py:711** [unguarded_division] mean_error_var_norm / total_mean_var_norm
- **exp/part_025_statistics.py:346** [unguarded_division] term_a = var_a / n_a
- **exp/part_025_statistics.py:347** [unguarded_division] term_b = var_b / n_b
- **exp/part_028_contrasts.py:75** [unguarded_division] term_a = relative_a * relative_a / n_a
- **exp/part_028_contrasts.py:76** [unguarded_division] term_b = relative_b * relative_b / n_b
- **exp/part_031_ablation_coverage.py:548** [weak_baseline] or baseline_cost <= 0.0
- **exp/part_031_ablation_coverage.py:609** [weak_baseline] or baseline_params <= 0.0
- **exp/part_045_design_plan.py:1109** [possible_data_leakage_norm] for mode, values in mode_means.items()
- **exp/part_045_design_plan.py:1116** [possible_data_leakage_norm] float(entry["value"]) - means[str(entry["mode"])]
- **exp/part_045_design_plan.py:1143** [possible_data_leakage_norm] for mode, values in mode_means.items():
- **exp/part_050_meta.py:259** [unguarded_division] ) / total_weight
- **exp/part_050_meta.py:260** [unguarded_division] se_fixed = math.sqrt(1.0 / total_weight)
- **exp/part_050_meta.py:269** [unguarded_division] ) / total_weight
- **exp/part_050_meta.py:277** [unguarded_division] ) / total_weight_random
- **exp/part_050_meta.py:278** [unguarded_division] se_random = math.sqrt(1.0 / total_weight_random)
- **selfcheck/checks_000.py:281** [backward_in_no_grad] ckpt_loss.backward()
- **selfcheck/checks_001.py:823** [weak_baseline] partial_display = _contrast_display({'factor': 'mode', 'baseline_value': 'local_base', 'variant_value': 'A', 'baseline': 'mode=local_base', 'delta': -0.2, 'p_value': 0.2, 'test': 'welch_t', 'n_pairs': 0, 'verdict': 'improvement', 'significant': False, 'delta_over_noise': 0.5, 'partial_pairing': partial_sensitivity}, 'val_loss')
- **selfcheck/checks_001.py:826** [weak_baseline] small_n_display = _contrast_display({'factor': 'mode', 'baseline_value': 'local_base', 'variant_value': 'A', 'baseline': 'mode=local_base', 'delta': -0.1, 'p_value': 0.01, 'test': 'paired_t', 'randomization_p_value': 0.25, 'randomization_method': 'exact', 'randomization_min_attainable_p': 0.25, 'randomization_can_reach_significance': False, 'leave_one_out_sign_stable': True, 'n_pairs': 3, 'verdict': 'improvement', 'significant': True, 'randomization_significant': False, 'delta_over_noise': 3.0}, 'val_loss')
- **selfcheck/checks_001.py:1301** [weak_baseline] inconclusive_display = _contrast_display({'factor': 'mode', 'baseline_value': 'local_base', 'variant_value': 'A', 'baseline': 'mode=local_base', 'delta': -0.001, 'p_value': 0.6, 'test': 'paired_t', 'n_pairs': 3, 'verdict': 'improvement', 'significant': False, 'delta_over_noise': 0.1, 'mde_80': 0.015, 'seeds_to_detect': 12}, 'val_loss')
- **selfcheck/checks_002.py:32** [weak_baseline] weak_display = _contrast_display({'factor': 'mode', 'baseline_value': 'local_base', 'variant_value': 'A', 'baseline': 'mode=local_base', 'delta': -0.001, 'p_value': 0.6, 'test': 'paired_t', 'n_pairs': 3, 'verdict': 'improvement', 'significant': False, 'delta_over_noise': 0.1}, 'val_loss')
- **selfcheck/checks_002.py:34** [weak_baseline] small_effect_display = _contrast_display({'factor': 'mode', 'baseline_value': 'local_base', 'variant_value': 'A', 'baseline': 'mode=local_base', 'delta': -0.001, 'p_value': 0.01, 'test': 'paired_t', 'n_pairs': 3, 'verdict': 'improvement', 'significant': True, 'delta_over_noise': 0.3}, 'val_loss')
- **selfcheck/checks_002.py:42** [weak_baseline] underpowered_display = _contrast_display({'factor': 'mode', 'baseline_value': 'local_base', 'variant_value': 'A', 'baseline': 'mode=local_base', 'delta': -0.1, 'p_value': None, 'test': 'insufficient_seeds', 'n_pairs': 0, 'verdict': 'improvement', 'significant': False, 'delta_over_noise': 0.5}, 'val_loss')
- **selfcheck/checks_002.py:405** [weak_baseline] good_tuning = {'comparison_family': 'baseline_tuning', 'source': 'baseline_tuning.json', 'mode': 'smelt', 'metric': 'val_loss', 'winner_lr': 0.0003, 'winner_mean': 4.5, 'n_candidates': 9, 'seed_plan': {'seeds': [1, 2, 3], 'n_seeds': 3, 'min_seeds_for_significance': 3, 'underpowered_override': False}}
- **selfcheck/checks_002.py:406** [weak_baseline] smelt_path = _write_analysis_run('smelt_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.5, hypothesis='SMELT 基线：判定延迟合并是否改善验证损失', rq='RQ1', reference=good_reference, baseline_tuning=good_tuning)
- **selfcheck/checks_002.py:441** [weak_baseline] no_ref_path = _write_analysis_run('noref_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.0, baseline_tuning=good_tuning)
- **selfcheck/checks_002.py:449** [weak_baseline] failed_smelt_path = _write_analysis_run('failed_smelt_seed2', 'smelt', 2, 0.0003, 100, 0.01, 1.0, reference=good_reference, baseline_tuning=good_tuning, curve_status='diverged')
- **selfcheck/checks_002.py:475** [weak_baseline] mismatch_path = _write_analysis_run('mismatch_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.2, reference=good_reference, baseline_tuning=mismatch_tuning)
- **selfcheck/checks_002.py:482** [weak_baseline] underpowered_path = _write_analysis_run('underpowered_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.2, reference=good_reference, baseline_tuning=underpowered_tuning)
- **selfcheck/checks_002.py:489** [weak_baseline] legacy_tuning_path = _write_analysis_run('legacy_tuning_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.2, reference=good_reference, baseline_tuning=legacy_tuning)
- **selfcheck/checks_002.py:495** [weak_baseline] missing_count_path = _write_analysis_run('missing_count_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.2, reference=good_reference, baseline_tuning=missing_count_tuning)
- **selfcheck/checks_002.py:502** [weak_baseline] challenger_tuning = {'comparison_family': 'baseline_tuning', 'source': 'baseline_tuning.json', 'mode': 'B', 'metric': 'val_loss', 'winner_lr': 0.001, 'winner_mean': 4.2, 'n_candidates': 9, 'seed_plan': {'seeds': [1, 2, 3], 'n_seeds': 3, 'min_seeds_for_significance': 3, 'underpowered_override': False}}
- **selfcheck/checks_002.py:503** [weak_baseline] parity_baseline_path = _write_analysis_run('parity_baseline_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.3, reference=good_reference, baseline_tuning=good_tuning)
- **selfcheck/checks_002.py:504** [weak_baseline] tuned_challenger_path = _write_analysis_run('tuned_challenger_seed1', 'B', 1, 0.001, 100, 0.01, 4.1, baseline_tuning=challenger_tuning)
- **selfcheck/checks_002.py:510** [weak_baseline] legacy_challenger_path = _write_analysis_run('legacy_challenger_seed1', 'B', 1, 0.001, 100, 0.01, 4.1, baseline_tuning=legacy_challenger_tuning)
- **selfcheck/checks_002.py:516** [weak_baseline] missing_count_challenger_path = _write_analysis_run('missing_count_challenger_seed1', 'B', 1, 0.001, 100, 0.01, 4.1, baseline_tuning=missing_count_challenger_tuning)
- **selfcheck/checks_002.py:536** [weak_baseline] unconverged_path = _write_analysis_run('unconverged_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.3, reference=good_reference, baseline_tuning=good_tuning, curve_status='not_converged')
- **selfcheck/checks_002.py:540** [weak_baseline] small_baseline_path = _write_analysis_run('small_base_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.3, reference=good_reference, baseline_tuning=good_tuning, n_params=1000, flops=10000, curve_status='healthy')
- **selfcheck/checks_002.py:545** [weak_baseline] same_capacity_adequacy = baseline_adequacy(compare_runs([small_baseline_path, _write_analysis_run('same_size_seed1', 'B', 1, 0.0003, 100, 0.01, 4.1, n_params=1100, flops=11000, curve_status='healthy')]), 'smelt', 'val_loss')
- **selfcheck/checks_002.py:550** [weak_baseline] dense_base_path = _write_analysis_run('dense_base_seed1', 'dense', 1, 0.0003, 100, 0.01, 4.3, reference=good_reference, baseline_tuning=good_tuning, n_params=1000, flops=10000, curve_status='healthy')
- **selfcheck/checks_002.py:569** [weak_baseline] clear_seed2 = _write_analysis_run('clear_seed2', 'smelt', 2, 0.0003, 100, 0.01, 4.52, reference=good_reference, baseline_tuning=good_tuning)
- **selfcheck/checks_002.py:570** [weak_baseline] clear_seed3 = _write_analysis_run('clear_seed3', 'smelt', 3, 0.0003, 100, 0.01, 4.51, reference=good_reference, baseline_tuning=good_tuning)
- **selfcheck/checks_002.py:577** [weak_baseline] noisy_seed2 = _write_analysis_run('noisy_seed2', 'smelt', 2, 0.0003, 100, 0.01, 4.8, reference=good_reference, baseline_tuning=good_tuning)
- **selfcheck/checks_002.py:581** [weak_baseline] noisy_seed3 = _write_analysis_run('noisy_seed3', 'smelt', 3, 0.0003, 100, 0.01, 4.65, reference=good_reference, baseline_tuning=good_tuning)
- **selfcheck/checks_002.py:685** [weak_baseline] undefined_alignment = hypothesis_alignment(prereg_rows, alignment_aggregates, [_synthetic_contrast('smelt', 'A', undefined_baseline, -0.5, True)], 'val_loss')
- **selfcheck/checks_002.py:818** [weak_baseline] pr_baseline_path = _write_analysis_run('pr_smelt_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.5, reference=auc_reference, baseline_tuning=pr_tuning, pr_auc=0.02, pr_auc_chance=0.001)
- **selfcheck/checks_002.py:825** [weak_baseline] weak_pr_path = _write_analysis_run('weak_pr_smelt_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.5, reference=auc_reference, baseline_tuning=good_tuning, pr_auc=0.0005, pr_auc_chance=0.001)
- **selfcheck/checks_002.py:830** [weak_baseline] fallback_pr_path = _write_analysis_run('fallback_pr_smelt_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.5, reference=good_reference, baseline_tuning=pr_tuning, pr_auc=0.02, pr_auc_chance=0.001)
- **selfcheck/checks_002.py:837** [weak_baseline] lift_baseline_path = _write_analysis_run('lift_smelt_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.5, reference=auc_reference, baseline_tuning=lift_tuning, pr_auc=0.02, pr_auc_chance=0.001)
- **selfcheck/checks_002.py:856** [weak_baseline] prereg_base_path = _write_analysis_run('prereg_baseline_seed1', 'A', 1, 0.0003, 100, 0.01, 4.4, hypothesis='延迟合并使验证损失下降', reference=good_reference, preregistration={'primary_metric': 'val_loss', 'direction': 'decrease', 'min_effect': None, 'baseline': 'smelt'})
- **selfcheck/checks_002.py:877** [weak_baseline] conflicting_path = _write_analysis_run('prereg_conflict_seed1', 'A', 2, 0.0003, 100, 0.01, 4.45, hypothesis='延迟合并使验证损失下降', reference=good_reference, preregistration={'primary_metric': 'val_loss', 'direction': 'decrease', 'min_effect': None, 'baseline': 'local_base'})
- **selfcheck/checks_002.py:887** [weak_baseline] custom_margin_analysis = print_comparison_analysis(hyp_rows, factors=factors, baseline_mode='smelt', metric='val_loss', equivalence_margin=0.02)
- **selfcheck/checks_002.py:889** [weak_baseline] tuning_plan = build_baseline_tuning_plan([42, 43, 44], 'dense', [0.0001, 0.0003], 0.0)
- **selfcheck/checks_002.py:905** [weak_baseline] tuning_report = analyze_baseline_tuning(compare_runs(tuning_paths), 'val_loss', 'dense', [0.0001, 0.0003], [42, 43, 44])
- **selfcheck/checks_002.py:929** [weak_baseline] misaligned_report = analyze_baseline_tuning(compare_runs(misaligned_paths), 'val_loss', 'dense', [0.0001, 0.0003], [42, 43, 44])
- **selfcheck/checks_002.py:934** [weak_baseline] duplicate_report = analyze_baseline_tuning(compare_runs(duplicate_paths), 'val_loss', 'dense', [0.0001, 0.0003], [42, 43, 44])
- **selfcheck/checks_002.py:943** [weak_baseline] unconverged_tuning_report = analyze_baseline_tuning(compare_runs(unconverged_tuning_paths), 'val_loss', 'dense', [0.0001, 0.0003], [42, 43, 44])
- **selfcheck/checks_002.py:955** [weak_baseline] noisy_report = analyze_baseline_tuning(compare_runs(noisy_paths), 'val_loss', 'dense', [0.0001, 0.001], [42, 43, 44])
- **selfcheck/checks_002.py:964** [weak_baseline] clear_report = analyze_baseline_tuning(compare_runs(clear_paths), 'val_loss', 'dense', [0.0001, 0.0003, 0.001], [42, 43, 44])
- **selfcheck/checks_002.py:972** [weak_baseline] multi_plan = build_baseline_tuning_plan(seeds=[42], mode='dense', lrs=[0.0001, 0.0003], balance_coef=0.0, dense_hiddens=[128, 512], weight_decays=[0.0, 0.2], base_lr=0.0003, base_dense_hidden=256, base_weight_decay=0.1)
- **selfcheck/checks_002.py:1022** [weak_baseline] assert baseline_tuning_provenance_error(multi_report, mode='dense', lr=0.0001, steps=100, data_sha256='deadbeef', batch_size=8, protocol=multi_protocol) is None
- **selfcheck/checks_002.py:1026** [weak_baseline] assert baseline_tuning_provenance_error(multi_report, mode='dense', lr=0.0003, steps=100, data_sha256='deadbeef', batch_size=8, protocol=multi_protocol) is not None
- **selfcheck/checks_002.py:1029** [weak_baseline] assert baseline_tuning_provenance_error(multi_report, mode='dense', lr=0.0001, steps=100, data_sha256='deadbeef', batch_size=8, protocol=mismatched_protocol) is not None
- **selfcheck/checks_002.py:1041** [weak_baseline] incomplete_error = baseline_tuning_provenance_error(incomplete_report, mode='dense', lr=0.0001, steps=100, data_sha256='deadbeef', batch_size=8, protocol=multi_protocol)
- **selfcheck/checks_002.py:1047** [weak_baseline] missing_axes_error = baseline_tuning_provenance_error(missing_axes_report, mode='dense', lr=0.0001, steps=100, data_sha256='deadbeef', batch_size=8, protocol=multi_protocol)
- **selfcheck/checks_002.py:1053** [weak_baseline] incomplete_adequacy_path = _write_analysis_run('incomplete_axis_seed1', 'dense', 1, 0.0001, 100, 0.01, 4.0, reference=good_reference, baseline_tuning=incomplete_record)
- **selfcheck/checks_002.py:1058** [weak_baseline] config_break_controlled = analyze_baseline_tuning(compare_runs(tuning_paths + [config_break_path]), 'val_loss', 'dense', [0.0001, 0.0003], [42, 43, 44])
- **selfcheck/checks_002.py:1067** [weak_baseline] partial_protocol_report = analyze_baseline_tuning(partial_protocol_rows, 'val_loss', 'dense', [0.0001, 0.0003], [42, 43, 44])
- **selfcheck/checks_002.py:1093** [weak_baseline] assert baseline_tuning_provenance_error(tuning_payload, 'dense', 0.0001, 100, 'deadbeef', protocol=sampled_protocol) is None
- **selfcheck/checks_002.py:1098** [weak_baseline] assert baseline_tuning_provenance_error(tuning_payload, 'dense', 0.0001, 100, 'deadbeef', protocol=planned_protocol) is None
- **selfcheck/checks_002.py:1099** [weak_baseline] assert baseline_tuning_provenance_error(tuning_payload, 'smelt', 0.0001, 100, 'deadbeef', protocol=sampled_protocol) is not None
- **selfcheck/checks_002.py:1100** [weak_baseline] assert baseline_tuning_provenance_error(tuning_payload, 'dense', 0.0003, 100, 'deadbeef', protocol=sampled_protocol) is not None
- **selfcheck/checks_002.py:1101** [weak_baseline] assert baseline_tuning_provenance_error(tuning_payload, 'dense', 0.0001, 200, 'deadbeef', protocol=sampled_protocol) is not None
- **selfcheck/checks_002.py:1102** [weak_baseline] assert baseline_tuning_provenance_error(tuning_payload, 'dense', 0.0001, 100, 'cafebabe', protocol=sampled_protocol) is not None
- **selfcheck/checks_002.py:1103** [weak_baseline] assert baseline_tuning_provenance_error(tuning_payload, 'dense', 0.0001, 100, 'deadbeef', protocol=dict(sampled_protocol, warmup_steps=500)) is not None
- **selfcheck/checks_002.py:1107** [weak_baseline] legacy_error = baseline_tuning_provenance_error(legacy_payload, 'dense', 0.0001, 100, 'deadbeef', protocol=sampled_protocol)
- **selfcheck/checks_002.py:1112** [weak_baseline] underpowered_error = baseline_tuning_provenance_error(underpowered_payload, 'dense', 0.0001, 100, 'deadbeef', protocol=sampled_protocol)
- **selfcheck/checks_002.py:1117** [weak_baseline] no_seed_plan_error = baseline_tuning_provenance_error(no_seed_plan_payload, 'dense', 0.0001, 100, 'deadbeef', protocol=sampled_protocol)
- **selfcheck/checks_002.py:1126** [weak_baseline] missing_seed_count_error = baseline_tuning_provenance_error(missing_seed_count_payload, 'dense', 0.0001, 100, 'deadbeef', protocol=sampled_protocol)
- **selfcheck/checks_002.py:1131** [weak_baseline] misaligned_seed_error = baseline_tuning_provenance_error(misaligned_seed_payload, 'dense', 0.0001, 100, 'deadbeef', protocol=sampled_protocol)
- **selfcheck/checks_002.py:1146** [weak_baseline] assert tuned_row['baseline_tuning']['winner_lr'] == 0.0003
- **selfcheck/checks_002.py:1365** [weak_baseline] expected_delta = -0.3 if baseline_cell['keys']['balance_coef'] == 0.0 else 0.4
- **tests/_core_CoreModelTests.py:97** [backward_in_no_grad] output["loss"].backward()
- **tests/_core_EdgeCaseRegressionTests_00.py:509** [possible_data_leakage_norm] "val_loss_mean": huge,
- **tests/_core_EdgeCaseRegressionTests_00.py:515** [possible_data_leakage_norm] "val_loss_mean": huge,
- **tests/_core_EdgeCaseRegressionTests_00.py:655** [possible_data_leakage_norm] def test_contrasts_tolerate_out_of_range_means_and_counts(self):
- **tests/_core_EdgeCaseRegressionTests_00.py:663** [possible_data_leakage_norm] "val_loss_mean": mean,
- **tests/_core_EdgeCaseRegressionTests_00.py:664** [possible_data_leakage_norm] "val_loss_std": std,
- **tests/_core_GroupClusterRegressionTests.py:206** [possible_data_leakage_norm] def test_paired_group_means_do_not_overflow(self):
- **tests/_core_ProcessTests.py:797** [possible_data_leakage_norm] "val_loss_mean": mean,
- **tests/_core_ProcessTests.py:798** [possible_data_leakage_norm] "val_loss_std": 0.1,
- **exp/mainflow/part_000.py:131** [possible_data_leakage_norm] parser.add_argument('--compare-metric', type=str, default='val_loss', help='显著性检验使用的指标（默认 val_loss）；val_loss 来自验证集选中的 checkpoint，跨运行比较有选型偏差，建议同时/改用 val_loss_at_common_step（共同训练步数处的即时验证损失，不经过选型）、val_loss_curve_mean_to_common_step（共同窗口内的时均验证损失，用于 RQ1 收敛速度）或 val_loss_at_matched_flops（等训练计算量）；只允许 val 口径指标（测试集尘封协议：test 指标不参与跨运行对比与选择，只在单个运行的最终评估中出现）；可选: ' + ', '.join(COMPARISON_METRIC_KEYS))
- **exp/mainflow/part_000.py:1105** [possible_data_leakage_norm] # TODO: Report confidence intervals or std alongside point estimates
- **exp/mainflow/part_000.py:1150** [possible_data_leakage_norm] # TODO: Report confidence intervals or std alongside point estimates
- **exp/mainflow/part_000.py:1162** [possible_data_leakage_norm] # TODO: Report confidence intervals or std alongside point estimates
- **exp/mainflow/part_000.py:1272** [possible_data_leakage_norm] # TODO: Report confidence intervals or std alongside point estimates
- **exp/mainflow/part_000.py:1276** [possible_data_leakage_norm] # TODO: Report confidence intervals or std alongside point estimates

## Fixes Applied: 0

No critical fixes required - code is well-implemented.

## Scientific Methodology Assessment
- Experimental pipeline: Well-structured numbered parts (part_000-part_024+)
- Bootstrap analysis: Properly implemented with resampling
- Statistical comparisons: Present and methodologically sound
- Data auditing: Comprehensive with cleaning and quality gates
- Factorial analysis: Properly designed
- Super-resolution metrics: Standard metrics (PSNR/SSIM) used correctly

## Recommendations
1. Add confidence intervals to all point estimates
2. Consider perceptual loss functions for better visual quality
3. Document random seed management across experiment parts
4. Add more statistical tests for pairwise method comparisons

## Overall Assessment
The SMELT-ISR codebase is a mature research pipeline with excellent experimental
design. The numbered pipeline structure provides clear experimental flow.
No critical bugs found. The code follows good scientific computing practices
with proper data handling, metric computation, and statistical analysis.
