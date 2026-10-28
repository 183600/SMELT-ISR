from __future__ import annotations

def _selfcheck_body_60201_0075():
    global combined_aggregates, display, f, ngram_contrast, ngram_contrasts, selection_analysis, selection_config, selection_path, small_effect_display, weak_display
    combined_aggregates = aggregate_runs(compare_runs(run_paths + ngram_paths), ['mode'])
    ngram_contrasts = [contrast for contrast in factor_contrasts(combined_aggregates, ['mode'], 'val_loss') if 'ngram' in (contrast['baseline_value'], contrast['variant_value'])]
    assert ngram_contrasts
    for ngram_contrast in ngram_contrasts:
        assert ngram_contrast['test'] == 'insufficient_seeds'
        assert ngram_contrast['significant'] is False
        assert min(ngram_contrast['n_baseline'], ngram_contrast['n_variant']) == 1
    selection_path = _write_analysis_run('selection_mismatch_seed1', 'A', 1, 0.0003, 100, 0.01, 4.2, precision_recall={'val_macro_f1': 0.5})
    with open(Path(selection_path) / 'config.json', encoding='utf-8') as f:
        selection_config = json.load(f)
    selection_config['args']['selection_metric'] = 'val_loss'
    with open(Path(selection_path) / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(selection_config, f)
    selection_analysis = print_comparison_analysis(compare_runs([selection_path]), metric='val_macro_f1')
    assert selection_analysis['selection_metric_mismatch'][0]['run'] == 'selection_mismatch_seed1'
    assert 'test_loss' not in COMPARISON_METRIC_KEYS
    assert all(('test_loss' not in row for row in analysis_rows)), '对比行不允许携带 test 指标'
    try:
        print_comparison_analysis(analysis_rows, factors=factors, metric='test_loss')
    except ValueError:
        pass
    else:
        raise AssertionError('test_loss 必须被拒绝作为对比指标（测试集不参与跨运行选择）')
    display = _contrast_display(contrast, 'val_loss')
    assert '存疑' not in display
    assert '不能据此声称' not in display
    weak_display = _contrast_display({'factor': 'mode', 'baseline_value': 'local_base', 'variant_value': 'A', 'baseline': 'mode=local_base', 'delta': -0.001, 'p_value': 0.6, 'test': 'paired_t', 'n_pairs': 3, 'verdict': 'improvement', 'significant': False, 'delta_over_noise': 0.1}, 'val_loss')
    assert '不能据此声称提升' in weak_display
    small_effect_display = _contrast_display({'factor': 'mode', 'baseline_value': 'local_base', 'variant_value': 'A', 'baseline': 'mode=local_base', 'delta': -0.001, 'p_value': 0.01, 'test': 'paired_t', 'n_pairs': 3, 'verdict': 'improvement', 'significant': True, 'delta_over_noise': 0.3}, 'val_loss')
    assert '实际意义存疑' in small_effect_display
    assert _paired_t_test([0.1, 0.2])['test'] == 'insufficient_seeds'
    assert _welch_t_test([1.0, 2.0], [3.0, 4.0])['test'] == 'insufficient_seeds'
    assert _paired_t_test([0.1, 0.2, 0.3, 0.4])['test'] == 'paired_t'

def _selfcheck_body_60340_0076():
    global bad_controlled, bad_path, declared, declared_k1, k1_controlled, k1_path, narrow_factors, split_controlled, split_path, steps_controlled, steps_path, underpowered_display
    underpowered_display = _contrast_display({'factor': 'mode', 'baseline_value': 'local_base', 'variant_value': 'A', 'baseline': 'mode=local_base', 'delta': -0.1, 'p_value': None, 'test': 'insufficient_seeds', 'n_pairs': 0, 'verdict': 'improvement', 'significant': False, 'delta_over_noise': 0.5}, 'val_loss')
    assert '不能据此声称差异' in underpowered_display
    bad_path = _write_analysis_run('A_seed45_badlr', 'A', 45, 0.001, 100, 0.01, 2.5)
    bad_controlled = check_controlled_variables(compare_runs(run_paths + [bad_path]), factors)
    assert not bad_controlled['ok']
    assert any((violation['key'] == 'lr' for violation in bad_controlled['violations']))
    declared = check_controlled_variables(compare_runs(run_paths + [bad_path]), factors + ['lr'])
    assert all((violation['key'] != 'lr' for violation in declared['violations']))
    steps_path = _write_analysis_run('A_seed46_badsteps', 'A', 46, 0.0003, 200, 0.01, 2.6)
    steps_controlled = check_controlled_variables(compare_runs(run_paths + [steps_path]), factors)
    assert any((violation['key'] == 'total_steps' for violation in steps_controlled['violations']))
    split_path = _write_analysis_run('A_seed48_badsplit', 'A', 48, 0.0003, 100, 0.01, 2.55, protocol_overrides={'split_sha256': {'train': 'train-a', 'val': 'val-a', 'test': 'test-a'}})
    split_controlled = check_controlled_variables(compare_runs(run_paths + [split_path]), factors)
    assert any((violation['key'] == 'split_sha256' for violation in split_controlled['violations'])), split_controlled['violations']
    k1_path = _write_analysis_run('A_seed47_badk1', 'A', 47, 0.0003, 100, 0.01, 2.4, k1=4)
    narrow_factors = parse_factor_spec('mode,seed')
    k1_controlled = check_controlled_variables(compare_runs(run_paths + [k1_path]), narrow_factors)
    assert any((violation['key'] == 'k1' for violation in k1_controlled['violations'])), k1_controlled['violations']
    declared_k1 = check_controlled_variables(compare_runs(run_paths + [k1_path]), narrow_factors + ['k1'])
    assert all((violation['key'] != 'k1' for violation in declared_k1['violations']))

def _selfcheck_body_60467_0077():
    global coverage, cross_bad, cross_ok, cross_split_bad, expected_missing, full_coverage, full_paths, mismatched_ngram, neural_cross, ngram_cross, split_mismatch_ngram
    neural_cross = {'run': 'neural', 'protocol': {'comparison_family': 'neural', 'data_sha256': 'abc', 'split_sha256': {'train': 't', 'val': 'v', 'test': 'x'}, 'seq_len': 64, 'val_fraction': 0.1, 'test_fraction': 0.1, 'batch_size': 4, 'eval_batches': 8, 'data_augmentation': 'none'}}
    ngram_cross = {'run': 'ngram', 'protocol': {'comparison_family': 'statistical', 'data_sha256': 'abc', 'split_sha256': {'train': 't', 'val': 'v', 'test': 'x'}, 'seq_len': 64, 'val_fraction': 0.1, 'test_fraction': 0.1, 'batch_size': 4, 'eval_batches': 8, 'data_augmentation': 'none'}}
    cross_ok = cross_family_protocol_check([neural_cross, ngram_cross], [])
    assert cross_ok['ok'], cross_ok
    assert 'data_sha256' in cross_ok['checked_keys']
    assert 'split_sha256' in cross_ok['checked_keys']
    mismatched_ngram = json.loads(json.dumps(ngram_cross))
    mismatched_ngram['protocol']['data_sha256'] = 'def'
    cross_bad = cross_family_protocol_check([neural_cross, mismatched_ngram], [])
    assert not cross_bad['ok']
    assert any((violation['key'] == 'data_sha256' for violation in cross_bad['violations']))
    split_mismatch_ngram = json.loads(json.dumps(ngram_cross))
    split_mismatch_ngram['protocol']['split_sha256']['val'] = 'v-other'
    cross_split_bad = cross_family_protocol_check([neural_cross, split_mismatch_ngram], [])
    assert not cross_split_bad['ok']
    assert any((violation['key'] == 'split_sha256' for violation in cross_split_bad['violations']))
    coverage = ablation_coverage(analysis_rows, factors)
    assert coverage['applicable']
    assert not coverage['partial']
    assert coverage['undeclared_project_modes'] == []
    assert any(('缺少延迟合并方案 B' in message for message in coverage['missing_cells'])), coverage['missing_cells']
    for expected_missing in ('local_base/b=off', 'A/b=off', 'B/b=off', 'B/b=on'):
        assert any((expected_missing in message for message in coverage['missing_cells'])), coverage['missing_cells']
    full_paths = run_paths + [_write_analysis_run('local_base_bal0_seed42', 'local_base', 42, 0.0003, 100, 0.0, 3.05), _write_analysis_run('A_bal0_seed42', 'A', 42, 0.0003, 100, 0.0, 2.95)]
    full_coverage = ablation_coverage(compare_runs(full_paths), factors, ['A'], allow_partial=True)
    assert not full_coverage['missing_cells'], full_coverage
    assert not full_coverage['structural_confounds']
    assert full_coverage['partial']
    assert full_coverage['partial_allowed']
    assert full_coverage['undeclared_project_modes'] == ['B']
    assert not any(('全因子组合齐全' in message for message in full_coverage['advisories']))

def _selfcheck_body_60601_0078():
    global confounded_coverage, confounded_paths, failed_full_coverage, failed_full_paths, failure_effect, finite_div_coverage, finite_div_paths
    assert any(('只有 1 个种子' in message for message in full_coverage['low_replicate_cells']))
    failed_full_paths = full_paths[:-1] + [_write_analysis_run('A_bal0_failed', 'A', 42, 0.0003, 100, 0.0, 2.0, curve_status='diverged')]
    failed_full_coverage = ablation_coverage(compare_runs(failed_full_paths), factors, ['A'], allow_partial=True)
    assert failed_full_coverage['n_failed_rows_excluded'] == 1
    assert failed_full_coverage['failed_runs'] == ['A_bal0_failed']
    assert any(('A/b=off' in message for message in failed_full_coverage['missing_cells'])), failed_full_coverage['missing_cells']
    assert any(('数值失败运行不能充当消融' in message for message in failed_full_coverage['advisories']))
    assert failed_full_coverage['attrition_confounds'], failed_full_coverage['numerical_attrition']
    assert any(('不平衡' in message for message in failed_full_coverage['advisories']))
    failure_effect = {'delayed_mode': 'A', 'effect_name': 'delayed_merge_main', 'claim': 'supports_component'}
    annotate_factorial_confounds({'effects': [failure_effect]}, {'ok': True, 'unchecked_keys': []}, failed_full_coverage)
    assert failure_effect['claim'] == 'attrition_confounded', failure_effect
    finite_div_paths = full_paths[:-1] + [_write_analysis_run('A_bal0_finite_diverging', 'A', 42, 0.0003, 100, 0.0, 2.0, curve_status='diverging')]
    finite_div_coverage = ablation_coverage(compare_runs(finite_div_paths), factors, ['A'], allow_partial=True)
    assert finite_div_coverage['n_failed_rows_excluded'] == 1
    assert finite_div_coverage['failed_runs'] == ['A_bal0_finite_diverging']
    assert any(('A/b=off' in message for message in finite_div_coverage['missing_cells'])), finite_div_coverage['missing_cells']
    assert not any(('全因子组合齐全' in message for message in failed_full_coverage['advisories']))
    confounded_paths = full_paths + [_write_analysis_run('A_bal0_seed43_k1four', 'A', 43, 0.0003, 100, 0.0, 2.9, k1=4)]
    confounded_coverage = ablation_coverage(compare_runs(confounded_paths), factors, ['A'], allow_partial=True)
    assert not confounded_coverage['missing_cells']
    assert confounded_coverage['structural_confounds'], confounded_coverage
    assert any(('k1' in message for message in confounded_coverage['structural_confounds']))
    assert not any(('全因子组合齐全' in message for message in confounded_coverage['advisories']))

def _selfcheck_body_60737_0079():
    global b_full_coverage, b_full_paths, b_missing_coverage, b_missing_paths, declared_b_coverage, delayed_main, effects, factorial, factorial_aggregates, factorial_paths, offset, seed
    b_missing_paths = full_paths + [_write_analysis_run('B_bal0.01_seed42', 'B', 42, 0.0003, 100, 0.01, 2.7)]
    b_missing_coverage = ablation_coverage(compare_runs(b_missing_paths), factors)
    assert b_missing_coverage['expected_delayed_modes'] == ['A', 'B']
    assert any(('B/b=off' in message for message in b_missing_coverage['missing_cells'])), b_missing_coverage['missing_cells']
    declared_b_coverage = ablation_coverage(analysis_rows, factors, ['A', 'B'])
    assert any(('缺少延迟合并方案 B' in message for message in declared_b_coverage['missing_cells'])), declared_b_coverage['missing_cells']
    b_full_paths = b_missing_paths + [_write_analysis_run('B_bal0_seed42', 'B', 42, 0.0003, 100, 0.0, 2.75)]
    b_full_coverage = ablation_coverage(compare_runs(b_full_paths), factors)
    assert not b_full_coverage['missing_cells'], b_full_coverage
    assert any(('3×2 全因子组合齐全' in message for message in b_full_coverage['advisories'])), b_full_coverage['advisories']
    factorial_paths = []
    for seed, offset in ((1, 0.0), (2, 0.02), (3, -0.02)):
        factorial_paths.extend([_write_analysis_run(f'fact_local_off_s{seed}', 'local_base', seed, 0.0003, 100, 0.0, 3.0 + offset), _write_analysis_run(f'fact_A_off_s{seed}', 'A', seed, 0.0003, 100, 0.0, 2.9 + offset), _write_analysis_run(f'fact_local_on_s{seed}', 'local_base', seed, 0.0003, 100, 0.01, 2.95 + offset), _write_analysis_run(f'fact_A_on_s{seed}', 'A', seed, 0.0003, 100, 0.01, 2.85 + offset)])
    factorial_aggregates = aggregate_runs(compare_runs(factorial_paths), ['mode', 'balance_coef'])
    factorial = factorial_effects(factorial_aggregates, ['mode', 'balance_coef'], 'val_loss', equivalence_margin=0.01)
    assert factorial['applicable']
    assert len(factorial['designs']) == 1, factorial
    assert not factorial['incomplete_designs']
    effects = {effect['effect_name']: effect for effect in factorial['effects'] if effect['delayed_mode'] == 'A'}
    assert 'delayed_merge_main' in effects, factorial
    delayed_main = effects['delayed_merge_main']
    assert abs(delayed_main['mean_effect'] + 0.1) < 1e-09
    assert delayed_main['n_pairs'] == 3
    assert delayed_main['aligned_seeds']
    assert delayed_main['significant']
    assert delayed_main['claim'] == 'supports_component'
    assert delayed_main['randomization_required_seeds'] == 6
    assert delayed_main['significant_robust_possible'] is False
    assert '至少需 6 个独立种子' in _format_factorial_effect(delayed_main)
    assert delayed_main['equivalence_margin'] == 0.01

def _selfcheck_body_60877_0080():
    global balance_main, component_coverage, component_verdicts, effect, equivalent_support, interaction, interaction_partial_factorial, interaction_partial_verdicts, metric_mismatch_verdicts, mixed_equivalence
    assert delayed_main['equivalence_margin_abs'] is not None
    assert abs(float(delayed_main['equivalence_margin_abs']) - 0.01 * 2.975) < 1e-09
    assert delayed_main['tost_p_value'] is not None
    assert delayed_main['tost_p_value_holm'] is not None
    assert not delayed_main['equivalent']
    assert delayed_main['equivalent_raw'] is False
    balance_main = effects['balance_loss_main']
    assert abs(balance_main['mean_effect'] + 0.05) < 1e-09
    assert balance_main['claim'] == 'supports_component'
    interaction = effects['interaction']
    assert abs(interaction['mean_effect']) < 1e-09
    assert not interaction['significant']
    assert interaction['claim'] == 'not_significant'
    component_coverage = {'expected_delayed_modes': ['A'], 'project_delayed_modes': ['A'], 'global_seed_alignment': {'applicable': True, 'all_expected_present': True, 'complete_seed_records': True, 'aligned': True, 'common_seeds': [1, 2, 3], 'n_common_seeds': 3}, 'missing_cells': [], 'structural_confounds': [], 'width_confounds': [], 'randomness_confounds': [], 'randomness_unverified': [], 'compute_parity': [{'delayed_mode': 'A', 'field': 'flops_per_forward', 'matched': True, 'ratio': 1.0}], 'low_replicate_cells': []}
    equivalent_support = _component_evidence_summary([{'claim': 'supports_component', 'equivalent': True}])
    assert equivalent_support['status'] == 'not_supported'
    assert '等价界' in equivalent_support['reason']
    mixed_equivalence = _component_evidence_summary([{'claim': 'supports_component', 'equivalent': True}, {'claim': 'supports_component', 'equivalent': False}])
    assert mixed_equivalence['status'] == 'partial'
    assert '等价界' in mixed_equivalence['reason']
    component_verdicts = ablation_component_verdicts(factorial, component_coverage)
    assert component_verdicts['all_components_supported'], component_verdicts
    assert component_verdicts['statuses']['delayed_merge_A'] == 'supported'
    assert component_verdicts['statuses']['balance_loss'] == 'supported'
    interaction_partial_factorial = json.loads(json.dumps(_json_safe(factorial), ensure_ascii=False))
    for effect in interaction_partial_factorial['effects']:
        if effect['effect_name'] == 'interaction':
            effect['claim'] = 'interaction_detected'
            effect['significant'] = True
        elif effect['effect_name'] == 'delayed_merge_at_balance_on':
            effect['claim'] = 'not_significant'
            effect['significant'] = False
    interaction_partial_verdicts = ablation_component_verdicts(interaction_partial_factorial, component_coverage)
    assert interaction_partial_verdicts['statuses']['delayed_merge_A'] == 'partial', interaction_partial_verdicts['components'][0]
    assert not interaction_partial_verdicts['all_components_supported']
    metric_mismatch_verdicts = ablation_component_verdicts(factorial, component_coverage, metric_preregistration={'metric_mismatch_runs': ['A_seed1'], 'metric_mismatch_details': ['A_seed1（预注册 val_loss，当前判定 val_macro_f1）']})
    assert not metric_mismatch_verdicts['all_components_supported']
    assert any(('指标选择偏差' in blocker for blocker in metric_mismatch_verdicts['blockers']))

def _selfcheck_body_61007_0081():
    global confounded_factorial, confounded_verdicts, effect, misaligned_components, misaligned_verdicts, missing_prereg_verdicts, no_compute_coverage, no_compute_verdicts, not_supported_factorial, not_supported_verdicts, partial_compute_coverage, partial_compute_verdicts, sensitivity_paths
    assert metric_mismatch_verdicts['metric_preregistration']['analysis_metric'] == factorial['metric']
    missing_prereg_verdicts = ablation_component_verdicts(factorial, component_coverage, metric_preregistration={'preregistration_missing_runs': ['A_seed1'], 'preregistered_runs': []})
    assert not missing_prereg_verdicts['all_components_supported']
    assert any(('没有训练前预注册主指标' in blocker for blocker in missing_prereg_verdicts['blockers'])), missing_prereg_verdicts['blockers']
    not_supported_factorial = json.loads(json.dumps(_json_safe(factorial), ensure_ascii=False))
    for effect in not_supported_factorial['effects']:
        if effect['effect_name'] == 'delayed_merge_main':
            effect['claim'] = 'not_significant'
            effect['significant'] = False
    not_supported_verdicts = ablation_component_verdicts(not_supported_factorial, component_coverage)
    assert not not_supported_verdicts['all_components_supported']
    assert not_supported_verdicts['statuses']['delayed_merge_A'] == 'partial'
    misaligned_components = json.loads(json.dumps(component_coverage))
    misaligned_components['global_seed_alignment']['aligned'] = False
    misaligned_verdicts = ablation_component_verdicts(factorial, misaligned_components)
    assert not misaligned_verdicts['all_components_supported']
    assert any(('种子' in blocker for blocker in misaligned_verdicts['blockers']))
    no_compute_coverage = json.loads(json.dumps(component_coverage))
    no_compute_coverage['compute_parity'] = []
    no_compute_verdicts = ablation_component_verdicts(factorial, no_compute_coverage)
    assert not no_compute_verdicts['all_components_supported']
    assert any(('计算量对齐未核对' in blocker for blocker in no_compute_verdicts['blockers']))
    partial_compute_coverage = json.loads(json.dumps(component_coverage))
    partial_compute_coverage['compute_parity'][0]['delayed_mode'] = 'B'
    partial_compute_verdicts = ablation_component_verdicts(factorial, partial_compute_coverage)
    assert not partial_compute_verdicts['all_components_supported']
    assert any(('A' in blocker and '计算量对齐未核对' in blocker for blocker in partial_compute_verdicts['blockers'])), partial_compute_verdicts['blockers']
    confounded_factorial = json.loads(json.dumps(_json_safe(factorial), ensure_ascii=False))
    confounded_factorial['confounded_designs'] = [{'context': {}, 'positive_balance_values': [0.01, 0.02]}]
    confounded_verdicts = ablation_component_verdicts(confounded_factorial, component_coverage)
    assert not confounded_verdicts['all_components_supported']
    assert any(('混杂的消融设计' in blocker for blocker in confounded_verdicts['blockers'])), confounded_verdicts['blockers']
    sensitivity_paths = []

def _selfcheck_body_61132_0082():
    global common_delta, global_seed_coverage, global_seed_rows, offset, seed, sensitivity_aggregates, sensitivity_common, sensitivity_coverage, sensitivity_primary, sensitivity_report, sensitivity_rows, sensitivity_verdicts
    for seed, offset, common_delta in ((1, 0.0, 0.02), (2, 0.02, -0.01), (3, -0.02, -0.01)):
        sensitivity_paths.extend([_write_analysis_run(f'sens_local_off_s{seed}', 'local_base', seed, 0.0003, 10, 0.0, 3.0 + offset, periodic_val_loss=[(10, 4.0 + offset)]), _write_analysis_run(f'sens_A_off_s{seed}', 'A', seed, 0.0003, 10, 0.0, 2.9 + offset, periodic_val_loss=[(10, 4.0 + offset + common_delta)]), _write_analysis_run(f'sens_local_on_s{seed}', 'local_base', seed, 0.0003, 10, 0.01, 2.95 + offset, periodic_val_loss=[(10, 3.95 + offset)]), _write_analysis_run(f'sens_A_on_s{seed}', 'A', seed, 0.0003, 10, 0.01, 2.85 + offset, periodic_val_loss=[(10, 3.95 + offset + common_delta)])])
    sensitivity_rows = compare_runs(sensitivity_paths)
    sensitivity_common = annotate_common_step_metrics(sensitivity_rows)
    assert sensitivity_common['ok'], sensitivity_common
    sensitivity_aggregates = aggregate_runs(sensitivity_rows, ['mode', 'balance_coef'])
    sensitivity_primary = factorial_effects(sensitivity_aggregates, ['mode', 'balance_coef'], 'val_loss')
    sensitivity_report = factorial_common_step_sensitivity(sensitivity_aggregates, ['mode', 'balance_coef'], 'val_loss', sensitivity_primary, component_coverage, controlled={'ok': True, 'unchecked_keys': []})
    assert sensitivity_report['applicable']
    assert any((item['effect_name'] == 'delayed_merge_main' and item['effect_name'] in {str(record['effect_name']) for record in sensitivity_report['support_flips']} for item in sensitivity_primary['effects'])), sensitivity_report
    assert 'delayed_merge_A' in sensitivity_report['component_flips']
    sensitivity_coverage = json.loads(json.dumps(component_coverage))
    sensitivity_coverage['common_step_sensitivity'] = sensitivity_report
    sensitivity_verdicts = ablation_component_verdicts(sensitivity_primary, sensitivity_coverage)
    assert not sensitivity_verdicts['all_components_supported']
    assert sensitivity_verdicts['statuses']['delayed_merge_A'] == 'inconclusive'
    assert any(('共同步数' in blocker for blocker in sensitivity_verdicts['blockers']))
    global_seed_rows = compare_runs(factorial_paths)
    global_seed_coverage = ablation_coverage(global_seed_rows, ['mode', 'balance_coef'], ['A'], allow_partial=True)

def _selfcheck_body_61261_0083():
    global _incomplete_cell, incomplete_delayed, incomplete_factorial
    assert global_seed_coverage['global_seed_alignment']['applicable']
    assert global_seed_coverage['global_seed_alignment']['all_expected_present']
    json.dumps(_json_safe(factorial), ensure_ascii=False)
    def _incomplete_cell(mode_name: str, balance: float, samples: List[Tuple[object, float]]) -> Dict[str, object]:
        values = [value for _, value in samples]
        return {'label': f'incomplete_{mode_name}_{balance}', 'keys': {'mode': mode_name, 'balance_coef': balance}, 'n_runs': len(samples), 'samples': {'val_loss': [{'seed': seed, 'run': f'incomplete_{mode_name}_{balance}', 'value': value} for seed, value in samples]}, 'val_loss_mean': sum(values) / len(values)}
    incomplete_factorial = factorial_effects([_incomplete_cell('local_base', 0.0, [(1, 3.0), (2, 3.02), (3, 2.98), (None, 3.0)]), _incomplete_cell('A', 0.0, [(1, 2.9), (2, 2.92), (3, 2.88)]), _incomplete_cell('local_base', 0.01, [(1, 2.95), (2, 2.97), (3, 2.93)]), _incomplete_cell('A', 0.01, [(1, 2.85), (2, 2.87), (3, 2.83)])], ['mode', 'balance_coef'], 'val_loss')
    incomplete_delayed = [effect for effect in incomplete_factorial['effects'] if effect['effect_name'] == 'delayed_merge_main'][0]
    assert incomplete_delayed['n_pairs'] == 3
    assert not incomplete_delayed['complete_seed_records']
    assert any(('未记录随机种子' in message for message in incomplete_factorial['advisories']))
    assert '未记录种子的运行' in _format_factorial_effect(incomplete_delayed)

def _selfcheck_body_61347_0084():
    global by_effect, interaction_effects, interaction_paths
    interaction_paths = [_write_analysis_run('ix_local_off_s7', 'local_base', 7, 0.0003, 100, 0.0, 3.0), _write_analysis_run('ix_B_off_s7', 'B', 7, 0.0003, 100, 0.0, 2.9), _write_analysis_run('ix_local_on_s7', 'local_base', 7, 0.0003, 100, 0.01, 2.9), _write_analysis_run('ix_B_on_s7', 'B', 7, 0.0003, 100, 0.01, 2.6), _write_analysis_run('ix_local_off_s8', 'local_base', 8, 0.0003, 100, 0.0, 3.01), _write_analysis_run('ix_B_off_s8', 'B', 8, 0.0003, 100, 0.0, 2.91), _write_analysis_run('ix_local_on_s8', 'local_base', 8, 0.0003, 100, 0.01, 2.91), _write_analysis_run('ix_B_on_s8', 'B', 8, 0.0003, 100, 0.01, 2.61), _write_analysis_run('ix_local_off_s9', 'local_base', 9, 0.0003, 100, 0.0, 2.99), _write_analysis_run('ix_B_off_s9', 'B', 9, 0.0003, 100, 0.0, 2.89), _write_analysis_run('ix_local_on_s9', 'local_base', 9, 0.0003, 100, 0.01, 2.89), _write_analysis_run('ix_B_on_s9', 'B', 9, 0.0003, 100, 0.01, 2.59)]
    interaction_effects = factorial_effects(aggregate_runs(compare_runs(interaction_paths), ['mode', 'balance_coef']), ['mode', 'balance_coef'], 'val_loss', equivalence_margin=1.0)
    by_effect = {effect['effect_name']: effect for effect in interaction_effects['effects']}
    assert abs(by_effect['interaction']['mean_effect'] + 0.2) < 1e-09, by_effect['interaction']

def _selfcheck_body_61473_0085():
    global coefficient, mode_name, multi_balance_effects, multi_balance_paths, parity_coverage, parity_entries, parity_paths, seed
    assert by_effect['interaction']['claim'] == 'interaction_detected'
    assert by_effect['interaction']['tost_p_value'] is not None
    assert by_effect['interaction']['tost_p_value_holm'] is not None
    assert by_effect['interaction']['equivalent']
    assert abs(by_effect['delayed_merge_at_balance_off']['mean_effect'] + 0.1) < 1e-09
    assert by_effect['delayed_merge_at_balance_off']['claim'] == 'not_practically_significant'
    assert abs(by_effect['delayed_merge_at_balance_on']['mean_effect'] + 0.3) < 1e-09
    assert abs(by_effect['balance_loss_at_local_base']['mean_effect'] + 0.1) < 1e-09
    assert abs(by_effect['balance_loss_at_delayed_merge']['mean_effect'] + 0.3) < 1e-09
    multi_balance_paths = []
    for seed in (1, 2, 3):
        for mode_name in ('local_base', 'A'):
            for coefficient in (0.01, 0.02):
                multi_balance_paths.append(_write_analysis_run(f'multi_{mode_name}_{coefficient}_s{seed}', mode_name, seed, 0.0003, 100, coefficient, 3.0 - 0.1 * coefficient))
    multi_balance_effects = factorial_effects(aggregate_runs(compare_runs(multi_balance_paths), ['mode', 'balance_coef']), ['mode', 'balance_coef'], 'val_loss')
    assert not multi_balance_effects['designs']
    assert not multi_balance_effects['effects']
    assert multi_balance_effects['confounded_designs']
    assert any(('多个正 balance_coef' in message for message in multi_balance_effects['advisories']))
    parity_paths = factorial_paths + [_write_analysis_run('fact_B_off_s1_cost', 'B', 1, 0.0003, 100, 0.0, 2.8, flops=3000)] + [_write_analysis_run(f'fact_local_off_cost_s{seed}', 'local_base', seed, 0.0003, 100, 0.0, 3.0, flops=1000) for seed in (1, 2, 3)]
    parity_coverage = ablation_coverage(compare_runs(parity_paths), factors)
    parity_entries = [entry for entry in parity_coverage['compute_parity'] if entry['delayed_mode'] == 'B']
    assert parity_entries, parity_coverage['compute_parity']
    assert not parity_entries[0]['matched']
    assert abs(parity_entries[0]['ratio'] - 3.0) < 1e-09
    assert any(('计算量' in message for message in parity_coverage['advisories']))

def _selfcheck_body_61590_0086():
    global alignment_entries, capacity_coverage, capacity_entries, capacity_factorial, capacity_paths, confound_factorial, misaligned_coverage, misaligned_paths
    capacity_paths = [_write_analysis_run('fact_B_on_capacity', 'B', 1, 0.0003, 100, 0.01, 2.8, flops=1000, n_params=3000), _write_analysis_run('fact_local_on_capacity', 'local_base', 1, 0.0003, 100, 0.01, 3.0, flops=1000, n_params=1000)]
    capacity_coverage = ablation_coverage(compare_runs(capacity_paths), factors)
    capacity_entries = [entry for entry in capacity_coverage['capacity_parity'] if entry['delayed_mode'] == 'B']
    assert capacity_entries, capacity_coverage
    assert not capacity_entries[0]['matched']
    assert abs(capacity_entries[0]['ratio'] - 3.0) < 1e-09
    assert any(('n_params' in message for message in capacity_coverage['advisories']))
    capacity_factorial = {'effects': [{'delayed_mode': 'B', 'claim': 'supports_component'}, {'delayed_mode': 'A', 'claim': 'supports_component'}], 'advisories': []}
    annotate_factorial_confounds(capacity_factorial, {'ok': True}, capacity_coverage)
    assert capacity_factorial['effects'][0]['claim'] == 'capacity_confounded'
    assert capacity_factorial['effects'][0]['capacity_confounded']
    assert capacity_factorial['effects'][1]['claim'] == 'supports_component'
    misaligned_paths = factorial_paths + [_write_analysis_run(f'fact_A_on_extra_s{seed}', 'A', seed, 0.0003, 100, 0.01, 2.85) for seed in (7, 8)]
    misaligned_coverage = ablation_coverage(compare_runs(misaligned_paths), factors)
    alignment_entries = [entry for entry in misaligned_coverage['seed_alignment'] if entry['delayed_mode'] == 'A']
    assert alignment_entries
    assert alignment_entries[0]['n_common_seeds'] == 3
    assert not alignment_entries[0]['aligned']
    assert any(('种子集合不一致' in message for message in misaligned_coverage['advisories']))
    confound_factorial = {'effects': [{'delayed_mode': 'A', 'claim': 'supports_component'}, {'delayed_mode': 'B', 'claim': 'interaction_detected'}, {'delayed_mode': 'A', 'claim': 'not_significant'}], 'advisories': []}
    annotate_factorial_confounds(confound_factorial, {'ok': False}, {'structural_confounds': ['k1 不一致'], 'compute_parity': [{'delayed_mode': 'B', 'matched': False}]})

def _selfcheck_body_61723_0087():
    global balance_a_path, balance_confounded_effect, balance_confounded_factorial, balance_new_base_path, balance_old_base_path, balance_smelt_path, compensated_balance, factorial_balance, new_balance_report, old_balance_report, wide_balance_a_path
    assert confound_factorial['effects'][0]['claim'] == 'confounded'
    balance_a_path = _write_analysis_run('balance_scale_a', 'A', 1, 0.001, 10, 0.01, val_loss=2.0, k1=2, k2=2)
    balance_old_base_path = _write_analysis_run('balance_scale_old_base', 'local_base', 1, 0.001, 10, 0.01, val_loss=2.1, k1=2, k2=2)
    old_balance_report = balance_strength_report(compare_runs([balance_a_path, balance_old_base_path]))
    assert old_balance_report['confounded']
    assert abs(float(old_balance_report['strength_ratio']) - 1.5) < 1e-09
    assert old_balance_report['advisories']
    balance_confounded_effect = {'claim': 'supports_component'}
    balance_confounded_factorial = {'effects': [balance_confounded_effect]}
    annotate_factorial_confounds(balance_confounded_factorial, {'ok': True}, {'balance_confounds': old_balance_report['advisories']})
    assert balance_confounded_effect['claim'] == 'balance_confounded'
    assert _component_evidence_summary([balance_confounded_effect])['status'] == 'blocked'
    balance_new_base_path = _write_analysis_run('balance_scale_new_base', 'local_base', 1, 0.001, 10, 0.01, val_loss=2.1, k1=2, k2=2, protocol_overrides={'balance_width_normalization': _BALANCE_WIDTH_NORMALIZATION})
    new_balance_report = balance_strength_report(compare_runs([balance_a_path, balance_new_base_path]))
    assert not new_balance_report['confounded']
    assert not new_balance_report['width_varied']
    balance_smelt_path = _write_analysis_run('balance_scale_smelt', 'smelt', 1, 0.001, 10, 0.01, val_loss=2.2, protocol_overrides={'top_k': 4})
    factorial_balance = factorial_balance_strength(compare_runs([balance_a_path, balance_new_base_path, balance_smelt_path]))
    assert not factorial_balance['confounded']
    assert not factorial_balance['width_varied']
    wide_balance_a_path = _write_analysis_run('balance_scale_wide_a', 'A', 1, 0.001, 10, 0.01 * 2.0 / 3.0, val_loss=2.0, k1=2, k2=4)
    compensated_balance = balance_strength_report(compare_runs([balance_a_path, wide_balance_a_path]))
    assert compensated_balance['width_varied']
    assert not compensated_balance['confounded']
    assert not compensated_balance['strength_varied']
    assert confound_factorial['effects'][1]['claim'] == 'compute_confounded'

def _selfcheck_body_61857_0088():
    global clean_factorial, randomness_factorial, wide_a_path, wide_base_path, wide_coverage, width_factorial
    assert confound_factorial['effects'][2]['claim'] == 'not_significant'
    assert len(confound_factorial['advisories']) == 3
    wide_a_path = _write_analysis_run('wide_a', 'A', 1, 0.001, 10, 0.01, val_loss=2.0, k1=3, k2=2, flops=1200)
    wide_base_path = _write_analysis_run('wide_base', 'local_base', 1, 0.001, 10, 0.0, val_loss=2.1, k1=3, k2=2, flops=1200)
    wide_coverage = ablation_coverage(compare_runs([wide_a_path, wide_base_path]), {'mode', 'balance_coef'}, ['A'])
    assert any(('第二轮专家宽度' in message for message in wide_coverage['advisories']))
    assert wide_coverage['width_confounds'], wide_coverage
    assert wide_coverage['width_confounds'][0]['delayed_mode'] == 'A'
    width_factorial = {'effects': [{'delayed_mode': 'A', 'claim': 'supports_component'}], 'advisories': []}
    annotate_factorial_confounds(width_factorial, {'ok': True}, wide_coverage)
    assert width_factorial['effects'][0]['claim'] == 'compute_confounded'
    clean_factorial = {'effects': [{'delayed_mode': 'A', 'claim': 'supports_component'}], 'advisories': []}
    annotate_factorial_confounds(clean_factorial, {'ok': True}, {'structural_confounds': [], 'compute_parity': []})
    assert clean_factorial['effects'][0]['claim'] == 'supports_component'
    randomness_factorial = {'effects': [{'delayed_mode': 'A', 'claim': 'supports_component'}, {'delayed_mode': 'B', 'claim': 'supports_component'}], 'advisories': []}
    annotate_factorial_confounds(randomness_factorial, {'ok': True}, {'structural_confounds': [], 'compute_parity': [], 'randomness_confounds': [{'type': 'training_batch', 'seed': 7, 'modes': ['A']}]})
    assert randomness_factorial['effects'][0]['claim'] == 'randomness_confounded'
    assert randomness_factorial['effects'][1]['claim'] == 'supports_component'
    assert any(('同种子' in message for message in randomness_factorial['advisories']))

def _selfcheck_body_61988_0089():
    global good_reference, irrelevant, irrelevant_coverage, unrelated, unrelated_coverage, unverified_factorial
    unverified_factorial = {'effects': [{'delayed_mode': 'A', 'claim': 'supports_component'}, {'delayed_mode': 'A', 'claim': 'not_significant'}], 'advisories': []}
    annotate_factorial_confounds(unverified_factorial, {'ok': True, 'unchecked_keys': ['device_type', 'warmup_steps']}, {'structural_confounds': [], 'compute_parity': []})
    assert unverified_factorial['effects'][0]['claim'] == 'unverified_controls'
    assert unverified_factorial['effects'][0]['uncontrolled_keys'] == ['device_type', 'warmup_steps']
    assert unverified_factorial['effects'][1]['claim'] == 'not_significant'
    assert any(('都没有记录' in message for message in unverified_factorial['advisories']))
    unrelated = [_write_analysis_run('smelt_s1', 'smelt', 1, 0.0003, 100, 0.01, 3.0), _write_analysis_run('B_s1', 'B', 1, 0.0003, 100, 0.01, 2.9)]
    unrelated_coverage = ablation_coverage(compare_runs(unrelated), factors)
    assert any(('缺少立即合并对照模式 local_base' in message for message in unrelated_coverage['missing_cells'])), unrelated_coverage['missing_cells']
    assert any(('B/b=off' in message for message in unrelated_coverage['missing_cells'])), unrelated_coverage['missing_cells']
    assert any(('计算量差异' in message for message in unrelated_coverage['advisories']))
    irrelevant = [_write_analysis_run('smelt_only_s1', 'smelt', 1, 0.0003, 100, 0.01, 3.0), _write_analysis_run('dense_fnn_only_s1', 'dense_fnn', 1, 0.0003, 100, 0.01, 3.1)]
    irrelevant_coverage = ablation_coverage(compare_runs(irrelevant), factors)
    assert not irrelevant_coverage['missing_cells']
    assert not irrelevant_coverage['partial']
    assert any(('消融覆盖不适用' in message for message in irrelevant_coverage['advisories']))
    assert not any(('未覆盖项目声明的延迟合并方案' in message for message in irrelevant_coverage['advisories']))
    good_reference = {'unigram_loss': 5.0, 'uniform_loss': 5.545, 'unigram_token_accuracy': 0.2, 'unigram_macro_f1': 0.01, 'unigram_weighted_f1': 0.15}

def _selfcheck_body_62118_0090():
    global a_hyp_path, adequacy_ok, disjoint_report, good_tuning, hyp_rows, legacy_path, missing_eval_report, missing_seed_report, mixed_seed_report, no_tuning_report, overlap_report, smelt_path
    good_tuning = {'comparison_family': 'baseline_tuning', 'source': 'baseline_tuning.json', 'mode': 'smelt', 'metric': 'val_loss', 'winner_lr': 0.0003, 'winner_mean': 4.5, 'n_candidates': 9, 'seed_plan': {'seeds': [1, 2, 3], 'n_seeds': 3, 'min_seeds_for_significance': 3, 'underpowered_override': False}}
    smelt_path = _write_analysis_run('smelt_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.5, hypothesis='SMELT 基线：判定延迟合并是否改善验证损失', rq='RQ1', reference=good_reference, baseline_tuning=good_tuning)
    a_hyp_path = _write_analysis_run('A_seed1', 'A', 1, 0.0003, 100, 0.01, 4.4, hypothesis='延迟合并使第二轮组合独立，验证损失至少改善 2%', rq='RQ1', reference=good_reference)
    legacy_path = _write_analysis_run('legacy_seed1', 'smelt', 9, 0.0003, 100, 0.01, 4.6, reference=good_reference)
    hyp_rows = compare_runs([smelt_path, a_hyp_path, legacy_path])
    adequacy_ok = baseline_adequacy(hyp_rows, 'smelt', 'val_loss')
    assert adequacy_ok['ok'] is True, adequacy_ok
    assert not adequacy_ok['straw_man']
    overlap_report = tuning_seed_overlap_report(hyp_rows)
    assert overlap_report['ok'] is False
    assert overlap_report['n_overlapping_runs'] == 1
    assert overlap_report['entries'][0]['overlap'] == [1]
    assert any(("winner's curse" in message for message in overlap_report['advisories']))
    disjoint_report = tuning_seed_overlap_report([{'run': 'smelt_seed42', 'mode': 'smelt', 'seed': 42, 'baseline_tuning': good_tuning}])
    assert disjoint_report['ok'] is True
    assert disjoint_report['n_overlapping_runs'] == 0
    no_tuning_report = tuning_seed_overlap_report([{'run': 'smelt_seed42', 'mode': 'smelt', 'seed': 42}])
    assert no_tuning_report['ok'] is None
    missing_seed_report = tuning_seed_overlap_report([{'run': 'legacy_seed1', 'mode': 'smelt', 'seed': 1, 'baseline_tuning': {'source': 'baseline_tuning.json', 'mode': 'smelt'}}])
    assert missing_seed_report['ok'] is None
    assert missing_seed_report['n_runs_with_tuning'] == 1
    assert missing_seed_report['n_unverifiable_runs'] == 1
    missing_eval_report = tuning_seed_overlap_report([{'run': 'smelt_seed1', 'mode': 'smelt', 'baseline_tuning': good_tuning}])
    assert missing_eval_report['ok'] is None
    assert missing_eval_report['n_verifiable_runs'] == 0
    mixed_seed_report = tuning_seed_overlap_report([{'run': 'smelt_seed42', 'mode': 'smelt', 'seed': 42, 'baseline_tuning': good_tuning}, {'run': 'legacy_seed1', 'mode': 'smelt', 'seed': 1, 'baseline_tuning': {'source': 'baseline_tuning.json'}}])
    assert mixed_seed_report['ok'] is None
    assert mixed_seed_report['n_verifiable_runs'] == 1
    assert mixed_seed_report['n_unverifiable_runs'] == 1

def _selfcheck_body_62249_0091():
    global all_failed_adequacy, failed_ngram_adequacy, failed_ngram_path, failed_smelt_path, mixed_failed_adequacy, ngram_adequacy, ngram_path, no_ref_adequacy, no_ref_path, weak_adequacy, weak_path
    weak_path = _write_analysis_run('weak_seed1', 'smelt', 1, 0.0003, 100, 0.01, 5.2, reference=good_reference)
    weak_adequacy = baseline_adequacy(compare_runs([weak_path]), 'smelt', 'val_loss')
    assert weak_adequacy['ok'] is False
    assert any(('unigram' in message for message in weak_adequacy['straw_man']))
    no_ref_path = _write_analysis_run('noref_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.0, baseline_tuning=good_tuning)
    no_ref_adequacy = baseline_adequacy(compare_runs([no_ref_path]), 'smelt', 'val_loss')
    assert no_ref_adequacy['ok'] is None
    assert any(('缺少参考基线' in message for message in no_ref_adequacy['advisories']))
    ngram_path = _write_analysis_run('ngram_val', 'ngram', 0, 0.0003, 100, 0.01, 4.3)
    ngram_adequacy = baseline_adequacy(compare_runs([smelt_path, ngram_path]), 'smelt', 'val_loss')
    assert ngram_adequacy['ok'] is False
    assert any(('n-gram' in message for message in ngram_adequacy['straw_man']))
    failed_smelt_path = _write_analysis_run('failed_smelt_seed2', 'smelt', 2, 0.0003, 100, 0.01, 1.0, reference=good_reference, baseline_tuning=good_tuning, curve_status='diverged')
    mixed_failed_adequacy = baseline_adequacy(compare_runs([smelt_path, failed_smelt_path]), 'smelt', 'val_loss')
    assert mixed_failed_adequacy['n_baseline_runs'] == 1
    assert mixed_failed_adequacy['n_baseline_runs_failed'] == 1
    assert mixed_failed_adequacy['baseline_mean'] == 4.5
    assert any(('数值失败' in message for message in mixed_failed_adequacy['advisories']))
    all_failed_adequacy = baseline_adequacy(compare_runs([failed_smelt_path]), 'smelt', 'val_loss')
    assert all_failed_adequacy['ok'] is False
    assert all_failed_adequacy['n_baseline_runs'] == 0
    assert all_failed_adequacy['baseline_mean'] is None
    assert any(('全部数值失败' in message for message in all_failed_adequacy['straw_man']))
    failed_ngram_path = _write_analysis_run('failed_ngram', 'ngram', 0, 0.0003, 100, 0.01, 3.0, curve_status='diverged', ngram_tuning_metric='val_loss')
    failed_ngram_adequacy = baseline_adequacy(compare_runs([smelt_path, failed_ngram_path]), 'smelt', 'val_loss')
    assert failed_ngram_adequacy['ngram_missing'] is True
    assert any(('数值失败' in message for message in failed_ngram_adequacy['advisories']))

def _selfcheck_body_62385_0092():
    global legacy_tuning, legacy_tuning_adequacy, legacy_tuning_path, mismatch_adequacy, mismatch_path, mismatch_tuning, missing_count_adequacy, missing_count_path, missing_count_tuning, underpowered_adequacy, underpowered_path, underpowered_tuning, untuned_adequacy, untuned_path
    untuned_path = _write_analysis_run('untuned_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.2, reference=good_reference)
    untuned_adequacy = baseline_adequacy(compare_runs([untuned_path]), 'smelt', 'val_loss')
    assert untuned_adequacy['tuned'] is False
    assert untuned_adequacy['ok'] is False
    assert untuned_adequacy['untuned']
    assert any(('调参' in message for message in untuned_adequacy['straw_man']))
    mismatch_tuning = dict(good_tuning)
    mismatch_tuning['metric'] = 'val_macro_f1'
    mismatch_path = _write_analysis_run('mismatch_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.2, reference=good_reference, baseline_tuning=mismatch_tuning)
    mismatch_adequacy = baseline_adequacy(compare_runs([mismatch_path]), 'smelt', 'val_loss')
    assert mismatch_adequacy['tuned'] is False
    assert mismatch_adequacy['tuning_metric_mismatch']
    assert any(('未对齐' in message for message in mismatch_adequacy['advisories']))
    underpowered_tuning = dict(good_tuning)
    underpowered_tuning['seed_plan'] = {'seeds': [1, 2], 'n_seeds': 2, 'min_seeds_for_significance': 3, 'underpowered_override': True}
    underpowered_path = _write_analysis_run('underpowered_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.2, reference=good_reference, baseline_tuning=underpowered_tuning)
    underpowered_adequacy = baseline_adequacy(compare_runs([underpowered_path]), 'smelt', 'val_loss')
    assert underpowered_adequacy['tuned'] is False
    assert underpowered_adequacy['ok'] is False
    assert any(('underpowered' in message for message in underpowered_adequacy['advisories']))
    legacy_tuning = dict(good_tuning)
    legacy_tuning.pop('seed_plan')
    legacy_tuning_path = _write_analysis_run('legacy_tuning_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.2, reference=good_reference, baseline_tuning=legacy_tuning)
    legacy_tuning_adequacy = baseline_adequacy(compare_runs([legacy_tuning_path]), 'smelt', 'val_loss')
    assert legacy_tuning_adequacy['tuned'] is False
    assert any(('seed_plan' in message for message in legacy_tuning_adequacy['advisories']))
    missing_count_tuning = dict(good_tuning)
    missing_count_tuning['seed_plan'] = {'seeds': [1, 2, 3], 'min_seeds_for_significance': 3, 'underpowered_override': False}
    missing_count_path = _write_analysis_run('missing_count_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.2, reference=good_reference, baseline_tuning=missing_count_tuning)
    missing_count_adequacy = baseline_adequacy(compare_runs([missing_count_path]), 'smelt', 'val_loss')
    assert missing_count_adequacy['tuned'] is False
    assert any(('n_seeds' in message for message in missing_count_adequacy['advisories']))

def _selfcheck_body_62528_0093():
    global challenger_tuning, legacy_challenger_path, legacy_challenger_tuning, legacy_parity, missing_count_challenger_path, missing_count_challenger_tuning, missing_count_parity, parity_baseline_path, tuned_challenger_path, tuned_parity, untuned_strong_path
    challenger_tuning = {'comparison_family': 'baseline_tuning', 'source': 'baseline_tuning.json', 'mode': 'B', 'metric': 'val_loss', 'winner_lr': 0.001, 'winner_mean': 4.2, 'n_candidates': 9, 'seed_plan': {'seeds': [1, 2, 3], 'n_seeds': 3, 'min_seeds_for_significance': 3, 'underpowered_override': False}}
    parity_baseline_path = _write_analysis_run('parity_baseline_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.3, reference=good_reference, baseline_tuning=good_tuning)
    tuned_challenger_path = _write_analysis_run('tuned_challenger_seed1', 'B', 1, 0.001, 100, 0.01, 4.1, baseline_tuning=challenger_tuning)
    tuned_parity = baseline_adequacy(compare_runs([parity_baseline_path, tuned_challenger_path]), 'smelt', 'val_loss')['tuning_parity']
    assert tuned_parity['challengers_with_own_tuning'] == ['tuned_challenger_seed1']
    assert not tuned_parity['challengers_without_tuning_parity']
    legacy_challenger_tuning = dict(challenger_tuning)
    legacy_challenger_tuning.pop('seed_plan')
    legacy_challenger_path = _write_analysis_run('legacy_challenger_seed1', 'B', 1, 0.001, 100, 0.01, 4.1, baseline_tuning=legacy_challenger_tuning)
    legacy_parity = baseline_adequacy(compare_runs([parity_baseline_path, legacy_challenger_path]), 'smelt', 'val_loss')['tuning_parity']
    assert not legacy_parity['challengers_with_own_tuning']
    assert [item['run'] for item in legacy_parity['challengers_without_tuning_parity']] == ['legacy_challenger_seed1']
    missing_count_challenger_tuning = dict(challenger_tuning)
    missing_count_challenger_tuning['seed_plan'] = {'seeds': [1, 2, 3], 'min_seeds_for_significance': 3, 'underpowered_override': False}
    missing_count_challenger_path = _write_analysis_run('missing_count_challenger_seed1', 'B', 1, 0.001, 100, 0.01, 4.1, baseline_tuning=missing_count_challenger_tuning)
    missing_count_parity = baseline_adequacy(compare_runs([parity_baseline_path, missing_count_challenger_path]), 'smelt', 'val_loss')['tuning_parity']
    assert not missing_count_parity['challengers_with_own_tuning']
    assert [item['run'] for item in missing_count_parity['challengers_without_tuning_parity']] == ['missing_count_challenger_seed1']
    untuned_strong_path = _write_analysis_run('untuned_strong_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.2, reference=good_reference)

def _selfcheck_body_62655_0094():
    global big_challenger_path, capacity_adequacy, missing_ngram_adequacy, no_baseline, same_capacity_adequacy, small_baseline_path, unconverged_adequacy, unconverged_path, untuned_strong_adequacy, weak_ngram_path
    weak_ngram_path = _write_analysis_run('ngram_weak_val', 'ngram', 0, 0.0003, 100, 0.01, 4.8)
    untuned_strong_adequacy = baseline_adequacy(compare_runs([untuned_strong_path, weak_ngram_path]), 'smelt', 'val_loss')
    assert untuned_strong_adequacy['tuned'] is False
    assert untuned_strong_adequacy['ok'] is True
    assert not untuned_strong_adequacy['straw_man']
    assert any(('调参' in message for message in untuned_strong_adequacy['advisories']))
    no_baseline = baseline_adequacy(hyp_rows, None, 'val_loss')
    assert no_baseline['ok'] is None
    assert any(('--compare-baseline' in message for message in no_baseline['advisories']))
    missing_ngram_adequacy = baseline_adequacy(compare_runs([mismatch_path]), 'smelt', 'val_loss')
    assert missing_ngram_adequacy['ngram_missing'] is True
    assert any(('未包含 n-gram' in message for message in missing_ngram_adequacy['advisories']))
    unconverged_path = _write_analysis_run('unconverged_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.3, reference=good_reference, baseline_tuning=good_tuning, curve_status='not_converged')
    unconverged_adequacy = baseline_adequacy(compare_runs([unconverged_path]), 'smelt', 'val_loss')
    assert unconverged_adequacy['unconverged_baselines']
    assert any(('not_converged' in message for message in unconverged_adequacy['advisories']))
    small_baseline_path = _write_analysis_run('small_base_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.3, reference=good_reference, baseline_tuning=good_tuning, n_params=1000, flops=10000, curve_status='healthy')
    big_challenger_path = _write_analysis_run('big_challenger_seed1', 'B', 1, 0.0003, 100, 0.01, 4.1, n_params=5000, flops=50000, curve_status='healthy')
    capacity_adequacy = baseline_adequacy(compare_runs([small_baseline_path, big_challenger_path]), 'smelt', 'val_loss')
    assert capacity_adequacy['capacity_confounds']
    assert any(('容量' in message or '等参数' in message for message in capacity_adequacy['advisories']))
    same_capacity_adequacy = baseline_adequacy(compare_runs([small_baseline_path, _write_analysis_run('same_size_seed1', 'B', 1, 0.0003, 100, 0.01, 4.1, n_params=1100, flops=11000, curve_status='healthy')]), 'smelt', 'val_loss')
    assert not same_capacity_adequacy['capacity_confounds']

def _selfcheck_body_62789_0095():
    global clear_adequacy, clear_comparison, clear_seed2, clear_seed3, dense_base_path, dense_capacity_adequacy, dense_config, f, noisy_seed2, single_comparison, single_ngram_adequacy
    dense_base_path = _write_analysis_run('dense_base_seed1', 'dense', 1, 0.0003, 100, 0.01, 4.3, reference=good_reference, baseline_tuning=good_tuning, n_params=1000, flops=10000, curve_status='healthy')
    with open(Path(dense_base_path) / 'config.json', encoding='utf-8') as f:
        dense_config = json.load(f)
    dense_config['config'].update({'top_k': 2, 'n_experts': 4, 'expert_hidden': 8, 'dense_fnn_hidden': None})
    with open(Path(dense_base_path) / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(dense_config, f)
    dense_capacity_adequacy = baseline_adequacy(compare_runs([dense_base_path, big_challenger_path]), 'dense', 'val_loss')
    assert dense_capacity_adequacy['capacity_confounds']
    assert dense_capacity_adequacy['baseline_capacity']['parameter_matched'] is False
    assert dense_capacity_adequacy['baseline_capacity']['parameter_matched_dense_fnn_hidden'] == 32
    assert any(('--dense-fnn-hidden 32' in message for message in dense_capacity_adequacy['advisories']))
    single_ngram_adequacy = baseline_adequacy(compare_runs([small_baseline_path, ngram_path]), 'smelt', 'val_loss')
    assert single_ngram_adequacy['ngram_deterministic'] is True
    assert any(('确定性基线' in message and 'bootstrap' in message for message in single_ngram_adequacy['advisories']))
    assert not any(('多种子重跑 n-gram' in message for message in single_ngram_adequacy['advisories']))
    single_comparison = single_ngram_adequacy['ngram_comparison']
    assert single_comparison is not None
    assert single_comparison['n_baseline_values'] == 1
    assert any(('种子噪声' in message for message in single_comparison['advisories']))
    clear_seed2 = _write_analysis_run('clear_seed2', 'smelt', 2, 0.0003, 100, 0.01, 4.52, reference=good_reference, baseline_tuning=good_tuning)
    clear_seed3 = _write_analysis_run('clear_seed3', 'smelt', 3, 0.0003, 100, 0.01, 4.51, reference=good_reference, baseline_tuning=good_tuning)
    clear_adequacy = baseline_adequacy(compare_runs([smelt_path, clear_seed2, clear_seed3, ngram_path]), 'smelt', 'val_loss')
    clear_comparison = clear_adequacy['ngram_comparison']
    assert clear_comparison['n_baseline_values'] == 3
    assert clear_comparison['significant'] is True
    assert clear_comparison['seeds_to_detect'] is not None
    assert any(('统计上可分辨' in message for message in clear_comparison['advisories']))
    noisy_seed2 = _write_analysis_run('noisy_seed2', 'smelt', 2, 0.0003, 100, 0.01, 4.8, reference=good_reference, baseline_tuning=good_tuning)

def _selfcheck_body_62922_0096():
    global ngram_registry, noisy_adequacy, noisy_comparison, noisy_seed3, registry, registry_by_mode, smelt_mentions, vague_path, vague_registry
    noisy_seed3 = _write_analysis_run('noisy_seed3', 'smelt', 3, 0.0003, 100, 0.01, 4.65, reference=good_reference, baseline_tuning=good_tuning)
    noisy_adequacy = baseline_adequacy(compare_runs([smelt_path, noisy_seed2, noisy_seed3, ngram_path]), 'smelt', 'val_loss')
    noisy_comparison = noisy_adequacy['ngram_comparison']
    assert noisy_comparison['significant'] is False
    assert noisy_comparison['mde_80'] is not None
    assert noisy_comparison['seeds_to_detect'] is not None
    assert any(('噪声内' in message for message in noisy_comparison['advisories']))
    registry = hypothesis_registry(hyp_rows)
    registry_by_mode = {cell['mode']: cell for cell in registry['cells']}
    assert registry_by_mode['smelt']['n_runs'] == 2
    assert len(registry_by_mode['smelt']['hypotheses']) == 1
    assert registry_by_mode['smelt']['research_questions'] == ['RQ1']
    assert 'legacy_seed1' in registry['missing_hypothesis_runs']
    ngram_registry = hypothesis_registry(compare_runs([ngram_path]))
    assert not ngram_registry['missing_hypothesis_runs']
    smelt_mentions = registry_by_mode['smelt']['hypotheses_metric_mentions']
    assert smelt_mentions[0]['metric_mentions'] == ['loss']
    assert not registry['hypotheses_without_metric_runs']
    vague_path = _write_analysis_run('vague_seed1', 'A', 2, 0.0003, 100, 0.01, 4.3, hypothesis='延迟合并的效果会更好', rq='RQ1')
    vague_registry = hypothesis_registry(compare_runs([vague_path]))
    assert vague_registry['hypotheses_without_metric_runs'] == ['vague_seed1']
    assert any(('判定指标' in message for message in vague_registry['advisories']))
    assert hypothesis_metric_mentions('验证损失至少改善 2%') == ['loss']
    assert 'f1' in hypothesis_metric_mentions('macro F1 提升且召回率不降')
    assert 'pr_auc' in hypothesis_metric_mentions('PR-AUC 上升')
    assert 'pr_auc_lift' in hypothesis_metric_mentions('PR-AUC lift 提升')
    assert 'pr_auc_lift' in hypothesis_metric_mentions('PR-AUC 高出随机水平')
    assert 'pr_auc_lift' not in hypothesis_metric_mentions('PR-AUC 提升')
    assert hypothesis_metric_mentions('效果更好') == []
    assert hypothesis_ambiguous_averaging_metrics('F1 提升') == ['f1']
    assert hypothesis_ambiguous_averaging_metrics('macro F1 提升') == []
    assert hypothesis_ambiguous_averaging_metrics('加权召回率不降') == []
    assert hypothesis_ambiguous_averaging_metrics('precision 与 recall 都提升') == ['precision', 'recall']
    assert hypothesis_metric_averaging_qualifier('val_macro_f1') == 'macro'
    assert hypothesis_metric_averaging_qualifier('val_macro_f1_at_common_step') == 'macro'
    assert hypothesis_metric_averaging_qualifier('val_loss') is None
    assert hypothesis_averaging_qualifier_mismatch('val_macro_f1', 'macro F1 提升') is None
    assert hypothesis_averaging_qualifier_mismatch('val_macro_f1', 'F1 提升') is None
    assert hypothesis_averaging_qualifier_mismatch('val_weighted_f1', 'macro F1 提升') is not None

def _selfcheck_body_63062_0097():
    global _prereg_args, ambiguous_path, ambiguous_registry, mismatched_averaging_path, mismatched_averaging_registry
    assert hypothesis_averaging_qualifier_mismatch('val_micro_f1', 'macro F1 与加权召回率都提升') is not None
    assert hypothesis_averaging_qualifier_mismatch('val_weighted_f1', 'macro F1 与加权召回率都提升') is None
    mismatched_averaging_path = _write_analysis_run('averaging_mismatch_seed1', 'A', 4, 0.0003, 100, 0.01, 4.3, hypothesis='延迟合并使 macro F1 提升', preregistration={'primary_metric': 'val_weighted_f1', 'direction': 'increase', 'min_effect': None})
    mismatched_averaging_registry = hypothesis_registry(compare_runs([mismatched_averaging_path]), 'val_weighted_f1')
    assert mismatched_averaging_registry['hypotheses_with_averaging_mismatch_runs'] == ['averaging_mismatch_seed1']
    assert any(('平均口径' in message for message in mismatched_averaging_registry['advisories']))
    ambiguous_path = _write_analysis_run('ambiguous_seed1', 'A', 4, 0.0003, 100, 0.01, 4.3, hypothesis='延迟合并使 F1 提升')
    ambiguous_registry = hypothesis_registry(compare_runs([ambiguous_path]), 'val_loss')
    assert ambiguous_registry['hypotheses_with_ambiguous_averaging_runs'] == ['ambiguous_seed1']
    assert any(('平均口径' in message for message in ambiguous_registry['advisories']))
    def _prereg_args(**overrides):
        base = {'compare': None, 'statistical_baseline': False, 'sanity_only': False, 'hypothesis': '延迟合并使验证损失至少下降 2%', 'primary_metric': None, 'hypothesis_direction': None, 'hypothesis_min_effect': None, 'hypothesis_baseline': None}
        base.update(overrides)
        return argparse.Namespace(**base)
    assert hypothesis_preregistration_error(_prereg_args()) is None
    assert hypothesis_preregistration_error(_prereg_args(hypothesis_direction='decrease')) is not None
    assert hypothesis_preregistration_error(_prereg_args(primary_metric='val_loss', hypothesis_direction='decrease', hypothesis_min_effect=-0.1)) is not None
    assert hypothesis_preregistration_error(_prereg_args(primary_metric='val_macro_f1', hypothesis_direction='increase')) is not None
    assert hypothesis_preregistration_error(_prereg_args(hypothesis='延迟合并使 macro F1 提升', primary_metric='val_macro_f1', hypothesis_direction='increase')) is None
    assert hypothesis_preregistration_error(_prereg_args(hypothesis='延迟合并使 macro F1 提升', primary_metric='val_macro_f1_at_common_step', hypothesis_direction='increase')) is None
    assert hypothesis_preregistration_error(_prereg_args(hypothesis='延迟合并使 macro F1 提升', primary_metric='val_weighted_f1', hypothesis_direction='increase')) is not None
    assert hypothesis_preregistration_error(_prereg_args(primary_metric='val_loss_at_common_step', hypothesis_direction='decrease', hypothesis_min_effect=0.02)) is None
    assert hypothesis_preregistration_error(_prereg_args(primary_metric='val_loss', hypothesis_direction='decrease', hypothesis_min_effect=0.02)) is None

def _selfcheck_body_63192_0098():
    global common_alignment, common_prereg_run, mismatch_registry, prereg_registry, prereg_rows, prereg_run, uncomputable
    assert hypothesis_preregistration_error(_prereg_args(compare=['some_run'], primary_metric='val_loss')) is not None
    assert hypothesis_preregistration_error(_prereg_args(statistical_baseline=True, primary_metric='val_loss')) is None
    assert hypothesis_preregistration_error(_prereg_args(statistical_baseline=True, primary_metric='val_macro_f1')) is None
    for uncomputable in ('routing_jaccard_branch', 'val_loss_at_matched_flops'):
        assert hypothesis_preregistration_error(_prereg_args(statistical_baseline=True, primary_metric=uncomputable)) is not None, uncomputable
    assert hypothesis_preregistration_error(_prereg_args(statistical_baseline=True, primary_metric='val_loss', hypothesis_direction='decrease')) is not None
    assert hypothesis_preregistration_error(_prereg_args(hypothesis_baseline='smelt')) is None
    assert hypothesis_preregistration_error(_prereg_args(primary_metric='val_loss', hypothesis_direction='decrease', hypothesis_baseline='local_base')) is None
    assert hypothesis_preregistration_error(_prereg_args(compare=['some_run'], hypothesis_baseline='smelt')) is not None
    assert hypothesis_preregistration_error(_prereg_args(statistical_baseline=True, hypothesis_baseline='smelt')) is not None
    prereg_run = _write_analysis_run('prereg_seed1', 'A', 5, 0.0003, 100, 0.01, 4.1, hypothesis='延迟合并使验证损失至少下降 2%', preregistration={'primary_metric': 'val_loss', 'direction': 'decrease', 'min_effect': 0.02})
    prereg_rows = compare_runs([prereg_run])
    prereg_registry = hypothesis_registry(prereg_rows, 'val_loss')
    assert prereg_registry['preregistered_runs'] == ['prereg_seed1']
    assert not prereg_registry['metric_mismatch_runs']
    mismatch_registry = hypothesis_registry(prereg_rows, 'val_macro_f1')
    assert mismatch_registry['metric_mismatch_runs'] == ['prereg_seed1']
    assert any(('预注册' in message for message in mismatch_registry['advisories']))
    common_prereg_run = _write_analysis_run('prereg_common_seed1', 'A', 6, 0.0003, 100, 0.01, 4.1, hypothesis='延迟合并使共同步数验证损失至少下降 2%', preregistration={'primary_metric': 'val_loss_at_common_step', 'direction': 'decrease', 'min_effect': 0.02})
    common_alignment = hypothesis_alignment(compare_runs([common_prereg_run]), [], [], 'val_loss_at_common_step')
    assert any(('频率加权' in message for message in common_alignment['advisories'])), common_alignment

def _selfcheck_body_63319_0099():
    global _synthetic_aggregate, _synthetic_contrast, alignment_aggregates, below_alignment, mismatch_alignment, missing_baseline_alignment, missing_baseline_contrast, refuted_alignment, supported_alignment, supported_contrast, undefined_alignment, undefined_baseline, undefined_check
    def _synthetic_aggregate(label: str, runs: List[str], mean: float) -> Dict[str, object]:
        return {'label': label, 'runs': list(runs), 'keys': {'mode': label}, 'n_runs': len(runs), 'samples': {'val_loss': []}, 'val_loss_mean': mean, 'val_loss_std': 0.01, 'costs': {}}
    def _synthetic_contrast(baseline_label: str, variant_label: str, baseline_mean: float, variant_mean: float, significant: bool) -> Dict[str, object]:
        return {'baseline': baseline_label, 'variant': variant_label, 'factor': 'mode', 'confounded': False, 'delta': variant_mean - baseline_mean, 'mean_baseline': baseline_mean, 'mean_variant': variant_mean, 'significant': significant, 'p_value_holm': 0.01 if significant else 0.5}
    alignment_aggregates = [_synthetic_aggregate('smelt', ['smelt_seed1'], 4.5), _synthetic_aggregate('A', ['prereg_seed1'], 4.4)]
    supported_contrast = _synthetic_contrast('smelt', 'A', 4.5, 4.4, True)
    supported_contrast.update({'ci95_low': -0.105, 'ci95_high': -0.095})
    supported_alignment = hypothesis_alignment(prereg_rows, alignment_aggregates, [supported_contrast], 'val_loss')
    assert supported_alignment['checks'][0]['status'] == 'supported', supported_alignment
    below_alignment = hypothesis_alignment(prereg_rows, alignment_aggregates, [_synthetic_contrast('smelt', 'A', 4.5, 4.48, True)], 'val_loss')
    assert below_alignment['checks'][0]['status'] == 'below_declared_min_effect'
    for undefined_baseline in (0.0, 1e-15):
        undefined_alignment = hypothesis_alignment(prereg_rows, alignment_aggregates, [_synthetic_contrast('smelt', 'A', undefined_baseline, -0.5, True)], 'val_loss')
        undefined_check = undefined_alignment['checks'][0]
        assert undefined_check['status'] == 'min_effect_undefined', undefined_alignment
        assert undefined_check['min_effect_abs'] is None
        assert undefined_alignment['min_effect_undefined']
    missing_baseline_contrast = _synthetic_contrast('smelt', 'A', 4.5, 4.4, True)
    missing_baseline_contrast.pop('mean_baseline')
    missing_baseline_alignment = hypothesis_alignment(prereg_rows, alignment_aggregates, [missing_baseline_contrast], 'val_loss')
    assert missing_baseline_alignment['checks'][0]['status'] == 'min_effect_undefined'
    refuted_alignment = hypothesis_alignment(prereg_rows, alignment_aggregates, [_synthetic_contrast('smelt', 'A', 4.5, 4.6, True)], 'val_loss')
    assert refuted_alignment['checks'][0]['status'] == 'refuted'
    assert refuted_alignment['refuted']
    mismatch_alignment = hypothesis_alignment(prereg_rows, alignment_aggregates, [_synthetic_contrast('smelt', 'A', 4.5, 4.6, False)], 'val_loss')

def _selfcheck_body_63453_0100():
    global imb_report, imb_rows, imbalanced_path, mde_alignment, mde_contrast, mixed_effect_aggregates, mixed_effect_alignment, mixed_effect_rows, other_metric_alignment, powered_alignment, powered_contrast, strict_prereg_run, underpowered_alignment, underpowered_check, underpowered_contrast
    assert mismatch_alignment['checks'][0]['status'] == 'direction_mismatch_not_significant'
    underpowered_contrast = _synthetic_contrast('smelt', 'A', 4.5, 4.4, False)
    underpowered_contrast.update({'se': 0.25, 'df': 3.0})
    underpowered_alignment = hypothesis_alignment(prereg_rows, alignment_aggregates, [underpowered_contrast], 'val_loss')
    underpowered_check = underpowered_alignment['checks'][0]
    assert underpowered_check['status'] == 'inconclusive_underpowered', underpowered_alignment
    assert float(underpowered_check['power_at_min_effect']) < _TARGET_POWER
    assert underpowered_alignment['inconclusive_underpowered']
    assert any(('功效不足' in message for message in underpowered_alignment['advisories']))
    powered_contrast = _synthetic_contrast('smelt', 'A', 4.5, 4.4, False)
    powered_contrast.update({'se': 0.005, 'df': 100.0})
    powered_alignment = hypothesis_alignment(prereg_rows, alignment_aggregates, [powered_contrast], 'val_loss')
    assert powered_alignment['checks'][0]['status'] == 'not_significant', powered_alignment
    mde_contrast = _synthetic_contrast('smelt', 'A', 4.5, 4.4, False)
    mde_contrast['mde_80_bonferroni'] = 0.5
    mde_alignment = hypothesis_alignment(prereg_rows, alignment_aggregates, [mde_contrast], 'val_loss')
    assert mde_alignment['checks'][0]['status'] == 'inconclusive_underpowered', mde_alignment
    assert mde_alignment['checks'][0]['mde_80_vs_min_effect'] == 0.5
    strict_prereg_run = _write_analysis_run('prereg_strict_seed2', 'A', 6, 0.0003, 100, 0.01, 4.4, hypothesis='延迟合并使验证损失至少下降 5%', preregistration={'primary_metric': 'val_loss', 'direction': 'decrease', 'min_effect': 0.05})
    mixed_effect_rows = compare_runs([prereg_run, strict_prereg_run])
    mixed_effect_aggregates = [_synthetic_aggregate('smelt', ['smelt_seed1'], 4.5), _synthetic_aggregate('A', ['prereg_seed1', 'prereg_strict_seed2'], 4.4)]
    mixed_effect_alignment = hypothesis_alignment(mixed_effect_rows, mixed_effect_aggregates, [_synthetic_contrast('smelt', 'A', 4.5, 4.365, True)], 'val_loss')
    assert mixed_effect_alignment['checks'][0]['min_effect'] == 0.05
    assert mixed_effect_alignment['checks'][0]['status'] == 'below_declared_min_effect'
    other_metric_alignment = hypothesis_alignment(prereg_rows, alignment_aggregates, [_synthetic_contrast('smelt', 'A', 4.5, 4.4, True)], 'val_macro_f1')
    assert other_metric_alignment['n_checks'] == 0
    assert other_metric_alignment['advisories']
    imbalanced_path = _write_analysis_run('imb_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.0, majority_accuracy=0.95)
    imb_rows = compare_runs([imbalanced_path])
    imb_report = imbalance_metric_advisory(imb_rows, 'val_token_accuracy')

def _selfcheck_body_63591_0101():
    global balanced_report, cm, common_acc_report, common_imb_report, confusion, extreme_path, extreme_pr, extreme_roc, extreme_rows, roc_report, sampled_auc_positions, single_auc_position
    assert imb_report['max_majority_accuracy'] == 0.95
    assert any(('PR-AUC' in message for message in imb_report['advisories']))
    common_imb_report = imbalance_metric_advisory(imb_rows, 'val_loss_at_common_step')
    assert any(('频率加权' in message for message in common_imb_report['advisories']))
    common_acc_report = imbalance_metric_advisory(imb_rows, 'val_token_accuracy_at_common_step')
    assert any(('多数类主导' in message for message in common_acc_report['advisories']))
    roc_report = imbalance_metric_advisory(imb_rows, 'val_roc_auc_macro')
    assert any(('ROC-AUC' in message for message in roc_report['advisories']))
    extreme_path = _write_analysis_run('extreme_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.0, majority_accuracy=0.9995, pr_auc_chance=0.001, pr_auc=0.02)
    extreme_rows = compare_runs([extreme_path])
    extreme_roc = imbalance_metric_advisory(extreme_rows, 'val_roc_auc_macro')
    assert extreme_roc['max_majority_accuracy'] == 0.9995
    assert extreme_roc['pr_auc_chance_mean'] == 0.001
    assert any(('随机水平 0.5' in message and 'PR-AUC' in message for message in extreme_roc['advisories']))
    extreme_pr = imbalance_metric_advisory(extreme_rows, 'val_pr_auc_macro')
    assert abs(float(extreme_pr['pr_auc_over_chance']) - 0.019) < 1e-12
    assert extreme_pr['pr_auc_worst_run'] == 'extreme_seed1'
    assert any(('随机水平' in message for message in extreme_pr['advisories']))
    balanced_report = imbalance_metric_advisory(compare_runs([no_ref_path]), 'val_loss')
    assert not balanced_report['advisories']
    confusion = torch.tensor([[8.0, 2.0], [3.0, 7.0]])
    cm = confusion_matrix_metrics(confusion)
    assert abs(cm['precision_macro'] - (8 / 11 + 7 / 9) / 2) < 1e-12
    assert abs(cm['recall_macro'] - 0.75) < 1e-12
    assert abs(cm['precision_weighted'] - (8 / 11 + 7 / 9) / 2) < 1e-12
    assert abs(cm['recall_weighted'] - 0.75) < 1e-12
    assert 'val_recall_macro' in COMPARISON_METRIC_KEYS
    assert 'val_precision_weighted' in COMPARISON_METRIC_KEYS
    assert hypothesis_metric_unverifiable(['recall'], None) == []
    assert hypothesis_metric_unverifiable(['precision'], None) == []
    assert hypothesis_metric_unverifiable(['jaccard'], routing_available=False) == ['jaccard']
    assert hypothesis_metric_unverifiable(['jaccard'], routing_available=True) == []
    assert hypothesis_metric_unverifiable(['jaccard'], routing_available=None) == []
    assert hypothesis_metric_unverifiable(['loss'], routing_available=False) == []
    sampled_auc_positions = _auc_uniform_sample_positions(1000, 100)
    assert sampled_auc_positions is not None
    assert sampled_auc_positions.numel() == 100
    assert int(sampled_auc_positions[0]) == 0
    assert int(sampled_auc_positions[-1]) == 999
    assert bool(torch.all(sampled_auc_positions[1:] > sampled_auc_positions[:-1]))
    assert _auc_uniform_sample_positions(1000, 1000) is None
    assert _auc_uniform_sample_positions(1000, 2000) is None
    single_auc_position = _auc_uniform_sample_positions(1000, 1)
    assert single_auc_position is not None
    assert int(single_auc_position[0]) == 500

def _selfcheck_body_63735_0102():
    global aggregates_by_mode, no_routing_hyp_path, no_routing_registry, routing_aggregates, routing_contrasts, routing_paths, routing_payload, routing_rows, seed, tied_auc, tied_probs, tied_scores
    tied_scores = torch.tensor([0.9, 0.8, 0.8])
    tied_probs = torch.stack([1.0 - tied_scores, tied_scores], dim=1)
    tied_auc = macro_auc_metrics(tied_probs, torch.tensor([0, 1, 1]), 2, min_support=1)
    assert abs(tied_auc['pr_auc_macro'] - 0.5) < 1e-12, tied_auc
    routing_payload = {'rq3_expert_overlap': {'summary': {'second_round_jaccard_mean': 0.3, 'second_round_slot_reuse_mean': 0.4, 'token_second_round_jaccard_mean': 0.25, 'branch_self_reuse_share': 0.5}}, 'rq4_load_and_entropy': {'summary': {'round1': {'load_entropy': 0.9}, 'round2': {'load_entropy': 0.85, 'dead_experts': 1, 'router_entropy_mean': 1.2}}}, 'branch_divergence': {'summary': {'branch_cosine_mean': 0.7}}}
    routing_paths = []
    for seed in (42, 43, 44):
        routing_paths.append(_write_analysis_run(f'route_A_seed{seed}', 'A', seed, 0.0003, 100, 0.01, 4.0, hypothesis='延迟合并让第二轮 Jaccard 重合度降低', rq='RQ3', routing=routing_payload))
    for seed in (42, 43, 44):
        routing_paths.append(_write_analysis_run(f'route_B_seed{seed}', 'B', seed, 0.0003, 100, 0.01, 3.9, hypothesis='延迟合并让第二轮 Jaccard 重合度降低', rq='RQ3', routing={**routing_payload, 'rq3_expert_overlap': {'summary': {'second_round_jaccard_mean': 0.15, 'second_round_slot_reuse_mean': 0.25, 'token_second_round_jaccard_mean': 0.1, 'branch_self_reuse_share': 0.3}}}))
    routing_rows = compare_runs(routing_paths)
    assert routing_rows[0]['routing_available'] is True
    assert routing_rows[0]['routing_jaccard_branch'] == 0.3
    assert routing_rows[0]['routing_r2_dead_experts'] == 1
    assert routing_rows[0]['routing_branch_cosine'] == 0.7
    routing_aggregates = aggregate_runs(routing_rows, ['mode'])
    aggregates_by_mode = {aggregate['keys']['mode']: aggregate for aggregate in routing_aggregates}
    assert abs(float(aggregates_by_mode['A']['routing_jaccard_branch_mean']) - 0.3) < 1e-12
    assert aggregates_by_mode['A']['routing_jaccard_branch_n'] == 3
    routing_contrasts = factor_contrasts(routing_aggregates, ['mode'], 'routing_jaccard_branch')
    assert routing_contrasts
    assert abs(float(routing_contrasts[0]['delta']) + 0.15) < 1e-12
    no_routing_hyp_path = _write_analysis_run('no_route_hyp_seed1', 'dense', 1, 0.0003, 100, 0.01, 4.0, hypothesis='第二轮 Jaccard 重合度降低', rq='RQ3')
    no_routing_registry = hypothesis_registry(compare_runs([no_routing_hyp_path]))
    assert no_routing_registry['hypotheses_with_unverifiable_metric_runs'] == ['no_route_hyp_seed1']

def _selfcheck_body_63864_0103():
    global auc_reference, fallback_pr_adequacy, fallback_pr_path, lift_adequacy, lift_baseline_path, lift_tuning, pr_adequacy, pr_baseline_path, pr_tuning, routing_registry, weak_lift_adequacy, weak_pr_adequacy, weak_pr_path
    assert any(('不可判定' in message for message in no_routing_registry['advisories']))
    routing_registry = hypothesis_registry(compare_runs([routing_paths[0]]))
    assert not routing_registry['hypotheses_with_unverifiable_metric_runs']
    auc_reference = dict(good_reference, unigram_pr_auc_macro_chance=0.001, unigram_roc_auc_macro=0.5)
    pr_tuning = dict(good_tuning)
    pr_tuning['metric'] = 'val_pr_auc_macro'
    pr_baseline_path = _write_analysis_run('pr_smelt_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.5, reference=auc_reference, baseline_tuning=pr_tuning, pr_auc=0.02, pr_auc_chance=0.001)
    pr_adequacy = baseline_adequacy(compare_runs([pr_baseline_path]), 'smelt', 'val_pr_auc_macro')
    assert pr_adequacy['ok'] is True, pr_adequacy
    assert pr_adequacy['checks']
    assert pr_adequacy['checks'][0]['trivial_reference'] == 0.001
    assert pr_adequacy['checks'][0]['trivial_reference_source'] == 'unigram_pr_auc_macro_chance'
    weak_pr_path = _write_analysis_run('weak_pr_smelt_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.5, reference=auc_reference, baseline_tuning=good_tuning, pr_auc=0.0005, pr_auc_chance=0.001)
    weak_pr_adequacy = baseline_adequacy(compare_runs([weak_pr_path]), 'smelt', 'val_pr_auc_macro')
    assert weak_pr_adequacy['ok'] is False
    assert any(('unigram' in message for message in weak_pr_adequacy['straw_man']))
    fallback_pr_path = _write_analysis_run('fallback_pr_smelt_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.5, reference=good_reference, baseline_tuning=pr_tuning, pr_auc=0.02, pr_auc_chance=0.001)
    fallback_pr_adequacy = baseline_adequacy(compare_runs([fallback_pr_path]), 'smelt', 'val_pr_auc_macro')
    assert fallback_pr_adequacy['ok'] is True
    assert 'val_pr_auc_macro_chance' in fallback_pr_adequacy['checks'][0]['trivial_reference_source']
    lift_tuning = dict(good_tuning)
    lift_tuning['metric'] = 'val_pr_auc_macro_lift'
    lift_baseline_path = _write_analysis_run('lift_smelt_seed1', 'smelt', 1, 0.0003, 100, 0.01, 4.5, reference=auc_reference, baseline_tuning=lift_tuning, pr_auc=0.02, pr_auc_chance=0.001)
    lift_adequacy = baseline_adequacy(compare_runs([lift_baseline_path]), 'smelt', 'val_pr_auc_macro_lift')
    assert lift_adequacy['ok'] is True, lift_adequacy
    assert lift_adequacy['checks'][0]['trivial_reference'] == 0.0
    weak_lift_adequacy = baseline_adequacy(compare_runs([weak_pr_path]), 'smelt', 'val_pr_auc_macro_lift')
    assert weak_lift_adequacy['ok'] is False
    assert any(('unigram' in message for message in weak_lift_adequacy['straw_man']))

def _selfcheck_body_64010_0104():
    global absent_baseline, absent_report, auto_baseline, conflict_baseline, conflict_report, conflict_rows, conflicting_path, full_analysis, mismatch_baseline, mismatch_report, only_variant, prereg_base_path, prereg_report, prereg_rows
    full_analysis = print_comparison_analysis(hyp_rows, factors=factors, baseline_mode='smelt', metric='val_loss')
    assert full_analysis['baseline_adequacy']['ok'] is True
    assert full_analysis['hypothesis_registry']['cells']
    assert full_analysis['baseline_preregistration']['ok'] is None
    assert full_analysis['baseline_preregistration']['declared_modes'] == []
    assert 'imbalance_advisory' in full_analysis
    assert full_analysis['equivalence_margin'] == _DEFAULT_EQUIVALENCE_MARGIN
    assert full_analysis['target_power'] == _TARGET_POWER
    assert 'power_equivalence' in full_analysis
    prereg_base_path = _write_analysis_run('prereg_baseline_seed1', 'A', 1, 0.0003, 100, 0.01, 4.4, hypothesis='延迟合并使验证损失下降', reference=good_reference, preregistration={'primary_metric': 'val_loss', 'direction': 'decrease', 'min_effect': None, 'baseline': 'smelt'})
    prereg_rows = compare_runs([smelt_path, prereg_base_path])
    auto_baseline = print_comparison_analysis(prereg_rows, factors=factors, metric='val_loss')
    prereg_report = auto_baseline['baseline_preregistration']
    assert prereg_report['auto_selected'] is True
    assert prereg_report['baseline_mode'] == 'smelt'
    assert prereg_report['ok'] is None
    assert prereg_report['runs_without_preregistration'] == ['smelt_seed1']
    assert prereg_report['advisories']
    assert auto_baseline['baseline_adequacy']['baseline_mode'] == 'smelt'
    mismatch_baseline = print_comparison_analysis(prereg_rows, factors=factors, baseline_mode='A', metric='val_loss')
    mismatch_report = mismatch_baseline['baseline_preregistration']
    assert mismatch_report['mismatch'] is True
    assert mismatch_report['ok'] is False
    assert any(('baseline shopping' in message for message in mismatch_report['advisories']))
    only_variant = compare_runs([prereg_base_path])
    absent_baseline = print_comparison_analysis(only_variant, factors=factors, metric='val_loss')
    absent_report = absent_baseline['baseline_preregistration']
    assert absent_report['baseline_in_runs'] is False
    assert absent_report['ok'] is None
    assert any(('不在本次对比集合中' in message for message in absent_report['advisories']))
    conflicting_path = _write_analysis_run('prereg_conflict_seed1', 'A', 2, 0.0003, 100, 0.01, 4.45, hypothesis='延迟合并使验证损失下降', reference=good_reference, preregistration={'primary_metric': 'val_loss', 'direction': 'decrease', 'min_effect': None, 'baseline': 'local_base'})
    conflict_rows = compare_runs([prereg_base_path, conflicting_path])
    conflict_baseline = print_comparison_analysis(conflict_rows, factors=factors, baseline_mode='smelt', metric='val_loss')
    conflict_report = conflict_baseline['baseline_preregistration']
    assert conflict_report['ok'] is False
    assert conflict_report['consistent'] is False
    assert any(('不一致' in message for message in conflict_report['advisories']))

def _selfcheck_body_64143_0105():
    global custom_margin_analysis, explicit_metric, fallback_metric, invalid_fallback_metric, resolved_metric, seed, tuning_paths, tuning_plan, tuning_report, value
    custom_margin_analysis = print_comparison_analysis(hyp_rows, factors=factors, baseline_mode='smelt', metric='val_loss', equivalence_margin=0.02)
    assert custom_margin_analysis['equivalence_margin'] == 0.02
    tuning_plan = build_baseline_tuning_plan([42, 43, 44], 'dense', [0.0001, 0.0003], 0.0)
    assert len(tuning_plan) == 6
    assert tuning_plan[0]['label'] == 'lr0.0001_seed42'
    assert tuning_plan[0]['mode'] == 'dense'
    assert tuning_plan[0]['lr'] == 0.0001
    assert tuning_plan[0]['seed'] == 42
    assert baseline_tuning_axis_candidates_error('--baseline-tuning-dense-hiddens', [128, 256], True) is None
    assert baseline_tuning_axis_candidates_error('--baseline-tuning-dense-hiddens', [0], True) is not None
    assert baseline_tuning_axis_candidates_error('--baseline-tuning-weight-decays', [0.0, 0.1], False) is None
    assert baseline_tuning_axis_candidates_error('--baseline-tuning-weight-decays', [-0.1], False) is not None
    assert baseline_tuning_axis_candidates_error('--baseline-tuning-weight-decays', [float('inf')], False) is not None
    tuning_paths = []
    for seed, value in ((42, 4.0), (43, 4.0), (44, 4.0)):
        tuning_paths.append(_write_analysis_run(f'tune_lr1e-4_seed{seed}', 'dense', seed, 0.0001, 100, 0.0, value))
    for seed, value in ((42, 4.0), (43, 4.2), (44, 3.8)):
        tuning_paths.append(_write_analysis_run(f'tune_lr3e-4_seed{seed}', 'dense', seed, 0.0003, 100, 0.0, value))
    tuning_report = analyze_baseline_tuning(compare_runs(tuning_paths), 'val_loss', 'dense', [0.0001, 0.0003], [42, 43, 44])
    assert tuning_report['comparison_family'] == 'baseline_tuning'
    assert tuning_report['winner']['lr'] == 0.0001
    assert abs(float(tuning_report['winner']['metric_mean']) - 4.0) < 1e-12
    assert tuning_report['training_config']['total_steps'] == 100
    assert tuning_report['seed_alignment']['aligned'] is True
    assert tuning_report['seed_alignment']['common_seeds'] == [42, 43, 44]
    assert any(('无法在统计上区分' in issue for issue in tuning_report['issues']))
    resolved_metric = resolve_baseline_tuning_metric(argparse.Namespace(baseline_tuning_metric=None, primary_metric='val_macro_f1', compare_metric='val_loss'))
    assert resolved_metric == ('val_macro_f1', 'primary_metric')
    explicit_metric = resolve_baseline_tuning_metric(argparse.Namespace(baseline_tuning_metric='val_pr_auc_macro', primary_metric='val_macro_f1', compare_metric='val_loss'))
    assert explicit_metric == ('val_pr_auc_macro', 'explicit')
    fallback_metric = resolve_baseline_tuning_metric(argparse.Namespace(baseline_tuning_metric=None, primary_metric='routing_jaccard_branch', compare_metric='val_loss'))
    assert fallback_metric == ('val_loss', 'compare_metric')
    invalid_fallback_metric = resolve_baseline_tuning_metric(argparse.Namespace(baseline_tuning_metric=None, primary_metric='routing_jaccard_branch', compare_metric='val_loss_at_matched_flops'))

def _selfcheck_body_64282_0106():
    global duplicate_paths, duplicate_report, misaligned_paths, misaligned_report, noisy_paths, seed, unconverged_tuning_paths, unconverged_tuning_report, value
    assert invalid_fallback_metric == ('val_loss', 'fallback_loss')
    misaligned_paths = []
    for seed, value in ((42, 4.0), (43, 4.0), (44, 4.0)):
        misaligned_paths.append(_write_analysis_run(f'misaligned_lr1e-4_seed{seed}', 'dense', seed, 0.0001, 100, 0.0, value))
    for seed, value in ((42, 4.5), (43, 4.5), (45, 4.5)):
        misaligned_paths.append(_write_analysis_run(f'misaligned_lr3e-4_seed{seed}', 'dense', seed, 0.0003, 100, 0.0, value))
    misaligned_report = analyze_baseline_tuning(compare_runs(misaligned_paths), 'val_loss', 'dense', [0.0001, 0.0003], [42, 43, 44])
    assert misaligned_report['seed_alignment']['aligned'] is False
    assert any(('种子集合不一致' in issue for issue in misaligned_report['issues']))
    duplicate_paths = list(tuning_paths)
    duplicate_paths.append(_write_analysis_run('tune_lr1e-4_seed42_repeat', 'dense', 42, 0.0001, 100, 0.0, 4.1))
    duplicate_report = analyze_baseline_tuning(compare_runs(duplicate_paths), 'val_loss', 'dense', [0.0001, 0.0003], [42, 43, 44])
    assert duplicate_report['seed_alignment']['aligned'] is False
    assert duplicate_report['seed_alignment']['duplicate_seeds']
    assert any(('伪重复' in issue for issue in duplicate_report['issues']))
    unconverged_tuning_paths = []
    for seed, value in ((42, 4.0), (43, 4.0), (44, 4.0)):
        unconverged_tuning_paths.append(_write_analysis_run(f'tune_unc_lr1e-4_seed{seed}', 'dense', seed, 0.0001, 100, 0.0, value, curve_status='not_converged'))
    for seed, value in ((42, 4.5), (43, 4.5), (44, 4.5)):
        unconverged_tuning_paths.append(_write_analysis_run(f'tune_unc_lr3e-4_seed{seed}', 'dense', seed, 0.0003, 100, 0.0, value, curve_status='healthy'))
    unconverged_tuning_report = analyze_baseline_tuning(compare_runs(unconverged_tuning_paths), 'val_loss', 'dense', [0.0001, 0.0003], [42, 43, 44])
    assert unconverged_tuning_report['winner']['lr'] == 0.0001
    assert unconverged_tuning_report['winner_unconverged_runs']
    assert any(('收敛' in issue for issue in unconverged_tuning_report['issues']))
    noisy_paths = []
    for seed, value in ((42, 4.0), (43, 4.05), (44, 3.95)):
        noisy_paths.append(_write_analysis_run(f'noisy_lr1e-4_seed{seed}', 'dense', seed, 0.0001, 100, 0.0, value))
    for seed, value in ((42, 3.97), (43, 4.0), (44, 3.94)):
        noisy_paths.append(_write_analysis_run(f'noisy_lr1e-3_seed{seed}', 'dense', seed, 0.001, 100, 0.0, value))

def _selfcheck_body_64420_0107():
    global _multi_run, changed, clear_paths, clear_report, item, lr, multi_axes, multi_base, multi_paths, multi_plan, noisy_report, seed, value
    noisy_report = analyze_baseline_tuning(compare_runs(noisy_paths), 'val_loss', 'dense', [0.0001, 0.001], [42, 43, 44])
    assert noisy_report['winner']['lr'] == 0.0001
    assert noisy_report['winner_selection'] == 'conservative_tie_break'
    assert noisy_report['tied_lrs'] == [0.0001, 0.001]
    assert any(('无法在统计上区分' in issue and '保守 winner' in issue for issue in noisy_report['issues']))
    clear_paths = []
    for lr, value in ((0.0001, 4.0), (0.0003, 4.5), (0.001, 4.2)):
        for seed in (42, 43, 44):
            clear_paths.append(_write_analysis_run(f'clear_lr{format(lr, 'g')}_seed{seed}', 'dense', seed, lr, 100, 0.0, value))
    clear_report = analyze_baseline_tuning(compare_runs(clear_paths), 'val_loss', 'dense', [0.0001, 0.0003, 0.001], [42, 43, 44])
    assert clear_report['winner']['lr'] == 0.0001
    assert clear_report['winner_selection'] == 'best_mean'
    assert clear_report['tied_lrs'] == [0.0001]
    assert clear_report['winner_at_grid_boundary'] is True
    assert not any(('无法在统计上区分' in issue for issue in clear_report['issues']))
    multi_axes = [{'name': 'lr', 'protocol_key': 'lr', 'label': '学习率', 'values': [0.0001, 0.0003, 0.001], 'base_value': 0.0003}, {'name': 'dense_fnn_hidden', 'protocol_key': 'dense_fnn_hidden', 'label': 'dense FFN 隐藏宽度', 'values': [128, 256, 512], 'base_value': 256}, {'name': 'weight_decay', 'protocol_key': 'weight_decay', 'label': '权重衰减', 'values': [0.0, 0.1, 0.2], 'base_value': 0.1}]
    multi_base = {'lr': 0.0003, 'dense_fnn_hidden': 256, 'weight_decay': 0.1}
    multi_plan = build_baseline_tuning_plan(seeds=[42], mode='dense', lrs=[0.0001, 0.0003], balance_coef=0.0, dense_hiddens=[128, 512], weight_decays=[0.0, 0.2], base_lr=0.0003, base_dense_hidden=256, base_weight_decay=0.1)
    assert sorted((str(item['label']) for item in multi_plan)) == sorted(['h128_seed42', 'h512_seed42', 'lr0.0001_seed42', 'lr0.0003_seed42', 'wd0_seed42', 'wd0.2_seed42'])
    for item in multi_plan:
        changed = [key for key, value in multi_base.items() if item.get(key) != value]
        assert len(changed) <= 1, item
    def _multi_run(name: str, lr: float, hidden: int, wd: float, value: float, seed: int) -> str:
        return _write_analysis_run(name, 'dense', seed, lr, 100, 0.0, value, protocol_overrides={'dense_fnn_hidden': hidden, 'weight_decay': wd})
    multi_paths = []

def _selfcheck_body_64556_0108():
    global axis_by_name, axis_name, axis_report, collision_alignment, collision_axes, collision_report, collision_rows, expected_values, metric_value, multi_protocol, multi_report, seed, value
    for seed in (42, 43, 44):
        multi_paths.append(_multi_run(f'multi_base_seed{seed}', 0.0003, 256, 0.1, 4.0, seed))
        multi_paths.append(_multi_run(f'multi_lr1e-4_seed{seed}', 0.0001, 256, 0.1, 3.9, seed))
        multi_paths.append(_multi_run(f'multi_lr1e-3_seed{seed}', 0.001, 256, 0.1, 4.2, seed))
        multi_paths.append(_multi_run(f'multi_h128_seed{seed}', 0.0003, 128, 0.1, 4.3, seed))
        multi_paths.append(_multi_run(f'multi_h512_seed{seed}', 0.0003, 512, 0.1, 3.7, seed))
        multi_paths.append(_multi_run(f'multi_wd0_seed{seed}', 0.0003, 256, 0.0, 4.1, seed))
        multi_paths.append(_multi_run(f'multi_wd0.2_seed{seed}', 0.0003, 256, 0.2, 3.8, seed))
    multi_report = analyze_baseline_tuning_axes(compare_runs(multi_paths), 'val_loss', 'dense', multi_axes, [42, 43, 44], multi_base)
    assert multi_report['multi_axis'] is True
    assert multi_report['design'] == 'coordinate'
    assert multi_report['axis_winners'] == {'lr': 0.0001, 'dense_fnn_hidden': 512, 'weight_decay': 0.2}
    assert multi_report['recommended_config'] == {'lr': 0.0001, 'dense_fnn_hidden': 512, 'weight_decay': 0.2}
    assert multi_report['recommended_config_evaluated'] is False
    assert any(('没有在本轮坐标式网格中被直接评估过' in issue for issue in multi_report['issues']))
    axis_by_name = {str(axis_report['name']): axis_report for axis_report in multi_report['axes']}
    for axis_name, expected_values in (('lr', [0.0001, 0.0003, 0.001]), ('dense_fnn_hidden', [128, 256, 512]), ('weight_decay', [0.0, 0.1, 0.2])):
        axis_report = axis_by_name[axis_name]
        assert sorted((candidate['value'] for candidate in axis_report['candidates'])) == expected_values
        assert all((candidate['n_runs'] == 3 for candidate in axis_report['candidates']))
        assert axis_report['seed_alignment']['aligned'] is True
    assert axis_by_name['lr']['winner']['value'] == 0.0001
    assert axis_by_name['lr']['winner']['lr'] == 0.0001
    assert multi_report['winner']['lr'] == 0.0001
    assert axis_by_name['dense_fnn_hidden']['winner']['value'] == 512
    assert axis_by_name['weight_decay']['winner']['value'] == 0.2
    collision_axes = [{'name': 'weight_decay', 'protocol_key': 'weight_decay', 'label': '权重衰减', 'values': [0.1, 0.2], 'base_value': 0.1000001}]
    collision_rows = []
    for seed in (42, 43, 44):
        for value, metric_value in ((0.1000001, 4.0), (0.1, 3.9), (0.2, 3.8)):
            collision_rows.append({'run': f'collision_wd{value}_seed{seed}', 'seed': seed, 'val_loss': metric_value, 'protocol': {'weight_decay': value}})
    collision_report = analyze_baseline_tuning_axes(collision_rows, 'val_loss', 'dense', collision_axes, [42, 43, 44], {'weight_decay': 0.1000001})
    collision_alignment = collision_report['axes'][0]['seed_alignment']
    assert collision_alignment['aligned'] is True, collision_alignment
    assert len(collision_alignment['candidate_seed_sets']) == 3
    multi_protocol = dict(multi_report['training_config'])
    multi_protocol['lr'] = 0.0001
    multi_protocol['weight_decay'] = 0.2
    multi_protocol['dense_fnn_hidden'] = 512
    assert baseline_tuning_provenance_error(multi_report, mode='dense', lr=0.0001, steps=100, data_sha256='deadbeef', batch_size=8, protocol=multi_protocol) is None

def _selfcheck_body_64687_0109():
    global axis_report, config_break_controlled, config_break_path, incomplete_adequacy, incomplete_adequacy_path, incomplete_axes, incomplete_error, incomplete_record, incomplete_report, mismatched_protocol, missing_axes_error, missing_axes_report, missing_protocol_key, multi_record, partial_protocol_rows
    assert baseline_tuning_provenance_error(multi_report, mode='dense', lr=0.0003, steps=100, data_sha256='deadbeef', batch_size=8, protocol=multi_protocol) is not None
    mismatched_protocol = dict(multi_protocol)
    mismatched_protocol['weight_decay'] = 0.1
    assert baseline_tuning_provenance_error(multi_report, mode='dense', lr=0.0001, steps=100, data_sha256='deadbeef', batch_size=8, protocol=mismatched_protocol) is not None
    multi_record = baseline_tuning_provenance_record(multi_report, 'baseline_tuning.json')
    assert multi_record['multi_axis'] is True
    assert multi_record['recommended_config'] == {'lr': 0.0001, 'dense_fnn_hidden': 512, 'weight_decay': 0.2}
    assert multi_record['recommended_config_evaluated'] is False
    assert _baseline_tuning_axis_error(multi_record) is None
    incomplete_report = dict(multi_report)
    incomplete_axes = [dict(axis_report) for axis_report in multi_report['axes']]
    for axis_report in incomplete_axes:
        if axis_report.get('protocol_key') == 'dense_fnn_hidden':
            axis_report['winner'] = None
    incomplete_report['axes'] = incomplete_axes
    incomplete_error = baseline_tuning_provenance_error(incomplete_report, mode='dense', lr=0.0001, steps=100, data_sha256='deadbeef', batch_size=8, protocol=multi_protocol)
    assert incomplete_error is not None
    assert 'dense_fnn_hidden' in incomplete_error
    assert '没有有效 winner' in incomplete_error
    missing_axes_report = dict(multi_report)
    missing_axes_report.pop('axes')
    missing_axes_error = baseline_tuning_provenance_error(missing_axes_report, mode='dense', lr=0.0001, steps=100, data_sha256='deadbeef', batch_size=8, protocol=multi_protocol)
    assert missing_axes_error is not None
    assert 'axes' in missing_axes_error
    incomplete_record = baseline_tuning_provenance_record(incomplete_report, 'incomplete_tuning.json')
    assert incomplete_record['multi_axis'] is True
    assert 'dense_fnn_hidden' not in incomplete_record['axis_winners']
    incomplete_adequacy_path = _write_analysis_run('incomplete_axis_seed1', 'dense', 1, 0.0001, 100, 0.01, 4.0, reference=good_reference, baseline_tuning=incomplete_record)
    incomplete_adequacy = baseline_adequacy(compare_runs([incomplete_adequacy_path]), 'dense', 'val_loss')
    assert incomplete_adequacy['tuned'] is False
    assert any(('调参来源不完整' in message for message in incomplete_adequacy['advisories']))
    config_break_path = _write_analysis_run('tune_configbreak_seed42', 'dense', 42, 0.0001, 200, 0.01, 3.5)
    config_break_controlled = analyze_baseline_tuning(compare_runs(tuning_paths + [config_break_path]), 'val_loss', 'dense', [0.0001, 0.0003], [42, 43, 44])
    assert any(('训练协议' in issue for issue in config_break_controlled['issues']))
    partial_protocol_rows = compare_runs(tuning_paths)
    missing_protocol_key = 'warmup_steps'
    assert all((missing_protocol_key in row['protocol'] for row in partial_protocol_rows))
    partial_protocol_rows[0]['protocol'].pop(missing_protocol_key)

def _selfcheck_body_64825_0110():
    global explicit_ngram_metric, fallback_ngram_metric, ngram_aligned_adequacy, ngram_aligned_path, ngram_mismatch_adequacy, ngram_mismatch_path, partial_protocol_report, resolved_ngram_metric, sampled_protocol, tuning_payload, tuning_source
    partial_protocol_report = analyze_baseline_tuning(partial_protocol_rows, 'val_loss', 'dense', [0.0001, 0.0003], [42, 43, 44])
    assert partial_protocol_report['training_config'][missing_protocol_key] is None
    assert any(('部分候选运行缺少训练协议字段' in issue for issue in partial_protocol_report['issues']))
    assert tuning_report['winner_selection'] == 'conservative_tie_break'
    assert tuning_report['tied_lrs'] == [0.0001, 0.0003]
    assert tuning_report['winner_at_grid_boundary'] is True
    assert any(('网格边界' in issue for issue in tuning_report['issues']))
    resolved_ngram_metric = resolve_ngram_tuning_metric(argparse.Namespace(ngram_tuning_metric=None, primary_metric='val_macro_f1'))
    assert resolved_ngram_metric == ('val_macro_f1', 'primary_metric')
    explicit_ngram_metric = resolve_ngram_tuning_metric(argparse.Namespace(ngram_tuning_metric='val_pr_auc_macro', primary_metric='val_macro_f1'))
    assert explicit_ngram_metric == ('val_pr_auc_macro', 'explicit')
    fallback_ngram_metric = resolve_ngram_tuning_metric(argparse.Namespace(ngram_tuning_metric=None, primary_metric='routing_jaccard_branch'))
    assert fallback_ngram_metric == ('val_loss', 'default_loss')
    ngram_mismatch_path = _write_analysis_run('ngram_loss_tuned_seed1', 'ngram', 1, 0.0003, 100, 0.0, 4.5, reference=auc_reference, pr_auc=0.02, pr_auc_chance=0.001, ngram_tuning_metric='val_loss')
    ngram_mismatch_adequacy = baseline_adequacy(compare_runs([ngram_mismatch_path]), 'ngram', 'val_pr_auc_macro')
    assert ngram_mismatch_adequacy['tuned'] is False
    assert ngram_mismatch_adequacy['ngram_tuning_metric_mismatch']
    assert any(('ngram-tuning-metric' in message for message in ngram_mismatch_adequacy['advisories']))
    ngram_aligned_path = _write_analysis_run('ngram_pr_tuned_seed1', 'ngram', 1, 0.0003, 100, 0.0, 4.5, reference=auc_reference, pr_auc=0.02, pr_auc_chance=0.001, ngram_tuning_metric='val_pr_auc_macro')
    ngram_aligned_adequacy = baseline_adequacy(compare_runs([ngram_aligned_path]), 'ngram', 'val_pr_auc_macro')
    assert ngram_aligned_adequacy['tuned'] is True
    assert not ngram_aligned_adequacy['ngram_tuning_metric_mismatch']
    tuning_source = write_baseline_tuning_report(Path(analysis_dir) / 'tuning_out', tuning_report)
    tuning_payload = load_baseline_tuning(tuning_source)
    assert tuning_payload['winner']['lr'] == 0.0001
    sampled_protocol = compare_runs([tuning_paths[0]])[0]['protocol']
    assert baseline_tuning_provenance_error(tuning_payload, 'dense', 0.0001, 100, 'deadbeef', protocol=sampled_protocol) is None

def _selfcheck_body_64967_0111():
    global legacy_error, legacy_payload, missing_seed_count_payload, no_seed_plan_error, no_seed_plan_payload, planned_protocol, underpowered_error, underpowered_payload
    planned_protocol = {key: value for key, value in sampled_protocol.items() if key not in _BASELINE_TUNING_RUNTIME_KEYS}
    assert baseline_tuning_provenance_error(tuning_payload, 'dense', 0.0001, 100, 'deadbeef', protocol=planned_protocol) is None
    assert baseline_tuning_provenance_error(tuning_payload, 'smelt', 0.0001, 100, 'deadbeef', protocol=sampled_protocol) is not None
    assert baseline_tuning_provenance_error(tuning_payload, 'dense', 0.0003, 100, 'deadbeef', protocol=sampled_protocol) is not None
    assert baseline_tuning_provenance_error(tuning_payload, 'dense', 0.0001, 200, 'deadbeef', protocol=sampled_protocol) is not None
    assert baseline_tuning_provenance_error(tuning_payload, 'dense', 0.0001, 100, 'cafebabe', protocol=sampled_protocol) is not None
    assert baseline_tuning_provenance_error(tuning_payload, 'dense', 0.0001, 100, 'deadbeef', protocol=dict(sampled_protocol, warmup_steps=500)) is not None
    assert baseline_tuning_provenance_error(tuning_payload, 'dense', 0.0001, 100, 'deadbeef') is not None
    legacy_payload = dict(tuning_payload)
    legacy_payload['training_config'] = {'total_steps': 100, 'batch_size': 8, 'data_sha256': 'deadbeef'}
    legacy_error = baseline_tuning_provenance_error(legacy_payload, 'dense', 0.0001, 100, 'deadbeef', protocol=sampled_protocol)
    assert legacy_error is not None
    assert '重新生成调参报告' in legacy_error
    underpowered_payload = dict(tuning_payload)
    underpowered_payload['seed_plan'] = dict(tuning_payload['seed_plan'], seeds=[42, 43], n_seeds=2, underpowered_override=True)
    underpowered_error = baseline_tuning_provenance_error(underpowered_payload, 'dense', 0.0001, 100, 'deadbeef', protocol=sampled_protocol)
    assert underpowered_error is not None
    assert 'underpowered' in underpowered_error
    no_seed_plan_payload = dict(tuning_payload)
    no_seed_plan_payload.pop('seed_plan')
    no_seed_plan_error = baseline_tuning_provenance_error(no_seed_plan_payload, 'dense', 0.0001, 100, 'deadbeef', protocol=sampled_protocol)
    assert no_seed_plan_error is not None
    assert 'seed_plan' in no_seed_plan_error
    missing_seed_count_payload = dict(tuning_payload)

def _selfcheck_body_65099_0112():
    global bogus_source, misaligned_seed_error, misaligned_seed_payload, missing_seed_count_error, provenance_record, tuned_row
    missing_seed_count_payload['seed_plan'] = dict(tuning_payload['seed_plan'])
    missing_seed_count_payload['seed_plan'].pop('n_seeds')
    missing_seed_count_error = baseline_tuning_provenance_error(missing_seed_count_payload, 'dense', 0.0001, 100, 'deadbeef', protocol=sampled_protocol)
    assert missing_seed_count_error is not None
    assert 'n_seeds' in missing_seed_count_error
    misaligned_seed_payload = dict(tuning_payload)
    misaligned_seed_payload['seed_alignment'] = {'aligned': False, 'candidate_seed_sets': {}, 'common_seeds': [], 'duplicate_seeds': {}, 'missing_seed_runs': []}
    misaligned_seed_error = baseline_tuning_provenance_error(misaligned_seed_payload, 'dense', 0.0001, 100, 'deadbeef', protocol=sampled_protocol)
    assert misaligned_seed_error is not None
    assert '种子集合' in misaligned_seed_error
    provenance_record = baseline_tuning_provenance_record(tuning_payload, str(tuning_source))
    assert provenance_record['winner_lr'] == 0.0001
    assert provenance_record['metric'] == 'val_loss'
    bogus_source = Path(analysis_dir) / 'bogus_tuning.json'
    bogus_source.write_text('{"comparison_family": "baseline_tuning", "mode": "dense", "metric": "val_loss"}', encoding='utf-8')
    try:
        load_baseline_tuning(bogus_source)
    except ValueError:
        pass
    else:
        raise AssertionError('缺少 winner 的调参报告必须被拒绝')
    tuned_row = next((row for row in compare_runs([smelt_path]) if row.get('run') == 'smelt_seed1'))
    assert tuned_row['baseline_tuning']['winner_lr'] == 0.0003

def _selfcheck_top_58785_0064():
    global analysis_dir
    with tempfile.TemporaryDirectory() as analysis_dir:
        _selfcheck_body_1_0066()
        _selfcheck_body_59084_0067()
        _selfcheck_body_59224_0068()
        _selfcheck_body_59371_0069()
        _selfcheck_body_59511_0070()
        _selfcheck_body_59644_0071()
        _selfcheck_body_59789_0072()
        _selfcheck_body_59921_0073()
        _selfcheck_body_60059_0074()
        _selfcheck_body_60201_0075()
        _selfcheck_body_60340_0076()
        _selfcheck_body_60467_0077()
        _selfcheck_body_60601_0078()
        _selfcheck_body_60737_0079()
        _selfcheck_body_60877_0080()
        _selfcheck_body_61007_0081()
        _selfcheck_body_61132_0082()
        _selfcheck_body_61261_0083()
        _selfcheck_body_61347_0084()
        _selfcheck_body_61473_0085()
        _selfcheck_body_61590_0086()
        _selfcheck_body_61723_0087()
        _selfcheck_body_61857_0088()
        _selfcheck_body_61988_0089()
        _selfcheck_body_62118_0090()
        _selfcheck_body_62249_0091()
        _selfcheck_body_62385_0092()
        _selfcheck_body_62528_0093()
        _selfcheck_body_62655_0094()
        _selfcheck_body_62789_0095()
        _selfcheck_body_62922_0096()
        _selfcheck_body_63062_0097()
        _selfcheck_body_63192_0098()
        _selfcheck_body_63319_0099()
        _selfcheck_body_63453_0100()
        _selfcheck_body_63591_0101()
        _selfcheck_body_63735_0102()
        _selfcheck_body_63864_0103()
        _selfcheck_body_64010_0104()
        _selfcheck_body_64143_0105()
        _selfcheck_body_64282_0106()
        _selfcheck_body_64420_0107()
        _selfcheck_body_64556_0108()
        _selfcheck_body_64687_0109()
        _selfcheck_body_64825_0110()
        _selfcheck_body_64967_0111()
        _selfcheck_body_65099_0112()

def _selfcheck_top_1_0113():
    _selfcheck_top_58785_0064()

def _selfcheck_top_65164_0114():
    global _write_cost_run
    def _write_cost_run(base_dir: str, name: str, mode: str, seed: int, val_loss: float, flops: float) -> str:
        run_dir = Path(base_dir) / name
        run_dir.mkdir()
        config_payload = {'config': {'mode': mode, 'top_k': 2, 'k1': 2, 'k2': 2, 'balance_coef': 0.01, 'dense_fnn_hidden': None}, 'args': {'seed': seed}, 'n_params': 100, 'efficiency': {'flops_per_forward': flops, 'latency_ms_per_forward': flops / 1000000000.0, 'training_ms_per_step': flops / 1000000000.0, 'training_peak_memory_mb': 100.0}}
        with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config_payload, f)
        write_jsonl_record(run_dir / 'metrics.jsonl', {'step': 'final', 'val_loss': val_loss})
        return str(run_dir)

def _selfcheck_body_65205_0116():
    global _Capture, base_costs, by_cost_label, capture, cost_aggregates, cost_contrast, cost_contrasts, cost_display, cost_paths, dup_cell, dup_costs, dup_paths, latency_costs, peak_config, peak_config_path, peak_row, ratios, report_text
    cost_paths = [_write_cost_run(cost_dir, 'local_base_seed42', 'local_base', 42, 3.0, 1000000000.0), _write_cost_run(cost_dir, 'local_base_seed43', 'local_base', 43, 3.1, 1000000000.0), _write_cost_run(cost_dir, 'local_base_seed44', 'local_base', 44, 3.2, 1000000000.0), _write_cost_run(cost_dir, 'A_seed42', 'A', 42, 2.8, 2000000000.0), _write_cost_run(cost_dir, 'A_seed43', 'A', 43, 2.85, 2000000000.0), _write_cost_run(cost_dir, 'A_seed44', 'A', 44, 2.9, 2000000000.0)]
    cost_aggregates = aggregate_runs(compare_runs(cost_paths), ['mode'])
    by_cost_label = {aggregate['label']: aggregate for aggregate in cost_aggregates}
    base_costs = by_cost_label['mode=local_base']['costs']
    assert base_costs['flops_per_forward']['mean'] == 1000000000.0
    peak_config_path = Path(cost_paths[0]) / 'config.json'
    peak_config = json.loads(peak_config_path.read_text(encoding='utf-8'))
    peak_config['efficiency']['actual_training_peak_memory_mb'] = 777.0
    peak_config_path.write_text(json.dumps(peak_config), encoding='utf-8')
    peak_row = compare_runs([cost_paths[0]])[0]
    assert peak_row['peak_memory_mb'] == 777.0
    assert peak_row['training_peak_memory_mb'] == 100.0
    cost_contrasts = factor_contrasts(cost_aggregates, ['mode'], 'val_loss')
    assert len(cost_contrasts) == 1
    cost_contrast = cost_contrasts[0]
    ratios = cost_contrast['cost']['ratios']
    assert abs(ratios['flops_per_forward'] - 2.0) < 1e-09
    assert abs(ratios['training_ms_per_step'] - 2.0) < 1e-09
    assert abs(ratios['n_params'] - 1.0) < 1e-09
    assert cost_contrast['cost_verdict'] == 'gain_with_cost', cost_contrast
    cost_display = _contrast_display(cost_contrast, 'val_loss')
    assert '效率倍率:' in cost_display
    assert '取舍' in cost_display
    assert '推理计时×2.00 [2.00,2.00]' in cost_display, cost_display
    assert base_costs['flops_per_forward']['n'] == 3
    assert base_costs['flops_per_forward']['std'] == 0.0
    assert base_costs['flops_per_forward']['coefficient_of_variation'] == 0.0
    latency_costs = base_costs['latency_ms_per_forward']
    assert float(latency_costs['ci95_low']) <= float(latency_costs['mean']) <= float(latency_costs['ci95_high'])
    assert latency_costs['ci95']['df'] == 2
    assert latency_costs['ci95']['method'] == 't_interval'
    assert abs(float(latency_costs['ci95_low']) - float(latency_costs['ci95_high'])) < 1e-12
    dup_paths = [_write_cost_run(cost_dir, 'dup_a', 'local_base', 42, 3.0, 1000000000.0), _write_cost_run(cost_dir, 'dup_b', 'local_base', 42, 3.0, 1000000000.0)]
    dup_cell = aggregate_runs(compare_runs(dup_paths), ['mode'])[0]
    dup_costs = dup_cell['costs']
    assert dup_cell['duplicate_seeds'] == [42]
    assert dup_costs['flops_per_forward']['n'] == 1, dup_costs
    assert 'ci95' not in dup_costs['flops_per_forward']
    class _Capture:

        def __init__(self) -> None:
            self.chunks: List[str] = []

        def write(self, text: str) -> int:
            self.chunks.append(text)
            return len(text)

        def flush(self) -> None:
            return None
    capture = _Capture()
    with contextlib.redirect_stdout(capture):
        print_comparison_analysis(compare_runs(cost_paths), factors=['mode'], metric='val_loss')
    report_text = ''.join(capture.chunks)
    assert '[efficiency-ci]' in report_text, report_text
    assert 'mode=local_base' in report_text

def _selfcheck_top_65204_0115():
    global cost_dir
    with tempfile.TemporaryDirectory() as cost_dir:
        _selfcheck_body_65205_0116()

def _selfcheck_top_1_0117():
    _selfcheck_top_65204_0115()

def _selfcheck_top_65347_0118():
    global _, high, low, noisy_cell_a, noisy_cell_b, noisy_cost, noisy_detail, noisy_text, partial_cell_a, partial_cell_b, partial_cost, partial_text, partial_verdict, unverifiable_cell_a
    noisy_cell_a = {'costs': {'training_ms_per_step': {'mean': 100.0, 'std': 30.0, 'n': 3}}}
    noisy_cell_b = {'costs': {'training_ms_per_step': {'mean': 120.0, 'std': 30.0, 'n': 3}}}
    noisy_cost, _ = _cost_comparison(noisy_cell_a, noisy_cell_b, 'improvement', True)
    noisy_detail = noisy_cost['uncertainty']['training_ms_per_step']
    assert noisy_detail['resolvable'] is True
    assert noisy_detail['ci_method'] == 'welch_t', noisy_detail
    assert abs(noisy_detail['ci_df'] - 3.874) < 0.01, noisy_detail
    assert noisy_detail['ci_t_critical'] > 2.8, noisy_detail
    low, high = noisy_detail['ratio_ci95']
    assert low < 1.0 < high, noisy_detail
    assert 'training_ms_per_step' in noisy_cost['timing_noise_fields']
    noisy_text = _contrast_display({'factor': 'mode', 'baseline': 'a', 'variant': 'b', 'metric': 'val_loss', 'samples_a': [{'seed': 1, 'value': 3.0}], 'samples_b': [{'seed': 1, 'value': 2.8}], 'delta': -0.2, 'verdict': '改善', 'cost': noisy_cost, 'cost_verdict': 'gain_balanced'}, 'val_loss')
    assert '计时噪声' in noisy_text, noisy_text
    partial_cell_a = {'costs': {'flops_per_forward': {'mean': 1000000000.0, 'std': 0.0, 'n': 3}, 'peak_memory_mb': {'mean': 100.0, 'std': 0.0, 'n': 3}}}
    partial_cell_b = {'costs': {'flops_per_forward': {'mean': 1100000000.0, 'std': 0.0, 'n': 3}, 'peak_memory_mb': {'mean': 100.0, 'std': 0.0, 'n': 3}, 'latency_ms_per_forward': {'mean': 100.0, 'std': 1.0, 'n': 3}}}
    partial_cost, partial_verdict = _cost_comparison(partial_cell_a, partial_cell_b, 'improvement', True)
    assert partial_verdict == 'gain_balanced', partial_cost
    assert partial_cost['baseline_missing_measured_fields'] == ['latency_ms_per_forward']
    partial_text = _contrast_display({'factor': 'mode', 'baseline': 'a', 'variant': 'b', 'metric': 'val_loss', 'samples_a': [{'seed': 1, 'value': 3.0}], 'samples_b': [{'seed': 1, 'value': 2.8}], 'delta': -0.2, 'verdict': '改善', 'cost': partial_cost, 'cost_verdict': 'gain_balanced'}, 'val_loss')
    assert '效率字段不可比' in partial_text, partial_text
    assert '基线缺 推理计时' in partial_text, partial_text
    unverifiable_cell_a = {'costs': {'n_params': {'mean': 1000000000.0, 'std': 0.0, 'n': 3}, 'latency_ms_per_forward': {'mean': 10.0, 'std': 0.0, 'n': 1}}}

def _selfcheck_top_65481_0119():
    global numerical_aggregate, numerical_report, numerical_rows, timing_only_cost, timing_only_verdict, unverifiable_cell_b, unverifiable_contrast, unverifiable_cost, unverifiable_verdict
    unverifiable_cell_b = {'costs': {'n_params': {'mean': 1000000000.0, 'std': 0.0, 'n': 3}, 'latency_ms_per_forward': {'mean': 100.0, 'std': 0.0, 'n': 1}}}
    unverifiable_cost, unverifiable_verdict = _cost_comparison(unverifiable_cell_a, unverifiable_cell_b, 'improvement', True)
    assert unverifiable_verdict == 'gain_balanced', unverifiable_cost
    assert unverifiable_cost['unverifiable_timing_fields'] == ['latency_ms_per_forward']
    assert unverifiable_cost['timing_noise_fields'] == []
    assert 'latency_ms_per_forward' not in unverifiable_cost['material_cost_fields']
    unverifiable_contrast = {'p_value': 0.01, 'significant': True, 'verdict': 'improvement', 'cost': unverifiable_cost, 'cost_verdict': 'gain_balanced'}
    _apply_multiple_comparison_correction([unverifiable_contrast])
    assert unverifiable_contrast['cost_verdict'] == 'gain_balanced', unverifiable_contrast
    timing_only_cost, timing_only_verdict = _cost_comparison({'costs': {'latency_ms_per_forward': {'mean': 10.0, 'std': 0.0, 'n': 1}}}, {'costs': {'latency_ms_per_forward': {'mean': 100.0, 'std': 0.0, 'n': 1}}}, 'improvement', True)
    assert timing_only_verdict == 'gain_with_cost', timing_only_cost
    assert timing_only_cost['material_cost_fields'] == []
    numerical_rows = [{'run': 'ok1', 'mode': 'A', 'seed': 1, 'val_loss': 3.0, 'numerical_status': 'ok'}, {'run': 'ok2', 'mode': 'A', 'seed': 2, 'val_loss': 2.0, 'numerical_status': 'ok'}, {'run': 'diverged', 'mode': 'A', 'seed': 3, 'val_loss': 9.0, 'curve_status': 'diverged', 'nonfinite_events': 4, 'numerical_status': 'failed'}, {'run': 'finite-diverging', 'mode': 'A', 'seed': 4, 'val_loss': 8.0, 'curve_status': 'diverging', 'nonfinite_events': 0, 'numerical_status': 'failed'}]
    numerical_aggregate = aggregate_runs(numerical_rows, ['mode'])[0]
    assert numerical_aggregate['n_runs'] == 2
    assert numerical_aggregate['val_loss_mean'] == 2.5
    assert numerical_aggregate['numerical_failures'][0]['run'] == 'diverged'
    assert {item['run'] for item in numerical_aggregate['numerical_failures']} == {'diverged', 'finite-diverging'}
    numerical_report = numerical_failure_report(numerical_rows)
    assert numerical_report['excluded_run_count'] == 2
    assert numerical_report['advisory']
    assert numerical_report['by_mode'][0]['mode'] == 'A'
    assert numerical_report['by_mode'][0]['failed_runs'] == 2
    assert not numerical_report['mode_failure_rates_imbalanced']

def _selfcheck_top_65612_0120():
    global a_value, a_values, aggregate_by_label, balance, baseline_cell, ci_aggregate, ci_block, ci_mean, ci_rows, ci_single, ci_std, context_aggregates, context_contrast, context_contrasts, context_rows, expected_delta, expected_half, imbalanced_report, seed, smelt_value, smelt_values, variant_cell
    imbalanced_report = numerical_failure_report(numerical_rows + [{'run': f'B_ok{seed}', 'mode': 'B', 'seed': seed, 'val_loss': 2.0, 'numerical_status': 'ok'} for seed in (1, 2, 3)])
    assert imbalanced_report['mode_failure_rates_imbalanced']
    assert '失败率不同' in str(imbalanced_report['advisory'])
    ci_rows = [{'run': f'cell_seed{seed}', 'mode': 'A', 'seed': seed, 'val_loss': value} for seed, value in enumerate([3.0, 2.0, 1.0], start=1)]
    ci_aggregate = aggregate_runs(ci_rows, ['mode'])[0]
    ci_block = ci_aggregate['ci95']['val_loss']
    assert ci_block is not None, ci_block
    ci_mean = ci_aggregate['val_loss_mean']
    ci_std = ci_aggregate['val_loss_std']
    assert ci_mean == 2.0 and ci_std == 1.0
    expected_half = ci_block['t_critical'] * ci_std / math.sqrt(3)
    assert ci_block['df'] == 2
    assert abs(ci_block['low'] - (ci_mean - expected_half)) < 1e-09
    assert ci_block['low'] < ci_mean < ci_block['high']
    assert ci_aggregate['val_loss_ci95_low'] == ci_block['low'] and ci_aggregate['val_loss_ci95_high'] == ci_block['high']
    ci_single = aggregate_runs(ci_rows[:1], ['mode'])[0]
    assert ci_single['ci95']['val_loss'] is None
    assert 'val_loss_ci95_low' not in ci_single
    context_rows = []
    for balance, smelt_values, a_values in ((0.0, [3.0, 3.2, 3.4], [2.8, 2.9, 3.0]), (0.01, [4.0, 4.2, 4.4], [4.5, 4.6, 4.7])):
        for seed, (smelt_value, a_value) in enumerate(zip(smelt_values, a_values), start=1):
            context_rows.append({'run': f'smelt_bal{balance}_seed{seed}', 'mode': 'smelt', 'balance_coef': balance, 'seed': seed, 'val_loss': smelt_value})
            context_rows.append({'run': f'A_bal{balance}_seed{seed}', 'mode': 'A', 'balance_coef': balance, 'seed': seed, 'val_loss': a_value})
    context_rows.append({'run': 'A_bal0.02_seed1', 'mode': 'A', 'balance_coef': 0.02, 'seed': 1, 'val_loss': 1.0})
    context_aggregates = aggregate_runs(context_rows, ['mode', 'balance_coef'])
    context_contrasts = compare_cells_to_baseline(context_aggregates, 'smelt', 'val_loss')
    aggregate_by_label = {aggregate['label']: aggregate for aggregate in context_aggregates}
    assert len(context_contrasts) == 2, context_contrasts
    for context_contrast in context_contrasts:
        baseline_cell = aggregate_by_label[context_contrast['baseline']]
        variant_cell = aggregate_by_label[context_contrast['variant']]
        assert baseline_cell['keys']['balance_coef'] == variant_cell['keys']['balance_coef'], context_contrast
        expected_delta = -0.3 if baseline_cell['keys']['balance_coef'] == 0.0 else 0.4
        assert abs(context_contrast['delta'] - expected_delta) < 1e-12, context_contrast

def _selfcheck_top_65733_0121():
    global ablation_plan, dominated_contrast, dominated_dir, dominated_paths, f, failed_row, failed_run, mode_plan, numerical_dir, seed_plan
    with tempfile.TemporaryDirectory() as numerical_dir:
        failed_run = Path(numerical_dir) / 'failed_run'
        failed_run.mkdir()
        with open(failed_run / 'config.json', 'w', encoding='utf-8') as f:
            json.dump({'config': {'mode': 'A'}, 'args': {'seed': 1}, 'n_params': 10}, f)
        write_jsonl_record(failed_run / 'metrics.jsonl', {'step': 'final', 'val_loss': 3.0, 'nonfinite_events': 2, 'numerical_stop': True, 'numerical_polluted': False, 'prior_numerical_stop': False})
        failed_row = compare_runs([str(failed_run)])[0]
        assert failed_row['numerical_status'] == 'failed'
        assert failed_row['nonfinite_events'] == 2
        assert failed_row['numerical_integrity_ok'] is False
    with tempfile.TemporaryDirectory() as dominated_dir:
        dominated_paths = [_write_cost_run(dominated_dir, 'local_base_seed42', 'local_base', 42, 3.0, 1000000000.0), _write_cost_run(dominated_dir, 'local_base_seed43', 'local_base', 43, 3.1, 1000000000.0), _write_cost_run(dominated_dir, 'local_base_seed44', 'local_base', 44, 3.2, 1000000000.0), _write_cost_run(dominated_dir, 'A_seed42', 'A', 42, 3.3, 2000000000.0), _write_cost_run(dominated_dir, 'A_seed43', 'A', 43, 3.35, 2000000000.0), _write_cost_run(dominated_dir, 'A_seed44', 'A', 44, 3.4, 2000000000.0)]
        dominated_contrast = factor_contrasts(aggregate_runs(compare_runs(dominated_paths), ['mode']), ['mode'], 'val_loss')[0]
        assert dominated_contrast['cost_verdict'] == 'dominated_regression', dominated_contrast
        assert '严格劣于基线' in _contrast_display(dominated_contrast, 'val_loss')
    seed_plan = build_sweep_plan(seeds=[1, 2], modes=None, ablation=False, balance_coef=0.01, default_mode='B')
    assert [item['label'] for item in seed_plan] == ['seed1', 'seed2']
    assert {item['mode'] for item in seed_plan} == {'B'}
    assert {item['balance_coef'] for item in seed_plan} == {0.01}
    mode_plan = build_sweep_plan(seeds=[7], modes=['local_base', 'A'], ablation=False, balance_coef=0.01, default_mode='B')
    assert [item['label'] for item in mode_plan] == ['local_base_seed7', 'A_seed7']
    ablation_plan = build_sweep_plan(seeds=[7], modes=None, ablation=True, balance_coef=0.01, default_mode='B')
    assert [item['label'] for item in ablation_plan] == ['local_base_bal0_seed7', 'local_base_bal0.01_seed7', 'A_bal0_seed7', 'A_bal0.01_seed7', 'B_bal0_seed7', 'B_bal0.01_seed7']
    assert [item['balance_coef'] for item in ablation_plan] == [0.0, 0.01, 0.0, 0.01, 0.0, 0.01]

def _selfcheck_top_65866_0122():
    global ablation_plan_a_only, missing_balance_axis, missing_mode_axis, provenance_mismatch, stripped, stripped_modes, unknown_spec
    ablation_plan_a_only = build_sweep_plan(seeds=[7], modes=None, ablation=True, balance_coef=0.01, default_mode='B', ablation_modes=['A'])
    assert [item['mode'] for item in ablation_plan_a_only] == ['local_base', 'local_base', 'A', 'A']
    stripped = strip_sweep_args(['--data', 'corpus.txt', '--out', 'runs/rq1', '--mode', 'B', '--seeds', '1', '2', '3', '--suite-ablation', '--ablation-modes', 'A', 'B', '--allow-underpowered', '--steps', '10', '--balance-coef', '0.02', '--skip-sanity-check'])
    assert stripped == ['--data', 'corpus.txt', '--steps', '10', '--skip-sanity-check']
    stripped_modes = strip_sweep_args(['--suite-modes', 'A', 'B', '--out=runs/x', '--seed=5', '--k1', '2'])
    assert stripped_modes == ['--k1', '2']
    for unknown_spec in ('', 'mode,not_a_factor', 'selection_metric', 'model_selection'):
        try:
            parse_factor_spec(unknown_spec)
        except ValueError:
            pass
        else:
            raise AssertionError('非法对比因子必须被拒绝: ' + repr(unknown_spec))
    assert suite_factor_error(parse_factor_spec('mode,balance_coef,seed'), None, True) is None
    assert suite_factor_error(parse_factor_spec('mode,seed'), ['A', 'B'], False) is None
    missing_balance_axis = suite_factor_error(parse_factor_spec('mode,seed'), None, True)
    assert missing_balance_axis is not None
    assert 'balance_coef' in missing_balance_axis
    missing_mode_axis = suite_factor_error(parse_factor_spec('balance_coef,seed'), ['A', 'B'], False)
    assert missing_mode_axis is not None
    assert 'mode' in missing_mode_axis
    assert suite_factor_error(parse_factor_spec('seed'), None, False) is None
    provenance_mismatch = check_run_provenance([{'run': 'r1', 'provenance': {'script_sha256': 'a' * 64, 'torch_version': '2.0.0', 'cuda_version': None, 'platform': 'linux'}}, {'run': 'r2', 'provenance': {'script_sha256': 'b' * 64, 'torch_version': '2.0.0', 'cuda_version': None, 'platform': 'linux'}}])
    assert not provenance_mismatch['ok']
    assert any((mismatch['key'] == 'script_sha256' for mismatch in provenance_mismatch['mismatches'])), provenance_mismatch['mismatches']

