from __future__ import annotations

def _selfcheck_top_55671_0039():
    global batch_mismatched, broken_name, broken_probe, changed_repeat, expert_ids, healthy_probe, loss_mismatch, mismatched, name, parameter, probe_model, probe_result, repeated_metrics, same_repeat, zero_name, zero_probe
    repeated_metrics = run_sanity_overfit(sanity_cfg, eval_dataset, eval_device, amp_dtype=None, batch_size=2, steps=200, lr=0.01, loss_threshold=0.5, acc_threshold=0.9, seed=321, grad_checkpoint=True)
    assert repeated_metrics['final_lm_loss'] == sanity_metrics['final_lm_loss'], (repeated_metrics['final_lm_loss'], sanity_metrics['final_lm_loss'])
    assert repeated_metrics['final_token_accuracy'] == sanity_metrics['final_token_accuracy']
    assert repeated_metrics['initial_lm_loss'] == sanity_metrics['initial_lm_loss']
    assert repeated_metrics['grad_probe']['grad_norm'] == grad_probe['grad_norm']
    assert repeated_metrics['batch_sha256'] == sanity_metrics['batch_sha256']
    assert len(sanity_metrics['batch_sha256']) == 64
    assert _batch_sha256(x, y) == _batch_sha256(x.clone(), y.clone())
    assert _batch_sha256(x, y) != _batch_sha256(y, x)
    same_repeat = sanity_repeat_report([sanity_metrics, repeated_metrics])
    assert same_repeat['identical'] is True, same_repeat
    assert same_repeat['repeats'] == 2
    mismatched = dict(repeated_metrics)
    mismatched['final_lm_loss'] = float(repeated_metrics['final_lm_loss']) + 1.0
    changed_repeat = sanity_repeat_report([sanity_metrics, mismatched])
    assert changed_repeat['identical'] is False
    loss_mismatch = next((item for item in changed_repeat['mismatches'] if item['key'] == 'final_lm_loss'))
    assert loss_mismatch['repeat'] == 2
    assert loss_mismatch['repeat_value'] == mismatched['final_lm_loss']
    batch_mismatched = dict(repeated_metrics)
    batch_mismatched['batch_sha256'] = '0' * 64
    assert any((item['key'] == 'batch_sha256' for item in sanity_repeat_report([sanity_metrics, batch_mismatched])['mismatches']))
    assert sanity_repeat_report([sanity_metrics])['identical'] is None
    probe_model = LanguageModel(Config(**common, mode='B'))
    probe_result = probe_model(x, y)
    probe_result['loss'].backward()
    healthy_probe = _probe_gradient_flow(probe_model)
    assert not healthy_probe['failures'], healthy_probe
    expert_ids = _expert_parameter_ids(probe_model)
    broken_name = None
    for name, parameter in probe_model.named_parameters():
        if id(parameter) not in expert_ids and parameter.grad is not None:
            parameter.grad = None
            broken_name = name
            break
    assert broken_name is not None
    broken_probe = _probe_gradient_flow(probe_model)
    assert 'missing_non_expert_gradients' in broken_probe['failures'], broken_probe
    assert broken_name in broken_probe['missing_non_expert_params']
    zero_name = None
    for name, parameter in probe_model.named_parameters():
        if id(parameter) not in expert_ids and parameter.grad is not None and bool(parameter.grad.abs().sum() > 0):
            parameter.grad = torch.zeros_like(parameter.grad)
            zero_name = name
            break
    assert zero_name is not None
    zero_probe = _probe_gradient_flow(probe_model)
    assert 'zero_non_expert_gradients' in zero_probe['failures'], zero_probe
    assert zero_name in zero_probe['zero_non_expert_params']

def _selfcheck_top_55802_0040():
    global _GlobalRngDependentDataset, expected_next, failed_metrics, loaded_payload, metadata, numpy_payload, original_forward, poisoned_forward, poisoned_metrics, rng_dependent_metrics, rng_dir, rng_path
    if np is not None:
        np.random.seed(123)
        np.random.rand(3)
        numpy_payload = _numpy_rng_state_payload()
        expected_next = float(np.random.rand())
        _restore_numpy_rng_state(numpy_payload)
        assert float(np.random.rand()) == expected_next
        with tempfile.TemporaryDirectory() as rng_dir:
            rng_path = Path(rng_dir) / 'rng.pt'
            torch.save({'numpy_rng_state': numpy_payload}, rng_path)
            loaded_payload = torch.load(rng_path)
        _restore_numpy_rng_state(loaded_payload['numpy_rng_state'])
        assert float(np.random.rand()) == expected_next
    failed_metrics = run_sanity_overfit(sanity_cfg, eval_dataset, eval_device, amp_dtype=None, batch_size=2, steps=1, lr=1e-09, loss_threshold=0.0, acc_threshold=1.0, seed=321)
    assert not failed_metrics['passed']
    assert failed_metrics['learned'] is False
    assert failed_metrics['grad_checkpoint'] is False
    original_forward = LanguageModel.forward
    def poisoned_forward(self, inputs, targets=None):
        output = original_forward(self, inputs, targets)
        if targets is not None:
            output['loss'] = output['loss'] * float('nan')
        return output
    LanguageModel.forward = poisoned_forward
    try:
        poisoned_metrics = run_sanity_overfit(sanity_cfg, eval_dataset, eval_device, amp_dtype=None, batch_size=2, steps=1, lr=1e-09, loss_threshold=0.0, acc_threshold=1.0, seed=321)
    finally:
        LanguageModel.forward = original_forward
    assert not poisoned_metrics['passed']
    assert math.isnan(poisoned_metrics['final_total_loss'])
    assert math.isfinite(poisoned_metrics['final_lm_loss'])
    assert poisoned_metrics['loss_source'].startswith("result['loss']")
    class _GlobalRngDependentDataset:

        def batch(self, split, batch_size, generator, device):
            torch.rand(1)
            inputs = torch.zeros((batch_size, 4), dtype=torch.int32, device=device)
            targets = torch.zeros((batch_size, 4), dtype=torch.uint8, device=device)
            return (inputs, targets)
    rng_dependent_metrics = run_sanity_overfit(sanity_cfg, _GlobalRngDependentDataset(), eval_device, amp_dtype=None, batch_size=2, steps=1, lr=1e-09, loss_threshold=0.0, acc_threshold=1.0, seed=321)
    assert not rng_dependent_metrics['passed']
    assert rng_dependent_metrics['data_sampling_uses_explicit_generator'] is False, rng_dependent_metrics
    metadata = collect_run_metadata(torch.device('cpu'))
    assert 'git_commit' in metadata
    assert 'torch_version' in metadata
    assert 'deterministic_algorithms' in metadata
    assert 'deterministic_algorithms_warn_only' in metadata
    assert 'allow_tf32_matmul' in metadata
    assert 'allow_tf32_cudnn' in metadata
    assert 'float32_matmul_precision' in metadata
    assert 'allow_fp16_reduced_precision_reduction' in metadata
    assert 'allow_bf16_reduced_precision_reduction' in metadata
    assert metadata['cwd'] == os.getcwd()
    assert metadata['python_executable'] == sys.executable
    assert 'cuda_visible_devices' in metadata
    assert 'pytorch_cuda_alloc_conf' in metadata
    assert 'device_uuid' in metadata
    assert 'device_index' in metadata
    assert metadata['cpu_model']
    assert metadata['cpu_count'] is None or metadata['cpu_count'] > 0
    assert metadata['torch_num_threads'] == torch.get_num_threads()
    assert metadata['torch_num_interop_threads'] == torch.get_num_interop_threads()
    assert metadata['script_sha256'] == hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    assert 'git_diff_sha256' in metadata
    if metadata['git_dirty']:
        assert metadata['git_diff_sha256']

def _selfcheck_top_55949_0041():
    global bad_probe, baseline, cfg_dir, cfg_path, curve_probe, merged, overfit_analysis, safe_dir, safe_path, safe_records, spark
    assert metadata['pythonhashseed_at_start'] == _PYTHONHASHSEED_AT_START
    assert json.dumps(_json_safe({'a': float('nan'), 'b': [float('inf'), 1.0]})) == '{"a": null, "b": [null, 1.0]}'
    with tempfile.TemporaryDirectory() as safe_dir:
        safe_path = Path(safe_dir) / 'metrics.jsonl'
        write_jsonl_record(safe_path, {'step': 1, 'train_loss': float('nan')})
        safe_records = read_metrics_records(safe_path)
    assert safe_records[0]['train_loss'] is None
    with tempfile.TemporaryDirectory() as cfg_dir:
        cfg_path = Path(cfg_dir) / 'config.json'
        cfg_path.write_text(json.dumps({'config': {'mode': 'B'}}), encoding='utf-8')
        update_run_config(Path(cfg_dir), 'final', {'step': 'final', 'val_loss': 1.25})
        update_run_config(Path(cfg_dir), 'curve_analysis', {'status': 'healthy'})
        merged = json.loads(cfg_path.read_text(encoding='utf-8'))
    assert merged['config']['mode'] == 'B'
    assert merged['final']['val_loss'] == 1.25
    assert merged['curve_analysis']['status'] == 'healthy'
    assert _sparkline([], 48) == ''
    assert _sparkline([1.0, 2.0, 3.0], 0) == ''
    assert len(_sparkline([1.0, 2.0, 3.0], 1)) == 1
    spark = _sparkline(list(range(100)), 48)
    assert len(spark) == 48
    assert spark[0] == _SPARK_BLOCKS[0]
    assert spark[-1] == _SPARK_BLOCKS[-1]
    baseline = math.log(256.0)
    overfit_analysis = analyze_training_curve([{'step': 10, 'train_loss': 5.0, 'val_loss': 5.1}, {'step': 20, 'train_loss': 4.0, 'val_loss': 4.5}, {'step': 30, 'train_loss': 3.0, 'val_loss': 4.4}, {'step': 40, 'train_loss': 2.0, 'val_loss': 4.6}, {'step': 50, 'train_loss': 1.5, 'val_loss': 4.9}], baseline_loss=baseline)
    assert overfit_analysis['status'] == 'overfitting', overfit_analysis
    assert overfit_analysis['best_val_step'] == 30
    curve_probe = curve_diagnostics_report([{'run': 'overfit_run', 'curve_available': True, 'curve_status': 'overfitting', 'curve_best_val_step': 30, 'curve_val_regression_from_best': 0.05, 'curve_generalization_gap_ratio_last': 0.2}, {'run': 'old_run', 'curve_available': False}])
    assert curve_probe['n_runs_with_curve'] == 1
    assert curve_probe['overfitting_runs'] == ['overfit_run']
    assert curve_probe['runs'][0]['val_regression_from_best'] == 0.05
    assert curve_probe['runs'][0]['generalization_gap_ratio_last'] == 0.2
    bad_probe = bad_case_comparison_report([{'run': 'a', 'bad_case_available': True, 'bad_case_tokens_evaluated': 100, 'bad_case_wrong_tokens': 10, 'bad_case_token_accuracy': 0.9, 'bad_case_error_category_counts': {'ambiguous': 4, 'systematic_confusion': 3, 'label_noise_suspect': 2, 'uncertain': 1}, 'bad_case_error_category_shares': {'ambiguous': 0.4, 'systematic_confusion': 0.3, 'label_noise_suspect': 0.2, 'uncertain': 0.1}, 'bad_case_top_confusions': [[65, 66, 5], [67, 68, 2]]}, {'run': 'b', 'bad_case_available': True, 'bad_case_tokens_evaluated': 200, 'bad_case_wrong_tokens': 20, 'bad_case_token_accuracy': 0.9, 'bad_case_error_category_counts': {'ambiguous': 2, 'systematic_confusion': 8, 'label_noise_suspect': 5, 'uncertain': 5}, 'bad_case_error_category_shares': {'ambiguous': 0.1, 'systematic_confusion': 0.4, 'label_noise_suspect': 0.25, 'uncertain': 0.25}, 'bad_case_top_confusions': [[65, 66, 7], [67, 68, 1]]}])
    assert bad_probe['n_runs_with_bad_cases'] == 2
    assert bad_probe['wrong_tokens'] == 30
    assert bad_probe['error_category_counts']['ambiguous'] == 6
    assert bad_probe['error_category_shares']['ambiguous'] == 0.2
    assert bad_probe['persistent_confusion_min_runs'] == 2
    assert bad_probe['persistent_confusions'][0]['target'] == 65
    assert bad_probe['persistent_confusions'][0]['runs_present'] == 2

def _selfcheck_top_56095_0042():
    global descending_analysis, diverged_analysis, even_prior_records, finite_diverging_analysis, gap_widening_analysis, grad_record, grad_spike_analysis, grad_spike_records, grad_step, healthy_analysis, underfit_analysis
    assert bad_probe['unstable_categories']
    underfit_analysis = analyze_training_curve([{'step': 10, 'train_loss': 5.5, 'val_loss': 5.55}, {'step': 20, 'train_loss': 5.49, 'val_loss': 5.54}, {'step': 30, 'train_loss': 5.48, 'val_loss': 5.53}], baseline_loss=baseline)
    assert underfit_analysis['status'] == 'underfitting', underfit_analysis
    descending_analysis = analyze_training_curve([{'step': 10, 'train_loss': 5.0, 'val_loss': 5.2}, {'step': 20, 'train_loss': 4.2, 'val_loss': 4.4}, {'step': 30, 'train_loss': 3.4, 'val_loss': 3.6}], baseline_loss=baseline)
    assert descending_analysis['status'] == 'not_converged', descending_analysis
    healthy_analysis = analyze_training_curve([{'step': 10, 'train_loss': 5.0, 'val_loss': 5.1}, {'step': 20, 'train_loss': 4.0, 'val_loss': 4.3}, {'step': 30, 'train_loss': 3.3, 'val_loss': 3.45}, {'step': 40, 'train_loss': 3.1, 'val_loss': 3.46}], baseline_loss=baseline)
    assert healthy_analysis['status'] == 'healthy', healthy_analysis
    gap_widening_analysis = analyze_training_curve([{'step': 10, 'train_loss': 5.0, 'val_loss': 5.0}, {'step': 20, 'train_loss': 4.2, 'val_loss': 4.5}, {'step': 30, 'train_loss': 3.5, 'val_loss': 4.45}, {'step': 40, 'train_loss': 2.8, 'val_loss': 4.44}, {'step': 50, 'train_loss': 2.2, 'val_loss': 4.43}], baseline_loss=baseline)
    assert gap_widening_analysis['status'] == 'healthy', gap_widening_analysis
    assert gap_widening_analysis['gap_widening_warning'] is True, gap_widening_analysis
    assert any(('过拟合前兆' in warning for warning in gap_widening_analysis['warnings'])), gap_widening_analysis['warnings']
    assert '早停' in str(gap_widening_analysis['recommendation'])
    diverged_analysis = analyze_training_curve([{'step': 10, 'train_loss': 5.0, 'val_loss': 5.1}, {'step': 20, 'train_loss': None, 'val_loss': None}, {'step': 30, 'train_loss': 4.9, 'grad_norm': None}], baseline_loss=baseline)
    assert diverged_analysis['status'] == 'diverged', diverged_analysis
    assert diverged_analysis['n_nonfinite_events'] == 2
    finite_diverging_analysis = analyze_training_curve([{'step': 10, 'train_loss': 5.0, 'val_loss': 5.1}, {'step': 20, 'train_loss': 4.8, 'val_loss': 4.9}, {'step': 30, 'train_loss': 3.9, 'val_loss': 4.0}, {'step': 40, 'train_loss': 4.3, 'val_loss': 4.5}], baseline_loss=baseline)
    assert finite_diverging_analysis['status'] == 'diverging', finite_diverging_analysis
    assert finite_diverging_analysis['finite_diverging'] is True
    assert finite_diverging_analysis['warning_flags']['diverging'] is True
    assert '学习率' in str(finite_diverging_analysis['recommendation'])
    assert 'grad-clip' in str(finite_diverging_analysis['recommendation'])
    grad_spike_records = []
    for grad_step in range(1, 41):
        grad_record = {'step': grad_step, 'train_loss': 5.0 - 0.02 * grad_step, 'grad_norm': 10.0 if grad_step >= 33 else 1.0}
        if grad_step in (10, 20, 30, 40):
            grad_record['val_loss'] = 5.0 - 0.01 * grad_step
        grad_spike_records.append(grad_record)
    grad_spike_analysis = analyze_training_curve(grad_spike_records, baseline_loss=baseline)
    assert grad_spike_analysis['grad_norm_spike_warning'] is True, grad_spike_analysis
    assert grad_spike_analysis['warning_flags']['grad_spike'] is True
    assert grad_spike_analysis['grad_norm_spike_ratio'] >= _CURVE_GRAD_SPIKE_RATIO
    assert any(('梯度' in warning for warning in grad_spike_analysis['warnings']))
    even_prior_records = []

def _selfcheck_top_56245_0043():
    global _, early_status, even_prior_analysis, gap_initial, gap_initial_status, gap_monitor_records, gap_warning, gap_warning_status, grad_record, grad_step, monitor_records, monitored, monitored_status, recovered_analysis, repeated_gap_status, repeated_status
    for grad_step in range(1, 41):
        grad_record = {'step': grad_step, 'train_loss': 5.0 - 0.02 * grad_step, 'grad_norm': float(grad_step) if grad_step <= 32 else 1000.0}
        if grad_step in (10, 20, 30, 40):
            grad_record['val_loss'] = 5.0 - 0.01 * grad_step
        even_prior_records.append(grad_record)
    even_prior_analysis = analyze_training_curve(even_prior_records, baseline_loss=baseline)
    assert even_prior_analysis['prior_grad_norm_median'] == 16.5, even_prior_analysis['prior_grad_norm_median']
    recovered_analysis = analyze_training_curve([{'step': 5, 'train_loss': None, 'grad_norm': None, 'val_loss': 5.0}, {'step': 6, 'train_loss': None, 'grad_norm': None, 'val_loss': 4.5}, {'step': 7, 'train_loss': None, 'grad_norm': None, 'val_loss': 4.0}, {'step': 8, 'train_loss': 3.5, 'grad_norm': 1.0}, {'step': 9, 'train_loss': 3.4, 'grad_norm': 1.0}], baseline_loss=baseline)
    assert recovered_analysis['status'] == 'healthy', recovered_analysis
    assert recovered_analysis['last_train_loss'] is None
    print_curve_analysis(recovered_analysis)
    monitor_records = [{'step': 10, 'train_loss': 5.0, 'val_loss': 5.1}, {'step': 20, 'train_loss': 4.0, 'val_loss': 4.5}, {'step': 30, 'train_loss': 3.0, 'val_loss': 4.4}, {'step': 40, 'train_loss': 2.0, 'val_loss': 4.6}, {'step': 50, 'train_loss': 1.5, 'val_loss': 4.9}]
    monitored, monitored_status = monitor_training_curve(monitor_records, baseline, None)
    assert monitored_status == 'overfitting', monitored
    assert monitored['status'] == 'overfitting'
    _, repeated_status = monitor_training_curve(monitor_records, baseline, 'overfitting')
    assert repeated_status is None
    _, early_status = monitor_training_curve([{'step': 10, 'train_loss': 5.0, 'val_loss': 5.1}, {'step': 20, 'train_loss': 4.8, 'val_loss': 4.9}], baseline, None)
    assert early_status is None
    gap_monitor_records = [{'step': 10, 'train_loss': 4.0, 'val_loss': 5.0}, {'step': 20, 'train_loss': 3.8, 'val_loss': 4.8}, {'step': 30, 'train_loss': 3.6, 'val_loss': 4.6}, {'step': 40, 'train_loss': 2.8, 'val_loss': 4.5}, {'step': 50, 'train_loss': 2.0, 'val_loss': 4.45}]
    gap_initial, gap_initial_status = monitor_training_curve(gap_monitor_records[:3], baseline, None)
    assert gap_initial_status == 'healthy', gap_initial
    assert gap_initial['warning_flags'] == {'nonfinite': False, 'gap_widening': False, 'diverging': False, 'grad_spike': False}, gap_initial
    gap_warning, gap_warning_status = monitor_training_curve(gap_monitor_records[:4], baseline, 'healthy', gap_initial['warning_flags'])
    assert gap_warning_status == 'healthy', gap_warning
    assert gap_warning['monitor_trigger'] == 'warning_change', gap_warning
    assert gap_warning['warning_flags']['gap_widening'] is True
    assert 'gap_widening' in gap_warning['new_warning_flags'], gap_warning
    _, repeated_gap_status = monitor_training_curve(gap_monitor_records[:5], baseline, 'healthy', gap_warning['warning_flags'])
    assert repeated_gap_status is None

def _selfcheck_top_56385_0044():
    global aux_nan_result, bad_data_check, diag_cfg, diag_model, diag_optimizer, diag_param, diag_report, diag_result, early_best, early_best_step, early_no_improve, recompute_check, uint8_range_check
    early_best, early_best_step, early_no_improve = early_stop_state_from_records([{'step': 10, 'val_loss': 5.0}, {'step': 20, 'val_loss': 4.0}, {'step': 30, 'val_loss': 4.1}, {'step': 40, 'val_loss': 4.05}], min_delta=0.0)
    assert early_best == 4.0
    assert early_best_step == 20
    assert early_no_improve == 2
    early_best, early_best_step, early_no_improve = early_stop_state_from_records([{'step': 10, 'val_loss': 5.0}, {'step': 20, 'val_loss': 4.0}, {'step': 'final', 'val_loss': 9.9}], min_delta=0.0)
    assert early_best == 4.0
    assert early_best_step == 20
    assert early_no_improve == 0
    assert selection_improved(4.0, 5.0, False, 0.0)
    assert not selection_improved(5.0, 5.0, False, 0.0)
    assert not selection_improved(4.9, 5.0, False, 0.2)
    assert selection_improved(0.5, 0.4, True, 0.0)
    assert not selection_improved(0.4, 0.4, True, 0.0)
    assert not selection_improved(float('nan'), 0.4, True, 0.0)
    assert selection_improved(0.62, 0.6, True, 0.01)
    assert not selection_improved(0.61, 0.6, True, 0.01)
    early_best, early_best_step, early_no_improve = early_stop_state_from_records([{'step': 10, 'val_macro_f1': 0.4}, {'step': 20, 'val_macro_f1': 0.55}, {'step': 30, 'val_macro_f1': 0.54}, {'step': 40}, {'step': 50, 'val_macro_f1': 0.56}, {'step': 60, 'val_macro_f1': 0.555}], min_delta=0.0, metric='val_macro_f1')
    assert abs(early_best - 0.56) < 1e-12
    assert early_best_step == 50
    assert early_no_improve == 1
    bad_data_check = _check_batch_integrity(torch.tensor([[0, 300]], dtype=torch.int32), torch.tensor([[1, 2]], dtype=torch.uint8), 256)
    assert not bad_data_check['ok']
    uint8_range_check = _check_batch_integrity(torch.tensor([[0, 255]], dtype=torch.uint8), torch.tensor([[0, 255]], dtype=torch.uint8), 256)
    assert uint8_range_check['ok'], uint8_range_check
    diag_cfg = Config(**common, mode='B')
    diag_model = LanguageModel(diag_cfg)
    diag_model.train()
    diag_optimizer = torch.optim.AdamW(diag_model.parameters(), lr=0.001)
    with torch.enable_grad():
        diag_result = diag_model(x, y)
        diag_result['loss'].backward()
    diag_param = next(diag_model.parameters())
    assert diag_param.grad is not None
    diag_param.grad.fill_(float('nan'))
    diag_report = diagnose_non_finite(diag_model, diag_optimizer, x, y, diag_result, torch.tensor(float('nan')), step=1, lr=0.001, amp_dtype=None, vocab_size=256)
    assert diag_report['checks']['data']['ok']
    assert diag_report['checks']['gradients']['n_nonfinite_grad_params'] >= 1
    recompute_check = diag_report['checks']['forward']['recompute_loss_fp32']
    assert recompute_check['available'] is True
    assert recompute_check['target_range_ok'] is True
    assert recompute_check['finite'] is True
    assert recompute_check['log_zero_probability_tokens'] == 0
    assert recompute_check['lm_loss'] is not None
    aux_nan_result = dict(diag_result)
    aux_nan_result['aux_loss'] = torch.tensor(float('nan'))

def _selfcheck_top_56521_0045():
    global _ScaleProbe, amp_check, amp_report, aux_nan_report, bad_case_model, bad_case_report, even_trend_report, forward_side_report, nan_loss_result, trend, trend_report, valid_categories
    aux_nan_report = diagnose_non_finite(diag_model, diag_optimizer, x, y, aux_nan_result, torch.tensor(float('nan')), step=1, lr=0.001, amp_dtype=None, vocab_size=256)
    assert any(('辅助负载均衡损失' in finding for finding in aux_nan_report['findings'])), aux_nan_report['findings']
    diag_model.zero_grad(set_to_none=True)
    trend_report = diagnose_non_finite(diag_model, diag_optimizer, x, y, diag_result, torch.tensor(float('nan')), step=2, lr=0.001, amp_dtype=None, vocab_size=256, recent_grad_norms=[(1, 1.0), (2, 1.1), (3, 0.9)])
    trend = trend_report['checks']['gradient_trend']
    assert trend['recent_median'] == 1.0, trend
    assert any(('符合梯度爆炸' in finding for finding in trend_report['findings'])), trend_report['findings']
    even_trend_report = diagnose_non_finite(diag_model, diag_optimizer, x, y, diag_result, torch.tensor(float('nan')), step=2, lr=0.001, amp_dtype=None, vocab_size=256, recent_grad_norms=[(1, 1.0), (2, 2.0), (3, 3.0), (4, 3.0)])
    assert even_trend_report['checks']['gradient_trend']['recent_median'] == 2.5, even_trend_report['checks']['gradient_trend']
    nan_loss_result = dict(diag_result)
    nan_loss_result['loss'] = torch.tensor(float('nan'))
    forward_side_report = diagnose_non_finite(diag_model, diag_optimizer, x, y, nan_loss_result, torch.tensor(2.0), step=3, lr=0.001, amp_dtype=None, vocab_size=256, recent_grad_norms=[(2, 1.0), (3, 1.1)])
    assert any(('损失侧' in finding for finding in forward_side_report['findings'])), forward_side_report['findings']
    assert not any(('符合梯度爆炸' in finding for finding in forward_side_report['findings'])), forward_side_report['findings']
    class _ScaleProbe:

        def __init__(self, scale: float):
            self.scale = scale

        def get_scale(self) -> float:
            return self.scale
    amp_report = diagnose_non_finite(diag_model, diag_optimizer, x, y, diag_result, torch.tensor(float('nan')), step=4, lr=0.001, amp_dtype=torch.float16, scaler=_ScaleProbe(32768.0), amp_scale_before_step=65536.0, vocab_size=256)
    amp_check = amp_report['checks']['amp']
    assert amp_check['scale_reduced_on_failure'] is True
    assert amp_check['grad_scale_before_step'] == 65536.0
    assert amp_check['grad_scale_at_diagnosis'] == 32768.0
    assert amp_check['scale_ratio_to_before'] == 0.5
    assert any(('GradScaler' in finding and '溢出' in finding for finding in amp_report['findings'])), amp_report['findings']
    bad_case_model = LanguageModel(Config(**common, mode='B')).eval()
    bad_case_report = analyze_bad_cases(bad_case_model, eval_dataset, batch_size=2, batches=2, device=eval_device, amp_dtype=None, split='val', top_n=3)
    valid_categories = set(_BAD_CASE_CATEGORIES)

def _selfcheck_top_56667_0046():
    global _, ambiguous_evidence, ambiguous_target_share, ambiguous_verified, calibration, calibration_probe, case, concentration, context_evidence, dominant_evidence, dominant_verified, empty_bad_case, error_counts, label_noise_check, unique_windows
    empty_bad_case = _empty_bad_case_report('val')
    assert empty_bad_case['tokens_evaluated'] == 0
    assert empty_bad_case['wrong_tokens'] == 0
    assert empty_bad_case['top_cases'] == []
    assert set(empty_bad_case['error_category_counts']) == valid_categories
    assert set(empty_bad_case['error_category_shares']) == valid_categories
    assert empty_bad_case['calibration']['ece'] is None
    assert empty_bad_case['label_noise_verification']['candidates'] == 0
    assert empty_bad_case['top_cases_unique_windows'] == 0
    assert empty_bad_case['top_cases_window_concentration'] is None
    assert not empty_bad_case['top_cases_concentration_warning']
    assert bad_case_report['tokens_evaluated'] > 0
    assert bad_case_report['split'] == 'val'
    assert bad_case_report['token_accuracy'] is not None
    assert 0.0 <= float(bad_case_report['token_accuracy']) <= 1.0
    assert len(bad_case_report['top_cases']) == min(3, int(bad_case_report['tokens_evaluated']))
    for case in bad_case_report['top_cases']:
        assert case['category'] in valid_categories
        assert case['context']
        assert 0.0 <= float(case['confidence']) <= 1.0
        assert 1 <= int(case['position']) <= int(eval_dataset.seq_len)
        assert int(case['window_repeat_count']) >= 1
    unique_windows = int(bad_case_report['top_cases_unique_windows'])
    assert 1 <= unique_windows <= len(bad_case_report['top_cases'])
    concentration = bad_case_report['top_cases_window_concentration']
    assert 0.0 < float(concentration) <= 1.0
    assert set(bad_case_report['category_counts'].keys()) <= valid_categories
    calibration = bad_case_report['calibration']
    assert 0.0 <= float(calibration['ece']) <= 1.0
    assert -1.0 <= float(calibration['overconfidence']) <= 1.0
    assert calibration['bins']
    assert int(calibration['high_confidence_error_count']) >= 0
    label_noise_check = bad_case_report['label_noise_verification']
    assert label_noise_check['context_length'] == _BAD_CASE_LABEL_CONTEXT
    assert label_noise_check['min_majority_share_in_context'] == _BAD_CASE_LABEL_MAJORITY_MIN_SHARE
    ambiguous_evidence = {'support': 20, 'majority_byte': 66, 'majority_share': 0.11, 'counts': {66: 2, 67: 2, 68: 2, 69: 2, 70: 2, 71: 2, 72: 2, 73: 2, 74: 2, 65: 2}}
    ambiguous_verified, ambiguous_target_share = _label_noise_context_verdict(ambiguous_evidence, 65)
    assert ambiguous_verified is False
    assert abs(ambiguous_target_share - 0.1) < 1e-09
    dominant_evidence = {'support': 20, 'majority_byte': 66, 'majority_share': 0.8, 'counts': {66: 16, 65: 1, 67: 1, 68: 1, 69: 1}}
    dominant_verified, _ = _label_noise_context_verdict(dominant_evidence, 65)
    assert dominant_verified is True
    for case in bad_case_report['top_cases']:
        assert 'label_noise_verified' in case
        assert 'train_context_support' in case
        assert 'train_target_share' in case
        if case['category'] == 'label_noise_suspect':
            assert case['label_noise_verified'] is True, case
    calibration_probe = _calibration_report(torch.tensor([0.9, 0.9, 0.1, 0.1]), torch.tensor([1, 0, 0, 1], dtype=torch.bool))
    assert abs(calibration_probe['ece'] - 0.4) < 1e-06, calibration_probe
    assert abs(calibration_probe['overconfidence'] - 0.0) < 1e-06
    context_evidence = _train_next_byte_evidence(eval_dataset, [bytes([10, 11]), bytes([255, 254])])
    assert context_evidence[bytes([10, 11])]['support'] == 1
    assert context_evidence[bytes([10, 11])]['majority_byte'] == 12
    assert context_evidence[bytes([255, 254])]['support'] == 0
    error_counts = bad_case_report['error_category_counts']
    assert set(error_counts.keys()) == valid_categories
    assert sum(error_counts.values()) == int(bad_case_report['wrong_tokens'])

def _selfcheck_top_56811_0047():
    global _FakeBadCaseDataset, bad_case_eval, concentrated_probe, frequency_rows, position_bucket, position_rows, rare_bucket, row, shared_window, stratified_probe
    if bad_case_report['wrong_tokens'] > 0:
        assert abs(sum((float(share) for share in bad_case_report['error_category_shares'].values())) - 1.0) < 1e-06
    bad_case_eval = evaluate(bad_case_model, eval_dataset, batch_size=2, batches=2, device=eval_device, amp_dtype=None)
    assert abs(float(bad_case_report['token_accuracy']) - float(bad_case_eval['val_token_accuracy'])) < 1e-06, (bad_case_report['token_accuracy'], bad_case_eval['val_token_accuracy'])
    assert bad_case_report['tokens_evaluated'] >= 1
    position_rows = bad_case_report['error_rate_by_position']
    frequency_rows = bad_case_report['error_rate_by_train_frequency']
    if position_rows:
        assert sum((int(row['tokens']) for row in position_rows)) == int(bad_case_report['tokens_evaluated'])
        assert sum((int(row['wrong']) for row in position_rows)) == int(bad_case_report['wrong_tokens'])
        for row in position_rows:
            assert 0.0 <= float(row['error_rate']) <= 1.0
    if frequency_rows:
        assert sum((int(row['tokens']) for row in frequency_rows)) == int(bad_case_report['tokens_evaluated'])
        assert sum((int(row['wrong']) for row in frequency_rows)) == int(bad_case_report['wrong_tokens'])
        assert bad_case_report['error_frequency_basis'] in ('train_target_share', 'eval_target_share')
    class _FakeBadCaseDataset:
        seq_len = 4
        effective_train_bytes = 1000

        @staticmethod
        def effective_train_byte_counts() -> torch.Tensor:
            counts = torch.zeros(256, dtype=torch.int64)
            counts[0] = 999
            counts[1] = 1
            return counts
    stratified_probe = _bad_case_report_from_buffers(torch.zeros(8), torch.tensor([0, 0, 1, 1, 0, 1, 0, 1], dtype=torch.uint8), torch.tensor([0, 0, 1, 0, 0, 0, 0, 1], dtype=torch.int16), torch.full((8,), 0.9), torch.zeros(8), 8, [], 'val', 0, 16, 8, 256, dataset=_FakeBadCaseDataset())
    assert stratified_probe['wrong_tokens'] == 2
    position_bucket = next((row for row in stratified_probe['error_rate_by_position'] if row['name'] == 'position_1_3'))
    assert position_bucket['tokens'] == 6, position_bucket
    assert position_bucket['wrong'] == 2, position_bucket
    assert abs(float(position_bucket['error_rate']) - 2.0 / 6.0) < 1e-06, position_bucket
    rare_bucket = next((row for row in stratified_probe['error_rate_by_train_frequency'] if row['name'] == 'train_share_0.1pct_to_1pct'))
    assert rare_bucket['tokens'] == 4, rare_bucket
    assert rare_bucket['wrong'] == 2, rare_bucket
    assert stratified_probe['error_frequency_basis'] == 'train_target_share'
    shared_window = torch.arange(6, dtype=torch.uint8)
    concentrated_probe = _bad_case_report_from_buffers(torch.tensor([5.0, 4.0, 3.0]), torch.tensor([0, 0, 0], dtype=torch.uint8), torch.tensor([1, 1, 1], dtype=torch.int16), torch.full((3,), 0.9), torch.full((3,), 0.5), 3, [(5.0, 0, shared_window, 0, 0, 1, 0.9, 0.5), (4.0, 1, shared_window, 1, 0, 1, 0.9, 0.5), (3.0, 2, shared_window, 2, 0, 1, 0.9, 0.5)], 'val', 3, 16, 8, 256, dataset=None)

def _selfcheck_top_56957_0048():
    global efficiency, efficiency_model, even_efficiency, expected_attention_flops, expected_n, original_probe_step, probe_attention_step, probe_generator, probe_module, probe_modules, probe_x, probe_y, scalar_key, summary, summary_key
    assert concentrated_probe['top_cases_unique_windows'] == 1
    assert concentrated_probe['top_cases_window_concentration'] == 1.0
    assert concentrated_probe['top_cases_concentration_warning']
    assert {case['position'] for case in concentrated_probe['top_cases']} == {1, 2, 3}
    assert all((case['window_repeat_count'] == 3 for case in concentrated_probe['top_cases']))
    efficiency_model = LanguageModel(Config(**common, mode='B'))
    efficiency = measure_efficiency(efficiency_model, eval_dataset, eval_device, batch_size=2, amp_dtype=None, warmup=0, iters=3)
    assert efficiency['n_params_total'] == sum((parameter.numel() for parameter in efficiency_model.parameters()))
    assert efficiency['n_params_non_embedding'] <= efficiency['n_params_total']
    assert efficiency['latency_ms_per_forward'] is not None
    assert float(efficiency['latency_ms_per_forward']) > 0.0
    assert efficiency['training_ms_per_step'] is not None
    assert float(efficiency['training_ms_per_step']) > 0.0
    assert '不含优化器状态' in efficiency['training_peak_memory_note']
    for summary_key, expected_n in (('latency_ms', 3), ('training_ms', 3)):
        summary = efficiency[summary_key]
        assert summary['n'] == expected_n, summary
        assert summary['median'] is not None
        assert summary['p95'] >= summary['median']
        assert summary['p99'] >= summary['p95']
        assert summary['std'] >= 0.0
        assert summary['max'] >= summary['min']
        if summary['mean'] and summary['mean'] > 0:
            assert summary['coefficient_of_variation'] is not None
    assert len(efficiency['timing_samples_seconds']['inference']) == 3
    assert len(efficiency['timing_samples_seconds']['training']) == 3
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert _percentile([], 0.95) is None
    assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5
    even_efficiency = measure_efficiency(efficiency_model, eval_dataset, eval_device, batch_size=2, amp_dtype=None, warmup=0, iters=4)
    for summary_key, scalar_key in (('latency_ms', 'latency_ms_per_forward'), ('training_ms', 'training_ms_per_step')):
        summary = even_efficiency[summary_key]
        assert summary['n'] == 4
        assert abs(float(summary['median']) - float(even_efficiency[scalar_key])) < 0.001, (summary, even_efficiency[scalar_key])
    assert efficiency['flops_per_forward'] is None or int(efficiency['flops_per_forward']) > 0
    assert efficiency['flops_attention'] is None or int(efficiency['flops_attention']) > 0
    expected_attention_flops = [0]
    probe_modules = []
    for probe_module in efficiency_model.modules():
        if not isinstance(probe_module, TransformerBlock):
            continue
        original_probe_step = probe_module.attention_step

        def probe_attention_step(inputs: torch.Tensor, *args, _original=original_probe_step, **kwargs) -> torch.Tensor:
            expected_attention_flops[0] += 4 * int(inputs.size(-2)) * int(inputs.numel())
            return _original(inputs, *args, **kwargs)
        probe_module.attention_step = probe_attention_step
        probe_modules.append(probe_module)
    probe_generator = torch.Generator().manual_seed(0)
    probe_x, probe_y = eval_dataset.batch('train', 2, probe_generator, eval_device)

def _selfcheck_top_57096_0049():
    global ckpt_args, ckpt_dir, ckpt_generator, ckpt_model, ckpt_optimizer, ckpt_path, ckpt_scheduler, compare_dir, comparison_rows, f, name, polluted_payload, probe_module, run_dir, saved_payload, val_loss
    try:
        with torch.no_grad():
            efficiency_model(probe_x, probe_y)
    finally:
        for probe_module in probe_modules:
            del probe_module.attention_step
    assert int(efficiency['flops_attention']) == expected_attention_flops[0]
    assert efficiency['measurement']['mode'] == 'B'
    with tempfile.TemporaryDirectory() as ckpt_dir:
        ckpt_model = LanguageModel(Config(**common, mode='A'))
        ckpt_optimizer = torch.optim.AdamW(ckpt_model.parameters(), lr=0.001)
        ckpt_scheduler = build_lr_schedule(ckpt_optimizer, 0, 1)
        ckpt_path = Path(ckpt_dir) / 'last.pt'
        ckpt_generator = torch.Generator().manual_seed(0)
        ckpt_args = argparse.Namespace(seed=7)
        assert save_checkpoint(str(ckpt_path), ckpt_model, ckpt_optimizer, None, ckpt_scheduler, 0, ckpt_generator.get_state(), ckpt_args, numerical_state={'nonfinite_events': 2, 'numerical_stop': True})
        saved_payload = load_checkpoint(str(ckpt_path), torch.device('cpu'))
        assert saved_payload['numerical_state']['nonfinite_events'] == 2
        assert _checkpoint_numerical_error(saved_payload) is None
        with torch.no_grad():
            next(ckpt_model.parameters()).view(-1)[0] = float('nan')
        assert not save_checkpoint(str(ckpt_path), ckpt_model, ckpt_optimizer, None, ckpt_scheduler, 1, ckpt_generator.get_state(), ckpt_args)
        assert not Path(str(ckpt_path) + '.tmp').exists()
        assert load_checkpoint(str(ckpt_path), torch.device('cpu'))['step'] == 0
    polluted_payload = {'model': {'weight': torch.tensor([1.0, float('nan')])}, 'optimizer': {'state': {}}}
    assert 'NaN/Inf' in str(_checkpoint_numerical_error(polluted_payload))
    assert _checkpoint_numerical_error({'model': {'weight': torch.tensor([1.0, 2.0])}}) is None
    assert _checkpoint_numerical_error({'model': {'step': torch.tensor([3], dtype=torch.int64)}}) is None
    with tempfile.TemporaryDirectory() as compare_dir:
        for name, val_loss in (('run_a', 3.0), ('run_b', 2.9)):
            run_dir = Path(compare_dir) / name
            run_dir.mkdir()
            with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
                json.dump({'config': {'mode': 'B'}, 'n_params': 100, 'efficiency': {'latency_ms_per_forward': 1.5}}, f)
            write_jsonl_record(run_dir / 'metrics.jsonl', {'step': 'final', 'val_loss': val_loss, 'val_ppl': 10.0})
        comparison_rows = compare_runs([str(Path(compare_dir) / 'run_a'), str(Path(compare_dir) / 'run_b')])
    assert len(comparison_rows) == 2
    assert comparison_rows[0]['val_loss'] == 3.0
    assert comparison_rows[1]['mode'] == 'B'
    assert comparison_rows[0]['latency_ms_per_forward'] == 1.5

def _selfcheck_top_57233_0050():
    global f, key, malformed_dir, malformed_routing_dir, malformed_rows, malformed_run, routing_rows, routing_run
    with tempfile.TemporaryDirectory() as malformed_dir:
        malformed_run = Path(malformed_dir) / 'malformed'
        malformed_run.mkdir()
        with open(malformed_run / 'config.json', 'w', encoding='utf-8') as f:
            json.dump({'config': ['not', 'a', 'dict'], 'args': {'seed': 1}}, f)
        write_jsonl_record(malformed_run / 'metrics.jsonl', {'step': 'final', 'val_loss': 3.1})
        malformed_rows = compare_runs([str(malformed_run)])
    assert len(malformed_rows) == 1
    assert malformed_rows[0].get('mode') is None
    assert malformed_rows[0]['selection_metric'] == 'val_loss'
    with tempfile.TemporaryDirectory() as malformed_routing_dir:
        routing_run = Path(malformed_routing_dir) / 'routing'
        routing_run.mkdir()
        with open(routing_run / 'config.json', 'w', encoding='utf-8') as f:
            json.dump({'config': {'mode': 'A'}, 'args': {'seed': 1}, 'routing_analysis': {'rq3_expert_overlap': ['not', 'a', 'dict'], 'rq4_load_and_entropy': {'summary': {'round1': 'bad', 'round2': 7}}, 'branch_divergence': {'summary': 3}}}, f)
        write_jsonl_record(routing_run / 'metrics.jsonl', {'step': 'final', 'val_loss': 3.2})
        routing_rows = compare_runs([str(routing_run)])
    assert routing_rows[0]['routing_available'] is True
    for key in _ROUTING_ROW_FIELDS:
        assert routing_rows[0][key] is None
    assert 'val_loss_at_matched_flops' in COMPARISON_METRIC_KEYS
    assert _COMPUTE_MATCHED_METRIC not in _BASELINE_TUNING_METRIC_KEYS

def _selfcheck_top_57307_0051():
    global by_run, curve, effective_dir, effective_path, effective_points, f, failed_dir, flops, loss, matched_dir, matched_paths, matched_report, matched_rows, name, run_dir, specs, step
    with tempfile.TemporaryDirectory() as matched_dir:
        specs = (('cheap', 100.0, [(10, 5.0), (20, 4.0), (30, 3.0)]), ('expensive', 150.0, [(10, 4.8), (25, 3.7), (30, 2.9)]))
        matched_paths = []
        for name, flops, curve in specs:
            run_dir = Path(matched_dir) / name
            run_dir.mkdir()
            with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
                json.dump({'config': {'mode': name}, 'efficiency': {'flops_per_forward': flops}}, f)
            for step, loss in curve:
                write_jsonl_record(run_dir / 'metrics.jsonl', {'step': step, 'val_loss': loss})
            write_jsonl_record(run_dir / 'metrics.jsonl', {'step': 'final', 'val_loss': 2.0})
            matched_paths.append(str(run_dir))
        failed_dir = Path(matched_dir) / 'failed_early'
        failed_dir.mkdir()
        with open(failed_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump({'config': {'mode': 'failed_early'}, 'efficiency': {'flops_per_forward': 100.0}}, f)
        write_jsonl_record(failed_dir / 'metrics.jsonl', {'step': 5, 'val_loss': 9.0})
        write_jsonl_record(failed_dir / 'metrics.jsonl', {'step': 10, 'val_loss': 8.0})
        write_jsonl_record(failed_dir / 'metrics.jsonl', {'step': 'curve_analysis', 'status': 'diverged'})
        write_jsonl_record(failed_dir / 'metrics.jsonl', {'step': 'final', 'val_loss': 8.0})
        matched_paths.append(str(failed_dir))
        matched_rows = compare_runs(matched_paths)
        matched_report = annotate_compute_matched_metrics(matched_rows)
    assert matched_report['ok'], matched_report
    assert abs(float(matched_report['target_flops']) - 9000.0) < 1e-09
    by_run = {str(row['run']): row for row in matched_rows}
    assert abs(float(by_run['cheap']['val_loss_at_matched_flops']) - 3.0) < 1e-09
    assert abs(float(by_run['expensive']['val_loss_at_matched_flops']) - 4.066666666666667) < 1e-09
    assert abs(float(by_run['expensive']['matched_compute_ratio']) - 2.0 / 3.0) < 1e-12
    assert any((item['interpolation'] == 'linear_between_evals' for item in matched_report['runs']))
    assert 'val_loss_at_matched_flops' not in by_run['failed_early']
    assert 'failed_early' in {str(item['run']) for item in matched_report['excluded_runs']}
    with tempfile.TemporaryDirectory() as effective_dir:
        effective_path = Path(effective_dir) / 'metrics.jsonl'
        write_jsonl_record(effective_path, {'step': 10, 'optimizer_steps': 5, 'val_loss': 5.0})
        write_jsonl_record(effective_path, {'step': 20, 'optimizer_steps': 10, 'val_loss': 4.0})
        effective_points = _val_curve_points(Path(effective_dir), 100.0)
        assert effective_points == [(1500.0, 5.0), (3000.0, 4.0)], effective_points
    assert 'val_loss_at_common_step' in COMPARISON_METRIC_KEYS

def _selfcheck_top_57445_0052():
    global by_run, common_dir, common_paths, common_report, common_rows, curve, f, failed_dir, final_loss, last_step, loss, name, run_dir, specs, step
    assert 'val_loss_at_common_step' not in _SELECTION_METRIC_KEYS
    assert 'val_loss_at_common_step' not in _NGRAM_TUNING_METRIC_KEYS
    with tempfile.TemporaryDirectory() as common_dir:
        specs = (('short', [(10, 5.0), (20, 4.5)], 9.0, 20), ('long', [(10, 4.8), (20, 4.2), (30, 4.0), (40, 3.9)], 2.0, 40))
        common_paths = []
        for name, curve, final_loss, last_step in specs:
            run_dir = Path(common_dir) / name
            run_dir.mkdir()
            with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
                json.dump({'config': {'mode': 'A'}, 'args': {'seed': 1}}, f)
            for step, loss in curve:
                write_jsonl_record(run_dir / 'metrics.jsonl', {'step': step, 'val_loss': loss})
            write_jsonl_record(run_dir / 'metrics.jsonl', {'step': 'final', 'val_loss': final_loss, 'selected_model': 'best_val', 'selected_step': last_step, 'trained_steps': last_step})
            common_paths.append(str(run_dir))
        failed_dir = Path(common_dir) / 'failed_early'
        failed_dir.mkdir()
        with open(failed_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump({'config': {'mode': 'A'}, 'args': {'seed': 2}}, f)
        write_jsonl_record(failed_dir / 'metrics.jsonl', {'step': 5, 'val_loss': 9.0})
        write_jsonl_record(failed_dir / 'metrics.jsonl', {'step': 10, 'val_loss': 8.0})
        write_jsonl_record(failed_dir / 'metrics.jsonl', {'step': 'curve_analysis', 'status': 'diverged'})
        write_jsonl_record(failed_dir / 'metrics.jsonl', {'step': 'final', 'val_loss': 8.0, 'selected_model': 'best_val', 'selected_step': 10, 'trained_steps': 10})
        common_paths.append(str(failed_dir))
        common_rows = compare_runs(common_paths)
        common_report = annotate_common_step_metrics(common_rows)
    assert common_report['ok'], common_report
    assert abs(float(common_report['target_step']) - 20.0) < 1e-09
    by_run = {str(row['run']): row for row in common_rows}
    assert abs(float(by_run['short']['val_loss_at_common_step']) - 4.5) < 1e-09
    assert abs(float(by_run['long']['val_loss_at_common_step']) - 4.2) < 1e-09
    assert 'val_loss_at_common_step' not in by_run['failed_early']
    assert 'failed_early' in {str(item['run']) for item in common_report['excluded_runs']}
    assert by_run['short']['val_loss_at_common_step'] != by_run['short']['val_loss']
    assert by_run['long']['val_loss_at_common_step'] != by_run['long']['val_loss']

def _selfcheck_top_57578_0053():
    global balanced_opportunity, bias_report, failed_bias, failed_bias_rows, imbalance, mixed_bias, mixed_rows, opportunity_bias, partial_opportunity
    bias_report = selection_bias_report(common_rows, 'val_loss', common_report)
    assert bias_report['applicable']
    assert len(bias_report['biased_runs']) == 2
    assert bias_report['unbiased_metric'] == 'val_loss_at_common_step'
    assert bias_report['unbiased_runs'] == 2
    assert bias_report['advisory']
    assert bias_report['eligible_runs'] == 2
    assert bias_report['selection_opportunities'] == []
    assert bias_report['opportunity_advisory'] is not None
    common_rows[0]['curve_n_eval_points'] = 3
    common_rows[1]['curve_n_eval_points'] = 5
    opportunity_bias = selection_bias_report(common_rows, 'val_loss', common_report)
    assert len(opportunity_bias['selection_opportunities']) == 2
    imbalance = opportunity_bias['opportunity_imbalance']
    assert imbalance is not None
    assert imbalance['min_eval_points'] == 3
    assert imbalance['max_eval_points'] == 5
    assert abs(float(imbalance['ratio']) - 5.0 / 3.0) < 1e-12
    assert '验证点数不同' in str(opportunity_bias['opportunity_advisory'])
    common_rows[1]['curve_n_eval_points'] = 3
    balanced_opportunity = selection_bias_report(common_rows, 'val_loss', common_report)
    assert balanced_opportunity['opportunity_imbalance'] is None
    common_rows[1]['curve_n_eval_points'] = 5
    common_rows[1].pop('curve_n_eval_points', None)
    partial_opportunity = selection_bias_report(common_rows, 'val_loss', common_report)
    assert len(partial_opportunity['selection_opportunities']) == 1
    assert partial_opportunity['missing_selection_opportunity_runs'] == ['long']
    assert '无法完整核对' in str(partial_opportunity['opportunity_advisory'])
    common_rows[1]['curve_n_eval_points'] = 5
    mixed_rows = list(common_rows) + [dict(common_rows[0], run='ngram-baseline', mode='ngram')]
    mixed_bias = selection_bias_report(mixed_rows, 'val_loss', common_report)
    assert mixed_bias['eligible_runs'] == 2
    assert len(mixed_bias['biased_runs']) == 2
    assert '/2 个运行' in str(mixed_bias['advisory'])
    failed_bias_rows = [row for row in common_rows if row.get('numerical_status') != 'failed'] + [dict(common_rows[0], run='failed-baseline', numerical_status='failed')]
    failed_bias = selection_bias_report(failed_bias_rows, 'val_loss', common_report)
    assert failed_bias['eligible_runs'] == 2
    assert len(failed_bias['biased_runs']) == 2
    assert failed_bias['failed_runs_excluded'] == ['failed-baseline']
    assert not selection_bias_report(common_rows, 'val_loss_at_common_step', common_report)['applicable']
    assert 'val_macro_f1_at_common_step' in COMPARISON_METRIC_KEYS
    assert 'val_macro_f1_at_common_step' not in _SELECTION_METRIC_KEYS
    assert 'val_macro_f1_at_common_step' not in _NGRAM_TUNING_METRIC_KEYS

def _selfcheck_top_57696_0054():
    global base, curve, derived, f, macro_bias, metric_by_run, metric_common, metric_dir, metric_paths, metric_rows, name, run_dir, score, step
    with tempfile.TemporaryDirectory() as metric_dir:
        metric_paths = []
        for name, curve in (('short', [(10, 0.5), (20, 0.6)]), ('long', [(10, 0.45), (20, 0.55), (30, 0.65)])):
            run_dir = Path(metric_dir) / name
            run_dir.mkdir()
            with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
                json.dump({'config': {'mode': 'A'}, 'args': {'seed': 1}}, f)
            for step, score in curve:
                write_jsonl_record(run_dir / 'metrics.jsonl', {'step': step, 'val_loss': 1.0, 'val_macro_f1': score})
            write_jsonl_record(run_dir / 'metrics.jsonl', {'step': 'final', 'val_loss': 0.5, 'val_macro_f1': 0.99, 'selected_model': 'best_val', 'selected_step': 20, 'trained_steps': 20})
            metric_paths.append(str(run_dir))
        metric_rows = compare_runs(metric_paths)
        metric_common = annotate_common_step_metrics(metric_rows)
        metric_by_run = {str(row['run']): row for row in metric_rows}
        assert abs(float(metric_by_run['short']['val_macro_f1_at_common_step']) - 0.6) < 1e-12
        assert abs(float(metric_by_run['long']['val_macro_f1_at_common_step']) - 0.55) < 1e-12
        macro_bias = selection_bias_report(metric_rows, 'val_macro_f1', metric_common)
        assert macro_bias['applicable']
        assert macro_bias['unbiased_metric'] == 'val_macro_f1_at_common_step'
        assert macro_bias['unbiased_runs'] == 2
        assert macro_bias['advisory']
        assert not selection_bias_report(metric_rows, 'val_macro_f1_at_common_step', metric_common)['applicable']
        assert 'val_pr_auc_macro_at_common_step' not in metric_common['metric_reports']
    for base in ('val_pr_auc_macro', 'val_pr_auc_macro_lift', 'val_roc_auc_macro'):
        derived = base + '_at_common_step'
        assert derived in COMPARISON_METRIC_KEYS, derived
        assert derived not in _SELECTION_METRIC_KEYS, derived
        assert derived not in _NGRAM_TUNING_METRIC_KEYS, derived
    assert _TRIVIAL_REFERENCE_FIELDS['val_pr_auc_macro_at_common_step'] == 'unigram_pr_auc_macro_chance'
    assert _TRIVIAL_REFERENCE_FIELDS['val_pr_auc_macro_lift_at_common_step'] == 'unigram_pr_auc_macro_lift'

def _selfcheck_body_57808_0056():
    global auc_by_run, auc_common, auc_paths, auc_rows, curve, f, lift, lift_bias, name, pr_auc, pr_auc_bias, roc_auc, run_dir, step
    auc_paths = []
    for name, curve in (('short', [(10, 0.3, 0.2, 0.6), (20, 0.4, 0.3, 0.7)]), ('long', [(10, 0.25, 0.15, 0.55), (20, 0.35, 0.25, 0.65), (30, 0.45, 0.35, 0.75)])):
        run_dir = Path(auc_dir) / name
        run_dir.mkdir()
        with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump({'config': {'mode': 'A'}, 'args': {'seed': 1}}, f)
        for step, pr_auc, lift, roc_auc in curve:
            write_jsonl_record(run_dir / 'metrics.jsonl', {'step': step, 'val_loss': 1.0, 'val_pr_auc_macro': pr_auc, 'val_pr_auc_macro_lift': lift, 'val_roc_auc_macro': roc_auc})
        write_jsonl_record(run_dir / 'metrics.jsonl', {'step': 'final', 'val_loss': 0.5, 'val_pr_auc_macro': 0.99, 'val_pr_auc_macro_lift': 0.89, 'val_roc_auc_macro': 0.99, 'selected_model': 'best_val', 'selected_step': 20, 'trained_steps': 20})
        auc_paths.append(str(run_dir))
    auc_rows = compare_runs(auc_paths)
    auc_common = annotate_common_step_metrics(auc_rows)
    auc_by_run = {str(row['run']): row for row in auc_rows}
    assert abs(float(auc_by_run['short']['val_pr_auc_macro_at_common_step']) - 0.4) < 1e-12
    assert abs(float(auc_by_run['long']['val_pr_auc_macro_at_common_step']) - 0.35) < 1e-12
    assert abs(float(auc_by_run['short']['val_pr_auc_macro_lift_at_common_step']) - 0.3) < 1e-12
    assert abs(float(auc_by_run['long']['val_roc_auc_macro_at_common_step']) - 0.65) < 1e-12
    assert auc_by_run['short']['val_pr_auc_macro_at_common_step'] != auc_by_run['short']['val_pr_auc_macro']
    pr_auc_bias = selection_bias_report(auc_rows, 'val_pr_auc_macro', auc_common)
    assert pr_auc_bias['applicable']
    assert pr_auc_bias['unbiased_metric'] == 'val_pr_auc_macro_at_common_step'
    assert pr_auc_bias['unbiased_runs'] == 2
    assert pr_auc_bias['advisory']
    lift_bias = selection_bias_report(auc_rows, 'val_pr_auc_macro_lift', auc_common)
    assert lift_bias['unbiased_metric'] == 'val_pr_auc_macro_lift_at_common_step'
    assert 'val_pr_auc_macro_at_common_step' in auc_common['metric_reports']
    assert 'val_roc_auc_macro_at_common_step' in auc_common['metric_reports']

def _selfcheck_top_57807_0055():
    global auc_dir
    with tempfile.TemporaryDirectory() as auc_dir:
        _selfcheck_body_57808_0056()

def _selfcheck_top_1_0057():
    _selfcheck_top_57807_0055()

def _selfcheck_top_57936_0058():
    global _, _BalanceScaleMoe, _BalanceScaleStub, aux1, aux2, balance_stub, large_diffs, p_at_010, p_at_050, paired_contrast, sign_flip, sign_flip_mc_a, sign_flip_mc_b, sign_flip_null, sign_flip_zero, stable_loo, unstable_loo
    class _BalanceScaleMoe:
        n_experts = 4
    class _BalanceScaleStub:
        moe = _BalanceScaleMoe()
        residual_scale = 1.0

        def __init__(self):
            self.calls = []

        def attention_step(self, value, scale=None, attn_mask=None):
            return value

        def moe_step(self, value, top_k, scale=None):
            self.calls.append(top_k)
            return (value, torch.tensor(float(top_k)))
    balance_stub = _BalanceScaleStub()
    _, aux1, aux2 = TransformerBlock.local_two_stage(balance_stub, torch.zeros(1, 1, 4), 2, 2, False)
    assert balance_stub.calls == [2, 4]
    assert abs(float(aux1) - 2.0) < 1e-09
    assert abs(float(aux2) - 2.0) < 1e-09
    assert _balance_width_profile({'mode': 'local_base', 'k1': 2, 'k2': 2, 'n_experts': 4, 'balance_coef': 0.01, 'balance_width_normalization': _BALANCE_WIDTH_NORMALIZATION})['effective_width'] == 2.0
    assert _balance_width_profile({'mode': 'local_base', 'k1': 2, 'k2': 2, 'n_experts': 4, 'balance_coef': 0.01, 'balance_width_normalization': 'none'})['effective_width'] == 3.0
    p_at_050 = _two_sided_t_p_value(2.7764451051977987, 4.0)
    p_at_010 = _two_sided_t_p_value(4.604094871349996, 4.0)
    assert p_at_050 is not None
    assert abs(p_at_050 - 0.05) < 1e-06, p_at_050
    assert p_at_010 is not None
    assert abs(p_at_010 - 0.01) < 1e-06, p_at_010
    assert _two_sided_t_p_value(0.0, 4.0) == 1.0
    assert abs(_regularized_incomplete_beta(0.3, 1.0, 1.0) - 0.3) < 1e-09
    assert abs((_two_sided_t_p_value(2.0, 7.0) or 0.0) - (_two_sided_t_p_value(-2.0, 7.0) or 0.0)) < 1e-12
    sign_flip = _paired_sign_flip_test([1.0, 2.0, 3.0])
    assert sign_flip['method'] == 'exact'
    assert abs(float(sign_flip['p_value']) - 0.25) < 1e-12
    assert abs(float(sign_flip['min_attainable_p']) - 0.25) < 1e-12
    assert sign_flip['can_reach_significance'] is False
    assert sign_flip['required_seeds_for_alpha'] == 6
    sign_flip_null = _paired_sign_flip_test([1.0, -1.0, 1.0])
    assert abs(float(sign_flip_null['p_value']) - 1.0) < 1e-12
    sign_flip_zero = _paired_sign_flip_test([0.0, 0.0, 0.0])
    assert float(sign_flip_zero['p_value']) == 1.0
    large_diffs = [0.1 * (index + 1) for index in range(15)]
    sign_flip_mc_a = _paired_sign_flip_test(large_diffs)
    sign_flip_mc_b = _paired_sign_flip_test(large_diffs)
    assert sign_flip_mc_a['method'] == 'monte_carlo'
    assert float(sign_flip_mc_a['p_value']) == float(sign_flip_mc_b['p_value'])
    assert 0.0 < float(sign_flip_mc_a['p_value']) < 1.0
    assert abs(float(sign_flip_mc_a['min_attainable_p']) - 1.0 / (_RANDOMIZATION_MONTE_CARLO_REPLICATES + 1)) < 1e-15
    stable_loo = _leave_one_out_effect_stability([1.0, 2.0, 3.0])
    assert stable_loo['leave_one_out_sign_stable'] is True
    unstable_loo = _leave_one_out_effect_stability([10.0, -1.0, -1.0])
    assert unstable_loo['leave_one_out_sign_stable'] is False
    assert abs(float(unstable_loo['leave_one_out_effect_min']) + 1.0) < 1e-12
    paired_contrast = _contrast_test([{'seed': 1, 'value': 3.0}, {'seed': 2, 'value': 3.0}, {'seed': 3, 'value': 3.0}], [{'seed': 1, 'value': 2.0}, {'seed': 2, 'value': 2.0}, {'seed': 3, 'value': 2.0}])
    assert paired_contrast['test'] == 'paired_t'
    assert paired_contrast['randomization_method'] == 'exact'
    assert paired_contrast['leave_one_out_sign_stable'] is True
    assert paired_contrast['randomization_can_reach_significance'] is False
    assert paired_contrast['randomization_required_seeds'] == 6
    assert paired_contrast['significant_robust_possible'] is False

def _selfcheck_top_58089_0059():
    global boot_a, insufficient_partial, loss_totals, partial_contrast, partial_display, partial_sensitivity, predicted_per_window, row, small_n_display, support_per_window, token_totals, tokens_per_window, tp_per_window, windows
    partial_contrast = _contrast_test([{'seed': 1, 'value': 3.0}, {'seed': 2, 'value': 3.3}, {'seed': 3, 'value': 2.7}, {'seed': 4, 'value': 3.1}], [{'seed': 1, 'value': 2.8}, {'seed': 2, 'value': 3.0}, {'seed': 3, 'value': 2.6}, {'seed': 5, 'value': 3.0}])
    assert partial_contrast['test'] == 'welch_t'
    assert partial_contrast['n_pairs'] == 0
    partial_sensitivity = partial_contrast['partial_pairing']
    assert partial_sensitivity['available'] is True
    assert partial_sensitivity['common_seeds'] == [1, 2, 3]
    assert partial_sensitivity['n_common'] == 3
    assert partial_sensitivity['test'] == 'paired_t'
    assert partial_sensitivity['p_value'] is not None
    assert abs(float(partial_sensitivity['effect']) + 0.2) < 1e-12
    assert partial_sensitivity['cohens_dz'] is not None
    insufficient_partial = _contrast_test([{'seed': 1, 'value': 3.0}, {'seed': 2, 'value': 3.1}, {'seed': 3, 'value': 3.2}, {'seed': 4, 'value': 3.3}], [{'seed': 3, 'value': 3.0}, {'seed': 4, 'value': 3.1}, {'seed': 5, 'value': 3.2}, {'seed': 6, 'value': 3.3}])
    assert insufficient_partial['partial_pairing']['available'] is False
    assert insufficient_partial['partial_pairing']['reason'] == 'insufficient_common_seeds'
    partial_display = _contrast_display({'factor': 'mode', 'baseline_value': 'local_base', 'variant_value': 'A', 'baseline': 'mode=local_base', 'delta': -0.2, 'p_value': 0.2, 'test': 'welch_t', 'n_pairs': 0, 'verdict': 'improvement', 'significant': False, 'delta_over_noise': 0.5, 'partial_pairing': partial_sensitivity}, 'val_loss')
    assert '共同种子配对敏感性' in partial_display
    assert '不替代主检验' in partial_display
    small_n_display = _contrast_display({'factor': 'mode', 'baseline_value': 'local_base', 'variant_value': 'A', 'baseline': 'mode=local_base', 'delta': -0.1, 'p_value': 0.01, 'test': 'paired_t', 'randomization_p_value': 0.25, 'randomization_method': 'exact', 'randomization_min_attainable_p': 0.25, 'randomization_can_reach_significance': False, 'leave_one_out_sign_stable': True, 'n_pairs': 3, 'verdict': 'improvement', 'significant': True, 'randomization_significant': False, 'delta_over_noise': 3.0}, 'val_loss')
    assert '无法达到 0.05 显著' in small_n_display
    windows = 6
    tokens_per_window = 10.0
    loss_totals = torch.tensor([10.0, 12.0, 11.0, 13.0, 9.0, 15.0], dtype=torch.float64)
    token_totals = torch.full((windows,), tokens_per_window, dtype=torch.float64)
    tp_per_window = torch.zeros((windows, 4), dtype=torch.int64)
    predicted_per_window = torch.zeros_like(tp_per_window)
    support_per_window = torch.zeros_like(tp_per_window)
    for row in range(windows):
        tp_per_window[row, row % 4] = 6
        predicted_per_window[row, row % 4] = 6
        support_per_window[row, row % 4] = 6
        tp_per_window[row, (row + 1) % 4] = 1
        predicted_per_window[row, (row + 1) % 4] = 2
        support_per_window[row, (row + 1) % 4] = 2
        support_per_window[row, (row + 2) % 4] = 2
    boot_a = _bootstrap_eval_intervals(loss_totals, token_totals, tp_per_window, predicted_per_window, support_per_window, replicates=300, seed=12345)

def _selfcheck_top_58213_0060():
    global boot_b, class_id, constant_boot, degenerate_boot, degenerate_support, perfect_boot, perfect_predicted, perfect_support, perfect_tp, point_loss, row, shuffled_totals, trend_boot, trend_pred, trend_rho, trend_sup, trend_tokens, trend_totals, trend_tp
    boot_b = _bootstrap_eval_intervals(loss_totals, token_totals, tp_per_window, predicted_per_window, support_per_window, replicates=300, seed=12345)
    assert boot_a == boot_b
    point_loss = float(loss_totals.sum() / token_totals.sum())
    assert boot_a['bootstrap_windows'] == windows
    assert boot_a['bootstrap_replicates'] == 300
    assert float(boot_a['loss_ci95_low']) <= point_loss <= float(boot_a['loss_ci95_high'])
    assert float(boot_a['token_accuracy_ci95_low']) <= float(boot_a['token_accuracy_ci95_high'])
    assert float(boot_a['macro_f1_ci95_low']) <= float(boot_a['macro_f1_ci95_high'])
    constant_boot = _bootstrap_eval_intervals(torch.full((windows,), 5.0, dtype=torch.float64), token_totals, tp_per_window, predicted_per_window, support_per_window, replicates=200, seed=7)
    assert abs(float(constant_boot['loss_ci95_low']) - 0.5) < 1e-12
    assert abs(float(constant_boot['loss_ci95_high']) - 0.5) < 1e-12
    assert constant_boot['window_lag1_autocorrelation'] is None
    assert float(constant_boot['window_ci_inflation']) == 1.0
    assert float(constant_boot['window_effective_windows']) == float(windows)
    perfect_tp = torch.zeros((windows, 4), dtype=torch.int64)
    perfect_predicted = torch.zeros_like(perfect_tp)
    perfect_support = torch.zeros_like(perfect_tp)
    for row in range(windows):
        for class_id in (0, 1, 2):
            perfect_tp[row, class_id] = 3
            perfect_predicted[row, class_id] = 3
            perfect_support[row, class_id] = 3
    perfect_tp[windows - 1, 3] = 1
    perfect_predicted[windows - 1, 3] = 1
    perfect_support[windows - 1, 3] = 1
    perfect_boot = _bootstrap_eval_intervals(loss_totals, token_totals, perfect_tp, perfect_predicted, perfect_support, replicates=300, seed=13)
    assert abs(float(perfect_boot['macro_f1_ci95_low']) - 1.0) < 1e-12
    assert abs(float(perfect_boot['macro_f1_ci95_high']) - 1.0) < 1e-12
    degenerate_support = perfect_support.clone()
    degenerate_support[:5] = 0
    degenerate_boot = _bootstrap_eval_intervals(loss_totals, token_totals, perfect_tp, perfect_predicted, degenerate_support, replicates=300, seed=17)
    assert 'macro_f1_ci95_low' in degenerate_boot
    assert math.isfinite(float(degenerate_boot['macro_f1_ci95_low']))
    assert math.isfinite(float(degenerate_boot['macro_f1_ci95_high']))
    assert int(degenerate_boot['macro_f1_bootstrap_degenerate']) > 0
    assert int(degenerate_boot['macro_f1_bootstrap_degenerate']) <= 300
    trend_totals = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], dtype=torch.float64)
    trend_tokens = torch.full((8,), 10.0, dtype=torch.float64)
    shuffled_totals = torch.tensor([10.0, 80.0, 20.0, 70.0, 30.0, 60.0, 40.0, 50.0], dtype=torch.float64)
    trend_tp = torch.zeros((8, 4), dtype=torch.int64)
    trend_pred = torch.zeros_like(trend_tp)
    trend_sup = torch.zeros_like(trend_tp)
    for row in range(8):
        trend_tp[row, row % 4] = 6
        trend_pred[row, row % 4] = 6
        trend_sup[row, row % 4] = 8
    trend_boot = _bootstrap_eval_intervals(trend_totals, trend_tokens, trend_tp, trend_pred, trend_sup, replicates=200, seed=11)
    trend_rho = trend_boot['window_lag1_autocorrelation']
    assert isinstance(trend_rho, float)

def _selfcheck_top_58355_0061():
    global alternating_boot, alternating_totals, bh_values, critical_2, critical_4, holm_values, paired_shuffled, paired_shuffled_boot, paired_trend, paired_trend_boot, shuffled_boot, shuffled_width, trend_effective, trend_width
    assert trend_rho >= _EVAL_WINDOW_AUTOCORRELATION, trend_rho
    assert float(trend_boot['window_ci_inflation']) > 1.0
    trend_effective = float(trend_boot['window_effective_windows'])
    assert 1.0 <= trend_effective < 8.0
    assert float(trend_boot['loss_ci95_low']) <= float(trend_totals.sum() / trend_tokens.sum()) <= float(trend_boot['loss_ci95_high'])
    trend_width = float(trend_boot['loss_ci95_high']) - float(trend_boot['loss_ci95_low'])
    shuffled_boot = _bootstrap_eval_intervals(shuffled_totals, trend_tokens, trend_tp, trend_pred, trend_sup, replicates=200, seed=11)
    assert float(shuffled_boot['window_ci_inflation']) == 1.0
    shuffled_width = float(shuffled_boot['loss_ci95_high']) - float(shuffled_boot['loss_ci95_low'])
    assert trend_width > shuffled_width
    assert float(shuffled_boot['window_effective_windows']) > 1.0
    alternating_totals = torch.tensor([10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0], dtype=torch.float64)
    alternating_boot = _bootstrap_eval_intervals(alternating_totals, trend_tokens, trend_tp, trend_pred, trend_sup, replicates=200, seed=11)
    assert float(alternating_boot['window_lag1_autocorrelation']) == -1.0
    assert float(alternating_boot['window_ci_inflation']) == 1.0
    assert float(alternating_boot['window_effective_windows']) == float(alternating_totals.numel())
    paired_trend = [0.1 * index for index in range(8)]
    paired_trend_boot = _paired_window_bootstrap_ci(paired_trend, seed=99)
    assert paired_trend_boot['available'] is True
    assert float(paired_trend_boot['lag1_autocorrelation']) >= _EVAL_WINDOW_AUTOCORRELATION
    assert float(paired_trend_boot['ci_inflation']) > 1.0
    assert float(paired_trend_boot['effective_windows']) < 8.0
    paired_shuffled = [0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5]
    paired_shuffled_boot = _paired_window_bootstrap_ci(paired_shuffled, seed=99)
    assert float(paired_shuffled_boot['ci_inflation']) == 1.0
    assert float(paired_trend_boot['high']) - float(paired_trend_boot['low']) > float(paired_shuffled_boot['high']) - float(paired_shuffled_boot['low'])
    assert _paired_window_bootstrap_ci(paired_trend, seed=99) == paired_trend_boot
    holm_values = _holm_bonferroni([0.03, 0.04, 0.5])
    assert abs(float(holm_values[0]) - 0.09) < 1e-12
    assert abs(float(holm_values[1]) - 0.09) < 1e-12
    assert float(holm_values[2]) == 0.5
    bh_values = _benjamini_hochberg([0.03, 0.04, 0.5])
    assert abs(float(bh_values[0]) - 0.06) < 1e-12
    assert abs(float(bh_values[1]) - 0.06) < 1e-12
    assert float(bh_values[2]) == 0.5
    assert all((float(b) <= float(h) + 1e-12 for h, b in zip(holm_values, bh_values)))
    assert _holm_bonferroni([0.02]) == [0.02]
    assert _benjamini_hochberg([0.02]) == [0.02]
    assert _holm_bonferroni([None, 0.01]) == [None, 0.01]
    assert _benjamini_hochberg([None, 0.01]) == [None, 0.01]
    assert _cost_verdict_from_ratios({'n_params': 1.0, 'latency_ms_per_forward': 10.0}, 'improvement', True) == 'gain_with_cost'
    assert _cost_verdict_from_ratios({'n_params': 1.0, 'latency_ms_per_forward': 10.0}, 'improvement', True, noisy_timing_fields=['latency_ms_per_forward']) == 'gain_balanced'
    assert _cost_verdict_from_ratios({'latency_ms_per_forward': 10.0}, 'improvement', True, noisy_timing_fields=['latency_ms_per_forward']) == 'gain_with_cost'
    critical_4 = _t_critical(4.0)
    assert critical_4 is not None
    assert abs(critical_4 - 2.7764451051977987) < 1e-06
    critical_2 = _t_critical(2.0)
    assert critical_2 is not None

def _selfcheck_top_58504_0062():
    global entry, equivalence_contrasts, expected_lower, expected_upper, first, half_effect_seeds, mde_large_var, mde_power, mde_value, normal_power, power_curve, second, synthetic_contrasts, third, tost_different, tost_equivalent, tost_manual, unit_effect_seeds, zero_df, zero_power
    assert abs(critical_2 - 4.302652729911275) < 1e-06
    synthetic_contrasts = [{'p_value': 0.03, 'significant': True, 'verdict': 'improvement', 'cost': {'ratios': {'n_params': 2.0}}, 'cost_verdict': 'gain_with_cost'}, {'p_value': 0.04, 'significant': True, 'verdict': 'improvement'}, {'p_value': 0.5, 'significant': False, 'verdict': 'tie'}, {'confounded': True, 'differing_factors': ['mode', 'k1'], 'cells': ['x', 'y']}]
    _apply_multiple_comparison_correction(synthetic_contrasts)
    first, second, third = synthetic_contrasts[:3]
    assert first['significant_raw'] is True
    assert not first['significant']
    assert abs(float(first['p_value_holm']) - 0.09) < 1e-12
    assert first['cost_verdict'] == 'gain_not_significant'
    assert second['significant_raw'] is True
    assert not second['significant']
    assert abs(float(second['p_value_holm']) - 0.09) < 1e-12
    assert not third['significant']
    assert 'p_value_holm' not in synthetic_contrasts[3]
    _apply_multiple_comparison_correction(synthetic_contrasts)
    assert first['significant_raw'] is True
    assert not first['significant']
    for zero_df in (2.0, 4.0, 10.0, 100.0):
        zero_power = _t_power(0.0, 1.0, zero_df)
        assert zero_power is not None
        assert abs(zero_power - _SIGNIFICANCE_LEVEL) < 1e-05, zero_power
    power_curve = [_t_power(effect, 1.0, 4.0) for effect in (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)]
    assert all((p is not None for p in power_curve))
    assert power_curve == sorted(power_curve)
    assert float(power_curve[-1]) > 0.999
    normal_power = _t_power(1.96, 1.0, 100000.0)
    assert normal_power is not None
    assert abs(normal_power - 0.50004) < 0.005, normal_power
    mde_value = _minimum_detectable_effect(0.1, 4.0)
    assert mde_value is not None and mde_value > 0.0
    mde_power = _t_power(mde_value, 0.1, 4.0)
    assert mde_power is not None
    assert abs(mde_power - _TARGET_POWER) < 0.001
    mde_large_var = _minimum_detectable_effect(0.3, 4.0)
    assert mde_large_var is not None and mde_large_var > mde_value
    assert _t_power(0.5, 0.0, 4.0) == 1.0
    assert _t_power(0.0, 0.0, 4.0) == _SIGNIFICANCE_LEVEL
    assert _minimum_detectable_effect(0.0, 4.0) == 0.0
    assert _t_power(1.0, 1.0, 0.0) is None
    assert _minimum_detectable_effect(0.1, 0.0) is None
    unit_effect_seeds = _seeds_to_detect(1.0, 1.0)
    assert unit_effect_seeds is not None
    assert 8 <= unit_effect_seeds <= 12, unit_effect_seeds
    half_effect_seeds = _seeds_to_detect(0.5, 1.0)
    assert half_effect_seeds is not None and half_effect_seeds > unit_effect_seeds
    assert _seeds_to_detect(0.0, 1.0) is None
    tost_equivalent = _tost_test(0.001, 0.01, 8.0, 0.05)
    assert float(tost_equivalent['tost_p_value']) < 0.001
    tost_different = _tost_test(0.2, 0.01, 8.0, 0.05)
    assert float(tost_different['tost_p_value']) > 0.9
    tost_manual = _tost_test(0.02, 0.05, 6.0, 0.04)
    expected_upper = _t_cdf((0.02 - 0.04) / 0.05, 6.0)
    expected_lower = 1.0 - _t_cdf((0.02 + 0.04) / 0.05, 6.0)
    assert expected_upper is not None and expected_lower is not None
    assert abs(float(tost_manual['tost_p_value']) - max(expected_upper, expected_lower)) < 1e-12
    assert _tost_test(0.001, 0.0, 8.0, 0.02)['tost_p_value'] == 0.0
    assert _tost_test(0.05, 0.0, 8.0, 0.02)['tost_p_value'] == 1.0
    equivalence_contrasts = [{'p_value': 0.2, 'significant': False, 'equivalent': True, 'tost_p_value': 0.03, 'verdict': 'tie'}, {'p_value': 0.3, 'significant': False, 'equivalent': True, 'tost_p_value': 0.04, 'verdict': 'tie'}]
    _apply_multiple_comparison_correction(equivalence_contrasts)
    for entry in equivalence_contrasts:
        assert entry['equivalent_raw'] is True
        assert not entry['equivalent']
        assert abs(float(entry['tost_p_value_holm']) - 0.06) < 1e-12
    _apply_multiple_comparison_correction(equivalence_contrasts[:1])
    assert equivalence_contrasts[0]['equivalent'] is True
    assert equivalence_contrasts[0]['equivalent_raw'] is True
    assert 'drop_train_corrupt_bytes' in COMPARISON_PROTOCOL_KEYS
    assert 'drop_train_corrupt_bytes' in _CROSS_FAMILY_PROTOCOL_KEYS
    assert 'allow_corrupt_test_labels' in COMPARISON_PROTOCOL_KEYS
    assert 'allow_corrupt_test_labels' in _CROSS_FAMILY_PROTOCOL_KEYS

def _selfcheck_top_58664_0063():
    global controlled_key, cross_family_key, factor_key, key, legacy_auc_protocol, legacy_neural_protocol, legacy_statistical_protocol
    assert 'clean_train_against_test' in COMPARISON_PROTOCOL_KEYS
    assert 'clean_train_against_test' in _CROSS_FAMILY_PROTOCOL_KEYS
    assert 'effective_train_sha256' in COMPARISON_PROTOCOL_KEYS
    assert 'effective_train_sha256' in _CROSS_FAMILY_PROTOCOL_KEYS
    assert 'split_isolation' in COMPARISON_PROTOCOL_KEYS
    assert 'split_isolation' in _CROSS_FAMILY_PROTOCOL_KEYS
    assert 'allow_eval_duplicates' in COMPARISON_PROTOCOL_KEYS
    assert 'allow_eval_duplicates' in _CROSS_FAMILY_PROTOCOL_KEYS
    assert 'eval_bootstrap' in COMPARISON_PROTOCOL_KEYS
    assert 'eval_bootstrap' in _CROSS_FAMILY_PROTOCOL_KEYS
    assert 'periodic_auc' in COMPARISON_PROTOCOL_KEYS
    assert 'periodic_auc' not in _CROSS_FAMILY_PROTOCOL_KEYS
    for controlled_key in SINGLE_FACTOR_DESIGN_KEYS:
        assert controlled_key in COMPARISON_PROTOCOL_KEYS
    for cross_family_key in ('split_policy', 'allow_eval_content_for_cleaning', 'allow_entity_overlap'):
        assert cross_family_key in _CROSS_FAMILY_PROTOCOL_KEYS
    assert 'n_kv_heads' not in _CROSS_FAMILY_PROTOCOL_KEYS
    assert 'residual_scaling' not in _CROSS_FAMILY_PROTOCOL_KEYS
    for factor_key in SINGLE_FACTOR_DESIGN_KEYS:
        assert factor_key in parse_factor_spec(factor_key)
    assert not _periodic_auc_enabled(False, 'val_loss')
    assert _periodic_auc_enabled(True, 'val_loss')
    assert _periodic_auc_enabled(False, 'val_pr_auc_macro')
    legacy_statistical_protocol = experiment_protocol_from_payload({'config': {'mode': 'ngram'}, 'args': {'seed': 7, 'deduplicate_train_groups': True, 'drop_train_corrupt_groups': True, 'drop_train_corrupt_bytes': True, 'drop_train_degenerate_groups': True, 'allow_data_leakage': True, 'allow_corrupt_test_labels': True}, 'data': {'sha256': 'deadbeef', 'split_sha256': {'train': 'a', 'val': 'b', 'test': 'c'}}, 'comparison_protocol': {'comparison_family': 'statistical', 'data_sha256': 'deadbeef'}})
    assert legacy_statistical_protocol['comparison_family'] == 'statistical'
    for key in ('deduplicate_train_groups', 'drop_train_corrupt_groups', 'drop_train_corrupt_bytes', 'drop_train_degenerate_groups', 'allow_data_leakage', 'allow_corrupt_test_labels'):
        assert legacy_statistical_protocol[key] is True, key
    assert legacy_statistical_protocol['eval_bootstrap'] is None
    assert legacy_statistical_protocol['periodic_auc'] is None
    assert legacy_statistical_protocol['effective_train_sha256'] is None
    assert legacy_statistical_protocol['split_isolation'] is None
    assert legacy_statistical_protocol['clean_train_against_test'] is True
    legacy_neural_protocol = experiment_protocol_from_payload({'config': {'mode': 'smelt'}, 'args': {'lr': 0.0003}})
    assert legacy_neural_protocol['periodic_auc'] is False
    legacy_auc_protocol = experiment_protocol_from_payload({'config': {'mode': 'smelt'}, 'args': {'lr': 0.0003, 'selection_metric': 'val_pr_auc_macro'}})
    assert legacy_auc_protocol['periodic_auc'] is True

def _selfcheck_body_hoist_58786_0065(name: str, mode: str, seed: int, lr: float, steps: int, balance: float, val_loss: float, hypothesis: str='', rq: str='', reference: Optional[Dict[str, object]]=None, majority_accuracy: Optional[float]=None, baseline_tuning: Optional[Dict[str, object]]=None, pr_auc_chance: Optional[float]=None, pr_auc: Optional[float]=None, k1: int=2, k2: int=2, protocol_overrides: Optional[Dict[str, object]]=None, routing: Optional[Dict[str, object]]=None, precision_recall: Optional[Dict[str, float]]=None, preregistration: Optional[Dict[str, object]]=None, curve_status: Optional[str]=None, periodic_val_loss: Optional[List[Tuple[int, float]]]=None, n_params: Optional[int]=None, flops: Optional[int]=None, ngram_tuning_metric: Optional[str]=None, eval_windows: Optional[Dict[str, object]]=None) -> str:
    run_dir = Path(analysis_dir) / name
    run_dir.mkdir()
    protocol: Dict[str, object] = {'comparison_family': 'neural', 'optimizer': 'AdamW', 'optimizer_betas': [0.9, 0.999], 'optimizer_eps': 1e-08, 'lr': lr, 'lr_schedule': 'linear_warmup_cosine_to_10pct', 'warmup_steps': 10, 'total_steps': steps, 'weight_decay': 0.1, 'grad_clip': 1.0, 'batch_size': 8, 'data_augmentation': 'none', 'seq_len': 8, 'd_model': 16, 'n_heads': 4, 'n_layers': 4, 'n_experts': 4, 'expert_hidden': 24, 'top_k': 2, 'residual_scale': 1.0, 'val_fraction': 0.05, 'test_fraction': 0.05, 'eval_interval': 10, 'eval_batches': 2, 'data_sha256': 'deadbeef', 'early_stop_patience': 0, 'early_stop_min_delta': 0.0, 'amp_dtype': 'float32', 'device_type': 'cpu', 'grad_checkpoint': False, 'compile': False}
    if protocol_overrides:
        protocol.update(protocol_overrides)
    config_payload = {'config': {'mode': mode, 'top_k': 2, 'k1': k1, 'k2': k2, 'balance_coef': balance, 'dense_fnn_hidden': None, 'n_experts': protocol.get('n_experts'), 'expert_hidden': protocol.get('expert_hidden'), 'seq_len': 8, 'd_model': 16, 'ngram_tuning_metric': ngram_tuning_metric}, 'args': {'seed': seed, 'lr': lr, 'steps': steps}, 'hypothesis': hypothesis, 'research_question': rq, 'hypothesis_preregistration': preregistration if isinstance(preregistration, dict) else {'primary_metric': None, 'direction': None, 'min_effect': None}, 'reference_baselines': reference, 'baseline_tuning': baseline_tuning, 'routing_analysis': routing, 'data': {'sha256': 'deadbeef'}, 'comparison_protocol': protocol}
    if n_params is not None:
        config_payload['n_params'] = n_params
    if flops is not None:
        config_payload['efficiency'] = {'flops_per_forward': flops}
    with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config_payload, f)
    for step, periodic_loss in periodic_val_loss or []:
        write_jsonl_record(run_dir / 'metrics.jsonl', {'step': step, 'val_loss': periodic_loss})
    if curve_status is not None:
        write_jsonl_record(run_dir / 'metrics.jsonl', {'step': 'curve_analysis', 'status': curve_status})
    final_record: Dict[str, object] = {'step': 'final', 'val_loss': val_loss}
    if majority_accuracy is not None:
        final_record['val_majority_accuracy'] = majority_accuracy
    if pr_auc_chance is not None:
        final_record['val_pr_auc_macro_chance'] = pr_auc_chance
    if pr_auc is not None:
        final_record['val_pr_auc_macro'] = pr_auc
    if pr_auc is not None and pr_auc_chance is not None:
        final_record['val_pr_auc_macro_lift'] = pr_auc - pr_auc_chance
    if precision_recall:
        final_record.update(precision_recall)
    write_jsonl_record(run_dir / 'metrics.jsonl', final_record)
    if eval_windows is not None:
        with open(run_dir / 'eval_windows.json', 'w', encoding='utf-8') as f:
            json.dump(eval_windows, f, ensure_ascii=False)
    return str(run_dir)

def _selfcheck_body_1_0066():
    global _write_analysis_run, amp_mismatch_controlled, amp_mismatch_path, analysis_rows, controlled, factors, mode_controlled, narrow_factors, run_paths, stale_stop_controlled, stale_stop_path
    _write_analysis_run = _selfcheck_body_hoist_58786_0065
    run_paths = [_write_analysis_run('local_base_seed42', 'local_base', 42, 0.0003, 100, 0.01, 3.0), _write_analysis_run('local_base_seed43', 'local_base', 43, 0.0003, 100, 0.01, 3.1), _write_analysis_run('local_base_seed44', 'local_base', 44, 0.0003, 100, 0.01, 3.2), _write_analysis_run('A_seed42', 'A', 42, 0.0003, 100, 0.01, 2.8), _write_analysis_run('A_seed43', 'A', 43, 0.0003, 100, 0.01, 2.85), _write_analysis_run('A_seed44', 'A', 44, 0.0003, 100, 0.01, 2.9)]
    factors = parse_factor_spec('mode,k1,k2,balance_coef,seed,dense_fnn_hidden')
    analysis_rows = compare_runs(run_paths)
    assert analysis_rows[0]['seed'] == 42
    assert analysis_rows[0]['balance_coef'] == 0.01
    assert analysis_rows[0]['protocol']['optimizer'] == 'AdamW'
    controlled = check_controlled_variables(analysis_rows, factors)
    assert controlled['ok'], controlled['violations']
    assert 'lr' in controlled['checked_keys']
    assert analysis_rows[0]['protocol']['mode'] == 'local_base'
    narrow_factors = parse_factor_spec('k1,k2,balance_coef,seed,dense_fnn_hidden')
    mode_controlled = check_controlled_variables(analysis_rows, narrow_factors)
    assert any((violation['key'] == 'mode' for violation in mode_controlled['violations'])), mode_controlled['violations']
    assert 'early_stop_patience' in controlled['checked_keys']
    assert 'amp_dtype' in controlled['checked_keys']
    assert 'device_type' in controlled['checked_keys']
    stale_stop_path = _write_analysis_run('A_seed42_stale_stop', 'A', 42, 0.0003, 100, 0.01, 2.85, protocol_overrides={'early_stop_patience': 5})
    stale_stop_controlled = check_controlled_variables(compare_runs(run_paths + [stale_stop_path]), factors)
    assert any((violation['key'] == 'early_stop_patience' for violation in stale_stop_controlled['violations'])), stale_stop_controlled['violations']
    amp_mismatch_path = _write_analysis_run('A_seed42_amp', 'A', 42, 0.0003, 100, 0.01, 2.85, protocol_overrides={'amp_dtype': 'bfloat16'})
    amp_mismatch_controlled = check_controlled_variables(compare_runs(run_paths + [amp_mismatch_path]), factors)
    assert any((violation['key'] == 'amp_dtype' for violation in amp_mismatch_controlled['violations'])), amp_mismatch_controlled['violations']

def _selfcheck_body_59084_0067():
    global a_cell, aggregates, by_label, contrast, contrasts, design_mismatch_controlled, design_mismatch_path, device_mismatch_controlled, device_mismatch_path, key, local_base_cell, missing_contrast, missing_rows
    device_mismatch_path = _write_analysis_run('A_seed42_device', 'A', 42, 0.0003, 100, 0.01, 2.85, protocol_overrides={'device_type': 'cuda'})
    device_mismatch_controlled = check_controlled_variables(compare_runs(run_paths + [device_mismatch_path]), factors)
    assert any((violation['key'] == 'device_type' for violation in device_mismatch_controlled['violations'])), device_mismatch_controlled['violations']
    design_mismatch_path = _write_analysis_run('A_seed42_design_mismatch', 'A', 42, 0.0003, 100, 0.01, 2.85, protocol_overrides={'n_kv_heads': 2, 'residual_scaling': 'paper', 'split_policy': 'chronological', 'allow_eval_content_for_cleaning': True, 'allow_entity_overlap': True})
    design_mismatch_controlled = check_controlled_variables(compare_runs(run_paths + [design_mismatch_path]), factors)
    for key in SINGLE_FACTOR_DESIGN_KEYS:
        assert any((violation['key'] == key for violation in design_mismatch_controlled['violations'])), (key, design_mismatch_controlled['violations'])
    aggregates = aggregate_runs(analysis_rows, ['mode'])
    assert len(aggregates) == 2
    by_label = {aggregate['label']: aggregate for aggregate in aggregates}
    local_base_cell = by_label['mode=local_base']
    a_cell = by_label['mode=A']
    assert local_base_cell['n_runs'] == 3
    assert abs(local_base_cell['val_loss_mean'] - 3.1) < 1e-09
    assert abs(local_base_cell['val_loss_std'] - 0.1) < 1e-09
    assert abs(a_cell['val_loss_mean'] - 2.85) < 1e-09
    assert abs(a_cell['val_loss_std'] - 0.05) < 1e-09
    contrasts = factor_contrasts(aggregates, ['mode'], 'val_loss')
    assert len(contrasts) == 1
    contrast = contrasts[0]
    assert contrast['test'] == 'paired_t'
    assert contrast['n_pairs'] == 3
    assert contrast['factor'] == 'mode'
    assert contrast['baseline_value'] == 'local_base'
    assert contrast['variant_value'] == 'A'
    assert abs(contrast['delta'] - -0.25) < 1e-09
    assert contrast['verdict'] == 'improvement'
    assert contrast['significant']
    assert contrast['p_value'] is not None and 0.005 < float(contrast['p_value']) < 0.05
    assert contrast['p_value_holm'] == contrast['p_value']
    assert contrast['p_value_bh'] == contrast['p_value']
    assert contrast['significant_raw'] is True
    assert contrast['ci95_low'] < contrast['effect'] < contrast['ci95_high']
    assert contrast['ci95_excludes_zero']
    assert '95%CI' in _contrast_display(contrast, 'val_loss')
    assert contrast['n_missing_baseline'] == 0
    assert contrast['n_missing_variant'] == 0
    assert contrast['differential_missingness'] is False
    missing_rows = [{'run': f'{mode}_seed{seed}', 'mode': mode, 'seed': seed, 'val_loss': value} for mode, values in (('local_base', [3.0, 3.1, 2.9]), ('A', [2.5, 2.6])) for seed, value in enumerate(values, start=1)]
    missing_rows.append({'run': 'A_seed3', 'mode': 'A', 'seed': 3})
    missing_contrast = factor_contrasts(aggregate_runs(missing_rows, ['mode']), ['mode'], 'val_loss')[0]
    assert missing_contrast['n_baseline'] == 3
    assert missing_contrast['n_variant'] == 2

def _selfcheck_body_59224_0068():
    global display_with_g, expected_correction, hedges, margin_consistent, margin_missing, margin_narrowed, margin_partial, margin_widened, noise_ratio, powered_contrast
    assert missing_contrast['n_missing_baseline'] == 0
    assert missing_contrast['n_missing_variant'] == 1
    assert missing_contrast['differential_missingness'] is True
    powered_contrast = factor_contrasts(aggregates, ['mode'], 'val_loss', equivalence_margin=0.01)[0]
    assert powered_contrast['mde_80'] is not None
    assert powered_contrast['mde_80'] > 0.0
    assert powered_contrast['mde_80_relative'] is not None
    assert powered_contrast['power_at_effect'] is not None
    assert powered_contrast['seeds_to_detect'] is not None
    assert powered_contrast['equivalence_margin'] == 0.01
    assert abs(float(powered_contrast['equivalence_margin_abs']) - 0.031) < 1e-09
    assert powered_contrast['equivalent'] is False
    assert powered_contrast['equivalent_raw'] is False
    assert float(powered_contrast['tost_p_value_holm']) > _SIGNIFICANCE_LEVEL
    assert 'tost_p_value' in powered_contrast
    noise_ratio = powered_contrast['delta_over_noise']
    hedges = powered_contrast['hedges_g']
    assert noise_ratio is not None
    assert hedges is not None
    expected_correction = _hedges_g_correction(powered_contrast['df'])
    assert expected_correction is not None
    assert 0.0 < expected_correction < 1.0
    assert abs(float(hedges) - float(noise_ratio) * expected_correction) < 1e-12
    assert abs(float(hedges)) < abs(float(noise_ratio))
    display_with_g = _contrast_display(powered_contrast, 'val_loss')
    assert 'Hedges g=' in display_with_g
    assert powered_contrast['cohens_dz'] is not None
    assert powered_contrast['hedges_gz'] is not None
    assert abs(float(powered_contrast['cohens_dz']) - abs(float(powered_contrast['t'])) / math.sqrt(float(powered_contrast['n_pairs']))) < 1e-12
    assert abs(float(powered_contrast['hedges_gz'])) < abs(float(powered_contrast['cohens_dz']))
    assert '配对 dz=' in display_with_g
    assert _paired_effect_size(-0.2, {'test': 'welch_t', 'se': 0.1, 'df': 4.0}) == {}
    assert _paired_effect_size(-0.2, {'test': 'paired_t', 'se': 0.1, 'df': 2.0}) == {}
    margin_consistent = equivalence_margin_preregistration([{'run': 'A_seed1', 'hypothesis_equivalence_margin': 0.01}, {'run': 'local_base_seed1', 'hypothesis_equivalence_margin': 0.01}], 0.01)
    assert margin_consistent['status'] == 'consistent'
    assert margin_consistent['ok'] is True
    assert margin_consistent['widened_post_hoc'] is False
    margin_partial = equivalence_margin_preregistration([{'run': 'A_seed1', 'hypothesis_equivalence_margin': 0.01}, {'run': 'old_run'}], 0.01)
    assert margin_partial['status'] == 'consistent'
    assert margin_partial['ok'] is None
    assert margin_partial['missing_runs'] == ['old_run']
    assert any(('无法核对全部运行' in advisory for advisory in margin_partial['advisories']))
    margin_widened = equivalence_margin_preregistration([{'run': 'A_seed1', 'hypothesis_equivalence_margin': 0.01}], 0.05)
    assert margin_widened['status'] == 'widened_post_hoc'
    assert margin_widened['ok'] is False
    assert margin_widened['widened_post_hoc'] is True
    assert any(('放宽' in advisory for advisory in margin_widened['advisories']))
    margin_narrowed = equivalence_margin_preregistration([{'run': 'A_seed1', 'hypothesis_equivalence_margin': 0.05}], 0.01)
    assert margin_narrowed['status'] == 'narrowed_post_hoc'
    assert margin_narrowed['widened_post_hoc'] is False
    margin_missing = equivalence_margin_preregistration([{'run': 'old_run'}], 0.01)

def _selfcheck_body_59371_0069():
    global _PowerCapture, baseline_partial, equivalent_contrast, margin_inconsistent, near_equal_aggregates, power_capture, power_text, rel_se
    assert margin_missing['status'] == 'unverifiable'
    assert margin_missing['ok'] is None
    margin_inconsistent = equivalence_margin_preregistration([{'run': 'A_seed1', 'hypothesis_equivalence_margin': 0.01}, {'run': 'A_seed2', 'hypothesis_equivalence_margin': 0.02}], 0.01)
    assert margin_inconsistent['status'] == 'inconsistent_declarations'
    assert margin_inconsistent['ok'] is False
    baseline_partial = resolve_preregistered_baseline([{'run': 'A_seed1', 'mode': 'A', 'hypothesis_baseline': 'smelt'}, {'run': 'old_run', 'mode': 'smelt'}], None)
    assert baseline_partial['baseline_mode'] == 'smelt'
    assert baseline_partial['auto_selected'] is True
    assert baseline_partial['runs_without_preregistration'] == ['old_run']
    assert baseline_partial['ok'] is None
    assert any(('无法核对这些运行' in advisory for advisory in baseline_partial['advisories']))
    assert powered_contrast['seeds_to_detect_basis'] == 'sd_point_estimate'
    rel_se = powered_contrast['seeds_to_detect_sd_relative_se']
    assert rel_se is not None
    assert abs(float(rel_se) - 0.5) < 1e-09
    class _PowerCapture:

        def __init__(self) -> None:
            self.chunks: List[str] = []

        def write(self, text: str) -> int:
            self.chunks.append(text)
            return len(text)

        def flush(self) -> None:
            return None
    power_capture = _PowerCapture()
    with contextlib.redirect_stdout(power_capture):
        print_comparison_analysis(analysis_rows, factors=['mode'], metric='val_loss', equivalence_margin=0.01)
    power_text = ''.join(power_capture.chunks)
    assert 'sd_point_estimate' in power_text
    assert '所需种子数' in power_text
    assert '95%CI' in _contrast_display(powered_contrast, 'val_loss')
    near_equal_aggregates = [{'label': 'mode=smelt', 'keys': {'mode': 'smelt'}, 'n_runs': 3, 'runs': ['smelt_seed42', 'smelt_seed43', 'smelt_seed44'], 'seeds': [42, 43, 44], 'duplicate_seeds': [], 'costs': {}, 'val_loss_mean': 3.0, 'val_loss_std': 0.0001, 'samples': {'val_loss': [{'seed': 42, 'value': 3.0}, {'seed': 43, 'value': 3.0001}, {'seed': 44, 'value': 2.9999}]}}, {'label': 'mode=local_base', 'keys': {'mode': 'local_base'}, 'n_runs': 3, 'runs': ['local_base_seed42', 'local_base_seed43', 'local_base_seed44'], 'seeds': [42, 43, 44], 'duplicate_seeds': [], 'costs': {}, 'val_loss_mean': 2.99985, 'val_loss_std': 0.00015, 'samples': {'val_loss': [{'seed': 42, 'value': 2.9998}, {'seed': 43, 'value': 3.0}, {'seed': 44, 'value': 2.99975}]}}]
    equivalent_contrast = factor_contrasts(near_equal_aggregates, ['mode'], 'val_loss', equivalence_margin=0.01)[0]
    assert equivalent_contrast['equivalent'] is True
    assert equivalent_contrast['equivalent_raw'] is True
    assert float(equivalent_contrast['tost_p_value_holm']) <= _SIGNIFICANCE_LEVEL

def _selfcheck_body_59511_0070():
    global _delta, _df, _se, boundary_tost, equivalent_display, needed_boundary, noisy_rows, precision_contrast, seeds_to_achieve, tighter_tost, wider_tost
    equivalent_display = _contrast_display(equivalent_contrast, 'val_loss')
    assert 'TOST 等价' in equivalent_display
    assert '无实质差异' in equivalent_display
    assert equivalent_contrast['equivalence_blocker'] is None
    assert equivalent_contrast['equivalence_margin_needed'] <= equivalent_contrast['equivalence_margin_abs']
    assert powered_contrast['equivalence_blocker'] == 'effect_exceeds_margin'
    assert 'equivalence_seeds_to_achieve' not in powered_contrast
    noisy_rows = [{'run': f'{mode}_seed{seed}', 'mode': mode, 'seed': seed, 'val_loss': value} for mode, values in (('local_base', [3.0, 3.5, 2.5]), ('A', [3.05, 3.45, 2.55])) for seed, value in enumerate(values, start=1)]
    precision_contrast = factor_contrasts(aggregate_runs(noisy_rows, ['mode']), ['mode'], 'val_loss', equivalence_margin=0.01)[0]
    assert precision_contrast['equivalent'] is False
    assert precision_contrast['equivalence_blocker'] == 'insufficient_precision'
    assert precision_contrast['equivalence_margin_needed'] > precision_contrast['equivalence_margin_abs']
    seeds_to_achieve = precision_contrast['equivalence_seeds_to_achieve']
    assert isinstance(seeds_to_achieve, int)
    assert seeds_to_achieve > 3
    _se, _df, _delta = (0.2, 5.0, 0.05)
    needed_boundary = abs(_delta) + _t_critical(_df, 2.0 * _SIGNIFICANCE_LEVEL) * _se
    boundary_tost = _tost_test(_delta, _se, _df, needed_boundary)
    assert boundary_tost
    assert abs(float(boundary_tost['tost_p_value']) - _SIGNIFICANCE_LEVEL) < 1e-09
    tighter_tost = _tost_test(_delta, _se, _df, needed_boundary * 0.9999)
    assert tighter_tost
    assert float(tighter_tost['tost_p_value']) > _SIGNIFICANCE_LEVEL
    wider_tost = _tost_test(_delta, _se, _df, needed_boundary * 1.0001)
    assert wider_tost
    assert float(wider_tost['tost_p_value']) < _SIGNIFICANCE_LEVEL
    assert powered_contrast['significance_family_size'] >= 1
    assert float(powered_contrast['alpha_bonferroni']) <= _SIGNIFICANCE_LEVEL
    assert float(powered_contrast['mde_80_bonferroni']) >= float(powered_contrast['mde_80']) - 1e-12
    assert int(powered_contrast['seeds_to_detect_bonferroni']) >= int(powered_contrast['seeds_to_detect'])
    assert float(powered_contrast['equivalence_margin_needed_bonferroni']) >= float(powered_contrast['equivalence_margin_needed']) - 1e-12
    assert equivalent_contrast['significance_family_size'] == 1
    assert equivalent_contrast['alpha_bonferroni'] == _SIGNIFICANCE_LEVEL
    assert abs(float(equivalent_contrast['mde_80_bonferroni']) - float(equivalent_contrast['mde_80'])) < 1e-15

def _selfcheck_body_59644_0071():
    global canonical_factor_contrasts, canonical_factor_rows, duplicate_aggregates, duplicate_analysis, duplicate_by_label, duplicate_contrast, duplicate_path, duplicate_rows, inconclusive_display
    assert abs(float(equivalent_contrast['equivalence_margin_needed_bonferroni']) - float(equivalent_contrast['equivalence_margin_needed'])) < 1e-12
    inconclusive_display = _contrast_display({'factor': 'mode', 'baseline_value': 'local_base', 'variant_value': 'A', 'baseline': 'mode=local_base', 'delta': -0.001, 'p_value': 0.6, 'test': 'paired_t', 'n_pairs': 3, 'verdict': 'improvement', 'significant': False, 'delta_over_noise': 0.1, 'mde_80': 0.015, 'seeds_to_detect': 12}, 'val_loss')
    assert '未显著不代表等价' in inconclusive_display
    assert '0.0150' in inconclusive_display
    assert '12 个种子' in inconclusive_display
    duplicate_path = _write_analysis_run('A_seed42_duplicate', 'A', 42, 0.0003, 100, 0.01, 2.82)
    duplicate_rows = compare_runs(run_paths + [duplicate_path])
    duplicate_aggregates = aggregate_runs(duplicate_rows, ['mode'])
    duplicate_by_label = {aggregate['label']: aggregate for aggregate in duplicate_aggregates}
    assert duplicate_by_label['mode=A']['duplicate_seeds'] == [42]
    assert duplicate_by_label['mode=local_base']['duplicate_seeds'] == []
    duplicate_contrast = factor_contrasts(duplicate_aggregates, ['mode'], 'val_loss')[0]
    assert duplicate_contrast['duplicate_seeds']
    assert duplicate_contrast['test'] == 'paired_t'
    assert duplicate_contrast['n_pairs'] == 3
    assert duplicate_contrast['pseudo_replication']['detected']
    assert duplicate_contrast['pseudo_replication']['collapsed']
    assert duplicate_contrast['n_baseline'] == 3
    assert duplicate_contrast['n_variant'] == 3
    assert duplicate_by_label['mode=A']['val_loss_n'] == 3
    assert duplicate_by_label['mode=A']['n_runs'] == 4
    assert duplicate_by_label['mode=A']['effective_seed_count'] == 3
    assert duplicate_by_label['mode=local_base']['effective_seed_count'] == 3
    assert '重复随机种子' in _contrast_display(duplicate_contrast, 'val_loss')
    canonical_factor_rows = [{'run': 'flag_true', 'flag': True, 'seed': 1, 'val_loss': 2.0}, {'run': 'flag_one', 'flag': 1, 'seed': 1, 'val_loss': 1.0}]
    canonical_factor_contrasts = factor_contrasts(aggregate_runs(canonical_factor_rows, ['flag']), ['flag'], 'val_loss')
    assert len(canonical_factor_contrasts) == 1
    assert canonical_factor_contrasts[0]['differing_factors'] == ['flag']
    duplicate_analysis = print_comparison_analysis(duplicate_rows, factors=factors, metric='val_loss')

def _selfcheck_body_59789_0072():
    global _window_payload, dense_losses, duplicate_window_cell_a, expected_window_diff, variant_losses, window_a_path, window_contrasts, window_dense_path, window_paired, window_per_seed, window_rows
    assert any((cell['cell'] == 'mode=A' and cell['duplicate_seeds'] == [42] for cell in duplicate_analysis['duplicate_seed_cells'])), duplicate_analysis['duplicate_seed_cells']
    def _window_payload(data_sha256: str, losses: List[float], hits: Optional[List[int]]=None, split_sha256: Optional[str]=None) -> Dict[str, object]:
        window_tokens = 7
        windows = [{'loss_sum': float(loss) * window_tokens, 'tokens': window_tokens, 'correct': int(hits[index] if hits is not None else 0)} for index, loss in enumerate(losses)]
        return {'split': 'val', 'data_sha256': data_sha256, 'split_sha256': split_sha256, 'seq_len': 8, 'batch_size': 8, 'batches': 1, 'n_windows': len(windows), 'overlap_fraction': 0.0, 'windows': windows}
    window_dense_path = _write_analysis_run('window_dense_seed42', 'dense', 42, 0.0003, 100, 0.0, 4.0, eval_windows=_window_payload('deadbeef', [4.0, 4.1, 4.2, 3.9]))
    window_a_path = _write_analysis_run('window_A_seed42', 'A', 42, 0.0003, 100, 0.0, 3.5, eval_windows=_window_payload('deadbeef', [3.5, 3.6, 3.7, 3.4]))
    window_rows = compare_runs([window_dense_path, window_a_path])
    assert window_rows[0]['eval_windows']['n_windows'] == 4
    window_contrasts = factor_contrasts(aggregate_runs(window_rows, ['mode']), ['mode'], 'val_loss')
    assert len(window_contrasts) == 1
    window_paired = window_contrasts[0]['window_paired']
    assert window_paired['available'] is True
    assert window_paired['n_seeds'] == 1
    dense_losses = [4.0, 4.1, 4.2, 3.9]
    variant_losses = [3.5, 3.6, 3.7, 3.4]
    if str(window_contrasts[0]['variant']) == 'mode=dense':
        expected_window_diff = (sum(dense_losses) - sum(variant_losses)) / len(dense_losses)
    else:
        expected_window_diff = (sum(variant_losses) - sum(dense_losses)) / len(dense_losses)
    assert abs(float(window_paired['mean_loss_diff']) - expected_window_diff) < 1e-12
    assert window_paired['direction_consistent'] is True
    assert window_paired['n_seeds_favoring_variant'] == (1 if expected_window_diff < 0.0 else 0)
    window_per_seed = window_paired['per_seed'][0]
    assert window_per_seed['status'] == 'ok'
    assert window_per_seed['n_windows'] == 4
    assert window_per_seed['accuracy_status'] == 'ok'
    assert window_per_seed['n_accuracy_windows'] == 4
    assert window_per_seed['fraction_variant_better'] == (1.0 if expected_window_diff < 0.0 else 0.0)
    assert '[window]' in _format_window_paired(window_contrasts[0])
    duplicate_window_cell_a = {'window_payloads': [{'seed': 42, 'run': 'dup_a42_first', 'payload': _window_payload('deadbeef', [4.0, 4.1, 4.2, 3.9])}, {'seed': 42, 'run': 'dup_a42_second', 'payload': _window_payload('deadbeef', [9.0, 9.0, 9.0, 9.0])}, {'seed': 43, 'run': 'dup_a43', 'payload': _window_payload('deadbeef', [4.2, 4.3, 4.4, 4.1])}]}

def _selfcheck_body_59921_0073():
    global ar1_trend_test, duplicate_window, duplicate_window_cell_b, only_duplicate_window, raw_trend_test, trend_base_losses, trend_pair, trend_variant_losses, window_mismatch, window_split_mismatch, window_trend_diffs
    duplicate_window_cell_b = {'window_payloads': [{'seed': 42, 'run': 'dup_b42_first', 'payload': _window_payload('deadbeef', [3.5, 3.6, 3.7, 3.4])}, {'seed': 42, 'run': 'dup_b42_second', 'payload': _window_payload('deadbeef', [0.5, 0.5, 0.5, 0.5])}, {'seed': 43, 'run': 'dup_b43', 'payload': _window_payload('deadbeef', [3.7, 3.8, 3.9, 3.6])}]}
    duplicate_window = _cell_window_paired_analysis(duplicate_window_cell_a, duplicate_window_cell_b)
    assert duplicate_window['available'] is True
    assert duplicate_window['duplicate_seeds_excluded'] == [42]
    assert [entry['seed'] for entry in duplicate_window['per_seed']] == [43]
    only_duplicate_window = _cell_window_paired_analysis({'window_payloads': duplicate_window_cell_a['window_payloads'][:2]}, {'window_payloads': duplicate_window_cell_b['window_payloads'][:2]})
    assert only_duplicate_window['available'] is False
    assert '重复随机种子' in only_duplicate_window['reason']
    window_trend_diffs = [0.02 + 0.01 * index + (0.0004 if index % 2 else -0.0004) for index in range(40)]
    raw_trend_test = _paired_t_test(window_trend_diffs)
    ar1_trend_test = _ar1_adjusted_mean_test(window_trend_diffs)
    assert raw_trend_test['p_value'] < 1e-06
    assert ar1_trend_test['applied'] is True
    assert ar1_trend_test['lag1_autocorrelation'] > 0.3
    assert ar1_trend_test['effective_windows'] < 3.0
    assert ar1_trend_test['p_value'] is None
    assert ar1_trend_test['note'] == 'effective_windows_below_three'
    trend_base_losses = [4.0 + 0.005 * index for index in range(40)]
    trend_variant_losses = [loss + diff for loss, diff in zip(trend_base_losses, window_trend_diffs)]
    trend_pair = paired_window_comparison(_window_payload('deadbeef', trend_base_losses), _window_payload('deadbeef', trend_variant_losses))
    assert trend_pair['status'] == 'ok'
    assert trend_pair['window_ar1_adjustment_applied'] is True
    assert trend_pair['window_p_value'] < 1e-06
    assert trend_pair['window_p_value_ar1'] is None
    assert 'AR(1) 校正 p' in _format_window_paired({'window_paired': {'available': True, 'mean_loss_diff': float(trend_pair['mean_loss_diff']), 'n_seeds': 1, 'n_seeds_favoring_variant': 0, 'n_seeds_favoring_baseline': 1, 'per_seed': [{'seed': 42, **trend_pair}]}})
    window_mismatch = paired_window_comparison(_window_payload('deadbeef', [4.0, 4.1]), _window_payload('other', [3.0, 3.1]))
    assert window_mismatch['status'] == 'protocol_mismatch'
    assert 'data_sha256' in window_mismatch['mismatched_keys']
    window_split_mismatch = paired_window_comparison(_window_payload('deadbeef', [4.0, 4.1], split_sha256='val-a'), _window_payload('deadbeef', [3.0, 3.1], split_sha256='val-b'))

def _selfcheck_body_60059_0074():
    global conflict_analysis, conflict_ngram_path, invalid_accuracy_b, invalid_accuracy_result, invalid_window_b, invalid_window_result, ngram_aggregates, ngram_cell, ngram_paths, ngram_rows, no_window_a_path, no_window_contrasts, no_window_dense_path, no_window_rows
    assert window_split_mismatch['status'] == 'protocol_mismatch'
    assert 'split_sha256' in window_split_mismatch['mismatched_keys']
    invalid_window_b = _window_payload('deadbeef', [3.0, 3.1, 3.2])
    invalid_window_b['windows'][0]['loss_sum'] = float('nan')
    invalid_window_b['windows'][1].pop('loss_sum')
    invalid_window_result = paired_window_comparison(_window_payload('deadbeef', [4.0, 4.1, 4.2]), invalid_window_b)
    assert invalid_window_result['status'] == 'missing_windows'
    assert invalid_window_result['invalid_window_indices'] == [0, 1]
    assert '不能只对剩余窗口' in invalid_window_result['reason']
    invalid_accuracy_b = _window_payload('deadbeef', [3.0, 3.1, 3.2])
    invalid_accuracy_b['windows'][2].pop('correct')
    invalid_accuracy_result = paired_window_comparison(_window_payload('deadbeef', [4.0, 4.1, 4.2]), invalid_accuracy_b)
    assert invalid_accuracy_result['status'] == 'ok'
    assert invalid_accuracy_result['n_windows'] == 3
    assert invalid_accuracy_result['accuracy_status'] == 'missing_windows'
    assert invalid_accuracy_result['invalid_accuracy_window_indices'] == [2]
    assert 'mean_accuracy_diff' not in invalid_accuracy_result
    assert '不能只对剩余窗口' in invalid_accuracy_result['accuracy_unavailable_reason']
    no_window_dense_path = _write_analysis_run('windowless_dense_seed42', 'dense', 42, 0.0003, 100, 0.0, 4.0)
    no_window_a_path = _write_analysis_run('windowless_A_seed42', 'A', 42, 0.0003, 100, 0.0, 3.5)
    no_window_rows = compare_runs([no_window_dense_path, no_window_a_path])
    assert 'eval_windows' not in no_window_rows[0]
    no_window_contrasts = factor_contrasts(aggregate_runs(no_window_rows, ['mode']), ['mode'], 'val_loss')
    assert len(no_window_contrasts) == 1
    assert no_window_contrasts[0]['window_paired']['available'] is False
    assert '逐窗口评估记录' in no_window_contrasts[0]['window_paired']['reason']
    assert _format_window_paired(no_window_contrasts[0]) == ''
    ngram_paths = [_write_analysis_run(f'ngram_seed{seed}', 'ngram', seed, 0.0003, 100, 0.01, 2.8) for seed in (42, 43, 44)]
    ngram_rows = compare_runs(ngram_paths)
    ngram_aggregates = aggregate_runs(ngram_rows, ['mode'])
    assert len(ngram_aggregates) == 1
    ngram_cell = ngram_aggregates[0]
    assert ngram_cell['deterministic'] is True
    assert ngram_cell['effective_seed_count'] == 1
    assert ngram_cell['val_loss_n'] == 1
    assert ngram_cell['val_loss_std'] is None
    assert len(ngram_cell['deterministic_repeat_runs']) == 3
    assert 'deterministic_repeat_conflicts' not in ngram_cell
    conflict_ngram_path = _write_analysis_run('ngram_conflict_seed45', 'ngram', 45, 0.0003, 100, 0.01, 2.79)
    conflict_analysis = print_comparison_analysis(compare_runs(ngram_paths + [conflict_ngram_path]), metric='val_loss')
    assert 'val_loss' in conflict_analysis['deterministic_conflict_cells'][0]['metrics']

