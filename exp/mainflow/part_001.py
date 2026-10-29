from __future__ import annotations

def _MAIN_part_005():
    global _, _current_training_peak_mb, _saved_counter, actual_training_peak_mb, bad_case_payload, bad_case_report, bad_cases_path, best_selection_step, best_selection_value, checkpoint_gen_state, checkpoint_numerical_state, consecutive_nonfinite, consecutive_text, current_lr, curve_analysis, early_stopped, end_step, error, eval_metrics, eval_windows_payload, evals_without_improvement, f, final_metrics, final_record, finite_grad, finite_step, flush_records, gaps, grad_norm, grad_norm_value, is_branch_point, last_completed_step, last_curve_status, last_curve_warning_flags, last_saved_step, live_analysis, live_status, loss, loss_value, metrics_file, min_val_loss, min_val_loss_step, nan_checks, nan_diagnosis, nan_record_pending, nonfinite_events, numerical_polluted, numerical_stop, optimizer_steps_completed, overlap_count, overlap_fraction, payload_val_loss, pending_records, pending_scalars, prefetched_batch, previous_status, prior_flags, prior_numerical_stop, prior_training_peak_mb, records, restore_error, result, routing_probe, save_at_steps, saved_checkpoint, saved_peak, saved_selected, scaler_scale_before_step, selected_payload, selected_step, selected_val_loss, selection_note, selection_policy, selection_value, starts, step, train_autocast, train_params, val_bad_case_collector, val_loss, val_window_records, x, y
    def _saved_counter(name: str) -> int:
        value = saved_numerical_state.get(name)
        if isinstance(value, int) and (not isinstance(value, bool)) and (value >= 0):
            return value
        return 0
    nonfinite_events = _saved_counter('nonfinite_events')
    optimizer_steps_completed = max(0, int(start_step) - int(nonfinite_events))
    consecutive_nonfinite = _saved_counter('consecutive_nonfinite')
    numerical_stop = False
    numerical_polluted = False
    prior_numerical_stop = bool(saved_numerical_state.get('numerical_stop') or saved_numerical_state.get('prior_numerical_stop'))
    if prior_numerical_stop:
        print(f'[nan] [warn] 检查点记录显示原运行曾在数值上停止（nonfinite_events={nonfinite_events}）：本次续跑会累计该历史；该运行不再具备解封测试集的资格')
    nan_diagnosis = None
    nan_record_pending = False
    prior_training_peak_mb = None
    saved_peak = _finite_float(saved_numerical_state.get('training_peak_memory_mb'))
    if saved_peak is not None and saved_peak >= 0.0:
        prior_training_peak_mb = saved_peak
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    def _current_training_peak_mb() -> Optional[float]:
        peaks = []
        if prior_training_peak_mb is not None:
            peaks.append(prior_training_peak_mb)
        if device.type == 'cuda':
            peaks.append(torch.cuda.max_memory_allocated(device) / 1000000.0)
        if not peaks:
            return None
        return round(max(peaks), 3)
    def checkpoint_numerical_state() -> Dict[str, object]:
        """当前数值状态随检查点存档，供 resume 与审计读取。"""
        return {'nonfinite_events': int(nonfinite_events), 'consecutive_nonfinite': int(consecutive_nonfinite), 'numerical_stop': bool(numerical_stop), 'numerical_polluted': bool(numerical_polluted), 'prior_numerical_stop': bool(prior_numerical_stop), 'nan_diagnosis': nan_diagnosis, 'training_peak_memory_mb': _current_training_peak_mb()}
    if args.resume is not None:
        _truncate_metrics_after_step(metrics_path, start_step)
    metrics_file = open(metrics_path, 'a' if args.resume is not None else 'w', encoding='utf-8')
    if args.resume is None:
        metrics_file.write(json.dumps(_json_safe({'step': 'reference_baselines', **tracking_header_fields(args, dataset, run_metadata, run_started_utc=tracking_provenance.get('run_started_utc'), comparison_protocol_sha256=tracking_provenance.get('comparison_protocol_sha256')), **reference_baselines}), ensure_ascii=False) + '\n')
        metrics_file.flush()
    if sanity_record is not None:
        metrics_file.write(json.dumps(_json_safe(sanity_record), ensure_ascii=False) + '\n')
        metrics_file.flush()
    model.train()
    train_params = list(model.parameters())
    train_autocast = torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_dtype is not None)
    pending_records = []
    pending_scalars = []
    def flush_records() -> List[Dict[str, object]]:
        if not pending_scalars:
            return []
        values = torch.stack(pending_scalars).view(-1, 2).tolist()
        records = []
        for meta, (lm, aux) in zip(pending_records, values):
            record = {'step': meta['step'], 'optimizer_steps': meta.get('optimizer_steps'), 'train_loss': lm if math.isfinite(lm) else None, 'aux_loss': aux if math.isfinite(aux) else None, 'lr': meta['lr'], 'grad_norm': meta['grad_norm']}
            extra = meta.get('extra')
            if extra is not None:
                record.update(extra)
            metrics_file.write(json.dumps(_json_safe(record), ensure_ascii=False) + '\n')
            records.append(record)
        pending_records.clear()
        pending_scalars.clear()
        curve_records.extend(records)
        metrics_file.flush()
        return records
    last_saved_step = None
    last_completed_step = start_step
    prefetched_batch = None
    checkpoint_gen_state = None
    for step in range(start_step, args.steps):
        current_lr = scheduler.get_last_lr()[0]
        optimizer.zero_grad(set_to_none=True)
        if prefetched_batch is None:
            x, y = dataset.batch('train', args.batch_size, train_generator, device)
        else:
            x, y = prefetched_batch
            prefetched_batch = None
        with train_autocast:
            result = model(x, y)
            loss = result['loss']
        del result['logits']
        checkpoint_gen_state = train_generator.get_state()
        scaler_scale_before_step = float(scaler.get_scale()) if scaler is not None else None
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if step + 1 < args.steps:
            prefetched_batch = dataset.batch('train', args.batch_size, train_generator, device)
        grad_norm = torch.nn.utils.clip_grad_norm_(train_params, args.grad_clip)
        grad_norm_value, loss_value = torch.stack((grad_norm.detach(), loss.detach())).tolist()
        finite_grad = math.isfinite(grad_norm_value)
        finite_step = finite_grad and math.isfinite(loss_value)
        if finite_step:
            if scaler is not None:
                scaler.step(optimizer)
            else:
                optimizer.step()
            scheduler.step()
            optimizer_steps_completed += 1
            model.invalidate_weight_caches()
            consecutive_nonfinite = 0
        else:
            nonfinite_events += 1
            consecutive_nonfinite += 1
            print(f'step {step + 1}: non-finite loss/gradient; optimizer and LR schedule skipped')
            if nan_diagnosis is None:
                nan_diagnosis = diagnose_non_finite(model, optimizer, x, y, result, grad_norm, step=step + 1, lr=current_lr, amp_dtype=amp_dtype, scaler=scaler, amp_scale_before_step=scaler_scale_before_step, vocab_size=cfg.vocab_size, recent_grad_norms=recent_grad_norms)
                print_non_finite_report(nan_diagnosis)
                nan_record_pending = True
                nan_checks = nan_diagnosis.get('checks', {})
                if nan_checks.get('parameters_nonfinite') or nan_checks.get('optimizer_state_nonfinite_keys'):
                    numerical_polluted = True
                    numerical_stop = True
            if consecutive_nonfinite >= 3:
                numerical_stop = True
        if scaler is not None:
            scaler.update()
        if finite_grad:
            recent_grad_norms.append((step + 1, grad_norm_value))
            if len(recent_grad_norms) > _NAN_GRAD_TREND_WINDOW:
                del recent_grad_norms[:-_NAN_GRAD_TREND_WINDOW]
        last_completed_step = step + 1
        pending_scalars.append(result['lm_loss'].detach())
        pending_scalars.append(result['aux_loss'].detach())
        del result, loss, x, y
        pending_records.append({'step': step + 1, 'optimizer_steps': optimizer_steps_completed, 'lr': current_lr, 'grad_norm': grad_norm_value if finite_grad else None})
        if nan_record_pending:
            pending_records[-1]['extra'] = {'nan_diagnosis': nan_diagnosis}
            nan_record_pending = False
        if numerical_stop:
            consecutive_text = '参数/优化器状态已被 NaN/Inf 污染' if numerical_polluted else '连续多次出现非有限损失/梯度'
            print(f'[nan] {consecutive_text}，在第 {step + 1} 步停止训练；请按排查路径处理后从污染前的检查点恢复（降低学习率、收紧 --grad-clip）。' + ('本次不会用受污染的参数覆盖 last.pt。' if numerical_polluted else ''))
            break
        if (step + 1) % args.eval_interval == 0:
            eval_metrics = evaluate(model, dataset, args.batch_size, args.eval_batches, device, amp_dtype=amp_dtype, compute_auc=_periodic_auc_enabled(getattr(args, 'periodic_auc', False), args.selection_metric))
            pending_records[-1].setdefault('extra', {}).update(eval_metrics)
            records = flush_records()
            print(records[-1])
            metrics_file.flush()
            model.train()
            live_analysis, live_status = monitor_training_curve(curve_records, curve_baseline_loss, last_curve_status, last_curve_warning_flags)
            if live_status is not None:
                previous_status = last_curve_status
                last_curve_status = live_status
                prior_flags = live_analysis.get('warning_flags')
                if isinstance(prior_flags, dict):
                    last_curve_warning_flags = {str(key): bool(value) for key, value in prior_flags.items()}
                print_curve_analysis(live_analysis)
                metrics_file.write(json.dumps(_json_safe({'step': 'curve_monitor', 'trigger': live_analysis.get('monitor_trigger', 'status_change'), 'previous_status': previous_status, 'reference_baseline_loss': curve_baseline_loss, **live_analysis}), ensure_ascii=False) + '\n')
                metrics_file.flush()
            selection_value = _finite_float(eval_metrics.get(args.selection_metric))
            val_loss = _finite_float(eval_metrics.get('val_loss'))
            if selection_value is None:
                pass
            elif selection_improved(selection_value, best_selection_value, metric_higher_is_better(args.selection_metric), args.early_stop_min_delta):
                saved_selected = save_selected_model(str(selected_model_path), model, step + 1, val_loss, args, selection_metric=args.selection_metric, selection_value=selection_value)
                if not saved_selected:
                    numerical_polluted = True
                    numerical_stop = True
                    print('[nan] 验证指标有限但模型权重含 NaN/Inf，拒绝写入 best.pt，停止训练')
                    break
                best_selection_value = selection_value
                best_selection_step = step + 1
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
            if args.early_stop_patience > 0 and evals_without_improvement >= args.early_stop_patience:
                early_stopped = True
                print(f'[early-stop] 选型指标 {args.selection_metric} 连续 {evals_without_improvement} 次未改善（最优 {best_selection_value:.4f} @ step {best_selection_step}），在第 {step + 1} 步提前结束训练；如需继续训练可调大 --steps 后 resume。')
                break
        save_at_steps = getattr(args, 'save_at_steps', None) or []
        is_branch_point = step + 1 in save_at_steps
        if (step + 1) % args.save_interval == 0 or is_branch_point:
            flush_records()
            saved_checkpoint = save_checkpoint(str(out_dir / 'last.pt'), model, optimizer, scaler, scheduler, step + 1, checkpoint_gen_state, args, numerical_state=checkpoint_numerical_state())
            if saved_checkpoint:
                last_saved_step = step + 1
                if is_branch_point:
                    save_checkpoint(str(out_dir / f'step_{step + 1}.pt'), model, optimizer, scaler, scheduler, step + 1, checkpoint_gen_state, args, numerical_state=checkpoint_numerical_state())
                    print(f'[wsd] 分支点检查点 step_{step + 1}.pt 已保存')
            else:
                numerical_polluted = True
                numerical_stop = True
                print('[nan] 存档前数值检查失败，停止训练并保留污染前的 last.pt')
                break
    flush_records()
    actual_training_peak_mb = _current_training_peak_mb()
    end_step = last_completed_step
    if last_saved_step != end_step and (not numerical_polluted):
        saved_checkpoint = save_checkpoint(str(out_dir / 'last.pt'), model, optimizer, scaler, scheduler, end_step, checkpoint_gen_state if checkpoint_gen_state is not None else train_generator.get_state(), args, numerical_state=checkpoint_numerical_state())
        if not saved_checkpoint:
            numerical_polluted = True
            numerical_stop = True
            print('[nan] 训练结束存档前数值检查失败：不写出受污染的 last.pt，并使用已有 best.pt 做评估')
    selection_policy = 'last'
    selection_note = None
    selected_val_loss = None
    if best_selection_step is not None:
        selected_payload = load_selected_model(str(selected_model_path), torch.device('cpu'))
        selected_step = None
        if selected_payload is not None:
            try:
                selected_step = int(selected_payload.get('step', -1))
            except (TypeError, ValueError, OverflowError):
                selected_step = None
        restore_error = None
        if selected_payload is not None and selected_step == best_selection_step:
            try:
                restore_selected_weights(model, selected_payload)
            except Exception as error:
                restore_error = str(error)
                selected_payload = None
        if selected_payload is not None and selected_step == best_selection_step:
            selection_policy = 'best_val'
            payload_val_loss = _finite_float(selected_payload.get('val_loss'))
            if payload_val_loss is not None:
                selected_val_loss = payload_val_loss
            print(f'[select] 使用验证集选中的模型做最终评估：step={best_selection_step}, {args.selection_metric}={best_selection_value:.4f}' + (f', val_loss={selected_val_loss:.4f}' if selected_val_loss is not None and args.selection_metric != 'val_loss' else ''))
        else:
            selection_note = 'best.pt 缺失、损坏或与验证记录不一致（如从旧检查点恢复）；最终评估回退到最后一步参数，模型未经验证集选择'
            if restore_error:
                selection_note += f'（恢复失败: {restore_error}）'
            print(f'[select] {selection_note}')
    routing_probe = None
    if not numerical_polluted and (not args.skip_routing_analysis):
        routing_probe = RoutingProbe(_unwrap_model(model).cfg, args.batch_size)
    val_bad_case_collector = None
    if args.bad_cases > 0 and (not numerical_polluted):
        val_bad_case_collector = _BadCaseCollector(dataset, 'val', args.batch_size, args.eval_batches, args.bad_cases, device=device)
    val_window_records = []
    val_group_stats_records = []
    final_metrics = evaluate(model, dataset, args.batch_size, args.eval_batches, device, amp_dtype=amp_dtype, compute_auc=True, routing_probe=routing_probe, bad_case_collector=val_bad_case_collector, bootstrap_replicates=getattr(args, 'eval_bootstrap', 0), window_sink=val_window_records, bootstrap_block=getattr(args, 'eval_bootstrap_block', 1), group_sink=val_group_stats_records)
    if val_window_records:
        overlap_fraction = None
        starts = dataset._fixed_starts.get('val', {}).get('starts')
        if starts is not None and len(starts) > 1:
            gaps = starts[1:] - starts[:-1]
            overlap_count = int((gaps < dataset.seq_len).sum().item())
            overlap_fraction = overlap_count / (len(starts) - 1)
        eval_windows_payload = {'split': 'val', 'data_sha256': dataset.data_sha256, 'split_sha256': dataset.split_sha256.get('val'), 'seq_len': int(dataset.seq_len), 'batch_size': int(args.batch_size), 'batches': int(args.eval_batches), 'n_windows': len(val_window_records), 'overlap_fraction': overlap_fraction, 'windows': val_window_records, 'group_stats': val_group_stats_records[0] if val_group_stats_records else None}
        with open(out_dir / 'eval_windows.json', 'w', encoding='utf-8') as f:
            json.dump(_json_safe(eval_windows_payload), f, ensure_ascii=False)
    min_val_loss, min_val_loss_step, _ = early_stop_state_from_records(read_metrics_records(metrics_path), 0.0, metric='val_loss')
    final_record = {'step': 'final', **final_metrics, 'early_stopped': early_stopped, 'best_val_loss': min_val_loss, 'best_val_step': min_val_loss_step, 'selection_metric': args.selection_metric, 'best_selection_value': best_selection_value, 'best_selection_step': best_selection_step, 'selected_model': selection_policy, 'selected_step': best_selection_step if selection_policy == 'best_val' else end_step, 'selected_val_loss': selected_val_loss if selection_policy == 'best_val' else None, 'selection_note': selection_note, 'trained_steps': max(0, int(end_step)), 'optimizer_steps': optimizer_steps_completed, 'nonfinite_events': nonfinite_events, 'numerical_stop': numerical_stop, 'numerical_polluted': numerical_polluted, 'prior_numerical_stop': prior_numerical_stop, 'numerical_integrity_ok': bool(not numerical_stop and (not numerical_polluted) and (not prior_numerical_stop) and (nonfinite_events == 0)), 'seed': args.seed, 'data_sha256': dataset.data_sha256, 'data_split_sha256': dict(dataset.split_sha256), 'data_effective_train_sha256': dataset.effective_train_sha256, 'script_sha256': run_metadata.get('script_sha256'), 'environment_id': run_metadata.get('environment_id'), 'git_commit': run_metadata.get('git_commit'), 'git_dirty': run_metadata.get('git_dirty'), 'run_started_utc': tracking_provenance.get('run_started_utc') or run_metadata.get('timestamp_utc'), 'run_segment_started_utc': run_metadata.get('timestamp_utc'), 'original_script_sha256': tracking_provenance.get('original_script_sha256') or run_metadata.get('script_sha256'), 'original_environment_id': tracking_provenance.get('original_environment_id') or run_metadata.get('environment_id'), 'original_git_commit': tracking_provenance.get('original_git_commit') or run_metadata.get('git_commit'), 'original_git_dirty': tracking_provenance.get('original_git_dirty'), 'resume_count': tracking_provenance.get('resume_count', 0), 'comparison_protocol_sha256': tracking_provenance.get('comparison_protocol_sha256'), 'run_completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    metrics_file.write(json.dumps(_json_safe(final_record), ensure_ascii=False) + '\n')
    metrics_file.flush()
    print(json.dumps(_json_safe({'final': final_metrics}), ensure_ascii=False))
    update_run_config(out_dir, 'final', final_record)
    update_run_config(out_dir, 'model_selection', {'policy': selection_policy, 'selection_metric': args.selection_metric, 'selected_metric_value': best_selection_value if selection_policy == 'best_val' else None, 'selected_step': best_selection_step if selection_policy == 'best_val' else end_step, 'selected_val_loss': selected_val_loss if selection_policy == 'best_val' else None, 'note': selection_note, 'evaluate_test_requested': bool(args.evaluate_test)})
    curve_analysis = analyze_training_curve(read_metrics_records(metrics_path), baseline_loss=curve_baseline_loss)
    metrics_file.write(json.dumps(_json_safe({'step': 'curve_analysis', 'reference_baseline_loss': curve_baseline_loss, **curve_analysis}), ensure_ascii=False) + '\n')
    metrics_file.flush()
    print_curve_analysis(curve_analysis)
    update_run_config(out_dir, 'curve_analysis', {'reference_baseline_loss': curve_baseline_loss, **curve_analysis})
    bad_cases_path = out_dir / 'bad_cases.json'
    bad_case_payload = {}
    if args.bad_cases > 0 and numerical_polluted:
        print('[bad-case] 参数/优化器状态已被 NaN/Inf 污染，跳过坏案例分析（输出不可信，不能作为改进线索）；请从污染前的检查点恢复并处理数值问题后再分析。')
    elif args.bad_cases > 0:
        bad_case_report = analyze_bad_cases(model, dataset, args.batch_size, args.eval_batches, device, amp_dtype=amp_dtype, split='val', top_n=args.bad_cases, collector=val_bad_case_collector)
        print_bad_cases(bad_case_report)
        bad_case_payload['val'] = bad_case_report
        metrics_file.write(json.dumps(_json_safe({'step': 'bad_cases', 'split': 'val', 'tokens_evaluated': bad_case_report['tokens_evaluated'], 'token_accuracy': bad_case_report['token_accuracy'], 'wrong_tokens': bad_case_report['wrong_tokens'], 'category_counts': bad_case_report['category_counts'], 'error_category_counts': bad_case_report['error_category_counts'], 'error_category_shares': bad_case_report['error_category_shares'], 'top_confusions': bad_case_report['top_confusions'], 'error_rate_by_position': bad_case_report['error_rate_by_position'], 'error_rate_by_train_frequency': bad_case_report['error_rate_by_train_frequency'], 'error_frequency_basis': bad_case_report['error_frequency_basis'], 'calibration': bad_case_report['calibration'], 'label_noise_verification': bad_case_report['label_noise_verification'], 'top_cases': bad_case_report['top_cases'], 'summary': bad_case_report['summary']}), ensure_ascii=False) + '\n')
        metrics_file.flush()

def _MAIN_part_006():
    global _, access_check, access_key, access_record, config_sha256, efficiency, environment_end, error, eval_duplicate_blocked, f, failure, intervention_batches, intervention_report, label_integrity_blocked, leakage_blocked, ledger_path, module, numerical_block, original_snapshot, reason, routing_report, shift_blocked, strict_blocked, test_audit, test_bad_case_collector, test_bad_cases, test_metrics, test_protocol_check, test_record
    if not args.skip_efficiency:
        efficiency = measure_efficiency(model, dataset, device, args.batch_size, amp_dtype=amp_dtype)
        if actual_training_peak_mb is not None:
            efficiency['actual_training_peak_memory_mb'] = actual_training_peak_mb
            efficiency['actual_training_peak_note'] = '训练循环真实峰值（含优化器状态与参数更新；resume 时取历史与本段的最大值）'
        update_run_config(out_dir, 'efficiency', efficiency)
        metrics_file.write(json.dumps(_json_safe({'step': 'efficiency', **efficiency}), ensure_ascii=False) + '\n')
        metrics_file.flush()
        print_efficiency(efficiency)
    if numerical_polluted:
        print('[routing] 参数/优化器状态已被 NaN/Inf 污染，跳过路由诊断（输出不可信，诊断结果没有意义）。')
    elif not args.skip_routing_analysis:
        routing_report = routing_probe.build_report(split='val')
        routing_report['batches_requested'] = args.eval_batches
        routing_report['batch_size'] = args.batch_size
        routing_report['selected_model'] = selection_policy
        routing_report['selected_step'] = best_selection_step if selection_policy == 'best_val' else end_step
        print_routing_analysis(routing_report)
        update_run_config(out_dir, 'routing_analysis', routing_report)
        with open(out_dir / 'routing.json', 'w', encoding='utf-8') as f:
            json.dump(_json_safe(routing_report), f, ensure_ascii=False, indent=2)
        metrics_file.write(json.dumps(_json_safe({'step': 'routing', **routing_report}), ensure_ascii=False) + '\n')
        metrics_file.flush()
        if args.routing_interventions:
            routing_probe.detach()
            intervention_batches = args.eval_batches if args.routing_intervention_batches == 0 else args.routing_intervention_batches if args.routing_intervention_batches is not None else min(5, args.eval_batches)
            if intervention_batches <= 0:
                print('[routing-intervention] batches<=0，跳过。')
            else:
                intervention_report = analyze_routing_interventions(model, dataset, args.batch_size, intervention_batches, device, amp_dtype, seed=int(args.seed))
                print_routing_interventions(intervention_report)
                update_run_config(out_dir, 'routing_interventions', intervention_report)
                with open(out_dir / 'routing_interventions.json', 'w', encoding='utf-8') as f:
                    json.dump(_json_safe(intervention_report), f, ensure_ascii=False, indent=2)
                metrics_file.write(json.dumps(_json_safe({'step': 'routing_interventions', **intervention_report}), ensure_ascii=False) + '\n')
                metrics_file.flush()
    numerical_block = numerical_polluted or numerical_stop or prior_numerical_stop
    if args.evaluate_test and numerical_block:
        reason = '参数/优化器状态已被 NaN/Inf 污染' if numerical_polluted else '本次训练因连续非有限损失/梯度而数值停止' if numerical_stop else '续跑前的原运行曾在数值上停止'
        print(f'[test] {reason}：本次不消耗唯一的测试集评估名额，测试集保持尘封。请从污染前的检查点恢复、修复数值稳定性后，再显式 --evaluate-test。')
    elif args.evaluate_test:
        if test_evaluation_sealed(out_dir, getattr(args, 'test_ledger', None)):
            print('[test] 输出目录已有测试集评估记录/封条；测试集在该目录的运行中已解封过一次，拒绝重复评估。如需新的最终结论，请使用全新的 --out 目录，并在确认模型与超参数冻结后再加 --evaluate-test。')
        else:
            test_audit = audit_data_splits(dataset, include_test=True)
            print_data_audit(test_audit)
            leakage_blocked = not args.allow_data_leakage and (not enforce_data_leakage_gate(test_audit, 'test'))
            eval_duplicate_blocked = not args.allow_eval_duplicates and (not enforce_evaluation_duplicate_gate(test_audit, 'test'))
            strict_blocked = args.strict_data_audit and (not enforce_strict_data_audit(test_audit, 'test'))
            label_integrity_blocked = not args.allow_corrupt_test_labels and (not enforce_test_label_integrity(test_audit, 'test'))
            test_audit['distribution_shift_gate'] = {'max_byte_js': args.max_byte_shift, 'max_bigram_js': args.max_bigram_shift, 'max_conditional_js': args.max_conditional_shift}
            shift_blocked = not enforce_distribution_shift_gate(test_audit, max_byte_js=args.max_byte_shift, max_bigram_js=args.max_bigram_shift, max_conditional_js=args.max_conditional_shift, context='test')
            if leakage_blocked or eval_duplicate_blocked or strict_blocked or shift_blocked or label_integrity_blocked:
                print('[test] 数据审计在测试集上发现泄露/损坏或分布偏移问题：为避免在污染数据上产出最终结论，本次拒绝解封测试集（不写入 test 记录，名额未消耗）。请先清洗数据后重跑。')
            else:
                test_protocol_check = probe_data_pipeline(dataset, device, include_test=True)
                if not test_protocol_check['ok']:
                    print('[test] 测试集数据管线检查未通过，拒绝解封：')
                    for failure in test_protocol_check['failures']:
                        print(f'  - {failure}')
                    raise SystemExit('测试集数据管线检查未通过：请先修复输入/标签错位或窗口越界，再评估测试集。')
                ledger_path = Path(args.test_ledger) if args.test_ledger else out_dir / 'test_access_ledger.jsonl'
                if args.test_ledger is None:
                    print('[test] [warn] 未指定 --test-ledger：访问账本只写在本次输出目录，无法阻止其它目录重复解封同一测试集')
                access_key = _test_access_key(dataset.data_sha256, dataset.split_sha256, dataset.test_fraction)
                access_check = check_test_access_ledger(ledger_path, access_key, out_dir)
                if _test_access_repeat_blocked(access_check) and (not args.allow_repeat_test_access):
                    raise SystemExit(_repeat_test_access_error(access_check))
                config_sha256 = None
                try:
                    config_sha256 = hashlib.sha256((out_dir / 'config.json').read_bytes()).hexdigest()
                except OSError:
                    pass
                access_record = {**access_key, 'out_dir': str(out_dir), 'study_id': args.study_id, 'seed': args.seed, 'script_sha256': run_metadata.get('script_sha256'), 'environment_id': run_metadata.get('environment_id'), 'git_commit': run_metadata.get('git_commit'), 'config_sha256': config_sha256, 'hypothesis': args.hypothesis, 'primary_metric': args.primary_metric, 'selected_step': best_selection_step if selection_policy == 'best_val' else end_step, 'model_selection': selection_policy, 'allow_repeat_access': bool(args.allow_repeat_test_access), 'accessed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'note': '测试集访问账本；同一 data/split 指纹的全部访问都应进入最终报告'}
                append_test_access_ledger(ledger_path, access_record)
                print(f'[test] 访问账本已记录: {ledger_path}（历史访问 {access_check['matching_accesses'] + 1} 次）')
                metrics_file.write(json.dumps(_json_safe({'step': 'test_started', 'split': 'test'}), ensure_ascii=False) + '\n')
                metrics_file.flush()
                write_test_evaluation_seal(out_dir, {'dataset_sha256': dataset.data_sha256, 'test_fraction': dataset.test_fraction, 'split_sha256': dict(dataset.split_sha256)})
                test_bad_case_collector = None
                if args.bad_cases > 0:
                    test_bad_case_collector = _BadCaseCollector(dataset, 'test', args.batch_size, args.eval_batches, args.bad_cases, device=device)
                test_metrics = evaluate(model, dataset, args.batch_size, args.eval_batches, device, amp_dtype=amp_dtype, split='test', compute_auc=True, bad_case_collector=test_bad_case_collector, bootstrap_replicates=getattr(args, 'eval_bootstrap', 0), bootstrap_block=getattr(args, 'eval_bootstrap_block', 1))
                if args.bad_cases > 0:
                    test_bad_cases = analyze_bad_cases(model, dataset, args.batch_size, args.eval_batches, device, amp_dtype=amp_dtype, split='test', top_n=args.bad_cases, collector=test_bad_case_collector)
                    print_bad_cases(test_bad_cases)
                    bad_case_payload['test'] = test_bad_cases
                    metrics_file.write(json.dumps(_json_safe({'step': 'bad_cases', 'split': 'test', 'tokens_evaluated': test_bad_cases['tokens_evaluated'], 'token_accuracy': test_bad_cases['token_accuracy'], 'wrong_tokens': test_bad_cases['wrong_tokens'], 'category_counts': test_bad_cases['category_counts'], 'error_category_counts': test_bad_cases['error_category_counts'], 'error_category_shares': test_bad_cases['error_category_shares'], 'top_confusions': test_bad_cases['top_confusions'], 'top_cases': test_bad_cases['top_cases'], 'summary': test_bad_cases['summary']}), ensure_ascii=False) + '\n')
                    metrics_file.flush()
                test_record = {'step': 'test', **test_metrics, 'data_audit': test_audit, 'data_protocol_check': test_protocol_check, 'seed': args.seed, 'data_sha256': dataset.data_sha256, 'data_split_sha256': dict(dataset.split_sha256), 'data_effective_train_sha256': dataset.effective_train_sha256, 'script_sha256': run_metadata.get('script_sha256'), 'git_commit': run_metadata.get('git_commit'), 'run_started_utc': tracking_provenance.get('run_started_utc') or run_metadata.get('timestamp_utc'), 'original_script_sha256': tracking_provenance.get('original_script_sha256') or run_metadata.get('script_sha256'), 'original_environment_id': tracking_provenance.get('original_environment_id') or run_metadata.get('environment_id'), 'original_git_commit': tracking_provenance.get('original_git_commit') or run_metadata.get('git_commit'), 'resume_count': tracking_provenance.get('resume_count', 0), 'comparison_protocol_sha256': tracking_provenance.get('comparison_protocol_sha256'), 'study_id': args.study_id, 'test_ledger': str(ledger_path), 'test_access': access_record, 'model_selection': selection_policy, 'selected_step': best_selection_step if selection_policy == 'best_val' else end_step, 'hypothesis': args.hypothesis, 'primary_metric': args.primary_metric, 'run_completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
                metrics_file.write(json.dumps(_json_safe(test_record), ensure_ascii=False) + '\n')
                metrics_file.flush()
                print(json.dumps(_json_safe({'test': test_metrics}), ensure_ascii=False))
                print('[test] 测试集已解封并完成唯一一次评估；请不要根据该结果继续调参，否则测试集将退化为隐式验证集。')
    if bad_case_payload:
        with open(bad_cases_path, 'w', encoding='utf-8') as f:
            json.dump(_json_safe(bad_case_payload), f, ensure_ascii=False, indent=2)
    metrics_file.close()
    try:
        original_snapshot = None
        try:
            original_snapshot, _ = load_environment_snapshot(out_dir)
        except (OSError, ValueError):
            original_snapshot = None
        environment_end = write_run_environment_end_record(out_dir, run_metadata, original_snapshot)
    except (OSError, ValueError) as error:
        environment_end = None
        print(f'[env][warn] 无法写入运行结束环境快照: {error}')
    if environment_end is not None:
        try:
            update_run_config(out_dir, 'environment_end', environment_end)
        except (OSError, ValueError) as error:
            print(f'[env][warn] 无法把结束环境摘要写入 config.json: {error}')
    dataset.clear_eval_batch_cache()
    for module in model.modules():
        if isinstance(module, SparseMoE):
            module.clear_padded_buffer_cache()

