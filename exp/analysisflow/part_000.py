from __future__ import annotations

def _PA_part_000():
    global _cost_text, _n_text, _stat, adequacy, aggregate, aggregates, alignment, arm, attrition_summary, autocorrelated_splits, balance_scale, baseline_mode, baseline_preregistration, cell, check, common_step, common_step_sensitivity, component_verdicts, compute_matched, compute_parity_failed, contrasts, controlled, cost_header, cost_rows, coverage, cross_violations, current_derived, curve_mean, delta, delta_text, design_violations, details, deterministic_cells, deterministic_conflict_cells, direction, direction_text, distinct, duplicate_seed_cells, effects, entry, factorial, factorial_balance_scale, factorial_compute_matched, factorial_coverage, factorial_fragility, factors, failure_rates, fragility_summary, group_keys, guard, header, high, higher_is_better, hypotheses, imbalance, item, key, line, low, margin_preregistration, matched_coverage, mean, message, metric_columns, metric_report, mismatch, missing_efficiency_cells, numeric, numerical_failures, order_audit, p_holm, p_text, parts, present_routing_fields, provenance, randomness, rate, rate_text, ratio, ratio_text, reference, resolved_baseline, routing_header, routing_summary_fields, row, rq_text, scale_guard_report, scale_tiers, selection_bias, selection_metric_mismatch, short, single_factor_design, tier, timing_ci_lines, tuning_seed_overlap, underpowered, value, values_text, violation
    '\n    打印完整的对比分析：受控变量检查、按种子聚合的\n    Mean ± Std、单因子显著性检验（含多重比较校正与效应量\n    95% 置信区间）、80% 功效的最小可检出效应（MDE）与\n    TOST 等价性检验，以及消融覆盖检查。\n\n    ``ablation_modes`` 声明本轮消融必须覆盖的延迟合并方案\n    （--suite-ablation --ablation-modes）；为 None 且运行集\n    涉及延迟合并时默认核对项目声明的全部方案（A、B）。\n    ``allow_partial_ablation`` 显式允许阶段性部分消融，但报告\n    仍记录 ``partial`` 与未覆盖方案，不会把部分覆盖说成完整。\n\n    ``equivalence_margin`` 是相对基线均值的等价界比例\n    （--equivalence-margin，默认 1%）：报告用 TOST 区分\n    「证据支持无实质差异」与「未发现显著差异（不确定）」，\n    并把 MDE 与观测效应所需的种子数列出；为 None 时只做\n    显著性检验（不推荐，不显著会被误读为等价）。\n\n    返回结构化结果，供写入汇总报告。\n    '
    if factors is None:
        factors = list(DEFAULT_COMPARISON_FACTORS)
    if metric not in COMPARISON_METRIC_KEYS:
        raise ValueError(f'未知的对比指标: {metric}（可用: ' + ', '.join(COMPARISON_METRIC_KEYS) + '）')
    baseline_preregistration = resolve_preregistered_baseline(rows, baseline_mode)
    resolved_baseline = baseline_preregistration.get('baseline_mode')
    baseline_mode = str(resolved_baseline) if resolved_baseline is not None else None
    margin_preregistration = equivalence_margin_preregistration(rows, equivalence_margin)
    compute_matched = annotate_compute_matched_metrics(rows)
    common_step = annotate_common_step_metrics(rows, target_step=common_step)
    curve_mean = annotate_curve_metrics(rows, target_step=common_step.get('target_step') if not common_step.get('target_step_data_dependent') else None)
    selection_bias = selection_bias_report(rows, metric, common_step)
    controlled = check_controlled_variables(rows, factors)
    single_factor_design = {'design_keys': list(SINGLE_FACTOR_DESIGN_KEYS), 'violations': [violation['key'] for violation in controlled['violations'] if violation['key'] in SINGLE_FACTOR_DESIGN_KEYS], 'unchecked_keys': [key for key in SINGLE_FACTOR_DESIGN_KEYS if key in controlled.get('unchecked_keys', [])], 'ok': not any((violation['key'] in SINGLE_FACTOR_DESIGN_KEYS for violation in controlled['violations']))}
    single_factor_design['advisory'] = '模型结构/数据口径在运行间不一致：这些运行不是同一个实验单元，不能作为单因子对比；请统一设置重跑，或把该键显式声明为实验因子后按多水平单因子分析' if not single_factor_design['ok'] else '单因子设计所需的模型结构/数据口径键均已核对一致' if not single_factor_design['unchecked_keys'] else '部分单因子设计键在所有运行中未记录，无法核对一致'
    group_keys = []
    for key in factors:
        if key == 'seed':
            continue
        distinct = {repr(_stable_factor_key(_row_factor_value(row, key))) for row in rows}
        if len(distinct) > 1:
            group_keys.append(key)
    aggregates = aggregate_runs(rows, group_keys)
    contrasts = factor_contrasts(aggregates, group_keys, metric, equivalence_margin=equivalence_margin)
    reference = compare_cells_to_baseline(aggregates, baseline_mode, metric, equivalence_margin=equivalence_margin)
    coverage = ablation_coverage(rows, factors, ablation_modes, allow_partial=allow_partial_ablation)
    balance_scale = balance_strength_report(rows)
    factorial_balance_scale = factorial_balance_strength(rows)
    coverage['balance_confounds'] = list(factorial_balance_scale.get('advisories') or []) if factorial_balance_scale.get('confounded') else []
    randomness = same_seed_randomness_check(rows)
    coverage['randomness_confounds'] = randomness['confounds']
    coverage['randomness_unverified'] = randomness['unverified']
    order_audit = run_order_drift_report(rows, metric)
    factorial = factorial_effects(aggregates, group_keys, metric, equivalence_margin=equivalence_margin)
    factorial_coverage = coverage
    if metric == _COMPUTE_MATCHED_METRIC:
        factorial_coverage = dict(coverage)
        factorial_coverage['compute_parity'] = []
        factorial_coverage['width_confounds'] = []
    annotate_factorial_confounds(factorial, controlled, factorial_coverage)
    factorial_compute_matched = None
    compute_parity_failed = any((not entry.get('matched') for entry in coverage.get('compute_parity', [])))
    if metric != _COMPUTE_MATCHED_METRIC and compute_parity_failed and (compute_matched.get('target_flops') is not None):
        factorial_compute_matched = factorial_effects(aggregates, group_keys, _COMPUTE_MATCHED_METRIC, equivalence_margin=equivalence_margin)
        matched_coverage = dict(coverage)
        matched_coverage['compute_parity'] = []
        matched_coverage['width_confounds'] = []
        annotate_factorial_confounds(factorial_compute_matched, controlled, matched_coverage)
    common_step_sensitivity = factorial_common_step_sensitivity(aggregates, group_keys, metric, factorial, coverage, controlled=controlled, equivalence_margin=equivalence_margin)
    coverage['common_step_sensitivity'] = common_step_sensitivity
    hypotheses = hypothesis_registry(rows, metric)
    component_verdicts = ablation_component_verdicts(factorial, coverage, metric_preregistration=hypotheses)
    duplicate_seed_cells = [{'cell': aggregate['label'], 'n_runs': aggregate['n_runs'], 'duplicate_seeds': aggregate['duplicate_seeds']} for aggregate in aggregates if aggregate.get('duplicate_seeds')]
    deterministic_conflict_cells = [{'cell': aggregate['label'], 'metrics': aggregate['deterministic_repeat_conflicts']} for aggregate in aggregates if aggregate.get('deterministic_repeat_conflicts')]
    alignment = hypothesis_alignment(rows, aggregates, contrasts, metric)
    adequacy = baseline_adequacy(rows, baseline_mode, metric)
    tuning_seed_overlap = tuning_seed_overlap_report(rows)
    imbalance = imbalance_metric_advisory(rows, metric)
    selection_metric_mismatch = [{'run': str(row.get('run')), 'mode': row.get('mode'), 'selection_metric': row.get('selection_metric')} for row in rows if metric in _SELECTION_METRIC_KEYS and str(row.get('mode')) != 'ngram' and (row.get('selection_metric') is not None) and (str(row.get('selection_metric')) != metric)] if metric in _SELECTION_METRIC_KEYS else []
    higher_is_better = metric_higher_is_better(metric)
    factorial_fragility = []
    if isinstance(factorial, dict):
        effects = factorial.get('effects')
        if isinstance(effects, list):
            factorial_fragility = [effect for effect in effects if isinstance(effect, dict)]
    fragility_summary = _fragility_summary(list(contrasts) + list(reference) + factorial_fragility, label='combined')
    attrition_summary = _attrition_bounds_summary(list(contrasts) + list(reference) + factorial_fragility, label='combined')
    print()
    print('[controlled] 受控变量检查：除声明因子外，优化器 / 学习率 / 训练轮数 / 数据增强等协议项必须逐项一致')
    print('  声明因子: ' + ', '.join(factors))
    if controlled['statistical_runs']:
        print('  统计基线（ngram，无优化器与训练轮数）不参与优化器类受控变量检查，但数据/评估口径仍需一致: ' + ', '.join(controlled['statistical_runs']))
    if not rows:
        print('  无运行目录可分析')
    elif not controlled['violations']:
        if controlled.get('unchecked_keys'):
            print(f'  已核对 {len(controlled['checked_keys'])} 项协议；另有 {len(controlled['unchecked_keys'])} 项在所有运行中都未记录，不能算作已控制')
        else:
            print(f'  OK：{len(controlled['checked_keys'])} 项协议在声明因子之外完全一致')
    else:
        for violation in controlled['violations']:
            values_text = ', '.join((f'{run}={_format_factor_value(value)}' for run, value in violation['values'].items()))
            print(f'  [violation] {violation['key']}: {values_text}')
        print('  以上差异会使对比结论无法归因到待验证模块；请统一这些设置后重跑，或在 --compare-factors 中显式声明为因子')
        design_violations = single_factor_design['violations']
        if design_violations:
            print('  [single-factor] 差异涉及模型结构/数据口径（' + ', '.join(design_violations) + '）：这些运行不是同一个实验单元，不能把它们当作「只改一个因子」的对比；请分别重跑以保持其余设置一致，或显式声明该键为实验因子后按多水平单因子分析')
    if controlled.get('unchecked_keys'):
        print('  [warn] 以下协议项所有运行都未记录，无法核对一致性: ' + ', '.join(controlled['unchecked_keys']))
    cross_violations = controlled.get('cross_family_violations') or []
    if cross_violations:
        for violation in cross_violations:
            values_text = ', '.join((f'{run}={_format_factor_value(value)}' for run, value in violation['values'].items()))
            print(f'  [violation] 跨 family {violation['key']}: {values_text}')
        print('  神经运行与 ngram 统计基线的数据/评估口径不一致：跨 family 的基线强弱判断无效，请先统一口径重跑')
    elif controlled.get('cross_family_checked_keys'):
        print(f'  OK：跨 family 的 {len(controlled['cross_family_checked_keys'])} 项数据/评估口径一致')
    if controlled.get('cross_family_unchecked_keys'):
        print('  [warn] 以下跨 family 口径项缺少记录，无法核对: ' + ', '.join(controlled['cross_family_unchecked_keys']))
    provenance = check_run_provenance(rows)
    numerical_failures = numerical_failure_report(rows)
    failure_rates = failure_rate_report(rows)
    scale_tiers = {}
    for row in rows:
        guard = row.get('scale_guard')
        if not isinstance(guard, dict):
            continue
        tier = str(guard.get('scale_tier'))
        scale_tiers.setdefault(tier, []).append(str(row.get('run')))
    scale_guard_report = {'tiers': {tier: len(names) for tier, names in scale_tiers.items()}, 'prototype_runs': scale_tiers.get('prototype', []), 'note': 'prototype 级（<20 token/param、<1 epoch、seq_len<512 或专家数≤8）只支持原型可行性；不能据此声称大规模 MoE/长上下文/推理任务的普遍结论'}
    print()
    print('[provenance] 代码版本与数值环境核对：对比结论要求同一份实验代码、同一 Python/torch/CUDA/cuDNN 与设备（跨实现/环境比较无法归因到方法本身）')
    if not rows:
        print('  无运行目录可分析')
    elif provenance['mismatches']:
        for mismatch in provenance['mismatches']:
            values_text = ', '.join((f'{run}={_format_factor_value(value)}' for run, value in mismatch['values'].items()))
            print(f'  [violation] {mismatch['key']}: {values_text}')
        print('  以上差异来自代码版本或运行环境，不是待验证的实验因子；请在同一份代码与同一环境下重跑后再比较')
    elif provenance['checked_keys']:
        print(f'  OK：{len(provenance['checked_keys'])} 项代码/环境记录完全一致')
    if provenance['unchecked_keys']:
        print('  [warn] 以下代码/环境项所有运行都未记录，无法核对: ' + ', '.join(provenance['unchecked_keys']))
    if provenance['hashseed_unset_runs']:
        print('  [warn] 以下运行启动时未固定 PYTHONHASHSEED（set_seed 的 setdefault 无法追溯地固定当前解释器；严格逐位复现请在命令行显式设置）: ' + ', '.join(provenance['hashseed_unset_runs']))
    if provenance['hashseed_unrecorded_runs']:
        print('  [warn] 以下运行未记录启动时 PYTHONHASHSEED 状态，无法核对是否固定（旧版本运行需在命令行显式设置后重跑）: ' + ', '.join(provenance['hashseed_unrecorded_runs']))
    if provenance['commits_differ']:
        print('  [note] 各运行的 git commit 不同但实验脚本指纹一致（差异来自与本实验无关的文件）；如需严格的同代码版本对比，请在同一 commit 上重跑。')
    print()
    print('[numerical] 数值完整性：NaN/Inf、数值停止、污染检查点续跑或 diverged 运行不进入聚合与显著性检验')
    if numerical_failures['failed']:
        for item in numerical_failures['failed']:
            print(f'  [excluded] {item['run']}（mode={item['mode']}, seed={item['seed']}） curve={item['curve_status']} nonfinite={item['nonfinite_events']} stop={item['numerical_stop']} polluted={item['numerical_polluted']} prior_stop={item['prior_numerical_stop']}')
        if numerical_failures.get('by_mode'):
            print('  各 mode 失败率: ' + '; '.join((f'{entry['mode']}={entry['failed_runs']}/{entry['attempted_runs']}' for entry in numerical_failures['by_mode'])))
        print('  [warn] ' + str(numerical_failures['advisory']))
    elif numerical_failures['unknown']:
        print('  [note] 以下运行缺少数值完整性记录（旧运行），无法确认是否发生过 NaN/Inf: ' + ', '.join(numerical_failures['unknown']))
    else:
        print('  OK：所有运行都记录了数值完整性且未发现失败')
    print()
    print('[failure] 失败率与成功条件性能：只比较成功运行会掩盖稳定性成本，失败率必须与成功条件指标一起报告')
    for arm in failure_rates['arms']:
        rate = arm.get('failure_rate')
        rate_text = 'n/a' if rate is None else _format_finite(rate, '.1%')
        print(f'  {arm['mode']}: planned={arm['planned_runs']} failed={arm['failed_runs']} unknown={arm['unknown_runs']} success={arm['success_runs']} failure_rate={rate_text}')
    if failure_rates['differential_failure']:
        print('  [warn] ' + str(failure_rates['differential_failure']['note']))
    print('  [note] ' + str(failure_rates['retry_policy']))
    if scale_guard_report['prototype_runs']:
        print('  [scale] [warn] 以下运行是原型级规模，结论不能外推到大规模/长上下文：' + ', '.join(scale_guard_report['prototype_runs'][:10]))
    print()
    print('[hypothesis] 科学假设登记（每个改动前写明预期效果与判定指标；缺假设的对比与排列组合超参数无法区分）')
    if not hypotheses['cells']:
        print('  无运行可检查')
    for cell in hypotheses['cells']:
        rq_text = ' / '.join(cell['research_questions']) if cell['research_questions'] else '-'
        if cell['hypotheses']:
            print(f'  {cell['mode']}（{cell['n_runs']} 个运行, RQ: {rq_text}）: ' + ' | '.join(cell['hypotheses']))
        else:
            print(f'  {cell['mode']}（{cell['n_runs']} 个运行）: 未登记假设')
    for message in hypotheses['advisories']:
        print(f'  [warn] {message}')
    print()
    print('[hypothesis-alignment] 预注册方向/最小效应核对（--primary-metric / --hypothesis-direction / --hypothesis-min-effect；观测 Δ = 变体 - 对照）')
    if not alignment['checks']:
        print('  无预注册对比可核对')
    else:
        for check in alignment['checks']:
            delta = check.get('delta')
            delta_text = f'Δ={_format_finite(delta, '+.4f')}' if isinstance(delta, (int, float)) else 'Δ=NA'
            direction = check.get('expected_direction')
            direction_text = '/'.join(direction) if isinstance(direction, list) else str(direction)
            p_holm = check.get('p_value_holm')
            p_text = f'Holm p={_format_finite(p_holm)}' if isinstance(p_holm, (int, float)) else 'Holm p=NA'
            print(f'  [{check['status']}] {check['baseline']}→{check['variant']}  {delta_text}  预期 {direction_text}  {p_text}')
    for message in alignment['advisories']:
        print(f'  [warn] {message}')
    print()
    print('[aggregate] 按实验单元聚合（Mean ± Std，样本标准差；std 反映不同随机种子的方差；CI 为均值的 t 分布 95% 置信区间，n<2 或确定性折叠后不给出；n(seed) 中括号为独立种子数，与运行数不同时说明存在伪重复或确定性折叠）')
    if selection_metric_mismatch:
        print(f'  [selection] [warn] 以下运行的 checkpoint 选型指标不是当前判定指标 {metric}：最终模型不是按假设的判定口径选出的，指标差异可能混入选型规则；请在训练时用 --selection-metric 对齐判定指标后重跑: ' + ', '.join((f'{item['run']}（{item['selection_metric']}）' for item in selection_metric_mismatch)))
    if selection_bias.get('advisory'):
        print('  [selection-bias] [warn] ' + str(selection_bias['advisory']) + '；挑选最优 checkpoint 的极值噪声会被误当成方法收益，不能把纯 val 选型结果当作无偏证据')
    elif selection_bias.get('applicable') and selection_bias.get('unknown_selection_runs'):
        print('  [selection-bias] [note] 以下运行缺少选型记录，无法确认 final 指标是否经过验证集选型: ' + ', '.join(selection_bias['unknown_selection_runs']))
    if selection_bias.get('opportunity_imbalance'):
        print('  [selection-opportunities] [warn] ' + str(selection_bias['opportunity_advisory']))
    elif selection_bias.get('opportunity_advisory'):
        print('  [selection-opportunities] [note] ' + str(selection_bias['opportunity_advisory']))
    if aggregates:
        metric_columns = ['val_loss']
        if metric != 'val_loss':
            metric_columns.append(metric)
        header = f'{'cell':<40} {'n(seed)':>8} ' + ' '.join((f'{column:>39}' for column in metric_columns))
        print(f'  判定指标（--compare-metric）: {metric}' + ('' if metric == 'val_loss' else '（表中同时给出 val_loss 以便对照）'))
        print(header)
        print('-' * len(header))

        def _stat(aggregate, base_metric: str) -> str:
            mean = aggregate.get(f'{base_metric}_mean')
            if mean is None:
                return '-'
            std = aggregate.get(f'{base_metric}_std')
            text = f'{_format_finite(mean)}' if std is None else f'{_format_finite(mean)}±{_format_finite(std)}'
            ci = (aggregate.get('ci95') or {}).get(base_metric)
            if ci:
                text += f' CI[{_format_finite(ci.get('low'))},{_format_finite(ci.get('high'))}]'
            return text

        def _n_text(aggregate) -> str:
            runs = int(aggregate['n_runs'])
            seeds = int(aggregate.get('effective_seed_count', runs) or runs)
            return str(runs) if seeds == runs else f'{runs}({seeds})'
        for aggregate in aggregates:
            print(f'{str(aggregate['label'])[:39]:<40} {_n_text(aggregate):>8} ' + ' '.join((f'{_stat(aggregate, column):>39}' for column in metric_columns)))
        underpowered = [aggregate['label'] for aggregate in aggregates if not aggregate.get('deterministic') and int(aggregate.get('effective_seed_count', 0)) < _RECOMMENDED_SEEDS]
        if underpowered:
            print(f'  统计可靠性不足：以下实验单元种子数 < {_RECOMMENDED_SEEDS}（建议 3~5 个不同随机种子并汇报 Mean ± Std）: ' + '; '.join(underpowered))
        deterministic_cells = [aggregate['label'] for aggregate in aggregates if aggregate.get('deterministic')]
        if deterministic_cells:
            print('  [note] 确定性基线（ngram）不做种子重复：换 seed 重跑是同一估计的复制，已折叠为 1 个观测；其评估不确定性应来自评估窗口/时间切分的 bootstrap 或多次切分，而不是种子数: ' + '; '.join(deterministic_cells))
        if deterministic_conflict_cells:
            print('  [warn] 以下确定性基线单元的不同 seed 运行给出了不同指标值：确定性管线本应逐位一致，可能存在数据/评估差异；已折叠为 1 个观测，但不能按多种子重复解读: ' + '; '.join((f'{item['cell']}（' + ', '.join(item['metrics']) + '）' for item in deterministic_conflict_cells)))
        if duplicate_seed_cells:
            details = '; '.join((f'{cell['cell']}（种子 ' + ', '.join((str(seed) for seed in cell['duplicate_seeds'])) + f' 各出现多次，共 {cell['n_runs']} 个运行）' for cell in duplicate_seed_cells))
            print('  [warn] 伪重复：以下实验单元含重复随机种子（复制的同一运行不是独立重复实验；已按独立种子折叠为一个观测，复制运行不增加 n）: ' + details)
        routing_summary_fields = ['routing_jaccard_branch', 'routing_slot_reuse_branch', 'routing_r2_load_entropy', 'routing_r2_dead_experts', 'routing_branch_cosine']
        present_routing_fields = [field for field in routing_summary_fields if metric != field and any((aggregate.get(f'{field}_mean') is not None for aggregate in aggregates))]
        if present_routing_fields:
            print()
            print('[routing] RQ3/RQ4 路由指标按实验单元聚合（Mean ± Std；可用 --compare-metric 直接选为判定指标做显著性检验）')
            routing_header = f'{'cell':<40} ' + ' '.join((f'{field.replace('routing_', ''):>16}' for field in present_routing_fields))
            print(routing_header)
            print('-' * len(routing_header))
            for aggregate in aggregates:
                print(f'{str(aggregate['label'])[:39]:<40} ' + ' '.join((f'{_stat(aggregate, field):>16}' for field in present_routing_fields)))
    else:
        print('  无有效指标可聚合')
    print()
    print('[metrics] 类别不平衡与指标口径提示')
    if not imbalance['advisories']:
        print('  OK：未发现多数类占比过半的运行，accuracy/micro 与 macro/PR-AUC 口径可对照解读')
    for message in imbalance['advisories']:
        print(f'  [note] {message}')
    autocorrelated_splits = {}
    for row in rows:
        for key, value in row.items():
            if not isinstance(key, str) or not key.endswith('_window_lag1_autocorrelation'):
                continue
            numeric = _finite_float(value)
            if numeric is not None and numeric >= _EVAL_WINDOW_AUTOCORRELATION:
                autocorrelated_splits[key] = numeric
    if autocorrelated_splits:
        print(f'  [warn] 以下评估 split 的窗口 lag-1 自相关 ρ̂ ≥ {_EVAL_WINDOW_AUTOCORRELATION}（bootstrap 区间已按 AR(1) 方差膨胀放宽；窗口不独立时未调整的区间会偏窄，跨实验比较请保持同一窗口口径）: ' + '; '.join((f'{key}={autocorrelated_splits[key]:.2f}' for key in sorted(autocorrelated_splits))))
    cost_rows = [aggregate for aggregate in aggregates if aggregate.get('costs')]
    if cost_rows:
        print()
        print('[efficiency] 各实验单元效率（运行结束时测量；参数量/FLOPs 可跨环境比较，其余字段受硬件影响，仅在同一环境内可比）')
        cost_header = f'{'cell':<40} {'params':>10} {'GFLOPs':>9} {'lat(ms)':>9} {'train(ms)':>10} {'mem(MB)':>9}'
        print(cost_header)
        print('-' * len(cost_header))

        def _cost_text(aggregate, key, scale=1.0, spec='.3f') -> str:
            entry = aggregate['costs'].get(key)
            if not entry or entry.get('mean') is None:
                return '-'
            return _format_finite(entry.get('mean'), spec, scale)
        for aggregate in cost_rows:
            print(f'{str(aggregate['label'])[:39]:<40} {_cost_text(aggregate, 'n_params', 1.0, '.0f'):>10} {_cost_text(aggregate, 'flops_per_forward', 1e-09):>9} {_cost_text(aggregate, 'latency_ms_per_forward'):>9} {_cost_text(aggregate, 'training_ms_per_step', 1.0, '.1f'):>10} {_cost_text(aggregate, 'peak_memory_mb', 1.0, '.1f'):>9}')
        timing_ci_lines = []
        for aggregate in cost_rows:
            parts = []
            for key, short in (('latency_ms_per_forward', 'lat'), ('training_ms_per_step', 'train')):
                entry = aggregate['costs'].get(key)
                if not isinstance(entry, dict):
                    continue
                low = entry.get('ci95_low')
                high = entry.get('ci95_high')
                mean = entry.get('mean')
                if not (isinstance(low, (int, float)) and isinstance(high, (int, float)) and isinstance(mean, (int, float))):
                    continue
                parts.append(f'{short} {_format_finite(mean, '.3f')} [{_format_finite(low, '.3f')},{_format_finite(high, '.3f')}]')
            if parts:
                timing_ci_lines.append(f'{aggregate['label']}: ' + '  '.join(parts))
        if timing_ci_lines:
            print('  [efficiency-ci] 计时均值的 95% CI（t 分布；区间宽度与均值同量级时，效率差异可能只是测量噪声）:')
            for line in timing_ci_lines:
                print(f'    {line}')
    missing_efficiency_cells = _missing_efficiency_cell_labels(aggregates)
    if missing_efficiency_cells:
        print('  [efficiency] 以下实验单元缺效率测量数据（旧运行或 --skip-efficiency），涉及它们的对比无法评估精度/效率权衡: ' + '; '.join(missing_efficiency_cells))
    print()
    print('[compute-matched] 等训练计算量的验证 loss（训练 FLOPs≈3×前向；共同计算量取各运行最后一个周期验证点的最小值，不做外推）')
    if not compute_matched['runs']:
        print('  无可用的等计算量对比')
    else:
        print(f'  共同计算量 FLOPs={_format_finite(compute_matched.get('target_flops'), '.3e')}')
        for item in compute_matched['runs']:
            ratio = item.get('compute_ratio')
            ratio_text = f'（用自身计算量的 {_format_finite(ratio, '.1%')}）' if isinstance(ratio, (int, float)) else ''
            print(f'  {str(item.get('run'))[:36]:<37} mode={str(item.get('mode')):<10} loss={_format_finite(item.get('value'))}{ratio_text} [{item['interpolation']}]')
        print('  可用 --compare-metric val_loss_at_matched_flops 把它作为判定指标做聚合与显著性检验；该点只代表共同计算量处，不等于完整训练结束时的排序')
    for message in compute_matched['advisories']:
        print(f'  [note] {message}')
    print()
    print('[common-step] 共同训练步数处的即时验证指标（不经过验证集 checkpoint 选择，用于选型无关复核）')
    if not common_step['runs']:
        print('  无可用的共同步数对比')
    else:
        print(f'  共同训练步数={_format_finite(common_step.get('target_step'), '.0f')}')
        for item in common_step['runs']:
            ratio = item.get('step_ratio')
            ratio_text = f'（占自身验证范围 {_format_finite(ratio, '.1%')}）' if isinstance(ratio, (int, float)) else ''
            print(f'  {str(item.get('run'))[:36]:<37} mode={str(item.get('mode')):<10} loss={_format_finite(item.get('value'))}{ratio_text} [{item['interpolation']}]')
        current_derived = metric if metric in _COMMON_STEP_METRICS else _common_step_metric_name(metric)
        if current_derived is not None and current_derived != _COMMON_STEP_METRIC:
            metric_report = _as_dict(_as_dict(common_step.get('metric_reports')).get(current_derived))
            print(f'  判定指标 {current_derived} 的共同步数取值：')
            for item in metric_report.get('runs', []):
                print(f'  {str(item.get('run'))[:36]:<37} mode={str(item.get('mode')):<10} value={_format_finite(item.get('value'))} [{item['interpolation']}]')
        print('  可用 --compare-metric <metric>_at_common_step 把对应指标作为判定口径做聚合与显著性检验；该点只代表共同步数处，不等于完整训练结束时的排序')
    for message in common_step['advisories']:
        print(f'  [note] {message}')
    print()
    print('[convergence] 共同训练窗口内的时均验证损失（周期曲线梯形积分/窗口长度；越低表示窗口内整体损失更低，常与更早收敛一致，但不能单独分离速度与最终水平）')

def _PA_part_001():
    global advisory, aligned_text, alpha_value, attempted, attrition_entries, baseline_mean, blocker, bounds_text, capacity_entries, capacity_parity_checked, capacity_parity_ok, compute_aligned, compute_parity_checked, compute_parity_ok, confound, contrast, cost_flagged, counts, coverage_text, current_seeds, curve_coverage, deleted, deleted_p, delta, delta_text, detail, details, direction, entry, equivalence_entries, equivalent_entries, expected_cells, family, family_contrasts, family_sizes, fragile_count, fragility_entries, fragility_flagged, full_factorial, global_alignment, global_alignment_info, global_seed_ok, high, holm, holm_text, inconclusive_entries, item, largest_family, level, low, margin_abs, margin_status, margin_text, mde, mde_relative, mde_text, message, mismatch_messages, missing, missingness_flagged, mixed_count, needed_text, not_evaluable_count, not_robust, parity, parity_entries, parity_mismatches, precision_limited, present, reason, seeds_needed, seeds_text, seeds_to_achieve, shape_text, significant_note, stable_count, summaries, tuned_runs, values, width_parity_ok, window_line, with_missing
    if not curve_mean['runs']:
        print('  无可用的曲线时均对比')
    else:
        print(f'  窗口 = [{_format_finite(curve_mean.get('start_step'), '.0f')}, {_format_finite(curve_mean.get('end_step'), '.0f')}]，起点取各运行第一个验证步的最大值，' + ('终点为训练前固定步' if curve_mean.get('end_rule') == 'preregistered_fixed_step' else '终点取各运行最后验证步的最小值（数据依赖）'))
        for item in curve_mean['runs']:
            curve_coverage = item.get('curve_coverage')
            coverage_text = f'（覆盖自身验证曲线 {_format_finite(curve_coverage, '.1%')}）' if isinstance(curve_coverage, (int, float)) else ''
            print(f'  {str(item.get('run'))[:36]:<37} mode={str(item.get('mode')):<10} mean_loss={_format_finite(item.get('value'))}{coverage_text} [start={item['start_interpolation']}, end={item['end_interpolation']}]')
        print(f'  可用 --compare-metric {_CURVE_MEAN_METRIC} 把它作为判定口径做聚合、显著性检验与 TOST；该指标同时受早期下降和最终水平影响，不能替代共同步数单点与 final 对照')
    for message in curve_mean['advisories']:
        print(f'  [note] {message}')
    print()
    direction = '越低越好' if not higher_is_better else '越高越好'
    print(f'[stats] 指标 {metric}（{direction}）；单因子对比（每个对比只改变一个因子；p 为原始值，显著性判定用家族内 Holm 校正后的 p）')
    family_sizes = sorted({int(size) for size in (contrast.get('significance_family_size') for contrast in contrasts) if isinstance(size, int) and (not isinstance(size, bool))})
    if any((size > 1 for size in family_sizes)):
        largest_family = max(family_sizes)
        print(f'  [power] 本分析家族最多 m={largest_family} 个检验；MDE/功效/所需种子与所需等价界默认按未校正 alpha={_SIGNIFICANCE_LEVEL:g} 计算，Holm 首步阈值为 alpha/m={_SIGNIFICANCE_LEVEL / largest_family:.4f}（同口径保守值见 mde_80_bonferroni / seeds_to_detect_bonferroni / equivalence_margin_needed_bonferroni；规划种子数与等价界时按后者）')
    if any((contrast.get('seeds_to_detect') is not None or contrast.get('equivalence_seeds_to_achieve') is not None for contrast in contrasts)):
        print('  [note] 所需种子数（seeds_to_detect / equivalence_seeds_to_achieve）按当前种子间标准差的点估计外推（seeds_to_detect_basis="sd_point_estimate"）；sd 自身估计不稳，其相对标准误 ≈ 1/√(2(n-1))（n=3 时约 50%）且被平方进所需种子数，实际所需可能明显更多，规划时留余量')
    if not contrasts:
        print('  无单因子对比（至少需要两个不同的实验单元）')
    else:
        for contrast in contrasts:
            if contrast.get('confounded'):
                print(f'  [confounded] {contrast['cells'][0]} vs {contrast['cells'][1]}: 同时改变了 ' + ', '.join(contrast['differing_factors']) + '，结论无法归因')
                continue
            print('  ' + _contrast_display(contrast, metric))
            window_line = _format_window_paired(contrast)
            if window_line:
                print(window_line)
        cost_flagged = [contrast for contrast in contrasts if contrast.get('cost_verdict') in ('gain_with_cost', 'dominated_regression')]
        if cost_flagged:
            summaries = []
            for contrast in cost_flagged:
                reason = '涨点但效率更低' if contrast['cost_verdict'] == 'gain_with_cost' else '变差且效率更低'
                summaries.append(f'{contrast['baseline']}→{contrast['variant']}（{reason}）')
            print('  [cost] 需按效率权衡的对比: ' + '; '.join(summaries))
        missingness_flagged = [contrast for contrast in contrasts if contrast.get('differential_missingness')]
        if missingness_flagged:
            print('  [warn] 以下对比两臂缺少该指标的运行数不同（完整案例分析建立在不同运行子集上，结论可能来自缺测而非方法差异；请核对被排除运行的原因）: ' + '; '.join((f'{contrast['baseline']}→{contrast['variant']}（缺 {contrast.get('n_missing_baseline', 0)} vs {contrast.get('n_missing_variant', 0)}，有效 {contrast.get('n_baseline', 0)} vs {contrast.get('n_variant', 0)}）' for contrast in missingness_flagged)))
    print()
    print('[fragility] 结果脆弱性检查（逐一/成组删除随机种子后重做配对 t 检验；alpha 为家族校正后阈值）')
    fragility_entries = fragility_summary.get('entries', [])
    if not fragility_entries:
        print('  无可用配对结果（旧运行缺少逐种子配对信息，或检验不是配对 t）')
    else:
        counts = fragility_summary.get('verdict_counts', {})
        print('  判定分布: ' + ', '.join((f'{key}={value}' for key, value in sorted(counts.items()))))
        stable_count = int(counts.get('stable', 0))
        fragile_count = int(counts.get('fragile', 0))
        mixed_count = int(counts.get('direction_mixed', 0))
        not_evaluable_count = int(counts.get('not_evaluable', 0))
        print(f'  稳健={stable_count}，脆弱={fragile_count}，方向混合={mixed_count}，不可判定={not_evaluable_count}')
        for message in fragility_summary.get('advisories', []):
            print(f'  [warn] {message}')
        fragility_flagged = [entry for entry in fragility_entries if entry.get('verdict') == 'fragile' and entry.get('significant')]
        for entry in fragility_flagged:
            deleted = entry.get('fragility_index')
            deleted_p = entry.get('fragility_index_p_value')
            alpha_value = entry.get('alpha')
            if isinstance(deleted, int) and isinstance(deleted_p, (int, float)):
                detail = f'删除 {deleted} 个种子后 p={_format_finite(deleted_p)}' + (f' > alpha={_format_finite(alpha_value)}' if isinstance(alpha_value, (int, float)) else '')
            elif isinstance(deleted, int):
                detail = f'删除 {deleted} 个种子后显著性消失'
            else:
                detail = '删除单个种子后效应方向翻转'
            print(f'  [fragile] {entry['baseline']}→{entry['variant']}: {detail}；符号方向 正/负={entry.get('n_positive')}/{entry.get('n_negative')}，不能作为稳健结论')
        print('  注：fragility 是描述性敏感性分析，不替代显著性检验；但「显著」若依赖少数种子，应增加独立种子或调查异常运行后再下结论')
    print()
    print('[attrition-bounds] 缺失/失败运行下的部分识别界（假设缺失结果落在两臂已观测结果的联合范围内；不是置信区间）')
    attrition_entries = attrition_summary.get('entries', [])
    if not attrition_entries:
        print('  无可用条目（确定性基线或没有任何已观测结果，无法计算缺失敏感性界）')
    else:
        counts = attrition_summary.get('verdict_counts', {})
        print('  判定分布: ' + ', '.join((f'{key}={value}' for key, value in sorted(counts.items()))))
        with_missing = attrition_summary.get('with_missing', [])
        not_robust = attrition_summary.get('not_robust', [])
        print(f'  有缺测/失败={len(with_missing)}，界跨零或无界={len(not_robust)}')
        for entry in not_robust:
            missing = entry.get('missing_runs')
            attempted = entry.get('attempted_runs')
            low = _finite_float(entry.get('effect_min'))
            high = _finite_float(entry.get('effect_max'))
            bounds_text = f'[{low:.4g}, {high:.4g}]' if low is not None and high is not None else '无有限界'
            print(f'  [warn] {entry['baseline']}→{entry['variant']}: 缺失 {missing}/{attempted}，{bounds_text}；完整案例的显著结论可能由「哪些运行留下了」驱动')
        if not not_robust and with_missing:
            print('  有缺测/失败运行的对比在观测支持范围内仍保持效应方向；这以缺失结果不超出观测范围为前提，不能替代补齐运行')
        for message in attrition_summary.get('advisories', []):
            if not not_robust:
                print(f'  [note] {message}')
        print('  注：这是有界结果假设下的部分识别，不替代显著性检验；缺失值可能超出观测范围时真实界为无穷宽，不能把「界内同向」写成「缺失无偏」')
    if baseline_mode:
        print(f'  指定基线 mode={baseline_mode} 的直接对比:')
        if not reference:
            print(f'    未找到 mode={baseline_mode} 的运行')
        for contrast in reference:
            print('    ' + _contrast_display(contrast, metric))
            window_line = _format_window_paired(contrast)
            if window_line:
                print('    ' + window_line.strip())
    print(f'  注：同一检验家族内已做多重比较校正——显著性判定以 Holm-Bonferroni（FWER）校正后为准，BH（FDR）校正值与 95% 置信区间写在对比报告里；结论以预设的关键对比为准，且种子数应 ≥{_RECOMMENDED_SEEDS}')
    equivalence_entries = []
    for family, family_contrasts in (('single_factor', contrasts), ('baseline', reference)):
        for contrast in family_contrasts:
            if contrast.get('confounded'):
                continue
            if contrast.get('test') not in ('paired_t', 'welch_t'):
                continue
            equivalence_entries.append({'family': family, 'baseline': contrast.get('baseline'), 'variant': contrast.get('variant'), 'delta': contrast.get('delta'), 'significant': bool(contrast.get('significant')), 'mde_80': contrast.get('mde_80'), 'mde_80_relative': contrast.get('mde_80_relative'), 'power_at_effect': contrast.get('power_at_effect'), 'seeds_to_detect': contrast.get('seeds_to_detect'), 'n_pairs': contrast.get('n_pairs'), 'equivalence_margin_abs': contrast.get('equivalence_margin_abs'), 'tost_p_value': contrast.get('tost_p_value'), 'tost_p_value_holm': contrast.get('tost_p_value_holm'), 'equivalent_raw': contrast.get('equivalent_raw'), 'equivalent': contrast.get('equivalent'), 'equivalence_margin_needed': contrast.get('equivalence_margin_needed'), 'equivalence_margin_needed_relative': contrast.get('equivalence_margin_needed_relative'), 'equivalence_blocker': contrast.get('equivalence_blocker'), 'equivalence_seeds_to_achieve': contrast.get('equivalence_seeds_to_achieve')})
    print()
    if equivalence_margin is None:
        print('[equivalence] 未设定等价界（equivalence_margin=None）：只报告显著性；不显著 ≠ 等价，不能据此主张无差异')
    else:
        print(f'[equivalence] 等价性与统计功效（等价界 = 基线均值的 ±{_format_finite(equivalence_margin, '.1f', 100.0)}%；不显著 ≠ 等价）')
        margin_status = margin_preregistration.get('status')
        if margin_status == 'consistent' and margin_preregistration.get('ok') is True:
            print('  等价界预注册: OK（当前界与训练前写入 hypothesis_preregistration 的取值一致）')
        else:
            for advisory in margin_preregistration.get('advisories', []):
                print('  [equivalence-preregistration] [warn] ' + str(advisory))
        if not equivalence_entries:
            print(f'  无可用检验（至少需要两个实验单元且每侧 ≥{_MIN_SEEDS_FOR_STATS} 个种子）')
        else:
            equivalent_entries = [entry for entry in equivalence_entries if entry['equivalent']]
            inconclusive_entries = [entry for entry in equivalence_entries if not entry['significant'] and (not entry['equivalent'])]
            for entry in equivalent_entries:
                holm = entry.get('tost_p_value_holm')
                holm_text = f'Holm p={_format_finite(holm, '.3f')}' if isinstance(holm, (int, float)) else 'p=NA'
                margin_abs = entry.get('equivalence_margin_abs')
                margin_text = f'（等价界 ±{_format_finite(margin_abs)}）' if isinstance(margin_abs, (int, float)) else ''
                significant_note = '；差异统计显著但落在等价界内（统计显著 ≠ 实际重要）' if entry['significant'] else ''
                delta = entry.get('delta')
                delta_text = f'Δ={_format_finite(delta, '+.4f')}' if isinstance(delta, (int, float)) else 'Δ=NA'
                print(f'  [equivalent] {entry['baseline']}→{entry['variant']}  {delta_text}  TOST {holm_text}{margin_text}：支持无实质差异{significant_note}')
            for entry in inconclusive_entries:
                mde = entry.get('mde_80')
                mde_text = f'80% 功效 MDE={_format_finite(mde)}' if isinstance(mde, (int, float)) else 'MDE=NA'
                mde_relative = entry.get('mde_80_relative')
                if isinstance(mde_relative, (int, float)):
                    mde_text += f'（相对 {_format_finite(mde_relative, '.2f', 100.0)}%）'
                seeds_needed = entry.get('seeds_to_detect')
                seeds_text = ''
                current_seeds = _current_seed_count(entry)
                if isinstance(seeds_needed, int) and current_seeds is not None:
                    seeds_text = f'；检出该效应约需 {seeds_needed} 个种子（当前 {_format_seed_count(current_seeds)} 个）'
                blocker = entry.get('equivalence_blocker')
                if blocker == 'effect_exceeds_margin':
                    needed_text = '（点估计本身已超出等价界：这不是种子不够，增加种子只会让「不等价」更确定）'
                elif blocker == 'insufficient_precision':
                    needed_text = f'（点估计在界内但噪声半宽越界：把界至少放宽到 ±{_format_finite(entry.get('equivalence_margin_needed'))} 才可能主张等价'
                    seeds_to_achieve = entry.get('equivalence_seeds_to_achieve')
                    if isinstance(seeds_to_achieve, int):
                        needed_text += f'，或在当前种子波动下约需 {seeds_to_achieve} 个种子'
                    needed_text += '）'
                else:
                    needed_text = ''
                delta = entry.get('delta')
                delta_text = f'Δ={_format_finite(delta, '+.4f')}' if isinstance(delta, (int, float)) else 'Δ=NA'
                print(f'  [inconclusive] {entry['baseline']}→{entry['variant']}  {delta_text}  {mde_text}{seeds_text}{needed_text}：不显著也说明不了等价')
            precision_limited = [entry for entry in equivalence_entries if entry.get('equivalence_blocker') == 'insufficient_precision' and (not entry.get('equivalent'))]
            if precision_limited:
                print('  [warn] 以下对比的等价判定卡在精度而不是效应：点估计落在等价界内，但种子噪声把置信区间推出了界外——TOST 未通过不能读成「两者不同」，应增加种子（而不是放宽等价界）: ' + '; '.join((f'{entry['baseline']}→{entry['variant']}' + (f'（约需 {entry['equivalence_seeds_to_achieve']} 个种子）' if isinstance(entry.get('equivalence_seeds_to_achieve'), int) else '') for entry in precision_limited)))
            if not equivalent_entries and (not inconclusive_entries):
                print('  所有对比均显著：本家族没有需要等价性解读的对比')
            print('  注：等价界必须事先声明（--equivalence-margin）；看到数据后再放宽等价界可以「制造」等价结论')
    print()
    print('[baseline-preregistration] 对照基线预注册核对（--hypothesis-baseline；事后更换对照基线属于 baseline shopping）')
    if not baseline_preregistration.get('declared_modes'):
        print('  [note] 没有任何运行预注册对照基线（--hypothesis-baseline）：无法排除事后挑选更弱的臂作为基线，建议训练前声明')
    else:
        print('  预注册对照基线: ' + ', '.join((str(mode) for mode in baseline_preregistration['declared_modes'])))
        if baseline_preregistration.get('ok') is True:
            print(f'  OK：分析使用的对照基线 {baseline_preregistration.get('baseline_mode')} 与训练前预注册一致')
    for message in baseline_preregistration.get('advisories', []):
        level = 'warn' if baseline_preregistration.get('ok') is False else 'note'
        print(f'  [{level}] {message}')
    print()
    print('[baseline] 基线合理性检查（复杂模型必须战胜精心调优过的基线；战胜稻草人说明不了问题）')
    if adequacy['baseline_mode'] is None:
        for message in adequacy['advisories']:
            print(f'  [note] {message}')
    else:
        baseline_mean = adequacy.get('baseline_mean')
        print(f'  基线 mode={adequacy['baseline_mode']}（{adequacy['n_baseline_runs']} 个运行）' + (f'，{metric} 均值 {_format_finite(baseline_mean)}' if baseline_mean is not None else ''))
        if adequacy['ok'] is True:
            print('  OK：基线在主指标上强于无上下文 unigram 参考（核对依据：各运行 config.json 的 reference_baselines）')
        if adequacy.get('baseline_mode') == 'ngram':
            print('  OK：统计基线在 val 上做过 阶数 × 平滑系数 网格调优（简单基线已充分调参）')
        elif adequacy.get('tuned') and (not adequacy.get('tuning_metric_mismatch')):
            tuned_runs = adequacy.get('tuned_runs') or []
            details = '; '.join((f'{item['run']}（调参口径={item.get('tuning_metric')}, winner lr={item.get('winner_lr')}）' for item in tuned_runs))
            print('  OK：基线记录了调参来源（--baseline-tuning-from）' + (f': {details}' if details else ''))
        elif adequacy.get('tuned'):
            print('  [warn] 基线记录了调参来源，但调参口径与当前对比指标不一致：不能证明基线在当前指标上被调强；请在该指标上重新调参，或改用与调参口径一致的对比指标')
        elif adequacy['baseline_mode'] is not None:
            print('  [untuned] 基线没有调参来源记录，也没有强于调优好的 n-gram：没调过参的弱基线只能证明复杂模型比稻草人强，先用 --suite-baseline-tuning 把基线调强')
        for message in adequacy['straw_man']:
            print(f'  [straw-man] {message}')
        mismatch_messages = set(adequacy.get('tuning_metric_mismatch') or [])
        for message in adequacy['advisories']:
            if message in mismatch_messages:
                continue
            print(f'  [note] {message}')
        parity = adequacy.get('tuning_parity') or {}
        parity_mismatches = parity.get('challengers_without_tuning_parity') or []
        if parity.get('baseline_winner_lrs'):
            print('  [tuning-parity] 基线调参 winner lr=' + ', '.join((_format_finite(lr, 'g') for lr in parity['baseline_winner_lrs'])) + f'；复用同一 lr 的挑战臂 {len(parity.get('challengers_using_baseline_lr') or [])} 个，有独立调参来源的挑战臂 {len(parity.get('challengers_with_own_tuning') or [])} 个')
        if parity_mismatches:
            print('  [tuning-parity] [warn] 以下挑战臂的学习率与调参基线不一致且没有独立调参来源：调参不对称，落败不能排除调参不足、胜出是保守下界')
            for item in parity_mismatches:
                print(f'    {item['run']}（{item['mode']}, lr={_format_finite(item.get('lr'), 'g')} vs 基线 winner lr={_format_finite(item.get('baseline_tuned_lr'), 'g')}）')
    print()
    print("[tuning-seeds] 调参种子与最终评估种子独立性核对（同一批种子既选 winner 又评估会把 winner's curse 带进对比）")
    if tuning_seed_overlap['ok'] is True:
        print('  OK：所有带调参来源的运行，其调参种子与最终评估种子均不相交')
    elif tuning_seed_overlap['n_runs_with_tuning'] == 0:
        print('  [note] 没有可核对的调参种子记录（无调参来源）：无法排除种子复用')
    else:
        for message in tuning_seed_overlap['advisories']:
            print(f'  [warn] {message}')
    expected_cells = coverage.get('expected_cells') or []
    shape_text = f'{len(expected_cells) // 2}×2' if expected_cells else '延迟合并 × 均衡损失'
    print()
    print(f'[ablation] 消融覆盖检查（{shape_text}：{coverage['legend']}）')
    if coverage.get('partial'):
        print('  [partial] 本轮是显式允许的部分消融：未覆盖项目方案 ' + ', '.join(coverage.get('undeclared_project_modes', [])) + '，结论不能支撑全部创新点各自有用')
    if coverage['missing_cells']:
        for message in coverage['missing_cells']:
            print(f'  [missing] {message}')
    if coverage['present_cells']:
        present = ', '.join((f'{cell['cell']}(n={cell['n_runs']})' for cell in coverage['present_cells']))
        print(f'  已覆盖组合: {present}')
    for message in coverage.get('structural_confounds', []):
        print(f'  [confounded] {message}')
    for entry in coverage.get('numerical_attrition', []):
        if int(entry.get('failed_runs', 0) or 0) <= 0:
            continue
        print(f'  [attrition] {entry['cell']}：失败 {entry['failed_runs']}/{entry['attempted_runs']}（可用 {entry['usable_runs']}）' + ('，失败运行: ' + ', '.join(entry.get('failed_run_names') or []) if entry.get('failed_run_names') else ''))
    for confound in coverage.get('attrition_confounds', []):
        print('  [confounded] 数值失败/脱落不平衡: ' + str(confound.get('details')) + '；幸存运行的均值不能归因到组件')
    for entry in coverage.get('seed_alignment', []):
        seeds_text = ', '.join((str(seed) for seed in entry['common_seeds']))
        if entry['aligned']:
            print(f'  种子对齐: {entry['delayed_mode']} 的 2×2 各格共享 {entry['n_common_seeds']} 个种子' + (f'（{seeds_text}）' if seeds_text else ''))
        else:
            print(f'  [warn] 种子未对齐: {entry['delayed_mode']} 的 2×2 各格种子集合不同，仅在 {entry['n_common_seeds']} 个共同种子上可配对')
    global_alignment = coverage.get('global_seed_alignment')
    if isinstance(global_alignment, dict) and global_alignment.get('applicable'):
        seeds_text = ', '.join((str(seed) for seed in global_alignment.get('common_seeds', [])))
        if global_alignment.get('aligned'):
            print(f'  全因子种子对齐: 所有消融格共享 {global_alignment.get('n_common_seeds', 0)} 个种子' + (f'（{seeds_text}）' if seeds_text else ''))
        elif global_alignment.get('all_expected_present'):
            print(f'  [warn] 全因子种子未对齐: 各格种子集合不同，共同种子只有 {global_alignment.get('n_common_seeds', 0)} 个；不同方案可能在不同种子子集上比较，不能拼成「各创新点各自有用」')
    for entry in coverage.get('compute_parity', []):
        if entry['matched']:
            continue
        print(f'  [confounded] 计算量不匹配: {entry['delayed_mode']} 的 {entry['field']} 是 local_base 的 {_format_finite(entry.get('ratio'), '.2f')} 倍，该方案的主效应混入计算量；请用 --compare-metric val_loss_at_matched_flops 在等训练计算量点复核')
    for entry in coverage.get('capacity_parity', []):
        if entry['matched']:
            continue
        print(f'  [confounded] 参数量不匹配: {entry['delayed_mode']} 的 n_params 是 local_base 的 {_format_finite(entry.get('ratio'), '.2f')} 倍，即使 FLOPs 对齐，多出的容量也可能带来收益，主效应不能归因到延迟合并')
    if coverage['low_replicate_cells']:
        for message in coverage['low_replicate_cells']:
            print(f'  [warn] {message}')
    for advisory in coverage['advisories']:
        print(f'  [note] {advisory}')
    print()
    print('[randomness] 同种子随机性核对（初始化指纹 / 首批训练数据指纹）')
    if randomness['groups_checked'] == 0:
        print('  [note] 没有同种子的非确定性运行可核对（至少需要同一 seed 的 2 次运行）')
    elif randomness['confounds']:
        for confound in randomness['confounds']:
            if confound['type'] == 'training_batch':
                detail = '首批训练数据顺序不同'
            elif confound['type'] == 'initialization_scheme':
                detail = '初始化方案不同'
            else:
                detail = str(confound.get('label')) + '初始化不同'
            values = ', '.join((f'{run}={_format_factor_value(value)}' for run, value in confound['values'].items()))
            print(f'  [confounded] 种子 {confound['seed']}：{detail}（{', '.join(confound['modes'])}）')
            print(f'    指纹: {values[:240]}')
        print('  同种子配对假设不成立：以上差异不是声明因子的效应，相关主效应/交互不能归因')
    else:
        print(f'  OK：{randomness['groups_checked']} 个种子分组的初始化与首批训练数据指纹一致')
    for message in randomness['unverified']:
        print(f'  [warn] {message}：无法确认同种子随机性对齐')
    full_factorial = 'mode' in factors and 'balance_coef' in factors and bool(expected_cells) and (not coverage.get('partial')) and (len(coverage['present_cells']) == len(expected_cells))
    parity_entries = coverage.get('compute_parity', [])
    compute_parity_checked = bool(parity_entries)
    compute_parity_ok = compute_parity_checked and all((entry.get('matched') for entry in parity_entries))
    width_parity_ok = not coverage.get('width_confounds')
    capacity_entries = coverage.get('capacity_parity') or []
    capacity_parity_ok = all((entry.get('matched') for entry in capacity_entries))
    capacity_parity_checked = bool(capacity_entries)
    compute_aligned = metric == _COMPUTE_MATCHED_METRIC or (compute_parity_ok and width_parity_ok and (capacity_parity_ok or not capacity_parity_checked))
    global_alignment_info = coverage.get('global_seed_alignment')
    global_seed_ok = bool(isinstance(global_alignment_info, dict) and global_alignment_info.get('applicable') and global_alignment_info.get('aligned'))
    if full_factorial and (not coverage['missing_cells']) and (not coverage['low_replicate_cells']) and (not coverage.get('structural_confounds')) and (not randomness['confounds']) and global_seed_ok and compute_aligned:
        aligned_text = '当前指标已按共同训练计算量对齐' if metric == _COMPUTE_MATCHED_METRIC else '计算量对齐'
        print(f'  OK：{len(expected_cells) // 2}×2 全因子组合齐全、种子数达标且{aligned_text}')
    elif full_factorial and (not global_seed_ok):
        print(f'  [warn] {len(expected_cells) // 2}×2 组合齐全，但全因子各格的随机种子集合不一致：不同方案的组件效应可能来自不同的种子子集，不能拼成「各创新点各自有用」')
    elif full_factorial and randomness['confounds']:
        print(f'  [warn] {len(expected_cells) // 2}×2 组合齐全，但同种子初始化/训练数据指纹不一致：主效应不能按同种子配对归因')
    elif full_factorial and (not coverage['missing_cells']) and (not coverage['low_replicate_cells']) and (not coverage.get('structural_confounds')) and (not compute_aligned) and (compute_parity_checked or not width_parity_ok):
        print(f'  [warn] {len(expected_cells) // 2}×2 组合齐全且种子数达标，但计算量/专家宽度不匹配：主效应须在等训练计算量指标上复核后才能归因')
    elif full_factorial and (not coverage['missing_cells']) and (not coverage['low_replicate_cells']) and (not coverage.get('structural_confounds')) and (metric != _COMPUTE_MATCHED_METRIC) and (not compute_parity_checked):
        print(f'  [note] {len(expected_cells) // 2}×2 组合齐全且种子数达标；但所有运行都缺 FLOPs/参数量记录，计算量对齐无法核对，主效应须以等训练计算量指标复核后才能归因')
    print()
    print('[balance-scale] 均衡损失强度核对（balance_coef × 平均路由宽度）')
    if not balance_scale['profiles']:
        print('  无可用运行（缺 mode/k1/k2/top_k/balance_coef 记录）')
    elif not balance_scale.get('compared'):
        if len(balance_scale['profiles']) < 2:
            print('  [note] 可比较的运行不足 2 个，无法核对均衡强度')
        else:
            print('  [note] 施加均衡压力（balance_coef>0）的有效臂不足 2 个（coef=0 对照臂不计入），无法核对均衡强度')
    elif balance_scale['confounded']:
        for advisory in balance_scale['advisories']:
            print(f'  [warn] {advisory}')
    elif balance_scale['width_varied']:
        print('  OK：路由宽度不同，但 balance_coef 已按宽度折算，均衡强度一致')
    elif balance_scale['strength_varied']:
        print('  OK：路由宽度一致；balance_coef 差异是显式实验因子，不构成宽度混杂')
    else:
        print('  OK：各运行的有效路由宽度与均衡强度一致')

def _PA_part_002():
    global _bad_value, _curve_value, _delta_text, advisory, aggregate, bad_case_comparison, bad_header, bad_run_width, blocker, category_names, component, curve_diagnostics, curve_header, curve_run_width, design, effect, entry, flag, item, min_runs, p_text, p_value, pairs_text, payloads, persistent, ranges, reason, rho, rho_text, shares, short_names, warning
    if factorial['designs'] or factorial['incomplete_designs']:
        print()
        print('[factorial] 全因子效应（按共同种子配对；主效应/交互同属一个检验家族，做 Holm 校正）')
        for design in factorial['designs']:
            print('  设计: ' + f'{design['delayed_mode']} vs local_base' + ('（' + ', '.join((f'{key}={_format_factor_value(value)}' for key, value in design['context'].items())) + '）' if design['context'] else ''))
        for design in factorial['incomplete_designs']:
            print('  [missing] ' + f'{design['delayed_mode']} vs local_base' + ' 缺格: ' + ', '.join(design['missing_cells']))
        for effect in factorial['effects']:
            print(_format_factorial_effect(effect))
        for advisory in factorial['advisories']:
            print(f'  [warn] {advisory}')
        print('  注：交互显著只说明一个因子的效应随另一个因子变化，不能单独解释为有益或有害；主效应结论以 Holm 校正后为准')
        if any((effect.get('tost_p_value_holm') is not None for effect in factorial['effects'])):
            print('  等价性：TOST p 值同族做 Holm/BH 校正；只有校正后仍 <= alpha 才可主张「无实质差异」，「不显著」本身不构成等价证据')
    if common_step_sensitivity['applicable']:
        print()
        print(f'[factorial common-step] 在 {common_step_sensitivity['common_step_metric']} 上复核选型偏差（无 checkpoint 选择的即时口径）')
        if common_step_sensitivity['effects']:

            def _delta_text(value: object) -> str:
                if value is None:
                    return '-'
                return _format_finite(value, '+.4f')
            for item in common_step_sensitivity['effects']:
                flag = ''
                if item['support_flip']:
                    flag = ' [FLIP]'
                elif item['sign_flip']:
                    flag = ' [SIGN-FLIP]'
                elif item['claim_flip']:
                    flag = ' [claim-changed]'
                print(f'  {item.get('delayed_mode')}/{item.get('effect_name')}: final={item['primary_claim']} (Δ={_delta_text(item.get('primary_mean_effect'))}), common-step={item['common_step_claim']} (Δ={_delta_text(item.get('common_step_mean_effect'))})' + flag)
        else:
            print('  无共同步数效应可比较')
        if common_step_sensitivity['support_flips']:
            print('  [warn] ' + str(common_step_sensitivity['advisory']))
        else:
            print('  ' + str(common_step_sensitivity['advisory']))
    if component_verdicts['applicable']:
        print()
        print('[ablation-verdict] 组件级联合判定（每个创新点必须有独立支持证据，不能把部分显著拼成「各自有用」）')
        for component in component_verdicts['components']:
            print(f'  [{component['status']}] {component['label']}' + '：' + '；'.join((str(reason) for reason in component['reasons'] if reason)))
        if component_verdicts['all_components_supported']:
            print('  OK：全部组件均获得无混杂的支持证据')
        else:
            print('  [not-supported] ' + str(component_verdicts['advisory']))
        for blocker in component_verdicts['blockers']:
            print(f'  [blocked] {blocker}')
    if factorial_compute_matched is not None:
        print()
        print(f'[factorial compute-matched] 在 {_COMPUTE_MATCHED_METRIC} 上重估全因子效应（已对齐共同训练计算量；单独成族，不替代完整训练曲线的排序）')
        if factorial_compute_matched['effects']:
            for effect in factorial_compute_matched['effects']:
                print(_format_factorial_effect(effect))
        else:
            print('  无可用效应（等计算量指标缺少数值或共同种子）')
        for advisory in factorial_compute_matched['advisories']:
            print(f'  [warn] {advisory}')
    curve_diagnostics = curve_diagnostics_report(rows)
    if curve_diagnostics['runs']:
        print()
        print('[curve] 跨运行训练曲线诊断（读取各运行 metrics.jsonl 的 curve_analysis；不是最终单点）')
        curve_run_width = max(20, max((len(str(entry.get('run'))) for entry in curve_diagnostics['runs']), default=3))
        curve_header = f'{'run':<{curve_run_width}} {'status':<14} {'best@':>7} {'regr%':>7} {'gap%':>7} {'recentVal%':>10} {'gradSpike':>9} {'lr/peak':>8}'
        print(curve_header)
        print('-' * len(curve_header))

        def _curve_value(value, spec='.2f'):
            return _format_finite(value, spec, 100.0)
        for entry in curve_diagnostics['runs']:
            print(f'{str(entry['run']):<{curve_run_width}} {str(entry.get('status'))[:13]:<14} {str(entry.get('best_val_step') or '-'):>7} {_curve_value(entry.get('val_regression_from_best')):>7} {_curve_value(entry.get('generalization_gap_ratio_last')):>7} {_curve_value(entry.get('recent_val_drop')):>10} {_curve_value(entry.get('grad_norm_spike_ratio')):>9} {_curve_value(entry.get('last_lr_over_peak')):>8}')
        if curve_diagnostics['overfitting_runs']:
            print('  [warn] 以下运行判定为过拟合：最终 val 指标不是其最优泛化点，比较时应同时看 best@/regr%：' + '; '.join(curve_diagnostics['overfitting_runs']))
        print('  ' + str(curve_diagnostics['note']))
    bad_case_comparison = bad_case_comparison_report(rows)
    if bad_case_comparison['runs']:
        print()
        print('[bad-case] 跨运行错误模式（val 固定窗口；类别按全部错误 token 加权，不是只看 top-N）')
        category_names = list(_BAD_CASE_CATEGORIES)
        short_names = {'ambiguous': 'ambig', 'systematic_confusion': 'sysConf', 'label_noise_suspect': 'labelNoise', 'uncertain': 'uncert'}
        bad_run_width = max(20, max((len(str(entry.get('run'))) for entry in bad_case_comparison['runs']), default=3))
        bad_header = f'{'run':<{bad_run_width}} {'acc':>7} {'wrong':>8} ' + ' '.join((f'{short_names.get(name, name[:9]):>10}' for name in category_names))
        print(bad_header)
        print('-' * len(bad_header))

        def _bad_value(value, spec='.4f'):
            return _format_finite(value, spec)
        for entry in bad_case_comparison['runs']:
            shares = entry['error_category_shares']
            print(f'{str(entry['run']):<{bad_run_width}} {_bad_value(entry.get('token_accuracy')):>7} {str(entry.get('wrong_tokens')):>8} ' + ' '.join((f'{_bad_value(shares.get(name)):>10}' for name in category_names)))
        persistent = bad_case_comparison['persistent_confusions']
        if persistent:
            min_runs = bad_case_comparison['persistent_confusion_min_runs']
            pairs_text = ' '.join((f"'{_render_byte(pair['target'])}'→'{_render_byte(pair['predicted'])}'(runs={pair['runs_present']},topSum={pair['top_count_sum']})" for pair in persistent[:8]))
            print(f'  [bad-case] 稳定混淆对（至少 {min_runs} 个运行的 top-10 中出现）: {pairs_text}')
        if bad_case_comparison['unstable_categories']:
            ranges = bad_case_comparison['category_share_ranges']
            print('  [bad-case] [warn] 错误类别占比跨运行极差 ≥20pp: ' + ', '.join((f'{name}={_format_finite(ranges.get(name, 0.0), '.1f', 100.0)}pp' for name in bad_case_comparison['unstable_categories'])) + '；失败模式可能由种子/运行差异驱动')
        print('  ' + str(bad_case_comparison['coverage_note']))
        print('  ' + str(bad_case_comparison['note']))
    for aggregate in aggregates:
        payloads = aggregate.pop('window_payloads', None)
        if not payloads:
            continue
        aggregate['eval_window_runs'] = [{'seed': entry.get('seed'), 'run': entry.get('run'), 'n_windows': entry.get('payload', {}).get('n_windows') if isinstance(entry.get('payload'), dict) else None, 'split': entry.get('payload', {}).get('split') if isinstance(entry.get('payload'), dict) else None} for entry in payloads if isinstance(entry, dict)]
    print()
    print('[order] 执行顺序/时间漂移诊断')
    if not order_audit.get('available'):
        print('  ' + str(order_audit.get('reason')))
    else:
        rho = order_audit.get('detrended_spearman')
        p_value = order_audit.get('detrended_p_value')
        rho_text = f'{_format_finite(rho, '+.2f')}' if isinstance(rho, (int, float)) else 'NA'
        p_text = f'{_format_finite(p_value, '.3f')}' if isinstance(p_value, (int, float)) else 'NA'
        print(f'  顺序方案={','.join(order_audit.get('order_schemes') or []) or 'NA'}，带顺序记录的运行={order_audit.get('n_runs')}，区组={order_audit.get('n_blocks')}；去模式均值后 ρ(order,{metric})={rho_text}（p={p_text}）')
        for warning in order_audit.get('warnings') or []:
            print(f'  [warn] {warning}')
        if order_audit.get('verdict') == 'insufficient':
            reason = order_audit.get('reason')
            if reason:
                print(f'  {reason}')
        elif order_audit.get('verdict') == 'ok':
            print('  未发现强顺序相关；随机化区组平衡了时间片，但描述性诊断不能证明没有漂移')
    raise _SC_RETURN({'factors': factors, 'metric': metric, 'controlled': controlled, 'single_factor_design': single_factor_design, 'provenance': provenance, 'numerical_failures': numerical_failures, 'failure_rates': failure_rates, 'scale_guard': scale_guard_report, 'hypothesis_registry': hypotheses, 'hypothesis_alignment': alignment, 'aggregates': aggregates, 'duplicate_seed_cells': duplicate_seed_cells, 'deterministic_conflict_cells': deterministic_conflict_cells, 'contrasts': contrasts, 'reference_contrasts': reference, 'baseline_preregistration': baseline_preregistration, 'baseline_adequacy': adequacy, 'tuning_seed_overlap': tuning_seed_overlap, 'selection_metric_mismatch': selection_metric_mismatch, 'imbalance_advisory': imbalance, 'ablation_coverage': coverage, 'balance_scale': balance_scale, 'factorial_effects': factorial, 'factorial_common_step_sensitivity': common_step_sensitivity, 'ablation_component_verdicts': component_verdicts, 'factorial_effects_compute_matched': factorial_compute_matched, 'missing_efficiency_cells': missing_efficiency_cells, 'curve_diagnostics': curve_diagnostics, 'bad_case_comparison': bad_case_comparison, 'compute_matched': compute_matched, 'common_step': common_step, 'curve_mean': curve_mean, 'selection_bias': selection_bias, 'run_order': order_audit, 'equivalence_margin': equivalence_margin, 'equivalence_margin_preregistration': margin_preregistration, 'target_power': _TARGET_POWER, 'power_equivalence': equivalence_entries, 'fragility_summary': fragility_summary, 'attrition_bounds_summary': attrition_summary, 'n_runs': len(rows)})

