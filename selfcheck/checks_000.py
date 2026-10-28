from __future__ import annotations

def _selfcheck_top_51350_0001():
    global common, grad, gradients, key, mode, model, named_dense, named_dense_state, named_moe, named_moe_state, named_shared, ordered_moe, result, x, y
    '\n    结构、梯度、因果性和等价性测试。\n    '
    set_seed(123)
    try:
        Config(vocab_size=256, seq_len=8, d_model=16, n_heads=4, n_layers=3, n_experts=3, expert_hidden=24).validate()
    except ValueError:
        pass
    else:
        raise AssertionError('非法 n_layers 在 -O 模式下也必须被拒绝')
    common = dict(vocab_size=256, seq_len=8, d_model=16, n_heads=4, n_layers=4, n_experts=3, expert_hidden=24, top_k=2, k1=2, k2=2)
    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))
    for mode in ('smelt', 'local_base', 'A', 'B', 'dense'):
        model = LanguageModel(Config(**common, mode=mode))
        result = model(x, y)
        assert result['logits'].shape == (2, 8, 256)
        assert torch.isfinite(result['loss'])
        result['loss'].backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(gradients) > 0
        for grad in gradients:
            assert torch.isfinite(grad).all()
    set_seed(123)
    named_moe = LanguageModel(Config(**common, mode='B', init_scheme='name_hash'))
    set_seed(123)
    named_dense = LanguageModel(Config(**common, mode='dense', init_scheme='name_hash'))
    named_moe_state = named_moe.state_dict()
    named_dense_state = named_dense.state_dict()
    named_shared = set(named_moe_state) & set(named_dense_state)
    assert named_shared
    for key in named_shared:
        assert torch.equal(named_moe_state[key], named_dense_state[key]), f'name_hash 初始化在共享参数 {key} 上未对齐'
    assert named_moe.init_fingerprints['shared_sha256'] == named_dense.init_fingerprints['shared_sha256']
    assert named_moe.init_fingerprints['moe_sha256'] != named_dense.init_fingerprints['moe_sha256']
    set_seed(123)
    ordered_moe = LanguageModel(Config(**common, mode='B', init_scheme='module_order'))
    set_seed(123)

def _selfcheck_top_51483_0002():
    global _BatchProbe, _batch_confound, _randomness_ok, _randomness_unverified, _rows_batch, _rows_ok, _rows_other_seed, _rows_unverified, _seed_split, _trace, _trace_a, _trace_b, _trace_c, ordered_dense, ordered_shared
    ordered_dense = LanguageModel(Config(**common, mode='dense', init_scheme='module_order'))
    ordered_shared = set(ordered_moe.state_dict()) & set(ordered_dense.state_dict())
    assert any((not torch.equal(ordered_moe.state_dict()[key], ordered_dense.state_dict()[key]) for key in ordered_shared)), 'module_order 下 MoE 与 dense 的共享层初始化应当不同（自测用于证明该历史方案确实存在初始化混杂）'
    assert ordered_moe.init_fingerprints['shared_sha256'] != ordered_dense.init_fingerprints['shared_sha256']
    _trace = {'seed': 123, 'batch_size': 2, 'batches': 2, 'sha256': 'batch-a'}
    _rows_ok = [{'run': 'r1', 'mode': 'A', 'seed': 123, 'init_fingerprints': dict(named_moe.init_fingerprints), 'training_trace': dict(_trace)}, {'run': 'r2', 'mode': 'B', 'seed': 123, 'init_fingerprints': dict(named_moe.init_fingerprints), 'training_trace': dict(_trace)}]
    _randomness_ok = same_seed_randomness_check(_rows_ok)
    assert _randomness_ok['ok'], _randomness_ok
    assert _randomness_ok['groups_checked'] == 1
    _rows_batch = [dict(row) for row in _rows_ok]
    _rows_batch[1] = dict(_rows_batch[1])
    _rows_batch[1]['training_trace'] = {**_trace, 'sha256': 'batch-b'}
    _batch_confound = same_seed_randomness_check(_rows_batch)
    assert not _batch_confound['ok']
    assert any((item['type'] == 'training_batch' for item in _batch_confound['confounds']))
    _rows_other_seed = [dict(row) for row in _rows_batch]
    _rows_other_seed[1] = dict(_rows_other_seed[1])
    _rows_other_seed[1]['seed'] = 124
    _seed_split = same_seed_randomness_check(_rows_other_seed)
    assert _seed_split['ok'], _seed_split
    _rows_unverified = [{'run': 'old-a', 'mode': 'A', 'seed': 7, 'init_fingerprints': None, 'training_trace': None}, {'run': 'old-b', 'mode': 'B', 'seed': 7, 'init_fingerprints': None, 'training_trace': None}]
    _randomness_unverified = same_seed_randomness_check(_rows_unverified)
    assert _randomness_unverified['ok']
    assert _randomness_unverified['unverified']
    class _BatchProbe:

        @staticmethod
        def batch(split, batch_size, generator, device):
            inputs = torch.randint(0, 256, (batch_size, 4), generator=generator)
            targets = torch.randint(0, 256, (batch_size, 4), generator=generator)
            return (inputs, targets)
    _trace_a = training_batch_fingerprint(_BatchProbe(), 5, 2)
    _trace_b = training_batch_fingerprint(_BatchProbe(), 5, 2)
    _trace_c = training_batch_fingerprint(_BatchProbe(), 6, 2)
    assert _trace_a['sha256'] == _trace_b['sha256']
    assert _trace_a['sha256'] != _trace_c['sha256']
    assert _trace_a['batches'] == 2

def _selfcheck_top_51618_0003():
    global RouteWidthProbe, _count_block_calls, block, dense_default, dense_explicit, local_probe, local_widths, middle_count, n_layers, original_route, probe, record_route
    def _count_block_calls(cfg: Config) -> int:
        probe_model = LanguageModel(cfg)
        calls = [0]
        for block in probe_model.blocks:
            original_step = block.attention_step

            def probe_attention_step(inputs: torch.Tensor, *args, _original=original_step, **kwargs) -> torch.Tensor:
                calls[0] += 1
                return _original(inputs, *args, **kwargs)
            block.attention_step = probe_attention_step
        with torch.no_grad():
            probe_model(x)
        return calls[0]
    n_layers = common['n_layers']
    middle_count = n_layers // 2
    assert _count_block_calls(Config(**common, mode='dense')) == n_layers
    assert _count_block_calls(Config(**common, mode='smelt')) == n_layers + middle_count
    dense_default = Config(**common, mode='dense')
    assert dense_default.effective_dense_fnn_hidden == common['top_k'] * common['expert_hidden']
    dense_explicit = Config(**common, mode='dense', dense_fnn_hidden=123)
    assert dense_explicit.effective_dense_fnn_hidden == 123
    try:
        Config(**common, mode='dense', dense_fnn_hidden=-1).validate()
    except ValueError:
        pass
    else:
        raise AssertionError('非法 dense_fnn_hidden 必须被拒绝')
    class RouteWidthProbe(LanguageModel):

        def __init__(self, cfg: Config):
            super().__init__(cfg)
            self.route_widths = []

        def run_independent_blocks(self, block: TransformerBlock, branches: torch.Tensor, top_k: int, scale=None, attn_mask=None):
            self.route_widths.append(top_k)
            return super().run_independent_blocks(block, branches, top_k, scale, attn_mask)
    probe = RouteWidthProbe(Config(**{**common, 'mode': 'B', 'k1': 1, 'k2': 2}))
    with torch.no_grad():
        probe(x)
    middle_count = len(probe.middle_blocks)
    assert probe.route_widths == [1] * (middle_count - 1) + [2] * middle_count
    local_probe = LanguageModel(Config(**{**common, 'n_experts': 4, 'mode': 'local_base', 'k1': 2, 'k2': 2}))
    local_widths = []
    for block in local_probe.blocks:
        original_route = block.moe.route

        def record_route(route_x, route_k, *, original_route=original_route):
            local_widths.append(route_k)
            return original_route(route_x, route_k)
        block.moe.route = record_route

def _selfcheck_top_51755_0004():
    global _, aux_moe, identical_metrics, middle_count, name, overlap_metrics, uniform_aux
    with torch.no_grad():
        local_probe(x)
    middle_count = len(local_probe.middle_blocks)
    assert local_widths == [common['top_k']] + [2, 4] * middle_count + [common['top_k']]
    aux_moe = SparseMoE(Config(**common, mode='smelt'))
    with torch.no_grad():
        aux_moe.router.weight.zero_()
    _, uniform_aux = aux_moe(torch.randn(2, 4, common['d_model']), top_k=2)
    torch.testing.assert_close(uniform_aux, torch.tensor(2.0, device=uniform_aux.device), rtol=1e-05, atol=1e-06)
    overlap_metrics = _routing_set_metrics(torch.tensor([[0, 1]]), torch.tensor([[1, 2]]), 4)
    assert overlap_metrics is not None
    torch.testing.assert_close(torch.tensor(overlap_metrics['second_round_jaccard_mean']), torch.tensor(1.0 / 3.0), rtol=1e-06, atol=1e-06)
    assert overlap_metrics['second_round_exact_match_share'] == 0.0
    assert abs(overlap_metrics['first_round_recall_mean'] - 0.5) < 1e-06
    assert abs(overlap_metrics['second_round_unique_reuse_mean'] - 0.5) < 1e-06
    assert abs(overlap_metrics['second_round_slot_reuse_mean'] - 0.5) < 1e-06
    identical_metrics = _routing_set_metrics(torch.tensor([[0, 1]]), torch.tensor([[0, 1]]), 4)
    assert identical_metrics is not None
    for name in ('second_round_jaccard_mean', 'second_round_exact_match_share', 'first_round_recall_mean', 'second_round_unique_reuse_mean', 'second_round_slot_reuse_mean'):
        assert abs(identical_metrics[name] - 1.0) < 1e-06
    assert _routing_shuffle_seed(7, 0) == _routing_shuffle_seed(7, 0)
    assert _routing_shuffle_seed(7, 0) != _routing_shuffle_seed(7, 1)
    assert _routing_shuffle_seed(7, 0) != _routing_shuffle_seed(8, 0)

def _selfcheck_body_51842_0006():
    global _, batch_index, block, block_report, event, events_by_block, metrics, middle_end, middle_start, name, probe, probe_x, round1_events, round2_events, routing_dataset, routing_device, routing_probe_model, routing_report
    routing_data.write(bytes(range(256)))
    routing_data.flush()
    routing_dataset = ByteDataset(routing_data.name, seq_len=8)
    routing_device = torch.device('cpu')
    routing_probe_model = LanguageModel(Config(**{**common, 'n_experts': 4, 'mode': 'B', 'k1': 2, 'k2': 2}))
    probe = RoutingProbe(routing_probe_model.cfg, 2)
    probe.attach(routing_probe_model)
    with torch.no_grad():
        for batch_index in range(2):
            probe_x, _ = routing_dataset.fixed_eval_batch('val', batch_index, 2, 2, routing_device)
            routing_probe_model(probe_x)
    probe.detach()
    assert all((module._routing_probe is None for module in routing_probe_model.modules() if isinstance(module, SparseMoE)))
    middle_start = routing_probe_model.cfg.middle_start
    middle_end = routing_probe_model.cfg.middle_end
    events_by_block = {}
    for event in probe.events:
        events_by_block.setdefault(event['block'], {}).setdefault(event['round'], []).append(event)
    for block in range(middle_start, middle_end):
        round1_events = events_by_block[block][1]
        round2_events = events_by_block[block][2]
        assert len(round1_events) == 2
        assert len(round2_events) == 2
        assert all((event['top_k'] == 2 for event in round1_events))
        assert all((event['top_k'] == 2 for event in round2_events))
        if block == middle_start:
            assert round1_events[0]['input_shape'] == [2, 8, 16]
            assert round2_events[0]['input_shape'] == [4, 8, 16]
        else:
            assert round1_events[0]['input_shape'][0] == 4
            assert round2_events[0]['input_shape'][0] == 4
            assert len(round2_events[0]['input_shape']) == 3
    routing_report = analyze_routing(routing_probe_model, routing_dataset, 2, 1, routing_device, split='val')
    assert routing_report['mode'] == 'B'
    assert routing_report['tokens_evaluated'] > 0
    for block in range(middle_start, middle_end):
        block_report = routing_report['per_block'][str(block)]
        for name in ('expert_overlap_branch_level', 'expert_overlap_token_level'):
            metrics = block_report[name]
            assert metrics is not None
            assert 0.0 <= metrics['second_round_jaccard_mean'] <= 1.0
            assert 0.0 <= metrics['second_round_unique_reuse_mean'] <= 1.0
        assert 0.0 <= block_report['expert_overlap_branch_level']['second_round_slot_reuse_mean'] <= 1.0
        if block == middle_start:
            assert block_report['branch_self_reuse_share'] is not None
            assert 0.0 <= block_report['branch_self_reuse_share'] <= 1.0
        assert block_report['branch_cosine_mean'] is not None
        assert -1.0 <= block_report['branch_cosine_mean'] <= 1.0
        assert block_report['round1']['dead_experts'] is not None
        assert block_report['round2']['dead_experts'] is not None

def _selfcheck_body_51968_0007():
    global a_model, a_report, block, block_report, dense_model, dense_report, smelt_model, smelt_report
    a_model = LanguageModel(Config(**{**common, 'n_experts': 4, 'mode': 'A', 'k1': 2, 'k2': 2}))
    a_report = analyze_routing(a_model, routing_dataset, 2, 1, routing_device, split='val')
    for block in range(middle_start, middle_end):
        block_report = a_report['per_block'][str(block)]
        assert block_report['expert_overlap_branch_level'] is not None
        assert 0.0 <= block_report['branch_self_reuse_share'] <= 1.0
    smelt_model = LanguageModel(Config(**{**common, 'n_experts': 4, 'mode': 'smelt'}))
    smelt_report = analyze_routing(smelt_model, routing_dataset, 2, 1, routing_device, split='val')
    assert smelt_report['branch_divergence']['summary']['branch_cosine_mean'] is None
    for block in range(middle_start, middle_end):
        assert smelt_report['per_block'][str(block)]['expert_overlap_branch_level'] is not None
    dense_model = LanguageModel(Config(**common, mode='dense'))
    dense_report = analyze_routing(dense_model, routing_dataset, 2, 1, routing_device, split='val')
    assert 'skipped' in dense_report

def _selfcheck_top_51841_0005():
    global routing_data
    with tempfile.NamedTemporaryFile() as routing_data:
        _selfcheck_body_51842_0006()
        _selfcheck_body_51968_0007()

def _selfcheck_top_1_0008():
    _selfcheck_top_51841_0005()

def _selfcheck_top_52048_0009():
    global a, b, branch, causal_model, ckpt_loss, ckpt_model, delayed, local_base, local_cfg, merge_counts, merge_perm, merged, plain_model, single_cfg, smelt, x_changed
    merge_perm = torch.tensor([1, 0, 2])
    merge_counts = [0, 2, 1]
    merged = _MergeExpertOutputs.apply(merge_perm, merge_counts, torch.empty(0, 4, dtype=torch.float32), torch.tensor([[1.0] * 4, [2.0] * 4], dtype=torch.bfloat16), torch.tensor([[3.0] * 4], dtype=torch.bfloat16))
    assert merged.dtype == torch.bfloat16
    assert torch.equal(merged, torch.tensor([[2.0] * 4, [1.0] * 4, [3.0] * 4], dtype=torch.bfloat16))
    single_cfg = dict(common, top_k=1, k1=1, k2=1)
    smelt = LanguageModel(Config(**single_cfg, mode='smelt')).eval()
    branch = LanguageModel(Config(**single_cfg, mode='B')).eval()
    branch.load_state_dict(smelt.state_dict())
    with torch.no_grad():
        a = smelt(x)['logits']
        b = branch(x)['logits']
    torch.testing.assert_close(a, b, rtol=1e-05, atol=1e-06)
    local_cfg = dict(common, k1=1)
    local_base = LanguageModel(Config(**local_cfg, mode='local_base')).eval()
    delayed = LanguageModel(Config(**local_cfg, mode='A')).eval()
    delayed.load_state_dict(local_base.state_dict())
    with torch.no_grad():
        a = local_base(x)['logits']
        b = delayed(x)['logits']
    torch.testing.assert_close(a, b, rtol=1e-05, atol=1e-06)
    ckpt_model = LanguageModel(Config(**common, mode='B')).eval()
    ckpt_model.grad_checkpoint = True
    plain_model = LanguageModel(Config(**common, mode='B')).eval()
    plain_model.load_state_dict(ckpt_model.state_dict())
    with torch.no_grad():
        a = ckpt_model(x)['logits']
        b = plain_model(x)['logits']
    torch.testing.assert_close(a, b, rtol=1e-05, atol=1e-06)
    ckpt_model.grad_checkpoint = True
    ckpt_model.train()
    ckpt_loss = ckpt_model(x, y)['loss']
    ckpt_loss.backward()
    assert torch.isfinite(ckpt_model.blocks[0].attn.qkv.weight.grad).all()
    causal_model = LanguageModel(Config(**common, mode='B')).eval()
    x_changed = x.clone()
    x_changed[:, -1] = (x_changed[:, -1] + 1) % 256

def _selfcheck_top_52201_0010():
    global _, block, branches, changed, logits_changed, logits_original, model, out1, out2
    with torch.no_grad():
        logits_original = causal_model(x)['logits']
        logits_changed = causal_model(x_changed)['logits']
    torch.testing.assert_close(logits_original[:, :-1], logits_changed[:, :-1], rtol=1e-05, atol=1e-06)
    model = LanguageModel(Config(**common, mode='B')).eval()
    block = model.blocks[model.middle_start]
    branches = torch.randn(2, 2, 8, 16)
    changed = branches.clone()
    changed[:, 1] += torch.randn_like(changed[:, 1])
    with torch.no_grad():
        out1, _ = model.run_independent_blocks(block, branches, 2)
        out2, _ = model.run_independent_blocks(block, changed, 2)
    torch.testing.assert_close(out1[:, 0], out2[:, 0], rtol=1e-05, atol=1e-06)
    assert not torch.allclose(out1[:, 1], out2[:, 1])

def _selfcheck_body_52263_0012():
    global _ShiftedTestDataset, eval_dataset, eval_device, first, probe_report, second, shifted_test_dataset, shifted_test_probe, test_probe_report, test_x, test_y, train_x, train_y
    data_file.write(bytes(range(256)))
    data_file.flush()
    eval_dataset = ByteDataset(data_file.name, seq_len=8)
    eval_device = torch.device('cpu')
    assert eval_dataset.data_sha256 == hashlib.sha256(bytes(range(256))).hexdigest()
    assert len(eval_dataset.train) == 232
    assert int(eval_dataset.train[0]) == 0
    assert int(eval_dataset.train[-1]) == 231
    assert int(eval_dataset.val[0]) == 232
    assert int(eval_dataset.val[-1]) == 243
    assert int(eval_dataset.test[0]) == 244
    assert int(eval_dataset.test[-1]) == 255
    assert eval_dataset.has_test
    try:
        eval_dataset.batch('test', 2, torch.Generator().manual_seed(0), eval_device)
    except ValueError:
        pass
    else:
        raise AssertionError('batch() 不允许访问测试集')
    train_x, train_y = eval_dataset.batch('train', 4, torch.Generator().manual_seed(7), eval_device)
    assert int(train_x.min()) >= 0
    assert int(train_x.max()) <= 231
    assert int(train_y.max()) <= 231
    assert torch.equal(train_x[:, 1:], train_y[:, :-1].to(torch.int32))
    test_x, test_y = eval_dataset.fixed_eval_batch('test', 0, 1, 2, eval_device)
    assert int(test_x.min()) >= 244
    assert int(test_y.max()) <= 255
    first = eval_dataset.fixed_val_batch(0, batches=2, batch_size=2, device=eval_device)
    second = eval_dataset.fixed_val_batch(0, batches=2, batch_size=2, device=eval_device)
    probe_report = probe_data_pipeline(eval_dataset, eval_device, batch_size=4, batches=2)
    assert probe_report['ok'], probe_report['failures']
    assert probe_report['train_checked']
    assert probe_report['val_fixed_checked']
    assert not probe_report['test_fixed_checked']
    test_probe_report = probe_data_pipeline(eval_dataset, eval_device, batch_size=4, batches=2, include_test=True)
    assert test_probe_report['ok'], test_probe_report['failures']
    assert test_probe_report['test_fixed_checked']
    class _ShiftedTestDataset(ByteDataset):

        def fixed_eval_batch(self, split, batch_index, batches, batch_size, device):
            x, y = ByteDataset.fixed_eval_batch(self, split, batch_index, batches, batch_size, device)
            if split == 'test':
                y = torch.roll(y, shifts=1, dims=1)
            return (x, y)
    shifted_test_dataset = _ShiftedTestDataset(data_file.name, seq_len=8)
    shifted_test_probe = probe_data_pipeline(shifted_test_dataset, eval_device, batch_size=4, batches=2, include_test=True)
    assert not shifted_test_probe['ok']
    assert any(('测试标签错位' in failure for failure in shifted_test_probe['failures'])), shifted_test_probe['failures']

def _selfcheck_body_52408_0013():
    global _OutOfRangeEvalDataset, _ShiftedLabelDataset, out_of_range_dataset, out_of_range_probe, shifted_dataset, shifted_report
    class _ShiftedLabelDataset(ByteDataset):

        def batch(self, split, batch_size, generator, device):
            x, y = ByteDataset.batch(self, split, batch_size, generator, device)
            return (x, torch.roll(y, shifts=1, dims=1))
    shifted_dataset = _ShiftedLabelDataset(data_file.name, seq_len=8)
    shifted_report = probe_data_pipeline(shifted_dataset, eval_device, batch_size=4, batches=2)
    assert not shifted_report['ok']
    assert any(('标签错位' in failure for failure in shifted_report['failures'])), shifted_report['failures']
    class _OutOfRangeEvalDataset(ByteDataset):

        def fixed_eval_batch(self, split, batch_index, batches, batch_size, device):
            x, y = ByteDataset.fixed_eval_batch(self, split, batch_index, batches, batch_size, device)
            if split == 'val':
                x = x + 256
            return (x, y)
    out_of_range_dataset = _OutOfRangeEvalDataset(data_file.name, seq_len=8)
    out_of_range_probe = probe_data_pipeline(out_of_range_dataset, eval_device, batch_size=4, batches=2)
    assert not out_of_range_probe['ok']
    assert any(('验证固定窗口输入 token 超出' in failure for failure in out_of_range_probe['failures'])), out_of_range_probe['failures']

def _selfcheck_top_52262_0011():
    global data_file
    with tempfile.NamedTemporaryFile() as data_file:
        _selfcheck_body_52263_0012()
        _selfcheck_body_52408_0013()

def _selfcheck_top_1_0014():
    _selfcheck_top_52262_0011()

def _selfcheck_top_52478_0015():
    global auc_cap_before, auc_metrics, capped_auc_metrics, eval_model, manual_class_sums, manual_counts, overfull_empty, overfull_first, overfull_last, val_metrics, val_starts
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    overfull_first = eval_dataset.fixed_val_batch(0, batches=3, batch_size=4, device=eval_device)
    overfull_last = eval_dataset.fixed_val_batch(1, batches=3, batch_size=4, device=eval_device)
    overfull_empty = eval_dataset.fixed_val_batch(2, batches=3, batch_size=4, device=eval_device)
    val_starts = eval_dataset._fixed_starts['val']['starts']
    assert len(overfull_first[1]) + len(overfull_last[1]) == len(val_starts)
    assert torch.equal(val_starts.sort().values, val_starts.unique(sorted=True))
    assert overfull_empty[0].numel() == 0
    assert overfull_empty[1].numel() == 0
    eval_model = LanguageModel(Config(**common, mode='B')).eval()
    val_metrics = evaluate(eval_model, eval_dataset, 2, 1, eval_device)
    assert 'val_loss' in val_metrics
    assert 'val_ppl' in val_metrics
    auc_metrics = evaluate(eval_model, eval_dataset, 2, 1, eval_device, compute_auc=True)
    assert int(auc_metrics['val_auc_tokens_evaluated']) > 0
    assert int(auc_metrics['val_auc_tokens_available']) == int(auc_metrics['val_auc_tokens_evaluated'])
    assert float(auc_metrics['val_auc_coverage_fraction']) == 1.0
    assert auc_metrics['val_auc_sampling'] == 'full'
    auc_cap_before = _SC_BASE['_AUC_MAX_TOKENS']
    _SC_BASE['_AUC_MAX_TOKENS'] = 8
    try:
        capped_auc_metrics = evaluate(eval_model, eval_dataset, 2, 1, eval_device, compute_auc=True)
    finally:
        _SC_BASE['_AUC_MAX_TOKENS'] = auc_cap_before
    assert capped_auc_metrics['val_auc_sampling'] == 'uniform_stride'
    assert int(capped_auc_metrics['val_auc_tokens_evaluated']) == 8
    assert int(capped_auc_metrics['val_auc_tokens_available']) > 8
    assert 0.0 < float(capped_auc_metrics['val_auc_coverage_fraction']) < 1.0
    assert 'val_macro_nll' in val_metrics
    assert 'val_macro_nll_chance' in val_metrics
    assert 'val_macro_nll_classes' in val_metrics
    assert abs(float(val_metrics['val_macro_nll_chance']) - math.log(256.0)) < 1e-12
    assert float(val_metrics['val_macro_nll']) > 0.0
    assert float(val_metrics['val_macro_nll_classes']) >= 1.0
    assert float(val_metrics['val_macro_nll_min_support']) >= 1.0
    manual_class_sums = torch.zeros(256, dtype=torch.float64)
    manual_counts = torch.zeros(256, dtype=torch.float64)

def _selfcheck_top_52613_0016():
    global empty_balanced, lift_probe, manual_balanced, manual_batch_index, manual_logits, manual_losses, manual_target, manual_weighted, manual_x, manual_y, synthetic_balanced
    with torch.no_grad():
        for manual_batch_index in range(2):
            manual_x, manual_y = eval_dataset.fixed_eval_batch('val', manual_batch_index, 2, 1, eval_device)
            if manual_x.size(0) == 0:
                break
            manual_logits = eval_model(manual_x, manual_y)['logits']
            manual_target = manual_y.reshape(-1).to(torch.int64)
            manual_losses = torch.nn.functional.cross_entropy(manual_logits.reshape(-1, 256).float(), manual_target, reduction='none')
            manual_class_sums.index_add_(0, manual_target, manual_losses.to(torch.float64))
            manual_counts.index_add_(0, manual_target, torch.ones_like(manual_losses, dtype=torch.float64))
    manual_weighted = float(manual_class_sums.sum() / manual_counts.sum())
    assert abs(manual_weighted - float(val_metrics['val_loss'])) < 0.0001 * max(1.0, abs(manual_weighted))
    manual_balanced = class_balanced_loss_metrics(manual_class_sums, manual_counts, 256)
    assert abs(float(manual_balanced['macro_nll']) - float(val_metrics['val_macro_nll'])) < 0.0001 * max(1.0, abs(manual_weighted))
    synthetic_balanced = class_balanced_loss_metrics(torch.tensor([2.0, 4.0, 0.0]), torch.tensor([2.0, 1.0, 0.0]), 256)
    assert abs(float(synthetic_balanced['macro_nll']) - 2.5) < 1e-12
    assert synthetic_balanced['macro_nll_classes'] == 2.0
    assert synthetic_balanced['macro_nll_min_support'] == 1.0
    assert abs(float(synthetic_balanced['macro_nll_chance']) - math.log(256.0)) < 1e-12
    empty_balanced = class_balanced_loss_metrics(torch.zeros(256, dtype=torch.float64), torch.zeros(256, dtype=torch.float64), 256)
    assert math.isnan(float(empty_balanced['macro_nll']))
    assert empty_balanced['macro_nll_classes'] == 0.0
    assert 'val_macro_nll' in COMPARISON_METRIC_KEYS
    assert 'val_macro_nll_at_common_step' in COMPARISON_METRIC_KEYS
    assert 'val_macro_nll' in _SELECTION_METRIC_KEYS
    assert 'val_pr_auc_macro' in _SELECTION_METRIC_KEYS
    assert 'val_pr_auc_macro_lift' in _SELECTION_METRIC_KEYS
    assert 'val_roc_auc_macro' in _SELECTION_METRIC_KEYS
    assert metric_higher_is_better('val_pr_auc_macro')
    assert metric_higher_is_better('val_pr_auc_macro_lift')
    assert metric_higher_is_better('val_roc_auc_macro')
    assert 'val_pr_auc_macro_lift' in COMPARISON_METRIC_KEYS
    lift_probe = macro_auc_metrics(torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.3, 0.7], [0.1, 0.9]]), torch.tensor([0, 0, 0, 1, 1, 1]), 2, min_support=1)
    assert abs(float(lift_probe['pr_auc_macro_lift']) - (float(lift_probe['pr_auc_macro']) - float(lift_probe['pr_auc_macro_chance']))) < 1e-12
    assert _TRIVIAL_REFERENCE_FIELDS['val_pr_auc_macro_lift'] == 'unigram_pr_auc_macro_lift'
    assert hypothesis_metric_unverifiable(['pr_auc_lift'], None) == []
    assert not metric_higher_is_better('val_macro_nll')
    assert not metric_higher_is_better('val_macro_nll_at_common_step')
    assert _TRIVIAL_REFERENCE_FIELDS['val_macro_nll'] == 'unigram_macro_nll'
    assert _TRIVIAL_REFERENCE_FIELDS['val_macro_nll_at_common_step'] == 'unigram_macro_nll'
    assert hypothesis_metric_unverifiable(['macro_nll'], None) == []

def _selfcheck_top_52750_0017():
    global boot_metrics_a, boot_metrics_b, byte_shift, clean_audit, high, low, name, shift_report, test_metrics
    boot_metrics_a = evaluate(eval_model, eval_dataset, 2, 1, eval_device, bootstrap_replicates=64)
    boot_metrics_b = evaluate(eval_model, eval_dataset, 2, 1, eval_device, bootstrap_replicates=64)
    assert boot_metrics_a == boot_metrics_b
    assert boot_metrics_a['val_loss'] == val_metrics['val_loss']
    assert boot_metrics_a['val_bootstrap_replicates'] == 64
    assert float(boot_metrics_a['val_loss_ci95_low']) <= float(boot_metrics_a['val_loss']) <= float(boot_metrics_a['val_loss_ci95_high'])
    assert float(boot_metrics_a['val_ppl_ci95_low']) <= float(boot_metrics_a['val_ppl']) <= float(boot_metrics_a['val_ppl_ci95_high'])
    assert float(boot_metrics_a['val_token_accuracy_ci95_low']) <= float(boot_metrics_a['val_token_accuracy']) <= float(boot_metrics_a['val_token_accuracy_ci95_high'])
    for name in ('macro_f1', 'weighted_f1', 'precision_macro', 'precision_weighted', 'recall_macro', 'recall_weighted', 'macro_nll'):
        low = boot_metrics_a.get(f'val_{name}_ci95_low')
        high = boot_metrics_a.get(f'val_{name}_ci95_high')
        assert low is not None, name
        assert high is not None, name
        assert math.isfinite(float(low)), name
        assert math.isfinite(float(high)), name
        assert float(low) <= float(high), name
    assert boot_metrics_a['val_eval_windows'] >= 1
    assert 0.0 <= float(boot_metrics_a['val_eval_window_overlap_fraction']) <= 1.0
    test_metrics = evaluate(eval_model, eval_dataset, 2, 1, eval_device, split='test')
    assert 'test_loss' in test_metrics
    assert 'val_loss' not in test_metrics
    clean_audit = audit_data_splits(eval_dataset, include_test=True)
    assert clean_audit['duplicate_windows']['train_val']['shared_signatures'] == 0
    assert clean_audit['duplicate_windows']['train_test']['shared_signatures'] == 0
    assert not any(('重复' in warning for warning in clean_audit['warnings']))
    assert clean_audit['leakage_violations'] == []
    assert enforce_data_leakage_gate(clean_audit)
    assert test_label_integrity_violations({'splits': {'test': {'invalid_tokens': 0}}}) == []
    assert enforce_test_label_integrity({'splits': {'test': {'invalid_tokens': 0}}})
    assert test_label_integrity_violations({'splits': {'train': {'utf8_replacements': 3}, 'test': {'invalid_tokens': 1, 'utf8_replacements': 2, 'nul_bytes': 4}}}) == ['test: 1 个越界 token（标签超出词表）', 'test: 2 个无法按 UTF-8 解码的字节（标签噪声）', 'test: 4 个 NUL 字节（二进制/填充污染）']
    assert not enforce_test_label_integrity({'splits': {'test': {'utf8_replacements': 1}}})
    shift_report = {'distribution_shift': {'train_val': 0.12}, 'distribution_shift_bigram': {'train_val': 0.03}, 'distribution_shift_conditional': {'train_val': 0.2, 'val_test': float('nan')}}
    assert distribution_shift_violations(shift_report) == []
    byte_shift = distribution_shift_violations(shift_report, max_byte_js=0.1)
    assert len(byte_shift) == 1
    assert 'train_val' in byte_shift[0]

def _selfcheck_top_52887_0018():
    global boundary_shift, boundary_shift_report, chunk_audit, chunk_dataset, chunk_duplicate, chunk_file, chunk_payload, chunk_rng, chunk_test, chunk_train, chunk_val, repeated_audit, repeated_dataset, repeated_file
    assert distribution_shift_violations(shift_report, max_bigram_js=0.04, max_conditional_js=0.21) == []
    assert enforce_distribution_shift_gate(shift_report, max_byte_js=0.12, max_bigram_js=0.03, max_conditional_js=0.2)
    assert not enforce_distribution_shift_gate(shift_report, max_byte_js=0.11)
    boundary_shift_report = {'distribution_shift': {'train_val': 0.02}, 'temporal_boundary_shift': {'pairs': {'train_tail_val_head': {'js': 0.2, 'left_bytes': 4096, 'right_bytes': 4096}}}}
    boundary_shift = distribution_shift_violations(boundary_shift_report, max_byte_js=0.1)
    assert len(boundary_shift) == 1
    assert '时间边界' in boundary_shift[0]
    assert not enforce_distribution_shift_gate(boundary_shift_report, max_byte_js=0.1)
    assert all((info['shared_chunks'] == 0 for info in clean_audit['content_chunks']['cross_split'].values()))
    with tempfile.NamedTemporaryFile() as repeated_file:
        repeated_file.write(bytes(range(64)) * 4)
        repeated_file.flush()
        repeated_dataset = ByteDataset(repeated_file.name, seq_len=8)
        repeated_audit = audit_data_splits(repeated_dataset, include_test=True)
    assert repeated_audit['duplicate_windows']['train_val']['shared_signatures'] > 0
    assert repeated_audit['duplicate_windows']['train_test']['shared_signatures'] > 0
    assert any(('重复' in warning for warning in repeated_audit['warnings']))
    assert repeated_audit['violations']
    assert any(('重复' in violation for violation in repeated_audit['violations']))
    assert repeated_audit['leakage_violations']
    assert not enforce_data_leakage_gate(repeated_audit)
    assert not enforce_strict_data_audit(repeated_audit)
    chunk_rng = random.Random(20241021)
    chunk_duplicate = chunk_rng.randbytes(512)
    chunk_train = chunk_rng.randbytes(5400)
    chunk_train = chunk_train[:500] + chunk_duplicate + chunk_train[1012:]
    chunk_val = chunk_rng.randbytes(100) + chunk_duplicate
    chunk_test = chunk_rng.randbytes(300)
    chunk_payload = chunk_train + chunk_val + chunk_test
    with tempfile.NamedTemporaryFile() as chunk_file:
        chunk_file.write(chunk_payload)
        chunk_file.flush()
        chunk_dataset = ByteDataset(chunk_file.name, seq_len=8, val_fraction=len(chunk_val) / len(chunk_payload), test_fraction=len(chunk_test) / len(chunk_payload))
        assert chunk_dataset.split_bounds == {'train': (0, len(chunk_train)), 'val': (len(chunk_train), len(chunk_train) + len(chunk_val)), 'test': (len(chunk_train) + len(chunk_val), len(chunk_payload))}
        chunk_audit = audit_data_splits(chunk_dataset, include_test=True, max_windows=1)
    assert chunk_audit['duplicate_windows']['train_val']['shared_signatures'] == 0

def _selfcheck_top_53026_0019():
    global chunk_coverage, clean_coverage, distance, info, order_audit, order_dataset, order_file, order_train, order_val, padded_audit, padded_dataset, padded_file, share
    assert chunk_audit['content_chunks']['cross_split']['train_val']['shared_chunks'] > 0
    assert chunk_audit['content_chunks']['cross_split']['train_val']['shared_bytes'] >= chunk_dataset.seq_len + 1
    assert any(('内容定义分块' in violation for violation in chunk_audit['leakage_violations']))
    assert not enforce_data_leakage_gate(chunk_audit)
    assert chunk_audit['content_chunks']['cross_split']['train_test']['shared_chunks'] == 0
    assert chunk_audit['content_chunks']['cross_split']['val_test']['shared_chunks'] == 0
    chunk_coverage = chunk_audit['leakage_scan_coverage']
    assert chunk_coverage['exhaustive'] is False
    assert '不等于' in chunk_coverage['residual_risk']
    assert chunk_coverage['per_split']['train']['stride'] > 1
    assert any(('不等于' in warning for warning in chunk_audit['warnings']))
    clean_coverage = clean_audit['leakage_scan_coverage']
    assert clean_coverage['exhaustive'] is True
    assert clean_coverage['residual_risk'] is None
    for info in clean_coverage['per_split'].values():
        assert info['stride'] == 1
        assert info['coverage'] == 1.0
    with tempfile.NamedTemporaryFile() as padded_file:
        padded_file.write(b'\x00' * 220 + bytes(range(36)))
        padded_file.flush()
        padded_dataset = ByteDataset(padded_file.name, seq_len=8)
        padded_audit = audit_data_splits(padded_dataset, include_test=True)
    assert any(('退化' in warning for warning in padded_audit['warnings']))
    for distance in clean_audit['distribution_shift'].values():
        assert math.isfinite(distance)
        assert 0.0 <= distance <= 1.0
    for distance in clean_audit['distribution_shift_bigram'].values():
        assert math.isfinite(distance)
        assert 0.0 <= distance <= 1.0
    for distance in clean_audit['distribution_shift_conditional'].values():
        assert math.isfinite(distance)
        assert 0.0 <= distance <= 1.0
    for share in clean_audit['unseen_transition_share'].values():
        assert math.isfinite(share)
        assert 0.0 <= share <= 1.0
    for share in clean_audit['seen_context_share'].values():
        assert math.isfinite(share)
        assert 0.0 <= share <= 1.0
    order_train = bytes(range(128)) * 8
    order_val = bytes(reversed(range(128))) * 8
    with tempfile.NamedTemporaryFile() as order_file:
        order_file.write(order_train + order_val)
        order_file.flush()
        order_dataset = ByteDataset(order_file.name, seq_len=8, val_fraction=0.5, test_fraction=0.0)
        order_audit = audit_data_splits(order_dataset)
    assert order_audit['distribution_shift']['train_val'] == 0.0
    assert order_audit['distribution_shift_bigram']['train_val'] > _AUDIT_BIGRAM_SHIFT_WARN
    assert any(('二元组分布偏移' in warning for warning in order_audit['warnings']))
    assert order_audit['distribution_shift_conditional']['train_val'] > _AUDIT_CONDITIONAL_SHIFT_WARN
    assert order_audit['unseen_transition_share']['train_val'] == 1.0

def _selfcheck_top_53172_0020():
    global corrupt_audit, corrupt_body, corrupt_dataset, corrupt_file, corrupt_rng, low_noise_audit, low_noise_body, low_noise_dataset, low_noise_file, low_noise_rng, val_noise_audit, val_noise_body, val_noise_dataset, val_noise_file, val_noise_rng
    assert order_audit['seen_context_share']['train_val'] == 1.0
    assert any(('标签分布偏移' in warning for warning in order_audit['warnings']))
    with tempfile.NamedTemporaryFile() as corrupt_file:
        corrupt_rng = random.Random(20241020)
        corrupt_body = bytes((corrupt_rng.choice(b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(1800)))
        corrupt_file.write(corrupt_body[:600] + b'\xff\xfe' + b'\x00' * 4 + corrupt_body[600:])
        corrupt_file.flush()
        corrupt_dataset = ByteDataset(corrupt_file.name, seq_len=8)
        corrupt_audit = audit_data_splits(corrupt_dataset, include_test=True)
    assert corrupt_audit['splits']['train']['utf8_replacements'] > 0
    assert corrupt_audit['splits']['train']['nul_bytes'] == 4
    assert any(('UTF-8' in warning for warning in corrupt_audit['warnings']))
    assert any(('NUL' in warning for warning in corrupt_audit['warnings']))
    assert corrupt_audit['violations']
    assert not enforce_strict_data_audit(corrupt_audit)
    assert corrupt_audit['leakage_violations'] == []
    assert enforce_data_leakage_gate(corrupt_audit)
    low_noise_rng = random.Random(20241021)
    low_noise_body = bytes((low_noise_rng.choice(b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(20000)))
    low_noise_body = low_noise_body[:10000] + b'\xff' + low_noise_body[10000:]
    with tempfile.NamedTemporaryFile() as low_noise_file:
        low_noise_file.write(low_noise_body)
        low_noise_file.flush()
        low_noise_dataset = ByteDataset(low_noise_file.name, seq_len=8)
        low_noise_audit = audit_data_splits(low_noise_dataset, include_test=True)
    assert low_noise_audit['splits']['train']['utf8_replacements'] == 1
    assert low_noise_audit['violations']
    assert not any(('UTF-8' in warning for warning in low_noise_audit['warnings']))
    assert not enforce_strict_data_audit(low_noise_audit)
    assert low_noise_audit['leakage_violations'] == []
    val_noise_rng = random.Random(20241022)
    val_noise_body = bytes((val_noise_rng.choice(b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(20000)))
    val_noise_body = val_noise_body[:18500] + b'\xff' + val_noise_body[18500:]
    with tempfile.NamedTemporaryFile() as val_noise_file:
        val_noise_file.write(val_noise_body)
        val_noise_file.flush()
        val_noise_dataset = ByteDataset(val_noise_file.name, seq_len=8)
        val_noise_audit = audit_data_splits(val_noise_dataset, include_test=False)
    assert val_noise_audit['val_label_noise'] == {'invalid_tokens': 0, 'utf8_replacements': 1, 'nul_bytes': 0, 'has_label_noise': True, 'used_for_selection': True}
    assert not any(('train: 检测到' in warning and 'UTF-8' in warning for warning in val_noise_audit['warnings']))
    assert any(('val: 验证集含标签噪声' in warning and '选型' in warning for warning in val_noise_audit['warnings']))

def _selfcheck_top_53314_0021():
    global boundary_audit, boundary_dataset, boundary_file, boundary_payload, invalid_scan, literal_audit, literal_body, literal_dataset, literal_file, literal_replacement, literal_rng, literal_scan, split_name, temporal_audit, temporal_dataset, temporal_file, temporal_pair, temporal_payload, temporal_shift
    assert val_noise_audit['violations']
    assert not enforce_strict_data_audit(val_noise_audit)
    literal_replacement = '�'.encode('utf-8')
    literal_scan = _scan_text_integrity(memoryview(bytearray(literal_replacement)), 0, len(literal_replacement), final=True)
    assert literal_scan['utf8_replacements'] == 0
    assert literal_scan['nul_bytes'] == 0
    invalid_scan = _scan_text_integrity(memoryview(bytearray(b'\xff')), 0, 1, final=True)
    assert invalid_scan['utf8_replacements'] == 1
    literal_rng = random.Random(20241022)
    literal_body = bytes((literal_rng.choice(b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(20000)))
    literal_body = literal_body[:10000] + literal_replacement + literal_body[10000:]
    with tempfile.NamedTemporaryFile() as literal_file:
        literal_file.write(literal_body)
        literal_file.flush()
        literal_dataset = ByteDataset(literal_file.name, seq_len=8)
        literal_audit = audit_data_splits(literal_dataset, include_test=True)
    assert literal_audit['splits']['train']['utf8_replacements'] == 0
    assert literal_audit['violations'] == []
    assert enforce_strict_data_audit(literal_audit)
    boundary_payload = b'a' * 161 + '€'.encode('utf-8') + b'b' * 16
    assert len(boundary_payload) == 180
    with tempfile.NamedTemporaryFile() as boundary_file:
        boundary_file.write(boundary_payload)
        boundary_file.flush()
        boundary_dataset = ByteDataset(boundary_file.name, seq_len=8)
        assert boundary_dataset.split_bounds['train'] == (0, 162)
        assert boundary_dataset.split_bounds['val'] == (162, 171)
        boundary_audit = audit_data_splits(boundary_dataset, include_test=True)
    for split_name in ('train', 'val', 'test'):
        assert boundary_audit['splits'][split_name]['utf8_replacements'] == 0
    assert not any(('UTF-8' in warning for warning in boundary_audit['warnings']))
    assert boundary_audit['violations'] == []
    assert enforce_strict_data_audit(boundary_audit)
    temporal_payload = b'A' * 29900 + b'B' * 100 + b'C' * 100 + b'A' * 9900
    with tempfile.NamedTemporaryFile() as temporal_file:
        temporal_file.write(temporal_payload)
        temporal_file.flush()
        temporal_dataset = ByteDataset(temporal_file.name, seq_len=8, val_fraction=0.25, test_fraction=0.0)
        temporal_audit = audit_data_splits(temporal_dataset, include_test=False)
    temporal_shift = temporal_audit['temporal_boundary_shift']
    assert 'test' not in temporal_audit['splits']
    assert set(temporal_shift['pairs']) == {'train_tail_val_head'}
    temporal_pair = temporal_shift['pairs']['train_tail_val_head']
    assert temporal_pair['js'] > 0.1
    assert temporal_pair['js'] > temporal_audit['distribution_shift']['train_val']
    assert any(('时间边界' in warning for warning in temporal_audit['warnings']))
    assert parse_group_separator('\\n\\n') == b'\n\n'
    assert parse_group_separator('<|end|>') == b'<|end|>'
    assert parse_group_separator('\\x00') == b'\x00'
    assert parse_group_separator('\\u2028') == '\u2028'.encode('utf-8')
    assert parse_group_separator('\\\\n') == b'\\n'

def _selfcheck_top_53462_0022():
    global duplicate_eval_file, duplicate_eval_groups, duplicate_eval_payload, duplicate_val_audit, duplicate_val_dataset, grouped_audit, grouped_dataset, grouped_file, grouped_payload, grouped_val, grouped_x, grouped_y, groups, invalid, separator
    for invalid in ('', '\\', '\\x1', '\\q', '\\ud800', '\ud800'):
        try:
            parse_group_separator(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f'非法群组分隔符必须报错: {invalid!r}')
    separator = b'\n\n'
    groups = [bytes([97 + index]) * 10 for index in range(5)]
    grouped_payload = separator.join(groups) + separator
    assert len(grouped_payload) == 5 * 12
    with tempfile.NamedTemporaryFile() as grouped_file:
        grouped_file.write(grouped_payload)
        grouped_file.flush()
        grouped_dataset = ByteDataset(grouped_file.name, seq_len=8, val_fraction=0.2, test_fraction=0.2, group_separator=separator)
        assert grouped_dataset.split_bounds['train'] == (0, 36)
        assert grouped_dataset.split_bounds['val'] == (36, 48)
        assert grouped_dataset.split_bounds['test'] == (48, 60)
        assert grouped_dataset.groups_per_split == {'train': 3, 'val': 1, 'test': 1}
        assert bytes(grouped_dataset.train[-2:]) == separator
        assert bytes(grouped_dataset.val) == groups[3] + separator
        assert bytes(grouped_dataset.test) == groups[4] + separator
        grouped_x, grouped_y = grouped_dataset.batch('train', 4, torch.Generator().manual_seed(3), eval_device)
        assert int(grouped_x.min()) >= 10
        assert int(grouped_x.max()) <= 99
        assert int(grouped_y.max()) <= 99
        grouped_val = grouped_dataset.fixed_eval_batch('val', 0, 1, 2, eval_device)[0]
        assert int(grouped_val.min()) == 10
        assert int(grouped_val.max()) == 100
        grouped_audit = audit_data_splits(grouped_dataset, include_test=True)
    assert grouped_audit['groups']['groups_total'] == 5
    assert grouped_audit['groups']['groups_per_split'] == {'train': 3, 'val': 1, 'test': 1}
    assert grouped_audit['violations'] == []
    assert grouped_audit['evaluation_violations'] == []
    assert enforce_evaluation_duplicate_gate(grouped_audit)
    duplicate_eval_groups = [b'A' * 10, b'B' * 10, b'C' * 10, b'D' * 10, b'D' * 10, b'E' * 10]
    duplicate_eval_payload = separator.join(duplicate_eval_groups) + separator
    with tempfile.NamedTemporaryFile() as duplicate_eval_file:
        duplicate_eval_file.write(duplicate_eval_payload)
        duplicate_eval_file.flush()
        duplicate_val_dataset = ByteDataset(duplicate_eval_file.name, seq_len=8, val_fraction=1.0 / 3.0, test_fraction=1.0 / 6.0, group_separator=separator)
        duplicate_val_audit = audit_data_splits(duplicate_val_dataset, include_test=True)
    assert duplicate_val_audit['leakage_violations'] == []
    assert duplicate_val_audit['groups']['content_duplicates']['per_split']['val']['within_split_duplicates'] == 1
    assert any(('val' in violation and '重复' in violation for violation in duplicate_val_audit['evaluation_violations']))

def _selfcheck_top_53597_0023():
    global _, begin, dedup_bias, dedup_dataset, dedup_file, dedup_groups, dedup_payload, dedup_x, duplicate_test_audit, duplicate_test_dataset, duplicate_test_file, duplicate_test_groups, duplicate_test_payload, end, no_test_dataset, no_test_file, no_test_probe, split_name
    assert not enforce_evaluation_duplicate_gate(duplicate_val_audit)
    duplicate_test_groups = [b'A' * 10, b'B' * 10, b'C' * 10, b'D' * 10, b'E' * 10, b'E' * 10]
    duplicate_test_payload = separator.join(duplicate_test_groups) + separator
    with tempfile.NamedTemporaryFile() as duplicate_test_file:
        duplicate_test_file.write(duplicate_test_payload)
        duplicate_test_file.flush()
        duplicate_test_dataset = ByteDataset(duplicate_test_file.name, seq_len=8, val_fraction=1.0 / 6.0, test_fraction=1.0 / 3.0, group_separator=separator)
        duplicate_test_audit = audit_data_splits(duplicate_test_dataset, include_test=True)
    assert duplicate_test_audit['leakage_violations'] == []
    assert any(('test' in violation and '重复' in violation for violation in duplicate_test_audit['evaluation_violations']))
    assert not enforce_evaluation_duplicate_gate(duplicate_test_audit)
    with tempfile.NamedTemporaryFile() as no_test_file:
        no_test_file.write(separator.join(groups))
        no_test_file.flush()
        no_test_dataset = ByteDataset(no_test_file.name, seq_len=8, val_fraction=0.2, test_fraction=0.0, group_separator=separator)
        assert not no_test_dataset.has_test
        assert no_test_dataset.split_bounds['test'][0] == no_test_dataset.split_bounds['test'][1] == no_test_dataset.num_bytes
        no_test_probe = probe_data_pipeline(no_test_dataset, eval_device, batch_size=2, batches=1)
    assert no_test_probe['ok'], no_test_probe['failures']
    dedup_groups = [b'A' * 10, b'B' * 10, b'A' * 10, b'C' * 10, b'B' * 10, b'D' * 10]
    dedup_payload = separator.join(dedup_groups) + separator
    with tempfile.NamedTemporaryFile() as dedup_file:
        dedup_file.write(dedup_payload)
        dedup_file.flush()
        dedup_dataset = ByteDataset(dedup_file.name, seq_len=8, val_fraction=1.0 / 6.0, test_fraction=1.0 / 6.0, group_separator=separator, deduplicate_train_groups=True)
    assert dedup_dataset.split_bounds['train'] == (0, 48)
    for split_name, (begin, end) in dedup_dataset.split_bounds.items():
        assert dedup_dataset.split_sha256[split_name] == hashlib.sha256(dedup_payload[begin:end]).hexdigest()
    assert dedup_dataset.train_excluded_spans == [(12, 22), (24, 34)]
    assert dedup_dataset.train_cleaning['groups_total'] == 4
    assert dedup_dataset.train_cleaning['groups_kept'] == 2
    assert dedup_dataset.train_cleaning['dropped_duplicate_of'] == {'val': 1, 'test': 0, 'train': 1}
    assert dedup_dataset.train_num_valid_starts == 10
    dedup_x, _ = dedup_dataset.batch('train', 32, torch.Generator().manual_seed(7), eval_device)
    assert int(dedup_x.min()) >= 10
    assert 66 not in set(dedup_x.flatten().tolist())
    assert dedup_dataset.effective_train_bytes == 28
    dedup_bias = dedup_dataset.train_cleaning['deletion_bias']
    assert dedup_bias['dropped_bytes'] == 20
    assert dedup_bias['kept_bytes'] == 28
    assert dedup_bias['train_bytes'] == 48
    assert dedup_bias['drop_fraction'] == round(20 / 48, 6)
    assert dedup_bias['excluded_span_lengths']['spans'] == 2
    assert dedup_dataset.effective_train_sha256 == _effective_train_fingerprint(dedup_dataset._raw, 48, dedup_dataset.train_excluded_spans)

def _selfcheck_top_53739_0024():
    global cleaned_ngram, dedup_counts, isolation, leaky_dedup_audit, leaky_dedup_dataset, overlapping_isolation, raw_ngram, sealed_dedup_audit, sealed_dedup_dataset, sealed_dedup_file, sealed_dedup_groups, sealed_dedup_payload
    assert dedup_dataset.effective_train_sha256 != _effective_train_fingerprint(dedup_dataset._raw, 48, [])
    isolation = dedup_dataset.split_isolation
    assert isolation['split_order'] == ['train', 'val', 'test']
    assert isolation['bounds']['train'] == [0, 48]
    assert isolation['contiguous'] is True
    assert isolation['covers_all_bytes'] is True
    assert isolation['overlap_bytes'] == 0
    assert isolation['policy'] == 'prefix_contiguous_no_shuffle'
    overlapping_isolation = _split_isolation_record({'train': (0, 30), 'val': (20, 50), 'test': (50, 60)}, 60)
    assert overlapping_isolation['contiguous'] is False
    assert overlapping_isolation['overlap_bytes'] == 10
    assert _split_isolation_record({'train': (0, 30), 'val': (30, 50), 'test': (50, 60)}, 61)['covers_all_bytes'] is False
    dedup_counts = dedup_dataset.effective_train_byte_counts()
    assert int(dedup_counts.sum()) == 28
    assert int(dedup_counts[ord('B')]) == 0
    raw_ngram = NgramBaseline(dedup_dataset.train, max_order=2)
    cleaned_ngram = NgramBaseline(dedup_dataset.train, max_order=2, exclude_spans=dedup_dataset.train_excluded_spans)
    assert raw_ngram.train_total == 48
    assert cleaned_ngram.train_total == 28
    sealed_dedup_groups = [b'A' * 10, b'B' * 10, b'C' * 10, b'V' * 10, b'A' * 10]
    sealed_dedup_payload = separator.join(sealed_dedup_groups) + separator
    with tempfile.NamedTemporaryFile() as sealed_dedup_file:
        sealed_dedup_file.write(sealed_dedup_payload)
        sealed_dedup_file.flush()
        sealed_dedup_dataset = ByteDataset(sealed_dedup_file.name, seq_len=4, val_fraction=0.2, test_fraction=0.2, group_separator=separator, deduplicate_train_groups=True)
        leaky_dedup_dataset = ByteDataset(sealed_dedup_file.name, seq_len=4, val_fraction=0.2, test_fraction=0.2, group_separator=separator, deduplicate_train_groups=True, clean_train_against_test=True)
        sealed_dedup_audit = audit_data_splits(sealed_dedup_dataset, include_test=True)
        leaky_dedup_audit = audit_data_splits(leaky_dedup_dataset, include_test=True)
    assert sealed_dedup_dataset.split_bounds == {'train': (0, 36), 'val': (36, 48), 'test': (48, 60)}
    assert sealed_dedup_dataset.train_excluded_spans == []
    assert sealed_dedup_dataset.train_cleaning['test_used_for_cleaning'] is False
    assert sealed_dedup_dataset.train_cleaning['cleaning_eval_splits'] == ['val']
    assert sealed_dedup_dataset.effective_train_bytes == 36
    assert sealed_dedup_dataset.effective_train_sha256 != leaky_dedup_dataset.effective_train_sha256
    assert sealed_dedup_audit['leakage_violations']
    assert not enforce_data_leakage_gate(sealed_dedup_audit)
    assert leaky_dedup_dataset.train_excluded_spans == [(0, 10)]
    assert leaky_dedup_dataset.train_cleaning['test_used_for_cleaning'] is True
    assert leaky_dedup_dataset.train_cleaning['cleaning_eval_splits'] == ['test', 'val']
    assert leaky_dedup_dataset.train_cleaning['dropped_duplicate_of']['test'] == 1
    assert leaky_dedup_dataset.effective_train_bytes == 26
    assert leaky_dedup_audit['leakage_violations'] == []

def _selfcheck_top_53881_0025():
    global key, ngram_auc, ngram_auc_available, ngram_auc_cap_before, ngram_batched_sink, ngram_boot, ngram_boot_repeat, ngram_boot_windows, ngram_capped_auc, ngram_window_sink, sink_correct, sink_loss, sink_tokens, value
    ngram_boot_windows = [torch.tensor([[65, 66, 65, 66, 65, 66, 65, 66, 65 + row % 2]], dtype=torch.uint8) for row in range(8)]
    ngram_window_sink = []
    ngram_boot = _ngram_extended_metrics(cleaned_ngram, ngram_boot_windows, alpha=1.0, order=2, split='val', include_auc=False, bootstrap_replicates=128, bootstrap_seed=2024, window_sink=ngram_window_sink)
    assert len(ngram_window_sink) == len(ngram_boot_windows)
    assert int(ngram_boot['val_bootstrap_windows']) == len(ngram_boot_windows)
    assert int(ngram_boot['val_bootstrap_replicates']) == 128
    assert float(ngram_boot['val_loss_ci95_low']) <= float(ngram_boot['val_loss']) <= float(ngram_boot['val_loss_ci95_high'])
    assert float(ngram_boot['val_macro_f1_ci95_low']) <= float(ngram_boot['val_macro_f1_ci95_high'])
    sink_tokens = sum((item['tokens'] for item in ngram_window_sink))
    sink_loss = sum((item['loss_sum'] for item in ngram_window_sink))
    sink_correct = sum((item['correct'] for item in ngram_window_sink))
    assert abs(sink_loss / sink_tokens - float(ngram_boot['val_loss'])) < 1e-12
    assert abs(sink_correct / sink_tokens - float(ngram_boot['val_token_accuracy'])) < 1e-12
    ngram_boot_repeat = _ngram_extended_metrics(cleaned_ngram, ngram_boot_windows, alpha=1.0, order=2, split='val', include_auc=False, bootstrap_replicates=128, bootstrap_seed=2024)
    for key, value in ngram_boot.items():
        assert ngram_boot_repeat[key] == value
    ngram_auc = _ngram_extended_metrics(cleaned_ngram, ngram_boot_windows, alpha=1.0, order=2, split='val', include_auc=True, bootstrap_replicates=0)
    ngram_auc_available = sum((int(window.size(0)) * (int(window.size(1)) - 1) for window in ngram_boot_windows))
    assert int(ngram_auc['val_auc_tokens_available']) == ngram_auc_available
    assert int(ngram_auc['val_auc_tokens_evaluated']) == ngram_auc_available
    assert float(ngram_auc['val_auc_coverage_fraction']) == 1.0
    assert ngram_auc['val_auc_sampling'] == 'full'
    ngram_auc_cap_before = _SC_BASE['_AUC_MAX_TOKENS']
    _SC_BASE['_AUC_MAX_TOKENS'] = 8
    try:
        ngram_capped_auc = _ngram_extended_metrics(cleaned_ngram, ngram_boot_windows, alpha=1.0, order=2, split='val', include_auc=True, bootstrap_replicates=0)
    finally:
        _SC_BASE['_AUC_MAX_TOKENS'] = ngram_auc_cap_before
    assert ngram_capped_auc['val_auc_sampling'] == 'uniform_stride'
    assert int(ngram_capped_auc['val_auc_tokens_evaluated']) == 8
    assert int(ngram_capped_auc['val_auc_tokens_available']) > 8
    assert 0.0 < float(ngram_capped_auc['val_auc_coverage_fraction']) < 1.0
    ngram_batched_sink = []

def _selfcheck_top_54012_0026():
    global batched_loss, batched_tokens, boundary_best, boundary_tuning, ngram_batched, one_se_base, one_se_candidates, one_se_diagnostics, one_se_n, one_se_near, one_se_selection, one_se_windows, one_se_worse, point_best, tie_best, tie_runner, tie_tuning, tied_keys
    ngram_batched = _ngram_extended_metrics(cleaned_ngram, [torch.cat(ngram_boot_windows[:2], dim=0)], alpha=1.0, order=2, split='val', include_auc=False, bootstrap_replicates=64, bootstrap_seed=2024, window_sink=ngram_batched_sink)
    assert len(ngram_batched_sink) == 2, ngram_batched_sink
    assert int(ngram_batched['val_bootstrap_windows']) == 2
    batched_tokens = sum((item['tokens'] for item in ngram_batched_sink))
    batched_loss = sum((item['loss_sum'] for item in ngram_batched_sink))
    assert abs(batched_loss / batched_tokens - float(ngram_batched['val_loss'])) < 1e-12
    tie_best = {'order': 2, 'alpha': 1.0, 'val_loss': 2.5}
    tie_runner = {'order': 1, 'alpha': 0.25, 'val_loss': 2.5}
    tie_tuning = _ngram_tuning_diagnostics(tie_best, tie_runner, 'val_loss', [0.25, 1.0], 3, candidate_count=6, valid_candidate_count=6)
    assert tie_tuning['metric_gap'] == 0.0
    assert any(('并列' in advisory for advisory in tie_tuning['advisories']))
    boundary_best = {'order': 3, 'alpha': 1.0, 'val_loss': 2.0}
    boundary_tuning = _ngram_tuning_diagnostics(boundary_best, None, 'val_loss', [0.25, 1.0], 3, candidate_count=6, valid_candidate_count=6)
    assert 'order=网格上界' in boundary_tuning['boundary_hits']
    assert 'alpha=网格上界' in boundary_tuning['boundary_hits']
    assert boundary_tuning['metric_gap'] is None
    assert boundary_tuning['advisories']
    one_se_n = 60
    one_se_base = [2.0 + 0.001 * (index * 7 % 13) for index in range(one_se_n)]
    one_se_near = [value + 0.0001 + 0.001 * (index % 3 - 1) for index, value in enumerate(one_se_base)]
    one_se_worse = [value + 0.01 + 1e-05 * (index % 2 - 0.5) for index, value in enumerate(one_se_base)]
    one_se_windows = {(3, 1.0): one_se_base, (1, 0.25): one_se_near, (2, 1.0): one_se_worse}
    one_se_candidates = [{'order': 1, 'alpha': 0.25, 'val_loss': sum(one_se_near) / one_se_n}, {'order': 2, 'alpha': 1.0, 'val_loss': sum(one_se_worse) / one_se_n}, {'order': 3, 'alpha': 1.0, 'val_loss': sum(one_se_base) / one_se_n}]
    point_best = one_se_candidates[2]
    one_se_selection = _select_ngram_candidate_one_se(point_best, one_se_candidates, one_se_windows, 'val_loss')
    assert one_se_selection is not None
    assert one_se_selection['available'] is True
    assert one_se_selection['selected_is_point_best'] is False
    assert one_se_selection['selected'] == {'order': 1, 'alpha': 0.25}
    tied_keys = {(row['order'], row['alpha']) for row in one_se_selection['tied']}
    assert (1, 0.25) in tied_keys
    assert (3, 1.0) in tied_keys
    assert (2, 1.0) not in tied_keys
    one_se_diagnostics = _ngram_tuning_diagnostics(one_se_candidates[0], point_best, 'val_loss', [0.25, 1.0], 3, candidate_count=3, valid_candidate_count=3, selection=one_se_selection)
    assert one_se_diagnostics['selection'] == one_se_selection
    assert any(('one-SE' in advisory for advisory in one_se_diagnostics['advisories']))

def _selfcheck_top_54147_0027():
    global _shared_random, dedup_audit, dedup_probe, dedup_reference, f1_base, f1_candidates, f1_near, f1_selection, f1_tied_keys, f1_windows, f1_worse, shared_alphabet, shared_fragment, shared_groups, shared_payload, shared_rng
    f1_base = [0.8 + 0.01 * (index % 3 - 1) for index in range(one_se_n)]
    f1_near = [value - 0.001 + 0.02 * (index % 2 - 0.5) for index, value in enumerate(f1_base)]
    f1_worse = [value - 0.2 + 0.01 * (index % 2 - 0.5) for index, value in enumerate(f1_base)]
    f1_windows = {(3, 1.0): f1_base, (1, 0.25): f1_near, (2, 1.0): f1_worse}
    f1_candidates = [{'order': 1, 'alpha': 0.25, 'val_macro_f1': sum(f1_near) / one_se_n}, {'order': 2, 'alpha': 1.0, 'val_macro_f1': sum(f1_worse) / one_se_n}, {'order': 3, 'alpha': 1.0, 'val_macro_f1': sum(f1_base) / one_se_n}]
    f1_selection = _select_ngram_candidate_one_se(f1_candidates[2], f1_candidates, f1_windows, 'val_macro_f1')
    assert f1_selection is not None
    assert f1_selection['available'] is True
    assert f1_selection['selected_is_point_best'] is False
    assert f1_selection['selected'] == {'order': 1, 'alpha': 0.25}
    f1_tied_keys = {(row['order'], row['alpha']) for row in f1_selection['tied']}
    assert (1, 0.25) in f1_tied_keys
    assert (3, 1.0) in f1_tied_keys
    assert (2, 1.0) not in f1_tied_keys
    assert _select_ngram_candidate_one_se(point_best, one_se_candidates, one_se_windows, 'val_pr_auc_macro') is None
    dedup_reference = compute_reference_baselines(dedup_dataset, batch_size=2, batches=1, device=eval_device)
    assert dedup_reference['unigram_train_bytes'] == 28
    assert dedup_reference['unigram_train_bytes_raw'] == 48
    assert dedup_reference['unigram_train_excluded_bytes'] == 20
    dedup_audit = audit_data_splits(dedup_dataset, include_test=True)
    assert dedup_audit['splits']['train']['effective_bytes'] == 28
    assert dedup_audit['splits']['train']['bytes'] == 48
    assert all((info['shared_chunks'] == 0 for info in dedup_audit['content_chunks']['cross_split'].values()))
    dedup_probe = probe_data_pipeline(dedup_dataset, eval_device, batch_size=4, batches=1)
    assert dedup_probe['ok'], dedup_probe['failures']
    assert dedup_probe['train_cleaning_checked']
    assert dedup_probe['group_boundaries_checked']
    shared_rng = random.Random(20241023)
    shared_alphabet = b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    def _shared_random(length: int) -> bytes:
        return bytes((shared_rng.choice(shared_alphabet) for _ in range(length)))
    shared_fragment = _shared_random(512)
    shared_groups = [_shared_random(1024), _shared_random(256) + shared_fragment + _shared_random(256), _shared_random(1024), _shared_random(1024), shared_fragment + _shared_random(512), _shared_random(1024)]
    shared_payload = separator.join(shared_groups) + separator

def _selfcheck_top_54278_0028():
    global _near_random, cleaned_shared_audit, cleaned_shared_dataset, near_base, near_core, near_groups, near_payload, near_variant, raw_shared_audit, raw_shared_dataset, shared_file, shared_probe, split_name
    with tempfile.NamedTemporaryFile() as shared_file:
        shared_file.write(shared_payload)
        shared_file.flush()
        raw_shared_dataset = ByteDataset(shared_file.name, seq_len=8, val_fraction=1.0 / 6.0, test_fraction=1.0 / 6.0, group_separator=separator)
        cleaned_shared_dataset = ByteDataset(shared_file.name, seq_len=8, val_fraction=1.0 / 6.0, test_fraction=1.0 / 6.0, group_separator=separator, drop_train_shared_chunks=True)
        raw_shared_audit = audit_data_splits(raw_shared_dataset, include_test=True)
        cleaned_shared_audit = audit_data_splits(cleaned_shared_dataset, include_test=True)
    assert raw_shared_audit['content_chunks']['cross_split']['train_val']['shared_chunks'] > 0
    assert raw_shared_audit['leakage_violations']
    assert not enforce_data_leakage_gate(raw_shared_audit)
    assert cleaned_shared_dataset.train_excluded_spans == [(1026, 2050)]
    assert cleaned_shared_dataset.train_cleaning['cleaning_kinds'] == ['shared_chunks']
    assert cleaned_shared_dataset.train_cleaning['shared_chunks']['groups_dropped'] == 1
    for split_name in ('val', 'test'):
        assert cleaned_shared_dataset.split_sha256[split_name] == raw_shared_dataset.split_sha256[split_name]
    assert cleaned_shared_audit['content_chunks']['cross_split']['train_val']['shared_chunks'] == 0
    assert cleaned_shared_audit['leakage_violations'] == []
    assert enforce_data_leakage_gate(cleaned_shared_audit)
    shared_probe = probe_data_pipeline(cleaned_shared_dataset, eval_device, batch_size=4, batches=1)
    assert shared_probe['ok'], shared_probe['failures']
    def _near_random(length: int, seed: int) -> bytes:
        generator = random.Random(seed)
        return bytes((generator.randrange(32, 127) for _ in range(length)))
    near_core = (b'alpha beta gamma delta epsilon zeta eta theta ' * 12)[:576]
    near_base = near_core + _near_random(640 - len(near_core), 20241101)
    near_variant = near_base.replace(b'alpha', b'omega', 1)
    assert len(near_base) == len(near_variant) == 640
    near_groups = [_near_random(640, 1), near_base, _near_random(640, 2), _near_random(640, 3), near_variant, _near_random(640, 4), _near_random(640, 5), _near_random(640, 6)]
    near_payload = separator.join(near_groups) + separator

def _selfcheck_top_54381_0029():
    global _run_random, audited_near_dataset, audited_near_report, cleaned_near_dataset, near_file, near_probe, near_stats, raw_near_dataset, run_payload, run_rng, run_symbols, split_name
    with tempfile.NamedTemporaryFile() as near_file:
        near_file.write(near_payload)
        near_file.flush()
        raw_near_dataset = ByteDataset(near_file.name, seq_len=8, val_fraction=0.25, test_fraction=0.25, group_separator=separator)
        cleaned_near_dataset = ByteDataset(near_file.name, seq_len=8, val_fraction=0.25, test_fraction=0.25, group_separator=separator, drop_train_near_duplicate_groups=True, near_duplicate_threshold=0.8, near_duplicate_shingle_bytes=8)
        audited_near_dataset = ByteDataset(near_file.name, seq_len=8, val_fraction=0.25, test_fraction=0.25, group_separator=separator, near_duplicate_audit=True, near_duplicate_threshold=0.8, near_duplicate_shingle_bytes=8)
        audited_near_report = audit_data_splits(audited_near_dataset)
    assert cleaned_near_dataset.split_bounds == raw_near_dataset.split_bounds
    for split_name in ('val', 'test'):
        assert cleaned_near_dataset.split_sha256[split_name] == raw_near_dataset.split_sha256[split_name]
    assert cleaned_near_dataset.train_excluded_spans == [(642, 1282)]
    assert cleaned_near_dataset.train_cleaning['cleaning_kinds'] == ['near_duplicate']
    near_stats = cleaned_near_dataset.train_cleaning['near_duplicates']
    assert near_stats['flagged_duplicate_of'] == {'val': 1, 'test': 0, 'train': 0}
    assert near_stats['groups_dropped'] == 1
    assert near_stats['examples'][0][3] >= 0.8
    assert cleaned_near_dataset.effective_train_bytes == raw_near_dataset.effective_train_bytes - 640
    assert cleaned_near_dataset.train_num_valid_starts is not None
    near_probe = probe_data_pipeline(cleaned_near_dataset, eval_device, batch_size=4, batches=1)
    assert near_probe['ok'], near_probe['failures']
    assert audited_near_dataset.train_excluded_spans == []
    assert audited_near_report['near_duplicates']['groups_flagged'] == 1
    assert audited_near_report['near_duplicates']['groups_dropped'] == 0
    assert audited_near_report['near_duplicates']['test_content_used'] is False
    assert any(('近重复' in warning for warning in audited_near_report['warnings']))
    run_rng = random.Random(20241024)
    run_symbols = b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXY'
    def _run_random(length: int) -> bytes:
        return bytes((run_rng.choice(run_symbols) for _ in range(length)))
    run_payload = _run_random(64) + b'\x00' * 200 + _run_random(640) + b'Z' * 96 + _run_random(1400)
    assert len(run_payload) == 2400

def _selfcheck_top_54493_0030():
    global _, cleaned_run_audit, cleaned_run_dataset, cleaned_run_ngram, cleaned_run_probe, cleaned_run_values, cleaned_run_x, corrupt_groups, corrupt_payload, raw_run_audit, raw_run_dataset, run_file, split_name
    with tempfile.NamedTemporaryFile() as run_file:
        run_file.write(run_payload)
        run_file.flush()
        raw_run_dataset = ByteDataset(run_file.name, seq_len=8, val_fraction=0.1, test_fraction=0.1)
        cleaned_run_dataset = ByteDataset(run_file.name, seq_len=8, val_fraction=0.1, test_fraction=0.1, drop_train_degenerate_runs=True)
        raw_run_audit = audit_data_splits(raw_run_dataset, include_test=True)
        cleaned_run_audit = audit_data_splits(cleaned_run_dataset, include_test=True)
        try:
            ByteDataset(run_file.name, seq_len=8, val_fraction=0.1, test_fraction=0.1, group_separator=separator, drop_train_degenerate_runs=True)
        except ValueError:
            pass
        else:
            raise AssertionError('drop_train_degenerate_runs 必须拒绝群组模式：否则会改动群组边界内的字节')
    assert raw_run_dataset.split_bounds == {'train': (0, 1920), 'val': (1920, 2160), 'test': (2160, 2400)}
    assert raw_run_audit['degenerate_runs']['runs'] == 2
    assert raw_run_audit['degenerate_runs']['degenerate_bytes'] == 296
    assert any(('退化连续段' in warning for warning in raw_run_audit['warnings']))
    assert cleaned_run_dataset.split_bounds == raw_run_dataset.split_bounds
    assert cleaned_run_dataset.train_excluded_spans == [(64, 264), (904, 1000)]
    assert cleaned_run_dataset.train_cleaning['cleaning_kinds'] == ['degenerate_runs']
    assert cleaned_run_dataset.train_cleaning['degenerate_runs']['runs_dropped'] == 2
    assert cleaned_run_dataset.train_cleaning['degenerate_runs']['dropped_bytes'] == 296
    assert cleaned_run_dataset.effective_train_bytes == 1624
    for split_name in ('val', 'test'):
        assert cleaned_run_dataset.split_sha256[split_name] == raw_run_dataset.split_sha256[split_name]
    assert cleaned_run_audit['degenerate_runs']['runs'] == 0
    assert enforce_strict_data_audit(cleaned_run_audit)
    assert cleaned_run_audit['leakage_violations'] == []
    cleaned_run_x, _ = cleaned_run_dataset.batch('train', 64, torch.Generator().manual_seed(17), eval_device)
    cleaned_run_values = set(cleaned_run_x.flatten().tolist())
    assert 0 not in cleaned_run_values
    assert 90 not in cleaned_run_values
    cleaned_run_ngram = NgramBaseline(cleaned_run_dataset.train, max_order=2, exclude_spans=cleaned_run_dataset.train_excluded_spans)
    assert cleaned_run_ngram.train_total == 1624
    cleaned_run_probe = probe_data_pipeline(cleaned_run_dataset, eval_device, batch_size=4, batches=1)
    assert cleaned_run_probe['ok'], cleaned_run_probe['failures']
    assert grouped_audit['degenerate_runs'] is None
    corrupt_groups = [b'A' * 10, b'B' * 10, b'\xff' + b'C' * 9, b'D' * 10, b'E' * 10, b'F' * 10]
    corrupt_payload = separator.join(corrupt_groups) + separator

def _selfcheck_top_54618_0031():
    global _, byte_body, byte_payload, byte_rng, cleaned_corrupt_audit, cleaned_corrupt_counts, cleaned_corrupt_dataset, cleaned_corrupt_probe, cleaned_corrupt_x, cleaned_ngram, cleaned_reference, corrupt_clean_file, raw_corrupt_audit, raw_corrupt_dataset
    with tempfile.NamedTemporaryFile() as corrupt_clean_file:
        corrupt_clean_file.write(corrupt_payload)
        corrupt_clean_file.flush()
        raw_corrupt_dataset = ByteDataset(corrupt_clean_file.name, seq_len=8, val_fraction=1.0 / 6.0, test_fraction=1.0 / 6.0, group_separator=separator)
        cleaned_corrupt_dataset = ByteDataset(corrupt_clean_file.name, seq_len=8, val_fraction=1.0 / 6.0, test_fraction=1.0 / 6.0, group_separator=separator, deduplicate_train_groups=True, drop_train_corrupt_groups=True)
        raw_corrupt_audit = audit_data_splits(raw_corrupt_dataset, include_test=True)
        cleaned_corrupt_audit = audit_data_splits(cleaned_corrupt_dataset, include_test=True)
    assert raw_corrupt_audit['splits']['train']['utf8_replacements'] > 0
    assert not enforce_strict_data_audit(raw_corrupt_audit)
    assert cleaned_corrupt_dataset.split_bounds == raw_corrupt_dataset.split_bounds
    assert cleaned_corrupt_dataset.train_excluded_spans == [(24, 34)]
    assert cleaned_corrupt_dataset.train_cleaning['groups_dropped'] == 1
    assert cleaned_corrupt_dataset.train_cleaning['dropped_duplicate_of'] == {'val': 0, 'test': 0, 'train': 0}
    assert cleaned_corrupt_dataset.train_cleaning['corruption']['groups_dropped'] == 1
    assert cleaned_corrupt_dataset.train_cleaning['corruption']['groups_with_invalid_utf8'] == 1
    assert cleaned_corrupt_dataset.effective_train_bytes == 38
    assert cleaned_corrupt_dataset.train_num_valid_starts == 22
    cleaned_corrupt_x, _ = cleaned_corrupt_dataset.batch('train', 32, torch.Generator().manual_seed(11), eval_device)
    assert 255 not in set(cleaned_corrupt_x.flatten().tolist())
    cleaned_corrupt_counts = cleaned_corrupt_dataset.effective_train_byte_counts()
    assert int(cleaned_corrupt_counts.sum()) == 38
    assert int(cleaned_corrupt_counts[ord('C')]) == 0
    cleaned_ngram = NgramBaseline(cleaned_corrupt_dataset.train, max_order=2, exclude_spans=cleaned_corrupt_dataset.train_excluded_spans)
    assert cleaned_ngram.train_total == 38
    cleaned_reference = compute_reference_baselines(cleaned_corrupt_dataset, batch_size=2, batches=1, device=eval_device)
    assert cleaned_reference['unigram_train_bytes'] == 38
    assert cleaned_corrupt_audit['splits']['train']['utf8_replacements'] == 0
    assert enforce_strict_data_audit(cleaned_corrupt_audit)
    assert cleaned_corrupt_audit['leakage_violations'] == []
    cleaned_corrupt_probe = probe_data_pipeline(cleaned_corrupt_dataset, eval_device, batch_size=4, batches=1)
    assert cleaned_corrupt_probe['ok'], cleaned_corrupt_probe['failures']
    assert cleaned_corrupt_probe['train_cleaning_checked']
    byte_rng = random.Random(20241022)
    byte_body = bytes((byte_rng.randrange(32, 127) for _ in range(1200)))
    byte_payload = byte_body[:101] + b'\xff' + byte_body[102:200] + b'\x00' + byte_body[201:]
    assert len(byte_payload) == 1200

def _selfcheck_top_54737_0032():
    global _, byte_file, cleaned_byte_audit, cleaned_byte_audit_train, cleaned_byte_dataset, cleaned_byte_ngram, cleaned_byte_probe, cleaned_byte_reference, cleaned_byte_values, cleaned_byte_x, degenerate_groups, degenerate_payload, raw_byte_audit, raw_byte_dataset
    with tempfile.NamedTemporaryFile() as byte_file:
        byte_file.write(byte_payload)
        byte_file.flush()
        raw_byte_dataset = ByteDataset(byte_file.name, seq_len=8, val_fraction=0.2, test_fraction=0.2)
        cleaned_byte_dataset = ByteDataset(byte_file.name, seq_len=8, val_fraction=0.2, test_fraction=0.2, drop_train_corrupt_bytes=True)
        raw_byte_audit = audit_data_splits(raw_byte_dataset, include_test=True)
        cleaned_byte_audit = audit_data_splits(cleaned_byte_dataset, include_test=True)
    assert raw_byte_audit['splits']['train']['utf8_replacements'] == 1
    assert raw_byte_audit['splits']['train']['nul_bytes'] == 1
    assert not enforce_strict_data_audit(raw_byte_audit)
    assert cleaned_byte_dataset.split_bounds == raw_byte_dataset.split_bounds
    assert cleaned_byte_dataset.split_sha256 == raw_byte_dataset.split_sha256
    assert cleaned_byte_dataset.train_excluded_spans == [(101, 102), (200, 201)]
    assert cleaned_byte_dataset.train_cleaning['cleaning_kinds'] == ['corrupt_bytes']
    assert cleaned_byte_dataset.train_cleaning['corruption_bytes']['invalid_utf8_bytes'] == 1
    assert cleaned_byte_dataset.train_cleaning['corruption_bytes']['nul_bytes'] == 1
    assert cleaned_byte_dataset.effective_train_bytes == 718
    assert cleaned_byte_dataset.train_num_valid_starts == 694
    cleaned_byte_audit_train = cleaned_byte_audit['splits']['train']
    assert cleaned_byte_audit_train['utf8_replacements'] == 0
    assert cleaned_byte_audit_train['nul_bytes'] == 0
    assert cleaned_byte_audit_train['effective_bytes'] == 718
    assert cleaned_byte_audit['train_cleaning']['cleaning_kinds'] == ['corrupt_bytes']
    assert enforce_strict_data_audit(cleaned_byte_audit)
    assert cleaned_byte_audit['leakage_violations'] == []
    cleaned_byte_x, _ = cleaned_byte_dataset.batch('train', 64, torch.Generator().manual_seed(12), eval_device)
    cleaned_byte_values = set(cleaned_byte_x.flatten().tolist())
    assert 255 not in cleaned_byte_values
    assert 0 not in cleaned_byte_values
    cleaned_byte_ngram = NgramBaseline(cleaned_byte_dataset.train, max_order=2, exclude_spans=cleaned_byte_dataset.train_excluded_spans)
    assert cleaned_byte_ngram.train_total == 718
    cleaned_byte_reference = compute_reference_baselines(cleaned_byte_dataset, batch_size=2, batches=1, device=eval_device)
    assert cleaned_byte_reference['unigram_train_bytes'] == 718
    cleaned_byte_probe = probe_data_pipeline(cleaned_byte_dataset, eval_device, batch_size=4, batches=1)
    assert cleaned_byte_probe['ok'], cleaned_byte_probe['failures']
    assert cleaned_byte_probe['train_cleaning_checked']
    degenerate_groups = [b'A' * 20, b'B' * 20, b'Z' * 100, b'C' * 20, b'D' * 20, b'E' * 20]
    degenerate_payload = separator.join(degenerate_groups) + separator

def _selfcheck_top_54856_0033():
    global _, aligned_audit, aligned_dataset, aligned_file, aligned_payload, cleaned_degenerate_audit, cleaned_degenerate_dataset, cleaned_degenerate_ngram, cleaned_degenerate_probe, cleaned_degenerate_x, degenerate_file, groups_five, health, raw_degenerate_audit, raw_degenerate_dataset, separator_five
    with tempfile.NamedTemporaryFile() as degenerate_file:
        degenerate_file.write(degenerate_payload)
        degenerate_file.flush()
        raw_degenerate_dataset = ByteDataset(degenerate_file.name, seq_len=8, val_fraction=0.1, test_fraction=0.1, group_separator=separator)
        cleaned_degenerate_dataset = ByteDataset(degenerate_file.name, seq_len=8, val_fraction=0.1, test_fraction=0.1, group_separator=separator, drop_train_degenerate_groups=True)
        raw_degenerate_audit = audit_data_splits(raw_degenerate_dataset, include_test=True)
        cleaned_degenerate_audit = audit_data_splits(cleaned_degenerate_dataset, include_test=True)
    assert raw_degenerate_dataset.split_bounds == {'train': (0, 168), 'val': (168, 190), 'test': (190, 212)}
    health = raw_degenerate_audit['groups']['content_health']
    assert health['long_groups_checked'] == 1
    assert health['degenerate_groups'] == 1
    assert health['max_top_byte_share'] == 1.0
    assert any(('退化群组' in warning for warning in raw_degenerate_audit['warnings']))
    assert cleaned_degenerate_dataset.split_bounds == raw_degenerate_dataset.split_bounds
    assert cleaned_degenerate_dataset.train_excluded_spans == [(44, 144)]
    assert cleaned_degenerate_dataset.train_cleaning['degenerate']['groups_dropped'] == 1
    assert cleaned_degenerate_dataset.train_cleaning['groups_dropped'] == 1
    assert cleaned_degenerate_dataset.effective_train_bytes == 68
    assert cleaned_degenerate_audit['groups']['content_health']['degenerate_groups'] == 0
    assert enforce_strict_data_audit(cleaned_degenerate_audit)
    assert cleaned_degenerate_audit['leakage_violations'] == []
    cleaned_degenerate_x, _ = cleaned_degenerate_dataset.batch('train', 32, torch.Generator().manual_seed(13), eval_device)
    assert ord('Z') not in set(cleaned_degenerate_x.flatten().tolist())
    cleaned_degenerate_ngram = NgramBaseline(cleaned_degenerate_dataset.train, max_order=2, exclude_spans=cleaned_degenerate_dataset.train_excluded_spans)
    assert cleaned_degenerate_ngram.train_total == 68
    cleaned_degenerate_probe = probe_data_pipeline(cleaned_degenerate_dataset, eval_device, batch_size=4, batches=1)
    assert cleaned_degenerate_probe['ok'], cleaned_degenerate_probe['failures']
    separator_five = b'<END>'
    groups_five = [bytes([65 + index * 5]) * 15 for index in range(4)]
    aligned_payload = separator_five.join(groups_five) + separator_five
    assert len(aligned_payload) == 80
    with tempfile.NamedTemporaryFile() as aligned_file:
        aligned_file.write(aligned_payload)
        aligned_file.flush()
        aligned_dataset = ByteDataset(aligned_file.name, seq_len=8, val_fraction=0.2, test_fraction=0.2, group_separator=separator_five)
    assert aligned_dataset.split_bounds['train'] == (0, 40)
    assert aligned_dataset.split_bounds['val'] == (40, 60)
    assert aligned_dataset.split_bounds['test'] == (60, 80)
    assert aligned_dataset.groups_per_split == {'train': 2, 'val': 1, 'test': 1}
    assert aligned_dataset.split_fractions == {'train': 0.5, 'val': 0.25, 'test': 0.25}
    aligned_audit = audit_data_splits(aligned_dataset, include_test=True)

def _selfcheck_top_54994_0034():
    global metrics_dir, metrics_file, no_sep_file, oversized_file, seal_dir, seal_root, sealed_aligned_audit
    assert aligned_audit['split_fractions'] == {'train': 0.5, 'val': 0.25, 'test': 0.25}
    assert any(('偏离名义值' in warning for warning in aligned_audit['warnings']))
    sealed_aligned_audit = audit_data_splits(aligned_dataset, include_test=False)
    assert 'test' not in sealed_aligned_audit['splits']
    assert sealed_aligned_audit['dataset_split_fractions']['test'] == 0.25
    assert any(('test: 实际占比' in warning for warning in sealed_aligned_audit['warnings']))
    with tempfile.NamedTemporaryFile() as oversized_file:
        oversized_file.write(b'x' * 100 + b'SEP' + b'y' * 5)
        oversized_file.flush()
        try:
            ByteDataset(oversized_file.name, seq_len=8, group_separator=b'SEP')
        except ValueError:
            pass
        else:
            raise AssertionError('群组无法整体划分时必须报错')
    with tempfile.NamedTemporaryFile() as no_sep_file:
        no_sep_file.write(b'z' * 120)
        no_sep_file.flush()
        try:
            ByteDataset(no_sep_file.name, seq_len=8, group_separator=b'<MISSING>')
        except ValueError:
            pass
        else:
            raise AssertionError('分隔符不存在时必须报错')
    with tempfile.TemporaryDirectory() as metrics_dir:
        metrics_file = Path(metrics_dir) / 'metrics.jsonl'
        assert not test_record_exists(metrics_file)
        write_jsonl_record(metrics_file, {'step': 'final', 'val_loss': 1.0})
        assert not test_record_exists(metrics_file)
        write_jsonl_record(metrics_file, {'step': 'test', 'test_loss': 1.0})
        assert test_record_exists(metrics_file)
    with tempfile.TemporaryDirectory() as seal_dir:
        seal_root = Path(seal_dir)
        assert not test_evaluation_sealed(seal_root)
        write_test_evaluation_seal(seal_root, {'dataset_sha256': 'abc'})
        assert test_evaluation_sealed(seal_root)
        assert _test_seal_path(seal_root).exists()
        write_jsonl_record(seal_root / 'metrics.jsonl', {'step': 'final', 'val_loss': 1.0})
        (seal_root / 'metrics.jsonl').unlink()
        assert test_evaluation_sealed(seal_root)

def _selfcheck_top_55096_0035():
    global base_lr, corrupt_selected_path, expected_lr, expected_state, group, new_optimizer, new_param, old_optimizer, old_param, old_scheduler, parameter, restored_state, resumed_scheduler, scheduler_state, selected_args, selected_dir, selected_path, selected_payload, selection_model
    with tempfile.TemporaryDirectory() as selected_dir:
        selected_path = Path(selected_dir) / 'best.pt'
        selection_model = LanguageModel(Config(**common, mode='B'))
        with torch.no_grad():
            for parameter in selection_model.parameters():
                parameter.add_(0.25)
        expected_state = {key: value.clone() for key, value in selection_model.state_dict().items()}
        selected_args = argparse.Namespace(seed=1, mode='B', selection_metric='val_macro_f1')
        save_selected_model(str(selected_path), selection_model, 7, 1.25, selected_args, selection_metric='val_macro_f1', selection_value=0.5)
        assert selected_path.exists()
        assert not selected_path.with_name(selected_path.name + '.tmp').exists()
        with torch.no_grad():
            for parameter in selection_model.parameters():
                parameter.zero_()
        selected_payload = load_selected_model(selected_path, torch.device('cpu'))
        assert selected_payload is not None
        assert int(selected_payload['step']) == 7
        assert abs(float(selected_payload['val_loss']) - 1.25) < 1e-12
        assert selected_payload['selection_metric'] == 'val_macro_f1'
        assert abs(float(selected_payload['selection_value']) - 0.5) < 1e-12
        restore_selected_weights(selection_model, selected_payload)
        restored_state = selection_model.state_dict()
        assert all((torch.equal(expected_state[key], restored_state[key]) for key in expected_state))
        assert load_selected_model(Path(selected_dir) / 'missing.pt', torch.device('cpu')) is None
        corrupt_selected_path = Path(selected_dir) / 'corrupt.pt'
        corrupt_selected_path.write_bytes(b'\x00\x01not a torch archive')
        assert load_selected_model(corrupt_selected_path, torch.device('cpu')) is None
    old_param = nn.Parameter(torch.randn(2))
    old_optimizer = torch.optim.AdamW([old_param], lr=0.1)
    old_scheduler = build_lr_schedule(old_optimizer, warmup_steps=0, total_steps=8)
    old_optimizer.step()
    old_scheduler.step()
    old_optimizer.step()
    old_scheduler.step()
    expected_lr = old_scheduler.get_last_lr()[0]
    scheduler_state = old_scheduler.state_dict()
    new_param = nn.Parameter(torch.randn(2))
    new_optimizer = torch.optim.AdamW([new_param], lr=999.0)
    new_optimizer.load_state_dict(old_optimizer.state_dict())
    for group, base_lr in zip(new_optimizer.param_groups, scheduler_state['base_lrs']):
        group['lr'] = base_lr
    resumed_scheduler = build_lr_schedule(new_optimizer, warmup_steps=0, total_steps=8, start_step=2)
    resumed_scheduler.load_state_dict(scheduler_state)

def _selfcheck_top_55229_0036():
    global _, effective_change_error, isolation_a, legacy_optimizer, legacy_param, legacy_scheduler, original, original_lrs, rebuilt, rebuilt_lrs, rebuilt_optimizer, rebuilt_param, split_change_error, warmup_optimizer, warmup_param, wrong, wrong_lrs, wrong_optimizer, wrong_param
    torch.testing.assert_close(new_optimizer.param_groups[0]['lr'], expected_lr)
    legacy_param = nn.Parameter(torch.randn(2))
    legacy_optimizer = torch.optim.AdamW([legacy_param], lr=0.1)
    legacy_optimizer.param_groups[0].pop('initial_lr', None)
    legacy_scheduler = build_lr_schedule(legacy_optimizer, warmup_steps=0, total_steps=8, start_step=2)
    assert legacy_scheduler.get_last_lr()[0] > 0.0, 'legacy resume must rebuild the LR schedule'
    warmup_param = nn.Parameter(torch.randn(2))
    warmup_optimizer = torch.optim.AdamW([warmup_param], lr=0.1)
    original = build_lr_schedule(warmup_optimizer, warmup_steps=3, total_steps=10)
    original_lrs = []
    for _ in range(10):
        original_lrs.append(warmup_optimizer.param_groups[0]['lr'])
        warmup_optimizer.step()
        original.step()
    wrong_param = nn.Parameter(torch.randn(2))
    wrong_optimizer = torch.optim.AdamW([wrong_param], lr=0.1)
    wrong = build_lr_schedule(wrong_optimizer, warmup_steps=5, total_steps=10)
    wrong_lrs = []
    for _ in range(10):
        wrong_lrs.append(wrong_optimizer.param_groups[0]['lr'])
        wrong_optimizer.step()
        wrong.step()
    assert not torch.allclose(torch.tensor(original_lrs), torch.tensor(wrong_lrs)), 'warmup mismatch must change the LR schedule'
    rebuilt_param = nn.Parameter(torch.randn(2))
    rebuilt_optimizer = torch.optim.AdamW([rebuilt_param], lr=0.1)
    rebuilt = build_lr_schedule(rebuilt_optimizer, warmup_steps=3, total_steps=10)
    rebuilt_lrs = []
    for _ in range(10):
        rebuilt_lrs.append(rebuilt_optimizer.param_groups[0]['lr'])
        rebuilt_optimizer.step()
        rebuilt.step()
    torch.testing.assert_close(torch.tensor(original_lrs), torch.tensor(rebuilt_lrs))
    assert resume_data_change_error(None, 'abc') is None
    assert resume_data_change_error('abc', 'abc') is None
    assert resume_data_change_error('abc', 'def') is not None
    assert resume_split_change_error(None, {'train': 'a'}) is None
    assert resume_split_change_error({}, {'train': 'a'}) is None
    assert resume_split_change_error({'train': 'a', 'val': 'b'}, {'train': 'a', 'val': 'b'}) is None
    split_change_error = resume_split_change_error({'train': 'a', 'val': 'b'}, {'train': 'a', 'val': 'c'})
    assert split_change_error is not None
    assert 'val' in split_change_error
    assert resume_effective_train_change_error(None, 'abc') is None
    assert resume_effective_train_change_error('abc', 'abc') is None
    effective_change_error = resume_effective_train_change_error('abc', 'def')
    assert effective_change_error is not None
    assert 'abc' in effective_change_error
    assert 'def' in effective_change_error
    isolation_a = _split_isolation_record({'train': (0, 30), 'val': (30, 50), 'test': (50, 60)}, 60)

def _selfcheck_top_55379_0037():
    global amp_change_error, captured_rng, expected_rng_draw, first_cuda, first_draw, invalid_seed, isolation_b, isolation_change_error, saved_python_state, saved_torch_state, second_cuda, second_draw
    isolation_b = _split_isolation_record({'train': (0, 28), 'val': (28, 50), 'test': (50, 60)}, 60)
    assert resume_split_isolation_change_error(None, isolation_a) is None
    assert resume_split_isolation_change_error(isolation_a, isolation_a) is None
    isolation_change_error = resume_split_isolation_change_error(isolation_a, isolation_b)
    assert isolation_change_error is not None
    assert '时序隔离' in isolation_change_error
    assert resume_amp_change_error({}, 'torch.bfloat16') is None
    assert resume_amp_change_error({'amp_dtype': None}, None) is None
    assert resume_amp_change_error({'amp_dtype': 'torch.float16'}, 'torch.float16') is None
    assert resume_amp_change_error({'amp_dtype': torch.float16}, 'torch.float16') is None
    amp_change_error = resume_amp_change_error({'amp_dtype': 'torch.float16'}, 'torch.bfloat16')
    assert amp_change_error is not None
    assert 'torch.float16' in amp_change_error
    assert 'torch.bfloat16' in amp_change_error
    set_seed(2024)
    first_draw = (random.random(), torch.rand(4), float(np.random.rand()) if np is not None else None)
    set_seed(2024)
    second_draw = (random.random(), torch.rand(4), float(np.random.rand()) if np is not None else None)
    assert first_draw[0] == second_draw[0]
    assert torch.equal(first_draw[1], second_draw[1])
    if np is not None:
        assert first_draw[2] == second_draw[2]
    assert _LAST_SEED_MANIFEST is not None
    assert _LAST_SEED_MANIFEST['seed'] == 2024
    assert _LAST_SEED_MANIFEST['torch_initial_seed'] == 2024
    if torch.cuda.is_available():
        assert _LAST_SEED_MANIFEST['cuda_initial_seeds']
        assert all((seed_value == 2024 for seed_value in _LAST_SEED_MANIFEST['cuda_initial_seeds']))
        torch.manual_seed(2024)
        first_cuda = torch.rand(4, device='cuda')
        torch.manual_seed(2024)
        second_cuda = torch.rand(4, device='cuda')
        assert torch.equal(first_cuda, second_cuda)
    set_seed(2024)
    saved_torch_state = torch.get_rng_state()
    saved_python_state = random.getstate()
    for invalid_seed in (-1, MAX_SEED + 1):
        try:
            set_seed(invalid_seed)
        except ValueError:
            pass
        else:
            raise AssertionError(f'非法种子 {invalid_seed} 必须被拒绝')
    assert torch.equal(torch.get_rng_state(), saved_torch_state)
    assert random.getstate() == saved_python_state
    set_seed(777)
    captured_rng = _capture_rng_state()
    expected_rng_draw = (random.random(), torch.rand(3), float(np.random.rand()) if np is not None else None)
    _restore_rng_state(captured_rng)
    assert random.random() == expected_rng_draw[0]

def _selfcheck_top_55519_0038():
    global _DuplicateGroupOptimizer, coverage, duplicate_coverage, grad_probe, partial_coverage, partial_optimizer, python_rng_before, rng_before, sanity_cfg, sanity_metrics, saved_matmul_precision, saved_tf32_cudnn, saved_tf32_matmul, tiny_probe_model
    assert torch.equal(torch.rand(3), expected_rng_draw[1])
    if np is not None:
        assert float(np.random.rand()) == expected_rng_draw[2]
    set_seed(11, strict_determinism=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    set_seed(11, strict_determinism=False)
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    saved_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    saved_tf32_cudnn = torch.backends.cudnn.allow_tf32
    saved_matmul_precision = torch.get_float32_matmul_precision() if hasattr(torch, 'get_float32_matmul_precision') else None
    try:
        configure_cuda_precision(False)
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        if hasattr(torch, 'get_float32_matmul_precision'):
            assert torch.get_float32_matmul_precision() == 'highest'
        configure_cuda_precision(True)
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        if hasattr(torch, 'get_float32_matmul_precision'):
            assert torch.get_float32_matmul_precision() == 'high'
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved_tf32_matmul
        torch.backends.cudnn.allow_tf32 = saved_tf32_cudnn
        if saved_matmul_precision is not None:
            torch.set_float32_matmul_precision(saved_matmul_precision)
    sanity_cfg = Config(**common, mode='B')
    rng_before = torch.get_rng_state()
    python_rng_before = random.getstate()
    sanity_metrics = run_sanity_overfit(sanity_cfg, eval_dataset, eval_device, amp_dtype=None, batch_size=2, steps=200, lr=0.01, loss_threshold=0.5, acc_threshold=0.9, seed=321, grad_checkpoint=True)
    assert sanity_metrics['passed'], sanity_metrics
    assert sanity_metrics['grad_checkpoint'] is True
    assert 'amp_dtype' in sanity_metrics
    assert sanity_metrics['data_sampling_uses_explicit_generator'] is True, sanity_metrics
    assert sanity_metrics['learned'] is True, sanity_metrics
    assert sanity_metrics['loss_reduction'] >= sanity_metrics['min_relative_improvement'], sanity_metrics
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert random.getstate() == python_rng_before
    assert sanity_metrics['gradient_flow_ok'] is True
    grad_probe = sanity_metrics['grad_probe']
    assert not grad_probe['failures'], grad_probe
    assert grad_probe['grad_norm'] > 0.0
    assert grad_probe['n_with_grad'] > 0
    assert sanity_metrics['optimizer_coverage_ok'] is True
    coverage = sanity_metrics['optimizer_coverage']
    assert coverage['missing_params'] == [], coverage
    assert coverage['duplicate_params'] == [], coverage
    assert coverage['foreign_param_count'] == 0, coverage
    assert sanity_metrics['parameter_update_ok'] is True
    assert sanity_metrics['pre_step_parameter_sha256'] != sanity_metrics['post_step_parameter_sha256']
    tiny_probe_model = LanguageModel(Config(**common, mode='B'))
    partial_optimizer = torch.optim.AdamW(tiny_probe_model.token_embedding.parameters(), lr=0.001)
    partial_coverage = _probe_optimizer_coverage(tiny_probe_model, partial_optimizer)
    assert partial_coverage['ok'] is False
    assert 'token_embedding.weight' not in partial_coverage['missing_params']
    assert partial_coverage['missing_params']
    class _DuplicateGroupOptimizer:
        param_groups = [{'params': list(tiny_probe_model.parameters())}, {'params': list(tiny_probe_model.parameters())}]
    duplicate_coverage = _probe_optimizer_coverage(tiny_probe_model, _DuplicateGroupOptimizer())
    assert duplicate_coverage['ok'] is False
    assert duplicate_coverage['duplicate_params']

