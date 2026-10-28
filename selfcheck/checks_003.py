from __future__ import annotations

def _selfcheck_top_65999_0123():
    global child_env, explicit_env, explicit_none_key, legacy_provenance, recorded_none, same_provenance, saved_hashseed, unrecorded_sha
    same_provenance = check_run_provenance([{'run': 'r1', 'provenance': {'script_sha256': 'a' * 64, 'torch_version': '2.0.0', 'platform': 'linux'}}, {'run': 'r2', 'provenance': {'script_sha256': 'a' * 64, 'torch_version': '2.0.0', 'platform': 'linux'}}])
    assert same_provenance['ok'], same_provenance['mismatches']
    assert 'script_sha256' in same_provenance['checked_keys']
    legacy_provenance = check_run_provenance([{'run': 'r1', 'provenance': {'git_commit': 'c1'}}, {'run': 'r2', 'provenance': {'git_commit': 'c2'}}])
    assert not legacy_provenance['commits_differ']
    assert legacy_provenance['hashseed_unset_runs'] == []
    assert legacy_provenance['hashseed_unrecorded_runs'] == ['r1', 'r2']
    recorded_none = check_run_provenance([{'run': 'r1', 'provenance': {'script_sha256': 'a' * 64, 'git_commit': 'c1', 'pythonhashseed_at_start': None}}, {'run': 'r2', 'provenance': {'script_sha256': 'a' * 64, 'git_commit': 'c2', 'pythonhashseed_at_start': None}}])
    assert recorded_none['commits_differ']
    assert recorded_none['hashseed_unset_runs'] == ['r1', 'r2']
    assert recorded_none['hashseed_unrecorded_runs'] == []
    assert _missing_efficiency_cell_labels([{'label': 'ngram', 'deterministic': True, 'costs': {}}, {'label': 'neural', 'deterministic': False, 'costs': {}}]) == ['neural']
    explicit_none_key = check_run_provenance([{'run': 'r1', 'provenance': {'script_sha256': 'a' * 64, 'cuda_version': None}}, {'run': 'r2', 'provenance': {'script_sha256': 'a' * 64, 'cuda_version': None}}])
    assert 'cuda_version' in explicit_none_key['checked_keys']
    assert 'cuda_version' not in explicit_none_key['unchecked_keys']
    unrecorded_sha = check_run_provenance([{'run': 'r1', 'provenance': {'git_commit': 'c1'}, 'provenance_recorded_keys': ['git_commit']}, {'run': 'r2', 'provenance': {'git_commit': 'c2'}, 'provenance_recorded_keys': ['git_commit']}])
    assert not unrecorded_sha['commits_differ']
    assert pythonhashseed_error(False) is None
    if _PYTHONHASHSEED_AT_START is None:
        assert pythonhashseed_error(True)
    else:
        assert pythonhashseed_error(True) is None
    saved_hashseed = os.environ.pop('PYTHONHASHSEED', None)
    try:
        child_env = _sweep_child_env(7)
        assert child_env['PYTHONHASHSEED'] == '7'
        assert child_env['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
        assert os.environ.get('PYTHONHASHSEED') is None
        os.environ['PYTHONHASHSEED'] = '1234'
        explicit_env = _sweep_child_env(7)
        assert explicit_env['PYTHONHASHSEED'] == '1234'
    finally:
        os.environ.pop('PYTHONHASHSEED', None)
        if saved_hashseed is not None:
            os.environ['PYTHONHASHSEED'] = saved_hashseed

def _selfcheck_top_66135_0124():
    global saved_env_hashseed, saved_rng
    if _PYTHONHASHSEED_AT_START is None:
        saved_rng = _capture_rng_state()
        saved_env_hashseed = os.environ.get('PYTHONHASHSEED')
        try:
            os.environ.pop('PYTHONHASHSEED', None)
            set_seed(101)
            assert os.environ.get('PYTHONHASHSEED') == '101'
            set_seed(102)
            assert os.environ.get('PYTHONHASHSEED') == '102'
        finally:
            _restore_rng_state(saved_rng)
            if saved_env_hashseed is None:
                os.environ.pop('PYTHONHASHSEED', None)
            else:
                os.environ['PYTHONHASHSEED'] = saved_env_hashseed
    print('self-test passed')

