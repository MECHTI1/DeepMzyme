"""Synthetic CPU contract/refusal tests; no campaign mutations or GPU actions."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

spec = importlib.util.spec_from_file_location('pmm_core_replay', Path(__file__).parents[1] / 'pmm_core_replay.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def row(target='six_class', probabilities=None):
    labels = r.NATIVE_LABELS[target]
    values = probabilities or {4: ['0.6', '0.1', '0.1', '0.2'],
                              5: ['0.6', '0.1', '0.1', '0.1', '0.1'],
                              6: ['0.6', '0.1', '0.1', '0.1', '0.05', '0.05']}[len(labels)]
    out = {'source_uid': 'one', 'native_element': 'MN', 'checkpoint_sha256': 'c' * 64,
           'pred_native': '0', 'pred_common4': '0', 'y_native': '0', 'y_common4': '0'}
    for label, value in zip(labels, values):
        out['p_native_' + label.replace(' ', '_')] = value
    for label, value in zip(r.FOUR, values[:3] + [str(sum(r.Decimal(v) for v in values[3:]))]):
        out['p_common4_' + label.replace(' ', '_')] = value
    return out


@pytest.mark.parametrize('target', r.NATIVE_LABELS)
def test_uniform_bound_all_native_vocabularies(target):
    original = row(target)
    replay = deepcopy(original)
    for prefix in ('p_native_', 'p_common4_'):
        replay[prefix + 'Mn'], replay[prefix + 'Cu'] = '0.60001', '0.09999'
    assert r.compare_rows([original], [replay], r.NATIVE_LABELS[target], r.POLICY)['qualified']
    replay['p_native_Mn'] = replay['p_common4_Mn'] = '0.60001001'
    replay['p_native_Cu'] = replay['p_common4_Cu'] = '0.09998999'
    assert not r.compare_rows([original], [replay], r.NATIVE_LABELS[target], r.POLICY)['qualified']


def test_five_class_native_co_ni_and_probability_first_collapse():
    example = row('five_class', ['0.1', '0.1', '0.1', '0.3', '0.4'])
    example.update(native_element='NI', y_native='4', y_common4='3', pred_native='4', pred_common4='3')
    r.validate_native_targets([example], 'five_class')
    r.validate_probabilities([example], r.FIVE, r.POLICY)
    example['y_native'] = '5'
    with pytest.raises(ValueError, match='Native target'):
        r.validate_native_targets([example], 'five_class')


@pytest.mark.parametrize('change', ['class_flip', 'uid', 'metadata', 'nan', 'negative', 'simplex', 'collapse', 'vocabulary'])
def test_probability_or_identity_changes_refused(change):
    before, after = row(), row()
    if change == 'class_flip':
        before = row(probabilities=['0.5000001', '0.4999999', '0', '0', '0', '0'])
        after = row(probabilities=['0.4999999', '0.5000001', '0', '0', '0', '0'])
        after.update(pred_native='1', pred_common4='1')
    elif change == 'uid':
        after['source_uid'] = 'other'
    elif change == 'metadata':
        after['checkpoint_sha256'] = 'd' * 64
    elif change == 'vocabulary':
        after['p_native_ZN'] = after.pop('p_native_Zn')
    else:
        key, value = {'nan': ('p_native_Mn', 'NaN'), 'negative': ('p_native_Mn', '-0.1'),
                      'simplex': ('p_native_Mn', '0.7'), 'collapse': ('p_common4_Mn', '0.6000003')}[change]
        after[key] = value
    with pytest.raises(ValueError):
        r.compare_rows([before], [after], r.SIX, r.POLICY)


def test_earliest_native_ba_checkpoint_and_history():
    history = [{'epoch': i, 'val_metal_balanced_acc': 0.8 if i in (10, 20) else 0.7} for i in range(1, 51)]
    receipt = {'selected_epoch': 10, 'selection_metric': 'val_metal_balanced_acc', 'selection_metric_value': 0.8}
    assert r.check_history(history, receipt)['epoch'] == 10
    with pytest.raises(ValueError, match='tie'):
        r.check_history(history, {**receipt, 'selected_epoch': 20})
    with pytest.raises(ValueError, match='Incomplete'):
        r.check_history(history[:-1], receipt)


def test_all_preserved_replays_required(tmp_path):
    current = tmp_path / 'independent_validation_replay'
    current.mkdir()
    for name in ('selected_checkpoint.json', 'val_predictions.csv'):
        (current / name).write_text('kept')
    old = tmp_path / '_incomplete_replays/failed_attempt'
    old.mkdir(parents=True)
    with pytest.raises(ValueError, match='Preserved replay'):
        r.replay_directories(tmp_path)
    for name in ('selected_checkpoint.json', 'val_predictions.csv'):
        (old / name).write_text('kept')
    assert len(r.replay_directories(tmp_path)) == 2


def test_source_drift_refused_before_any_scientific_import(tmp_path):
    before = set(sys.modules)
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src/changed.py').write_text('x=1')
    with pytest.raises(ValueError, match='source hash'):
        r.bind_frozen_source(tmp_path)
    assert set(sys.modules) == before


def test_import_from_wrong_tree_refused(tmp_path, monkeypatch):
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src/core_fixture.py').write_text('x=1')
    digest = r.hashlib.sha256(b'src/core_fixture.py\0x=1').hexdigest()
    monkeypatch.setattr(r, 'SOURCE_SHA256', digest)
    module = ModuleType('core_fixture')
    module.__file__ = '/another/source/core_fixture.py'
    monkeypatch.setitem(sys.modules, 'core_fixture', module)
    with pytest.raises(ValueError, match='another tree'):
        r.bind_frozen_source(tmp_path)


def test_prospective_contract_is_bound_and_historical_scope_exact():
    assert r.POLICY['prospective_folds'] == [1, 2, 3, 4]
    assert len(r.HISTORICAL_PINS) == 9
    assert all('__none__fold0__seed42' in name for name in r.HISTORICAL_PINS)
    assert len(r.HISTORICAL_FAILED) == 2
    assert r.POLICY_SHA256 == r.hashlib.sha256(json.dumps(r.POLICY, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def test_failed_historical_unit_cannot_be_accepted_without_diagnostic(tmp_path):
    name = 'only_gvp__five_class__none__fold0__seed42'
    pin = r.HISTORICAL_PINS[name]
    receipt = {'selected_checkpoint_sha256': pin['checkpoint_sha256'],
               'validation_predictions': {'sha256': pin['predictions_sha256']}}
    with pytest.raises(ValueError, match='diagnostic index'):
        r.historical_qualification(tmp_path, name, {}, receipt, False, 1492)
    with pytest.raises(ValueError, match='original strict status'):
        r.historical_qualification(tmp_path, name, {}, receipt, True, 1492)
    receipt['selected_checkpoint_sha256'] = 'd' * 64
    with pytest.raises(ValueError, match='nine frozen artifacts'):
        r.historical_qualification(tmp_path, name, {}, receipt, False, 1492)


def test_historical_strict_pass_keeps_its_original_status(tmp_path):
    name = 'only_esm__four_class__none__fold0__seed42'
    pin = r.HISTORICAL_PINS[name]
    receipt = {'selected_checkpoint_sha256': pin['checkpoint_sha256'],
               'validation_predictions': {'sha256': pin['predictions_sha256']}}
    q, _ = r.historical_qualification(tmp_path, name, {}, receipt, True, 1492)
    assert q['original_strict_status'] == 'passed'
    assert q['post_observation'] and q['acceptance_basis'] == 'retrospective_core_agreement'
    assert not q['diagnostic_required']


def fixture_diagnostic(tmp_path):
    import torch
    rows = [row('five_class')]
    identity = {'source_tree_sha256': r.SOURCE_SHA256, 'cohort_sha256': 'a' * 64,
                'fold_membership_sha256': 'f' * 64, 'target_scheme': 'five_class'}
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    snapshot = prepared / 'input_snapshot.pt'
    snapshot.write_bytes(b'input snapshot test fixture')
    tool = tmp_path / 'diagnostic.py'
    tool.write_bytes((Path(__file__).parents[1] / 'audit_pmm_replay.py').read_bytes())
    fields = {'x': {'sha256': 'a' * 64}, 'y_metal': {'sha256': 'b' * 64}}
    cache = {'path': '/original/cache.pkl', 'sha256': 'd' * 64, 'fields': fields}
    manifest = {'schema': 'pmm-replay-diagnostic-v1', 'status': 'predictive_inputs_matched',
                'campaign_run_identity': identity, 'cohort_file': {'sha256': 'a' * 64},
                'fold_file': {'sha256': 'f' * 64}, 'tool_sha256': r.sha(tool),
                'certifies_fit': False, 'training_performed': False, 'normalization_refitted': False,
                'excluded_from_predictive_equality': [], 'unmatched_uids': [],
                'input_files': {'/original/best_model_checkpoint.pt': 'c' * 64},
                'snapshot_sha256': r.sha(snapshot), 'n_examples': 1, 'n_batches': 1,
                'batches': [{'fresh_sha256': 'b' * 64, 'cached_sha256': 'b' * 64, 'field_differences': {}}],
                'matched_cache_files': [cache],
                'examples': [{'source_uid': 'one', 'fresh_fields': fields, 'matching_cache_files': [cache['path']],
                              'matching_cache_sha256': [cache['sha256']], 'raw_field_differences': {}}]}
    manifest_path = prepared / 'input_manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    refs = {view: torch.tensor([[float(rows[0][f'p_{view}_{label.replace(" ", "_")}']) for label in labels]], dtype=torch.float64)
            for view, labels in [('native', r.FIVE), ('common4', r.FOUR)]}
    summary = {'source_tree_sha256': r.SOURCE_SHA256, 'run_name': 'only_gvp__five_class__none__fold0__seed42',
               'checkpoint_sha256': 'c' * 64, 'predictive_inputs_matched': True, 'original_variable': False,
               'all_classes_stable': True, 'no_mutation': True, 'original_max_abs_probability_difference': 0,
               'evidence_files': [{'path': str(p.relative_to(tmp_path)), 'sha256': r.sha(p)} for p in (tool, manifest_path)],
               'processes': []}
    comparison = {view: {'changed_predictions': 0, 'max_abs_difference': 0.0} for view in refs}
    for condition in ('original', 'strict'):
        for number in range(2):
            directory = tmp_path / f'{condition}_{number}'
            directory.mkdir()
            results, preserved = [], []
            for index in range(5):
                result = {view: value.float() for view, value in refs.items()}
                result['y_native'] = torch.tensor([0])
                path = directory / f'pass_{index}.pt'
                torch.save(result, path)
                results.append(result)
                preserved.append({'pass': index, 'path': path.name, 'sha256': r.sha(path)})
            arrays = directory / 'forward_passes.pt'
            torch.save({'ordered_uids': ['one'], 'passes': results, 'saved_probabilities': refs}, arrays)
            report = {'schema': manifest['schema'], 'condition': condition, 'planned_repeats': 5, 'completed_repeats': 5,
                      'status': 'diagnosis_complete', 'deterministic_algorithms': condition == 'strict',
                      'deterministic_warn_only': False, 'preserved_passes': preserved,
                      'input_manifest_sha256': r.sha(manifest_path), 'tool_sha256': r.sha(tool),
                      'input_hashes_before_and_after': ['b' * 64], 'model_state_sha256_before_and_after': 'e' * 64,
                      'comparisons_to_saved': [comparison] * 5, 'pairwise_comparisons': [comparison] * 10,
                      'forward_passes_sha256': r.sha(arrays)}
            path = directory / 'evaluation_report.json'
            path.write_text(json.dumps(report))
            summary['processes'].append({'condition': condition, 'status': 'passed', 'completed_passes': 5,
                                         'report_path': str(path.relative_to(tmp_path)), 'report_sha256': r.sha(path)})
    path = tmp_path / 'diagnostic_summary.json'
    path.write_text(json.dumps(summary))
    args = dict(identity=identity, run_name=summary['run_name'], checkpoint_sha256='c' * 64, n_examples=1, original_rows=rows)
    return path, summary, args


def test_exact_input_stable_repeat_diagnostic_accepted(tmp_path):
    path, _, args = fixture_diagnostic(tmp_path)
    _, files = r.diagnostic_gate(path, **args)
    assert sum(Path(item['path']).name.startswith('pass_') for item in files) == 20


@pytest.mark.parametrize('change', ['input_mutation', 'model_mutation', 'warn_only', 'missing_process',
                                   'missing_pass', 'wrong_identity', 'raw_target_changed', 'exclude_target',
                                   'probability_bound', 'nan', 'tampered_pass'])
def test_diagnostic_refuses_incomplete_or_mutated_evidence(tmp_path, change):
    path, summary, args = fixture_diagnostic(tmp_path)
    item = summary['processes'][0]
    report_path = tmp_path / item['report_path']
    report = r.read_json(report_path)
    manifest_path = tmp_path / 'prepared/input_manifest.json'
    manifest = r.read_json(manifest_path)
    if change == 'input_mutation':
        report['input_hashes_before_and_after'] = ['f' * 64]
    elif change == 'model_mutation':
        report['model_state_sha256_before_and_after'] = 'f' * 64
    elif change == 'warn_only':
        report['deterministic_warn_only'] = True
    elif change == 'missing_process':
        summary['processes'].pop()
    elif change == 'missing_pass':
        report['preserved_passes'].pop()
    elif change == 'wrong_identity':
        args['identity'] = {**args['identity'], 'target_scheme': 'six_class'}
    elif change == 'raw_target_changed':
        manifest['examples'][0]['fresh_fields']['y_metal']['sha256'] = 'e' * 64
    elif change == 'exclude_target':
        manifest['excluded_from_predictive_equality'] = ['y_metal']
    elif change in ('probability_bound', 'nan'):
        report['comparisons_to_saved'][0]['native']['max_abs_difference'] = float('nan') if change == 'nan' else 1.1e-5
    else:
        (report_path.parent / report['preserved_passes'][0]['path']).write_bytes(b'tampered')
    manifest_path.write_text(json.dumps(manifest))
    for bound in summary['evidence_files']:
        if bound['path'] == 'prepared/input_manifest.json':
            bound['sha256'] = r.sha(manifest_path)
    for process in summary['processes']:
        p = tmp_path / process['report_path']
        content = report if process is item else r.read_json(p)
        content['input_manifest_sha256'] = r.sha(manifest_path)
        p.write_text(json.dumps(content))
        process['report_sha256'] = r.sha(p)
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        r.diagnostic_gate(path, **args)


def test_array_recomputation_rejects_forged_success_summary(tmp_path):
    import torch
    path, summary, args = fixture_diagnostic(tmp_path)
    item = summary['processes'][0]
    report_path = tmp_path / item['report_path']
    report = r.read_json(report_path)
    pass_path = report_path.parent / report['preserved_passes'][0]['path']
    result = torch.load(pass_path, weights_only=False)
    result['native'][0, 0] += 0.0001
    result['native'][0, 1] -= 0.0001
    torch.save(result, pass_path)
    report['preserved_passes'][0]['sha256'] = r.sha(pass_path)
    report_path.write_text(json.dumps(report))
    item['report_sha256'] = r.sha(report_path)
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='array exceeds'):
        r.diagnostic_gate(path, **args)


def test_replay_discovery_includes_sibling_incomplete_and_external_backup(tmp_path):
    run = tmp_path / 'runs/core_unit'
    current = run / 'independent_validation_replay'
    sibling = run / 'independent_validation_replay.incomplete.123'
    backup = tmp_path / 'runtime/old_backup/runs/core_unit/independent_validation_replay'
    for directory in (current, sibling, backup):
        directory.mkdir(parents=True)
        for name in ('selected_checkpoint.json', 'val_predictions.csv'):
            (directory / name).write_text('preserved')
    assert set(r.replay_directories(run, tmp_path)) == {current, sibling, backup}
    (backup / 'val_predictions.csv').unlink()
    with pytest.raises(ValueError, match='lacks predictions'):
        r.replay_directories(run, tmp_path)


def test_archived_fit_does_not_silently_count_as_new_unit(tmp_path):
    run = tmp_path / 'runs/core_unit'
    replay = run / 'independent_validation_replay'
    replay.mkdir(parents=True)
    for name in ('selected_checkpoint.json', 'val_predictions.csv'):
        (replay / name).write_text('preserved')
    (tmp_path / 'runs/_incomplete_attempts/core_unit__old').mkdir(parents=True)
    with pytest.raises(ValueError, match='archived training attempt'):
        r.replay_directories(run, tmp_path)


def test_policy_mutation_refused_before_loading_source(tmp_path, monkeypatch):
    monkeypatch.setitem(r.POLICY, 'probability_atol', '0.1')
    with pytest.raises(ValueError, match='policy mutated'):
        r.validate_core_unit(tmp_path, tmp_path, 'only_gvp', 'five_class', 1, source_root=tmp_path)


def test_historical_failed_exports_are_pinned_not_just_final_checkpoint():
    assert len(r.HISTORICAL_PINS['only_gvp__six_class__none__fold0__seed42']['replay_prediction_sha256s']) == 2
    assert len(r.HISTORICAL_PINS['only_gvp__five_class__none__fold0__seed42']['replay_prediction_sha256s']) == 1
    for pin in r.HISTORICAL_PINS.values():
        assert pin['replay_prediction_sha256s'] and all(len(digest) == 64 for digest in pin['replay_prediction_sha256s'])


def test_boolean_or_nonfinite_metric_cannot_masquerade_as_numeric_match():
    assert not r.same_metrics(1.0, True)
    assert not r.same_metrics(False, 0)
    assert not r.same_metrics(float('nan'), float('nan'))
