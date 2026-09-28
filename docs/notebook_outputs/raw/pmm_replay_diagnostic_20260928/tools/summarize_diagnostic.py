"""Aggregate every predeclared diagnostic pass; never certify a fit."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent.parent
UNIT = 'only_gvp__six_class__none__fold0__seed42'
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
read = lambda p: json.loads(Path(p).read_text())
manifest = read(ROOT/'prepared_recovered/input_manifest.json')
assert manifest['status'] == 'predictive_inputs_matched'
assert manifest['n_examples'] == 1492 and manifest['n_batches'] == 94
assert sha(ROOT/'prepared_recovered/input_snapshot.pt') == manifest['snapshot_sha256']
binding = {'prepared_recovered/input_manifest.json': sha(ROOT/'prepared_recovered/input_manifest.json')}
processes, passes, failures = {}, {'original': [], 'strict': []}, []
original_hashes, model_hash = None, None
for condition in ('original', 'strict'):
    for process in (1, 2):
        name = f'{condition}_{process}'
        path = ROOT/name/'evaluation_report.json'
        report = read(path)
        binding[str(path.relative_to(ROOT))] = sha(path)
        assert report['input_manifest_sha256'] == binding['prepared_recovered/input_manifest.json']
        assert report['tool_sha256'] == manifest['tool_sha256']
        assert report['condition'] == condition and report['planned_repeats'] == 5
        assert report['deterministic_warn_only'] is False
        assert report['deterministic_algorithms'] == (condition == 'strict')
        processes[name] = report
        if report['status'] != 'diagnosis_complete':
            failures.append({'process': name, 'report': report})
            continue
        assert report['completed_repeats'] == 5
        arrays_path = ROOT/name/'forward_passes.pt'
        assert sha(arrays_path) == report['forward_passes_sha256']
        binding[str(arrays_path.relative_to(ROOT))] = sha(arrays_path)
        arrays = torch.load(arrays_path, map_location='cpu', weights_only=False)
        assert arrays['ordered_uids'] == [r['source_uid'] for r in manifest['examples']]
        assert len(arrays['passes']) == 5
        for item in arrays['passes']:
            for view in ('native', 'common4'):
                assert torch.isfinite(item[view]).all()
            passes[condition].append(item)
        if original_hashes is None:
            original_hashes = report['input_hashes_before_and_after']
            model_hash = report['model_state_sha256_before_and_after']
        assert report['input_hashes_before_and_after'] == original_hashes
        assert report['model_state_sha256_before_and_after'] == model_hash
assert len(passes['original']) == 10
uids = [r['source_uid'] for r in manifest['examples']]
references = [CAMPAIGN/'runs'/UNIT/'val_predictions.csv',
              CAMPAIGN/'runs'/UNIT/'independent_validation_replay/val_predictions.csv']
references += sorted((CAMPAIGN/'runs'/UNIT).glob('independent_validation_replay.incomplete.*/val_predictions.csv'))
# Some frozen runners archive incomplete replays with an underscore suffix.
references += sorted((CAMPAIGN/'runs'/UNIT).glob('independent_validation_replay_incomplete_*/val_predictions.csv'))
first = CAMPAIGN/'runtime/fold0_completion_20260927/failed_replay_attempt1_backup/runs'/UNIT/'independent_validation_replay/val_predictions.csv'
if first.is_file():
    references.append(first)
references = list(dict.fromkeys(references))
unique = {}
for path in references:
    unique.setdefault(sha(path), path)
assert len(unique) == 3, f'Expected original plus two distinct replay exports, got {list(unique.values())}'
columns = {'native': ['p_native_'+x for x in ('Mn','Cu','Zn','Fe','Co','Ni')],
           'common4': ['p_common4_'+x for x in ('Mn','Cu','Zn','Class_VIII')]}
comparisons = []
for digest, path in unique.items():
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    by_uid = {row['source_uid']: row for row in rows}
    assert len(rows) == len(by_uid) == len(uids) and set(by_uid) == set(uids)
    for condition, results in passes.items():
        for index, result in enumerate(results):
            for view, keys in columns.items():
                reference = torch.tensor([[float(by_uid[uid][key]) for key in keys] for uid in uids], dtype=torch.float64)
                difference = (result[view].double()-reference).abs()
                comparisons.append(dict(reference=str(path.relative_to(CAMPAIGN)), reference_sha256=digest,
                                        condition=condition, pass_index=index, view=view,
                                        maximum_abs_difference=float(difference.max()),
                                        changed_predictions=int((result[view].argmax(1) != reference.argmax(1)).sum())))
pairwise = {}
for condition, results in passes.items():
    summary = {}
    for view in ('native', 'common4'):
        differences = [(a[view]-b[view]).abs().max().item() for a,b in itertools.combinations(results,2)]
        changes = [(a[view].argmax(1)!=b[view].argmax(1)).sum().item() for a,b in itertools.combinations(results,2)]
        summary[view] = dict(maximum_abs_difference=max(differences, default=0),
                             maximum_changed_predictions=max(changes,default=0),
                             nonidentical_pairs=sum(x > 0 for x in differences),
                             pairs=len(differences))
    pairwise[condition] = summary
original_comparisons = [r for r in comparisons if r['condition']=='original']
excluded_counts = {}
for row in manifest['examples']:
    differences = {key for key in row['fresh_fields'] if row['fresh_fields'][key] != row['cached_fields'][key]}
    for key in differences:
        excluded_counts[key] = excluded_counts.get(key,0)+1
assert set(excluded_counts) <= set(manifest['excluded_from_predictive_equality'])
out = dict(schema='pmm-replay-diagnostic-summary-v1', certifies_fit=False,
           predictive_inputs_matched=True, n_examples=1492, n_batches=94,
           original_variable=any(x['maximum_abs_difference'] > 0 for x in pairwise['original'].values()),
           all_classes_stable=all(r['changed_predictions']==0 for r in comparisons)
               and all(v['maximum_changed_predictions']==0 for c in pairwise.values() for v in c.values()),
           no_mutation=True,
           maximum_original_probability_difference=max(r['maximum_abs_difference'] for r in original_comparisons),
           completed_original_passes=len(passes['original']), completed_strict_passes=len(passes['strict']),
           strict_failures=failures, strict_results_reported=True,
           pairwise=pairwise, comparisons_to_all_preserved_exports=comparisons,
           input_model_and_artifact_checks_passed=True,
           excluded_metadata_difference_counts=excluded_counts,
           ambiguous_identical_content_examples=manifest['ambiguous_identical_content_examples'],
           retrospective_limit=manifest['retrospective_limit'],
           artifact_sha256=binding, source_tree_sha256=manifest['campaign_run_identity']['source_tree_sha256'],
           model_state_sha256=model_hash, tool_sha256=manifest['tool_sha256'],
           summary_tool_sha256=sha(__file__), training_performed=False, held_out_access=False)
out['disabled_ec_cpu_check'] = manifest['disabled_ec_cpu_check']
out['previous_failed_manifest_sha256'] = manifest['previous_failed_manifest_sha256']
out['run_name'] = UNIT
out['checkpoint_sha256'] = next(value for path,value in manifest['input_files'].items()
                                if path.endswith('/best_model_checkpoint.pt'))
out['original_max_abs_probability_difference'] = out['maximum_original_probability_difference']
out['processes'] = []
for name, report in processes.items():
    passed = report['status'] == 'diagnosis_complete'
    if not passed:
        assert report['condition'] == 'strict'
        assert 'determin' in report.get('error','').lower()
    out['processes'].append(dict(condition=report['condition'],
        status='passed' if passed else 'unsupported',
        completed_passes=report.get('completed_repeats',0),
        report_path=f'{name}/evaluation_report.json',
        report_sha256=binding[f'{name}/evaluation_report.json']))
binding['prepared_recovered/input_snapshot.pt'] = manifest['snapshot_sha256']
binding['failed_input_manifest.json'] = manifest['previous_failed_manifest_sha256']
binding['audit_pmm_replay_v1.py'] = manifest['previous_tool_sha256']
binding['summarize_diagnostic.py'] = sha(__file__)
binding['audit_pmm_replay.py'] = manifest['tool_sha256']
binding['run_diagnostic.sh'] = sha(ROOT/'run_diagnostic.sh')
binding['protocol.md'] = sha(ROOT/'protocol.md')
binding['run_recovered_diagnostic.sh'] = sha(ROOT/'run_recovered_diagnostic.sh')
binding['protocol_v1_1.md'] = sha(ROOT/'protocol_v1_1.md')
out['evidence_files'] = [dict(path=path,sha256=digest) for path,digest in binding.items()]
destination = ROOT/'diagnostic_summary.json'
with destination.open('x') as handle:
    json.dump(out, handle, indent=2)
    handle.write('\n')
print(json.dumps({k:out[k] for k in ('predictive_inputs_matched','original_variable','all_classes_stable',
                                  'maximum_original_probability_difference','completed_original_passes',
                                  'completed_strict_passes','pairwise')}))
