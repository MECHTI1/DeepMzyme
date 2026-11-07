"""CPU contracts for prebilling POSIX placement and authenticated preparation."""
import copy
import json
from pathlib import Path

import pytest

from ec_clean30 import phase3, views, workstation as ws


def placement():
    campaign = '/workspace/' + phase3.CAMPAIGN_ID
    config = {'artifact_cache': {'configured': True}, 'runpod': {'posix_root': '/opt/deepmzyme'},
              'remote_campaign': campaign, 'remote_python': '/opt/deepmzyme/main_env/bin/python',
              'embedding': {'esm1b': {'clean_python': '/opt/deepmzyme/clean_env/bin/python'}},
              'runtime_environments': {}, 'stage': []}
    sizes = {}
    for key, name, size in [('fit', 'main_env', 100), ('esm1b', 'clean_env', 200)]:
        archive = f'{campaign}/g2_inputs/{name}.tar.gz'
        config['runtime_environments'][key] = {'archive': archive}
        config['stage'].append({'kind': 'tar', 'remote': archive, 'extract_to': f'/opt/deepmzyme/{name}'})
        sizes[archive] = size
    return config, sizes


def test_fold0_lexical_campaign_path_is_retained_through_posix_view_alias(tmp_path):
    campaign = tmp_path / "network" / phase3.CAMPAIGN_ID
    local_views = tmp_path / "posix" / "views"
    campaign.mkdir(parents=True)
    (local_views / "fold0").mkdir(parents=True)
    (campaign / "views").symlink_to(local_views, target_is_directory=True)
    lexical = campaign / "views/fold0"
    assert lexical.resolve() == local_views / "fold0"
    assert views.view_location_problem(lexical, smoke=False, fold=0) is None
    assert views.campaign_root_of(lexical, smoke=False, fold=0) == campaign
    view = views.View(lexical, {"kind": views.KIND_FOLD_VIEW, "fold": 0}, "a" * 64)
    assert view.default_runs_dir == campaign / "runs/fold0"
    assert views.view_location_problem(tmp_path / "campaign/views/fold0", smoke=False, fold=0) is not None


def test_capacity_plan_counts_local_runtimes_and_view_without_bulk_archives():
    config, sizes = placement()
    request = ws._posix_storage_plan(config, sizes, 300)
    assert request['component_bytes'] == {'main_env': 100, 'clean_env': 200, 'views': 300}
    assert request['required_free_bytes'] == 750 + 5 * ws.GIB
    assert request['campaign_root'].endswith(phase3.CAMPAIGN_ID)
    assert ws._posix_storage_plan({}, {}, 0) is None


@pytest.mark.parametrize('change', ['network_runtime', 'wrong_python', 'bulk_local', 'extra_local_extraction',
                                   'view_alias', 'request_alias', 'no_cache'])
def test_placement_refuses_unbudgeted_or_inconsistent_paths(change):
    config, sizes = placement()
    if change == 'network_runtime':
        config['stage'][0]['extract_to'] = '/workspace/main_env'
    elif change == 'wrong_python':
        config['remote_python'] = '/workspace/main_env/bin/python'
    elif change == 'bulk_local':
        config['stage'].append({'kind': 'file', 'remote': '/opt/deepmzyme/large-weight.pt'})
    elif change == 'extra_local_extraction':
        config['stage'].append({'kind': 'tar', 'remote': config['remote_campaign'] + '/weights.tar',
                                'extract_to': '/opt/deepmzyme/weights'})
    elif change == 'view_alias':
        config['stage'].append({'kind': 'file', 'remote': config['remote_campaign'] + '/views/extra'})
    elif change == 'request_alias':
        config['stage'].append({'kind': 'tar', 'remote': config['remote_campaign'] + '/weights.tar',
                                'extract_to': config['remote_campaign'] + '/g2_requests/extra'})
    else:
        config.pop('artifact_cache')
    with pytest.raises(ws.WorkstationError):
        ws._posix_storage_plan(config, sizes, 300)


def test_posix_preparation_authenticates_helper_and_records_receipt(tmp_path, monkeypatch):
    config, sizes = placement()
    config.update(state_root=str(tmp_path / 'state'), bootstrap_python='/usr/bin/python3')
    repo = tmp_path / 'repo'
    helper = repo / 'src/ec_clean30/runpod_bootstrap.py'
    helper.parent.mkdir(parents=True)
    helper.write_text('# pinned storage helper\n')
    monkeypatch.setattr(phase3, 'REPO_ROOT', repo)
    digest = ws._sha(helper)
    ready = {'posix_storage': ws._posix_storage_plan(config, sizes, 300),
             'source': {'source_sha256': {'src/ec_clean30/runpod_bootstrap.py': digest}}}
    uploads = {}

    class Transport:
        def upload(self, local, remote):
            uploads[remote] = Path(local).read_bytes()

        def stat_file(self, remote):
            return {'sha256': digest, 'size_bytes': len(uploads[remote])}

        def remote(self, argv, **kwargs):
            assert argv[2:4] == ['prepare-posix-storage', '--request']
            assert json.loads(uploads[argv[4]]) == ready['posix_storage']
            return json.dumps({**ready['posix_storage'], 'status': 'passed',
                               'helper_sha256': digest, 'free_bytes': 8 * ws.GIB})

    ws._prepare_runpod_storage(config, Transport(), ready, {'session_id': 'owned'})
    assert json.loads((tmp_path / 'state/posix_storage.json').read_text())['helper_sha256'] == digest
    changed = copy.deepcopy(ready)
    changed['source']['source_sha256']['src/ec_clean30/runpod_bootstrap.py'] = '0' * 64
    uploads.clear()
    with pytest.raises(ws.WorkstationError, match='published pin'):
        ws._prepare_runpod_storage(config, Transport(), changed, {'session_id': 'owned'})
    assert not uploads
    with pytest.raises(ws.WorkstationError, match='prebilling capacity plan'):
        ws._prepare_runpod_storage(config, Transport(), {}, {'session_id': 'owned'})
