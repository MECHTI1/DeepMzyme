"""Rehydrate a frozen metal pilot on a new, explicitly owned Colab VM.

The host supplies /content/pilot_setup.json and already verified archives.
This script preserves the historical source and manifest, rechecks materialized
inputs, and never launches a training run. Host teardown is still mandatory.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import urllib.request


def main():
    config = json.loads(Path('/content/pilot_setup.json').read_text())
    output, root = Path(config['output_dir']), Path(config['repo_root'])
    deadline = config['allocation_started_epoch'] + config['recovery_setup_limit_seconds']

    def remaining():
        value = deadline - time.time()
        if value <= 0:
            raise TimeoutError('Cross-session preparation time allowance exhausted')
        return value

    def digest(path):
        value = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                remaining()
                value.update(chunk)
        return value.hexdigest()

    def extract(path, checksum, destination):
        if digest(path) != checksum:
            raise ValueError(f'Archive SHA-256 mismatch: {path}')
        with tarfile.open(path) as archive:
            for member in archive:
                remaining()
                archive.extract(member, destination, filter='data')

    extract(config['source_archive'], config['source_sha256'], root)
    snapshot = json.loads((root / 'code_snapshot_manifest.json').read_text())
    for name, checksum in snapshot['files'].items():
        if digest(root / name) != checksum:
            raise ValueError(f'Frozen source mismatch: {name}')
    extract(config['restoration_archive'], config['restoration_sha256'], output)
    if digest(output / 'campaign_manifest.json') != config['original_manifest_sha256']:
        raise ValueError('Restored scientific campaign manifest changed')
    sys.path.insert(0, str(root / 'scripts'))
    from colab_metal_pilot_bootstrap import cuda_check
    before = cuda_check()
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                    str(root / 'requirements/colab-overlay.txt')], check=True, timeout=remaining())
    code = ('import sys; sys.path.insert(0, ' + repr(str(root / 'scripts')) + '); '
            'from colab_metal_pilot_bootstrap import cuda_check; cuda_check()')
    after = json.loads(subprocess.check_output([sys.executable, '-c', code], text=True, timeout=remaining()))
    if before['torch'] != after['torch']:
        raise ValueError('Dependency setup replaced stock PyTorch')
    bundle = Path('/content/pilot_data_bundle.tar.gz')
    if not bundle.exists() or digest(bundle) != config['bundle_sha256']:
        print('Downloading the pinned v12 bundle', flush=True)
        temporary = bundle.with_suffix('.partial')
        with urllib.request.urlopen(config['bundle_url'], timeout=min(60, remaining())) as source, temporary.open('wb') as sink:
            while chunk := source.read(4 * 1024 * 1024):
                remaining()
                sink.write(chunk)
        temporary.replace(bundle)
    extract(bundle, config['bundle_sha256'], '/content/deepmzyme_bundle')
    Path('/content/v12_ready.json').write_text(json.dumps(dict(
        sha256=config['bundle_sha256'], verified_epoch=time.time(), url=config['bundle_url'])))
    extract(config['overlay_archive'], config['overlay_sha256'], '/content/metal_architecture_pilot_features')
    sys.path.insert(0, str(root / 'src'))
    import run_metal_architecture_pilot as pilot
    pilot.verify_manifest(root, output)
    pilot.storage_check(output, '/content/persistence_receipt.json')
    pilot.allocation_elapsed(output, config['allocation_started_epoch'])
    original_cohort = pilot.cohort_identity(pilot.read(output / 'expected_split.json'))
    original_cache = pilot.read(output / 'training_cache_audit.json')
    original_features = {item['path']: (item['sha256'], item['bytes']) for item in original_cache['files']}
    for name in ('readiness.json', 'training_cache_audit.json', 'expected_split.json', 'pooling_diagnostics.json'):
        pilot.save(output / 'cross_session_recovery' / ('previous_' + name), pilot.read(output / name))
    # The first allocation's profile window is already closed. This is a new
    # recovery preparation check, charged to the shared allocation ledger.
    command = [sys.executable, '-c', 'import sys; sys.path.insert(0, ' + repr(str(root / 'src')) + '); '
               'import run_metal_architecture_pilot as p; from pathlib import Path; '
               'p._preflight_worker(Path(' + repr(str(root)) + '), Path(' + repr(str(output)) + '), '
               'run_name="cross_session_preflight_run")']
    subprocess.run(command, check=True, timeout=remaining(),
                   env={**os.environ, 'DEEPGM_METAL_LABEL_SCHEME': 'six_class'})
    if pilot.cohort_identity(pilot.read(output / 'expected_split.json')) != original_cohort:
        raise ValueError('Cross-session retained cohort changed')
    refreshed = pilot.read(output / 'training_cache_audit.json')
    actual_features = {item['path']: (item['sha256'], item['bytes']) for item in refreshed['files']}
    if actual_features != original_features:
        raise ValueError('Cross-session feature content changed')
    pilot.verify_training_cache(output)
    completed = pilot.completed_rows(output)
    if len(completed) != 10:
        raise ValueError(f'Expected seven verified smokes and three full runs, got {len(completed)}')
    elapsed = pilot.allocation_elapsed(output, config['allocation_started_epoch'])
    pilot.save(output / 'cross_session_recovery' / 'readiness.json', dict(
        status='passed', original_manifest_sha256=config['original_manifest_sha256'],
        cohort_sha256=pilot.fingerprint(original_cohort), feature_content_identical=True,
        cache_timestamps_reaudited=True, verified_completed_runs=len(completed),
        allocated_seconds=elapsed, setup=config, cuda=after,
        setup_seconds=time.time() - config['allocation_started_epoch'], held_out_evaluation=False))
    print('CROSS_SESSION_RECOVERY_READY', flush=True)


if __name__ == '__main__':
    main()
