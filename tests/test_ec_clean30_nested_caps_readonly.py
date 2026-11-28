"""Nested CLEAN caps on the supervised read-only backend, through the environment that phase 4 really builds.

The capped CLEAN worker starts three kinds of capped child with an environment rebuilt by the comparator's
``build_clean_env``: the distance map, the training step (through the launcher) and the exports (through the
launcher). Each must stay on the worker's cores under its own memory limit. Kernel-facing checks are replaced as in
``test_ec_clean30_container_caps.READONLY_DRIVER``; the wrapper, its core claims, the affinity,
``phase4._clean_fit``, ``build_clean_env`` and ``_run`` are the real code. CPU only, no data.

The last test checks that a probe record carrying the nested self-test still passes the checks phase 3 adds.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ec_clean30 import container_caps as caps
from test_ec_clean30_container_caps import REBUILT_ENVIRONMENT, _helper, _need_cores, _prefix, _records, kernel

REPO = Path(caps.__file__).resolve().parents[2]
GIB = 1024**3

WORKER = r"""
import json, os, sys
from pathlib import Path
sys.path[:0] = [{repo!r}, {repo!r} + '/src', {repo!r} + '/tests']
from project_paths import install_relocated_module_aliases
install_relocated_module_aliases()
from ec_clean30 import container_caps as caps, phase4
clean = phase4.clean
tmp = Path(sys.argv[1]); request = json.loads((tmp / 'request.json').read_text())
work, attempt = tmp / 'work', tmp / 'attempt'; work.mkdir(); attempt.mkdir()
(work / clean.PREPARE_MANIFEST).write_text(json.dumps({{'training_name': 'fold0', 'training_table': {{'sha256': '1' * 64}},
    'embedding_farm': {{'base': {{'farm_sha256': '2' * 64}}}}, 'input_dim': 1280}}))
REPORT = ("import json,os; print('CHILD ' + json.dumps({{'step': %r, 'affinity': sorted(os.sched_getaffinity(0)), "
          "'scope': os.environ.get('DEEPMZYME_CAPS_SCOPE')}}))")
roots = [tmp / 'forbidden']

def fit(**kwargs):
    if kwargs.get('dry_run'):
        return {{'steps': [{{'step': 'distance_map', 'hash_seed': 0, 'command': [sys.executable, '-c', REPORT % 'map']}},
                          {{'step': 'train', 'hash_seed': 42, 'command': ['not-run']}}]}}
    # As run_fit does: the launcher stands for the private interpreter, under the rebuilt environment.
    env = clean.build_clean_env(clean_clone=Path(kwargs['clean_clone']), hash_seed=42, threads=kwargs['threads'],
                                forbidden_roots=kwargs['extra_forbidden_roots'], device=kwargs.get('device', 'cpu'))
    code = clean._run([str(kwargs['clean_python']), '-c', REPORT % 'train'], cwd=work, env=env,
                      log_path=work / 'logs' / 'train.log')
    return {{'recipe': clean.PUBLISHED_RECIPE, 'returncode': code}}

phase4._clean_caps = lambda request: None                       # the prefixes name this test's stand-in
clean.run_fit = fit
clean.verify_distance_map = lambda *a, **k: {{'sha256': '3' * 64}}
phase4.phase3.runtime_precision.read_policy_log = lambda *a, **k: {{}}
out = {{'worker_scope': os.environ.get(caps.SCOPE_ENV), 'worker_affinity': sorted(os.sched_getaffinity(0)),
       'kept_names': list(clean.PASSTHROUGH_ENV)}}
try:
    result, launcher = phase4._clean_fit(request, work, attempt, roots)
    out['train_returncode'] = result['returncode']
    # As run_export does (validation export and in-sample sanity export).
    env = clean.build_clean_env(clean_clone=Path(request['clean_clone']), hash_seed=42, threads=request['threads'],
                                forbidden_roots=roots, device='cpu')
    out['export_returncode'] = clean._run([str(launcher), '-c', REPORT % 'export'], cwd=work, env=env,
                                          log_path=attempt / 'export.log')
    out['child_environment_names_the_scope'] = caps.SCOPE_ENV in env
except phase4.Phase4Error as exc:
    out['error'] = str(exc)
print('WORKER ' + json.dumps(out), flush=True)
raise SystemExit(0 if 'error' not in out and not out['train_returncode'] and not out['export_returncode'] else 2)
"""


def _log(path):
    text = path.read_text() if path.is_file() else ""
    return {**_records(text), "text": text,
            "child": next((json.loads(line[6:]) for line in text.splitlines() if line.startswith("CHILD ")), None)}


@pytest.mark.parametrize("worker_cores,step_cores", [(4, 4), (4, 2), (2, 2)])
def test_capped_clean_steps_stay_on_the_capped_workers_cores_through_the_rebuilt_environment(
        tmp_path, worker_cores, step_cores):
    _need_cores(worker_cores)
    _helper(tmp_path)
    root = tmp_path / "ro"
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER.format(repo=str(REPO)))
    request = {"provider": "runpod", "threads": 4, "memory_limit_bytes": 24 * GIB,
               "cap_prefix": _prefix(tmp_path, step_cores, 24 * GIB),
               "distance_cap_prefix": _prefix(tmp_path, step_cores, 8 * GIB),
               "clean_cap_profile": {"cgroup_root": str(root), "cpu_cores": 4, "memory_bytes": 24 * GIB,
                                     "distance_map_memory_bytes": 8 * GIB},
               "clean_python": sys.executable, "clean_clone": str(tmp_path / "clone"),
               "training_script": str(tmp_path / "train.py"), "distance_helper": str(tmp_path / "map.py")}
    (tmp_path / "request.json").write_text(json.dumps(request))
    # As run_unit does: the worker itself runs under the larger CLEAN cap, with the caller's environment.
    outer = subprocess.run([*_prefix(tmp_path, worker_cores, 24 * GIB), sys.executable, str(worker), str(tmp_path)],
                           capture_output=True, text=True, timeout=300,
                           env={key: value for key, value in os.environ.items()
                                if key not in (caps.SCOPE_ENV, "CUDA_VISIBLE_DEVICES")})
    started = _records(outer.stderr)["started"]
    report = json.loads(next((line[7:] for line in outer.stdout.splitlines() if line.startswith("WORKER ")), "{}"))
    cores, scope = started["cpu_affinity"], Path(started["cgroup"]).name
    assert len(cores) == worker_cores and started["before"]["enclosing_scope"] is None
    assert report["worker_scope"] == scope and report["worker_affinity"] == cores
    logs = {"map": _log(tmp_path / "attempt/distance_map.log"), "train": _log(tmp_path / "work/logs/train.log"),
            "export": _log(tmp_path / "attempt/export.log")}
    # A refused step leaves one line in its log and nothing else, as distance_map.log did on the read-only host.
    for step in ("map", "train", "export"):
        assert "container caps refused" not in logs[step]["text"], (step, logs[step]["text"].strip()[-300:])
    assert "error" not in report and outer.returncode == 0, (report, logs["map"]["text"][-300:], outer.stderr[-600:])
    for step, limit in (("map", 8 * GIB), ("train", 24 * GIB), ("export", 24 * GIB)):
        record = logs[step]
        assert record["finished"] is not None and record["finished"]["returncode"] == 0, (step, record["text"][-300:])
        assert record["started"]["backend"] == caps.READONLY_BACKEND
        assert record["started"]["memory_limit_bytes"] == limit
        assert record["started"]["limits"]["memory"]["limit_bytes"] == limit
        assert record["started"]["before"]["enclosing_scope"] == scope, step
        assert record["started"]["before"]["enclosing_claim"]["pid"] == int(scope.split("-")[2]), step
        assert record["started"]["cpu_affinity"] == cores[:step_cores], step
        assert record["child"]["step"] == step and record["child"]["affinity"] == cores[:step_cores]
        # The step's own wrapper names itself to its child, as every capped command on this backend does.
        assert record["child"]["scope"] == Path(record["started"]["cgroup"]).name
    # The comparator's allow-list is what removes the variable: no step got it from the worker.
    assert report["child_environment_names_the_scope"] is False
    assert tuple(report["kept_names"]) == REBUILT_ENVIRONMENT == caps.NESTED_ENVIRONMENT


def test_probe_record_with_the_nested_test_passes_the_runtime_validators(kernel):
    """Every later check of a probe record: the helper's own validator, then the bindings phase 3 adds.

    phase3._verify_container_caps reads helper_sha256, hostname, boot_id, python_executable and cgroup_root, and
    the runtime binds the whole record by the hash of its canonical bytes; an added field must leave all of
    these intact, and a record without the nested test must no longer pass.
    """
    from ec_clean30 import phase3

    evidence = caps.probe(kernel[0])
    source = {"source_sha256": {"src/ec_clean30/container_caps.py":
                                hashlib.sha256(Path(caps.__file__).read_bytes()).hexdigest()}}
    for cores, gib, esm1b in ((2, 8, False), (4, 24, True)):
        prefix = caps.cap_prefix(sys.executable, phase3.REPO_ROOT, evidence["cgroup_root"],
                                 cpu_cores=cores, memory_bytes=gib * GIB)
        phase3._verify_container_caps(source, evidence, prefix, python=sys.executable, esm1b=esm1b)
    # The record as a later step reads it back from its file.
    stored = json.loads(phase3._bytes(evidence))
    assert "nested_test" in stored and phase3._bytes(stored) == phase3._bytes(evidence)
    phase3._verify_container_caps(source, stored, prefix, python=sys.executable, esm1b=True)
    without = deepcopy(stored)
    del without["nested_test"]
    with pytest.raises(caps.CapError):
        phase3._verify_container_caps(source, without, prefix, python=sys.executable, esm1b=True)
