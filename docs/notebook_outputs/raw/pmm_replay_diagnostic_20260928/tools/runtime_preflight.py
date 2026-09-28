import json
from pathlib import Path
import shutil
import sys
import torch

sys.path.insert(0, '/home/mechti/projects/DeepMzyme/src')
from benchmarking.pmm_ion_campaign import source_tree_sha256

assert sys.executable == '/home/mechti/venvs/deepmzyme/bin/python'
assert source_tree_sha256() == 'adc95c42261448dc9d35572a138a3b8349de79124d9708618f511c27a848dd23'
assert torch.cuda.is_available()
assert 'L4' in torch.cuda.get_device_name()
x = torch.ones((16, 16), device='cuda')
assert (x @ x).eq(16).all().item()
report = dict(python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
              gpu=torch.cuda.get_device_name(), cuda_matmul='passed',
              source_tree_sha256=source_tree_sha256(),
              free_gib=shutil.disk_usage('/home/mechti/deepmzyme_runs').free/2**30)
assert report['free_gib'] > 10
Path(__file__).with_name('runtime_preflight.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
