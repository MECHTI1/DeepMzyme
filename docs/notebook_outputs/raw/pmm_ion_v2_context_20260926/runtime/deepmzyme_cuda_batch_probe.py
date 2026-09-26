"""Operational timing only: mutate a copied model in memory; never save weights."""
import argparse
import hashlib
import json
import os
import pickle
import resource
import statistics
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-copy', type=Path, required=True)
parser.add_argument('--cache-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
args = parser.parse_args()
import torch
from torch_geometric.loader import DataLoader
from training.campaign_runtime import load_campaign_prediction_components
from training.graph_dataset import PocketGraphDataset
from training.loop import train_epoch
from training.raw_graph_cache import GRAPH_CACHE_ENV, _cache_namespace

checkpoint_sha = hashlib.sha256(args.checkpoint_copy.read_bytes()).hexdigest()
saved = torch.load(args.checkpoint_copy, map_location='cpu', weights_only=False)
cfg = saved['config']
assert cfg['task'] == 'metal' and cfg['metal_loss_function'] == 'cross_entropy'
assert cfg['batch_size'] == 16 and cfg.get('grad_accum_steps', 1) == 1 and cfg.get('num_workers', 0) == 0
assert not args.output.exists(), 'Choose a new timing output; overwriting is forbidden'
model, normalization, options = load_campaign_prediction_components(saved, device=args.device)
os.environ[GRAPH_CACHE_ENV] = str(args.cache_root)
namespace = _cache_namespace(options)
assert namespace is not None and namespace.is_dir(), 'Matching raw graph cache namespace is absent'
files = sorted(namespace.glob('*.pkl'), key=lambda path: (path.stat().st_size, path.name))
def read_graph(path):
    checksum, separator, payload = path.read_bytes().partition(b'\n')
    assert separator and hashlib.sha256(payload).hexdigest().encode() == checksum, f'Corrupt cache: {path}'
    key, graph = pickle.loads(payload)
    assert key == path.stem and graph.x_esm.shape[1] == cfg['esm_dim'], f'Wrong cache input: {path}'
    return graph, {'key': key, 'payload_sha256': checksum.decode()}
eligible = []
for path in files:
    graph, _ = read_graph(path)
    if int(graph.y_ec.item()) >= 0 and bool(graph.x_esm.abs().sum() > 0) == cfg['require_esm_embeddings']:
        eligible.append(path)
assert len(eligible) >= 16, 'Need at least16 normal-training graphs with the saved ESM presence'
selected = [eligible[round(index * (len(eligible) - 1) / 15)] for index in range(16)]
graphs, identities = map(list, zip(*(read_graph(path) for path in selected)))
# Precomputed graphs are cloned/normalized on every access; no raw pockets are read.
dataset = PocketGraphDataset([None] * 16, precomputed_data=graphs, normalization_stats=normalization, **options)
loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0,
                    pin_memory=bool(cfg.get('pin_memory')) and args.device == 'cuda',
                    generator=torch.Generator().manual_seed(int(cfg['seed'])))
gvp_lr = cfg.get('gvp_learning_rate')
if gvp_lr is not None and gvp_lr != cfg['learning_rate'] and hasattr(model, 'layers'):
    trunk, head = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (trunk if name.startswith(('layers.', 'node_scalar_encoder.', 'edge_scalar_encoder.')) else head).append(parameter)
    groups = [{'params': trunk, 'lr': gvp_lr, 'weight_decay': cfg.get('gvp_weight_decay') if cfg.get('gvp_weight_decay') is not None else cfg['weight_decay']},
              {'params': head, 'lr': cfg['learning_rate'], 'weight_decay': cfg['weight_decay']}]
else:
    groups = model.parameters()
optimizer = torch.optim.AdamW(groups, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
amp = bool(cfg.get('use_amp')) and args.device == 'cuda'
scaler = torch.amp.GradScaler('cuda') if amp else None
def sync():
    if args.device == 'cuda': torch.cuda.synchronize()
def step():
    return train_epoch(model, loader, optimizer, args.device, grad_clip_norm=cfg['grad_clip_norm'],
                       grad_accum_steps=1, use_amp=amp, scaler=scaler)
for _ in range(5): step()
sync()
if args.device == 'cuda': torch.cuda.reset_peak_memory_stats()
timings = []
for _ in range(20):
    started = time.perf_counter()
    step()
    sync()
    timings.append(time.perf_counter() - started)
result = {'operational_timing_only': True, 'weights_saved': False, 'canonical_fit': False,
          'checkpoint_copy_sha256': checkpoint_sha, 'cache_namespace': str(namespace), 'cache_files': identities,
          'device': args.device, 'model_architecture': cfg['model_architecture'], 'fusion_mode': cfg.get('fusion_mode'),
          'warmup_batches': 5, 'timed_batches': 20, 'batch_size': 16, 'seconds_per_batch': timings,
          'mean_seconds': statistics.mean(timings), 'median_seconds': statistics.median(timings),
          'batch_nodes': sum(g.num_nodes for g in graphs), 'batch_edges': sum(g.edge_index.shape[1] for g in graphs),
          'cpu_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
          'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0,
          'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved() if args.device == 'cuda' else 0,
          'notes': 'One representative repeated batch; includes clone/normalization/collation/transfer and saved training loss. Excludes graph construction, validation, export and checkpoint I/O; use overheads separately. Cold AdamW state was warmed; no scheduler or checkpoint writes.'}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open('x') as stream: json.dump(result, stream, indent=2)
print(json.dumps({key: result[key] for key in ('mean_seconds', 'median_seconds', 'batch_nodes', 'batch_edges', 'cuda_peak_allocated_bytes')}))
