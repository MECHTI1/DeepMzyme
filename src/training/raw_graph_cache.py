"""Opt-in, local cache of raw radius-only graphs, before any fitted normalization.

Set DEEPMZYME_GRAPH_CACHE_DIR to a persistent directory. Cache entries are trusted
local pickle files; checksums detect corruption, not hostile cache writers. Graph
options, source, library versions and complete pocket contents bind every result.
RING is bypassed because its external file contents are not part of a pocket.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import torch
import torch_geometric
from torch_geometric.data import Data

from data_structures import PocketRecord
from training.parallel_loading import _CompactTensorPickler

GRAPH_CACHE_ENV = "DEEPMZYME_GRAPH_CACHE_DIR"
_SRC_ROOT = Path(__file__).resolve().parents[1]


def _compact_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    _CompactTensorPickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    return buffer.getvalue()


def _source_identity() -> str:
    digest = hashlib.sha256()
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        digest.update(str(path.relative_to(_SRC_ROOT)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _cache_namespace(options: dict[str, Any]) -> Path | None:
    root = os.environ.get(GRAPH_CACHE_ENV, "").strip()
    if not root or options.get("use_ring_edges") or options.get("require_ring_edges"):
        return None
    import label_schemes

    identity = {"schema": 1, "source_sha256": _source_identity(), "options": options,
                "label_scheme": label_schemes.ACTIVE_METAL_LABEL_SCHEME,
                "torch": torch.__version__, "torch_geometric": torch_geometric.__version__}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return Path(root).expanduser() / digest


def build_cached_graphs(pockets: list[PocketRecord], options: dict[str, Any],
                        builder: Callable[..., Data]) -> list[Data] | None:
    """Return fresh raw graphs in input order, or None when caching is disabled."""
    namespace = _cache_namespace(options)
    if namespace is None:
        return None
    namespace.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    graphs, hits = [], 0
    for index, pocket in enumerate(pockets, start=1):
        key = hashlib.sha256(_compact_bytes(pocket)).hexdigest()
        path = namespace / f"{key}.pkl"
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            graph = builder(pocket, **options)
            payload = _compact_bytes((key, graph))
            content = hashlib.sha256(payload).hexdigest().encode() + b"\n" + payload
            partial = None
            try:
                with tempfile.NamedTemporaryFile(dir=namespace, prefix=f".{key}.", suffix=".tmp", delete=False) as handle:
                    partial = Path(handle.name)
                    handle.write(content)
                os.replace(partial, path)
            finally:
                if partial is not None:
                    partial.unlink(missing_ok=True)
        else:
            checksum, separator, payload = content.partition(b"\n")
            if not separator or hashlib.sha256(payload).hexdigest().encode() != checksum:
                raise ValueError(f"Raw graph cache integrity check failed: {path}; remove this entry to rebuild")
            stored_key, graph = pickle.loads(payload)
            if stored_key != key or not isinstance(graph, Data):
                raise ValueError(f"Raw graph cache input identity differs: {path}")
            hits += 1
        graphs.append(graph)
        if index % 500 == 0 and index < len(pockets):
            print(f"[GRAPH-CACHE] processed={index}/{len(pockets)} hits={hits} misses={index - hits} "
                  f"seconds={perf_counter() - started:.2f}", flush=True)
    print(f"[GRAPH-CACHE] hits={hits} misses={len(graphs) - hits} "
          f"seconds={perf_counter() - started:.2f} namespace={namespace}", flush=True)
    return graphs
