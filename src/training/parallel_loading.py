"""Parallel structure parsing for pocket loading.

Parsing a structure set is embarrassingly parallel (each structure file is parsed
independently) but was single-threaded: on a 12-vCPU Colab VM, parsing 6,443 structures
ran on one core for ~40 minutes while the GPU sat idle.

Guarantees:
- Results come back in input order, so the pocket list -- and therefore every
  downstream split, fold and metric -- is identical to the serial path.
- Workers set the active metal label scheme explicitly instead of relying on inheriting
  the parent's module globals: a spawned/forkserver worker re-imports ``label_schemes``
  and would otherwise silently fall back to the default scheme.
- Results travel between processes as plain pickled bytes, so torch never passes
  thousands of tensors through shared-memory file descriptors.
- Each tensor is pickled as only the bytes it views (``_CompactTensorPickler``), so the
  parent never receives more data than serial parsing would hold and unpickles cheaply.

Worker count (first match wins):
1. an explicit ``load_workers`` argument (``--load-workers`` on the CLI);
2. the ``DEEPMZYME_LOAD_WORKERS`` environment variable;
3. ``load_workers`` in the per-machine file ``~/.config/deepmzyme/runtime.json``
   (``$XDG_CONFIG_HOME`` is honoured). Use it to keep a shared workstation from being
   saturated without changing the project default;
4. every CPU core available to this process.
The values ``0`` and ``"all"`` mean every available core; ``1`` is the serial path.

Parse cache: set ``DEEPMZYME_PARSE_CACHE_DIR`` to keep one file per parsed structure so
later folds and models skip re-parsing. Entries are keyed by the structure file's
content, every load setting, the metal label scheme and the source of every module under
``src/``, so a code or setting change never reuses stale results. Feature files read
during parsing (ESM embeddings, ring/external features) are keyed by path only: clear the
cache after regenerating them in place.
"""

from __future__ import annotations

import hashlib
import io
import json
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import torch

from training.structure_loading import StructureLoadError, load_structure_pockets

LOAD_WORKERS_ENV = "DEEPMZYME_LOAD_WORKERS"
START_METHOD_ENV = "DEEPMZYME_LOAD_START_METHOD"
RUNTIME_CONFIG_ENV = "DEEPMZYME_RUNTIME_CONFIG"
PARSE_CACHE_ENV = "DEEPMZYME_PARSE_CACHE_DIR"
_SRC_ROOT = Path(__file__).resolve().parents[1]

# Below this many structures a process pool costs more than it saves, and small
# test/smoke sets keep exercising the plain in-process path.
MIN_STRUCTURES_FOR_PARALLEL = 32


def available_cpu_count() -> int:
    """CPU cores this process may run on (respects affinity masks and cgroup pinning)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def runtime_config_path() -> Path:
    override = os.environ.get(RUNTIME_CONFIG_ENV)
    if override:
        return Path(override).expanduser()
    config_home = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return config_home / "deepmzyme" / "runtime.json"


def _parse_worker_value(raw: Any, *, source: str) -> int:
    if isinstance(raw, str) and raw.strip().lower() in {"all", "auto"}:
        return 0
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid load worker count {raw!r} from {source}; use an integer >= 0 or 'all'") from exc
    if value < 0:
        raise ValueError(f"Invalid load worker count {value} from {source}; use an integer >= 0 or 'all'")
    return value


def resolve_load_workers(requested: int | str | None = None) -> tuple[int, str]:
    """Return ``(worker_count, source)`` for structure parsing.

    ``worker_count`` is always >= 1 and never exceeds the available cores.
    """
    cores = available_cpu_count()
    if requested is not None:
        value, source = _parse_worker_value(requested, source="--load-workers"), "--load-workers"
    elif os.environ.get(LOAD_WORKERS_ENV, "").strip():
        value, source = _parse_worker_value(os.environ[LOAD_WORKERS_ENV], source=LOAD_WORKERS_ENV), LOAD_WORKERS_ENV
    else:
        value, source = 0, "all available cores"
        config_path = runtime_config_path()
        if config_path.is_file():
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(f"Could not read machine runtime config {config_path}: {exc}") from exc
            if isinstance(payload, dict) and payload.get("load_workers") is not None:
                value = _parse_worker_value(payload["load_workers"], source=str(config_path))
                source = str(config_path)
    workers = cores if value == 0 else min(value, cores)
    return max(1, workers), source


def _choose_start_method() -> str:
    """Prefer ``fork`` (cheap, Linux) unless CUDA is already initialised in this process.

    Forking after CUDA initialisation is unsafe, so fall back to ``spawn``, which also
    works on macOS and Windows. ``DEEPMZYME_LOAD_START_METHOD`` overrides the choice.
    """
    override = os.environ.get(START_METHOD_ENV, "").strip()
    if override:
        return override
    if "fork" not in multiprocessing.get_all_start_methods():
        return "spawn"
    try:
        if torch.cuda.is_initialized():
            return "spawn"
    except Exception:  # CUDA probing failed: parsing is CPU-only anyway
        pass
    return "fork"


def resolve_parse_cache_dir(load_kwargs: dict[str, Any], metal_label_scheme: str) -> Path | None:
    """Cache directory for these exact load settings, or ``None`` when caching is off."""
    root = os.environ.get(PARSE_CACHE_ENV, "").strip()
    if not root:
        return None
    digest = hashlib.sha256()
    for source in sorted(_SRC_ROOT.rglob("*.py")):
        digest.update(str(source.relative_to(_SRC_ROOT)).encode())
        digest.update(source.read_bytes())
    digest.update(f"scheme={metal_label_scheme};".encode())
    for key in sorted(load_kwargs):
        value = load_kwargs[key]
        if isinstance(value, dict):
            value = sorted(repr(item) for item in value.items())
        digest.update(f"{key}={value!r};".encode())
    return Path(root).expanduser() / digest.hexdigest()[:20]


_WORKER_LOAD_KWARGS: dict[str, Any] | None = None
_WORKER_CACHE_DIR: Path | None = None


def _init_worker(load_kwargs: dict[str, Any], metal_label_scheme: str, cache_dir: Path | None) -> None:
    from training.access_guard import install_guard_from_environment

    install_guard_from_environment()
    global _WORKER_LOAD_KWARGS, _WORKER_CACHE_DIR
    _WORKER_LOAD_KWARGS = load_kwargs
    _WORKER_CACHE_DIR = cache_dir

    from label_schemes import configure_active_metal_label_scheme

    configure_active_metal_label_scheme(metal_label_scheme)
    # N workers x M intra-op threads would oversubscribe the machine.
    torch.set_num_threads(1)


def _rebuild_tensor(data: bytes, dtype_name: str, shape: tuple[int, ...], requires_grad: bool) -> torch.Tensor:
    dtype = getattr(torch, dtype_name)
    if data:
        tensor = torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)
    else:
        tensor = torch.empty(shape, dtype=dtype)
    return tensor.requires_grad_(True) if requires_grad else tensor


class _CompactTensorPickler(pickle.Pickler):
    """Pickle each CPU tensor as its raw viewed bytes plus dtype and shape.

    Two problems with torch's own tensor pickling, both measured on ion examples:
    - it ships a tensor's entire backing storage. Residue ESM embeddings are 960-float
      views into a per-chain matrix, so the matrix was repeated once per residue: 8
      examples holding 5 MB in memory pickled to 326 MB, and 160 structures exhausted a
      15 GB workstation;
    - it serialises every storage through ``torch.save``. Examples hold thousands of tiny
      tensors, so the parent spent ~40% of the parse time just unpickling, serially,
      which capped the speedup near 2x regardless of worker count.
    Values, dtype, shape and ``requires_grad`` are preserved exactly.
    """

    def reducer_override(self, obj: Any) -> Any:
        if (
            type(obj) is torch.Tensor
            and obj.layout == torch.strided
            and obj.device.type == "cpu"
            and not obj.is_quantized
        ):
            flat = obj.detach().contiguous().reshape(-1)
            data = flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b""
            dtype_name = str(obj.dtype).removeprefix("torch.")
            return _rebuild_tensor, (data, dtype_name, tuple(obj.shape), obj.requires_grad)
        return NotImplemented


def _load_serialized(
    structure_path: Path, load_kwargs: dict[str, Any], cache_dir: Path | None
) -> tuple[str, bytes | str]:
    """Parse one structure (or read its cache entry) and return it as compact pickled bytes."""
    cache_file = None
    if cache_dir is not None:
        content_hash = hashlib.sha256(structure_path.read_bytes()).hexdigest()[:20]
        cache_file = cache_dir / f"{structure_path.stem}__{content_hash}.pkl"
        try:
            return "ok", cache_file.read_bytes()
        except FileNotFoundError:
            pass
    try:
        result = load_structure_pockets(structure_path=structure_path, **load_kwargs)
    except StructureLoadError as exc:
        return "invalid", str(exc)
    buffer = io.BytesIO()
    _CompactTensorPickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(result)
    payload = buffer.getvalue()
    if cache_file is not None:
        # Write-then-rename: a crash mid-write never leaves a truncated entry behind.
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        partial = cache_file.with_name(f"{cache_file.name}.{os.getpid()}.partial")
        partial.write_bytes(payload)
        os.replace(partial, cache_file)
    return "ok", payload


def _load_in_worker(structure_path: Path) -> tuple[str, bytes | str]:
    assert _WORKER_LOAD_KWARGS is not None, "parallel loading worker was not initialised"
    return _load_serialized(structure_path, _WORKER_LOAD_KWARGS, _WORKER_CACHE_DIR)


def iter_structure_load_results(
    structure_files: list[Path],
    load_kwargs: dict[str, Any],
    *,
    workers: int,
) -> Iterator[tuple[Path, tuple[str, Any]]]:
    """Yield ``(structure_path, outcome)`` in input order.

    ``outcome`` is ``("ok", (pockets, fallbacks, skipped))`` or
    ``("invalid", StructureLoadError)``. Any other exception propagates, as it would
    from the serial path.
    """
    import label_schemes

    # The scheme the parent is actually using right now (the module global), not a default.
    metal_label_scheme = label_schemes.ACTIVE_METAL_LABEL_SCHEME
    cache_dir = resolve_parse_cache_dir(load_kwargs, metal_label_scheme)
    if cache_dir is not None:
        print(f"[LOAD] parse cache: {cache_dir}", flush=True)

    if workers <= 1 or len(structure_files) < MIN_STRUCTURES_FOR_PARALLEL:
        for structure_path in structure_files:
            if cache_dir is not None:
                status, payload = _load_serialized(structure_path, load_kwargs, cache_dir)
                outcome = pickle.loads(payload) if status == "ok" else StructureLoadError(payload)
                yield structure_path, (status, outcome)
                continue
            try:
                yield structure_path, ("ok", load_structure_pockets(structure_path=structure_path, **load_kwargs))
            except StructureLoadError as exc:
                yield structure_path, ("invalid", exc)
        return

    context = multiprocessing.get_context(_choose_start_method())
    # Many small chunks keep workers evenly busy despite very uneven structure sizes,
    # while amortising inter-process overhead.
    chunksize = max(1, min(16, len(structure_files) // (workers * 8)))
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(load_kwargs, metal_label_scheme, cache_dir),
    ) as executor:
        for structure_path, (status, payload) in zip(
            structure_files,
            executor.map(_load_in_worker, structure_files, chunksize=chunksize),
        ):
            if status == "ok":
                yield structure_path, ("ok", pickle.loads(payload))
            else:
                yield structure_path, ("invalid", StructureLoadError(payload))
