# Environment contracts

DeepMzyme deliberately separates the reproducible Linux development/test
environment from the managed Google Colab GPU environment.

## Linux x86_64 development and CPU tests

The canonical contract is Python 3.12 plus [`../uv.lock`](../uv.lock). It pins
the CPU PyTorch wheel and every resolved dependency. From a fresh clone:

```bash
uv python install 3.12
uv sync --frozen
uv run python -c "import sys, torch; print(sys.version); print(torch.__version__)"
uv run pytest
uv run python tests/smoke_checks.py
```

The default environment includes the test and reporting groups. Add ESMC only
when embedding generation is needed:

```bash
uv sync --frozen --group esm
```

This CPU contract is suitable for source validation, CLI checks, tests, and
small CPU smoke work. It is not a promise that a local CUDA driver can use the
locked CPU PyTorch wheel.

## Existing workstation and optional local CUDA

The existing `/home/mechti/miniconda3/envs/DeepMzyme` prefix is a real execution
path, but it is not the canonical lock and must not be described as a fresh
reconstruction of it. Record its actual runtime metadata in every run.

A separately locked local CUDA environment is not supported yet. To add one,
create a new hardware-specific contract and validate its PyTorch wheel against
the intended driver and GPU architecture; do not change the CPU or Colab
contracts in place.

## Colab overlay

[`colab-overlay.txt`](colab-overlay.txt) contains only packages that may be
installed over an importable stock Colab PyTorch stack. It intentionally omits
`torch`. The notebook verifies the stock PyTorch/CUDA architecture before and
after applying this overlay. Never install the Linux CPU lock in Colab.

## Compatibility file

[`../src/requirements.txt`](../src/requirements.txt) remains for older local
commands. It is pinned, but it is not the transitive lock; use `uv.lock` for a
fresh reproducible environment and `colab-overlay.txt` in Colab.

