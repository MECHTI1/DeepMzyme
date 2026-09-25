from __future__ import annotations

from training.access_guard import install_guard_from_environment
from training.config import parse_args


def main() -> None:
    # Development-only campaigns pass held-out roots through the environment.
    install_guard_from_environment()
    config = parse_args()
    from training.run import run_training

    run_training(config)


if __name__ == "__main__":
    main()
