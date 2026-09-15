from __future__ import annotations

from training.config import parse_args


def main() -> None:
    config = parse_args()
    from training.run import run_training

    run_training(config)


if __name__ == "__main__":
    main()
