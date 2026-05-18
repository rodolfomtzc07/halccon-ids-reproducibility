# 05_scripts/run_experiment.py (este es el entrypoint limpio del pipeline)
from pathlib import Path
import argparse

from src.experiment.run_experiment import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run a reproducible HALCCON experiment.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)

    run_dir = run_experiment(config_path)
    print(f"Run directory created at: {run_dir}")


if __name__ == "__main__":
    main()