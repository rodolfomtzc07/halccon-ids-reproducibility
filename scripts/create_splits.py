# 05_scripts/create_splits.py
from pathlib import Path
import argparse

from src.data.splitters import create_and_save_splits
from src.utils.io import read_yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create reproducible dataset splits."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to data YAML config file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)

    data_config = read_yaml(config_path)
    outputs = create_and_save_splits(data_config)

    print("Splits created successfully:")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
    