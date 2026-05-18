from pathlib import Path
import argparse

from src.data.preprocess import preprocess_variant
from src.utils.io import read_yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Litnet data into tensor artifacts for a specific encoding variant."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to data YAML config file.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        help="Encoding variant name defined in data config (e.g. catboost_label, label_encoding).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)

    data_config = read_yaml(config_path)

    if "encoding_variants" not in data_config:
        raise KeyError("El YAML no contiene la clave 'encoding_variants'.")

    if args.variant not in data_config["encoding_variants"]:
        available = list(data_config["encoding_variants"].keys())
        raise KeyError(
            f"Variant '{args.variant}' no encontrada en el YAML. "
            f"Variantes disponibles: {available}"
        )

    output_dir = preprocess_variant(data_config, args.variant)

    print("\n[OK] Variant preprocessed successfully")
    print(f"[OK] Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()