# src/data/splitters.py Este archivo hará el split estratificado reproducible a partir del CSV oficial de la muestra 10%.
#leer la muestra oficial 10%
#partirla de forma estratificada
#guardar train.csv, val.csv, test.csv
#guardar evidencia de distribución por clase
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import write_json


def load_source_dataframe(source_csv: str | Path) -> pd.DataFrame:
    source_csv = Path(source_csv)
    if not source_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo fuente: {source_csv}")
    return pd.read_csv(source_csv)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "", regex=False)
    return df


def clean_target_labels(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df = df.copy()
    if target_column not in df.columns:
        raise KeyError(f"La columna target '{target_column}' no existe en el DataFrame.")
    df[target_column] = df[target_column].astype(str).str.strip()
    return df


def stratified_train_val_test_split(
    df: pd.DataFrame,
    target_column: str,
    train_size: float,
    val_size: float,
    test_size: float,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train_size + val_size + test_size debe sumar 1.0")

    if target_column not in df.columns:
        raise KeyError(f"La columna target '{target_column}' no existe en el DataFrame.")

    df = clean_column_names(df)
    df = clean_target_labels(df, target_column)

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        stratify=df[target_column],
        random_state=random_seed,
    )

    val_relative = val_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_relative),
        stratify=temp_df[target_column],
        random_state=random_seed,
    )

    return train_df, val_df, test_df


def official_train_test_with_internal_val_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    train_size_within_official_train: float,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (0.0 < train_size_within_official_train < 1.0):
        raise ValueError("train_size_within_official_train debe estar en el intervalo (0, 1).")

    train_df = clean_column_names(train_df)
    test_df = clean_column_names(test_df)

    train_df = clean_target_labels(train_df, target_column)
    test_df = clean_target_labels(test_df, target_column)

    if target_column not in train_df.columns:
        raise KeyError(f"La columna target '{target_column}' no existe en el training-set oficial.")
    if target_column not in test_df.columns:
        raise KeyError(f"La columna target '{target_column}' no existe en el testing-set oficial.")

    train_part, val_part = train_test_split(
        train_df,
        test_size=(1.0 - train_size_within_official_train),
        stratify=train_df[target_column],
        random_state=random_seed,
    )

    return train_part, val_part, test_df


def class_distribution(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    counts = df[target_column].value_counts(dropna=False).sort_index()
    proportions = df[target_column].value_counts(normalize=True, dropna=False).sort_index()

    return pd.DataFrame({
        "class": counts.index,
        "count": counts.values,
        "proportion": proportions.values,
    })


def save_split_outputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    splits_dir: str | Path,
) -> Dict[str, Path]:
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    train_dist = class_distribution(train_df, target_column)
    val_dist = class_distribution(val_df, target_column)
    test_dist = class_distribution(test_df, target_column)

    train_dist.to_csv(splits_dir / "train_distribution.csv", index=False)
    val_dist.to_csv(splits_dir / "val_distribution.csv", index=False)
    test_dist.to_csv(splits_dir / "test_distribution.csv", index=False)

    write_json(
        {
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "target_column": target_column,
        },
        splits_dir / "split_summary.json",
    )

    return {
        "train_csv": train_csv,
        "val_csv": val_csv,
        "test_csv": test_csv,
    }


def create_and_save_stratified_splits(data_config: dict) -> Dict[str, Path]:
    source_csv = data_config["paths"]["source_csv"]
    splits_dir = data_config["paths"]["splits_dir"]
    target_column = data_config["target"]["column"]

    split_cfg = data_config["split"]

    df = load_source_dataframe(source_csv)

    train_df, val_df, test_df = stratified_train_val_test_split(
        df=df,
        target_column=target_column,
        train_size=split_cfg["train_size"],
        val_size=split_cfg["val_size"],
        test_size=split_cfg["test_size"],
        random_seed=split_cfg["random_seed"],
    )

    return save_split_outputs(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_column=target_column,
        splits_dir=splits_dir,
    )


def create_and_save_official_train_test_with_internal_val_splits(data_config: dict) -> Dict[str, Path]:
    source_train_csv = data_config["paths"]["source_train_csv"]
    source_test_csv = data_config["paths"]["source_test_csv"]
    splits_dir = data_config["paths"]["splits_dir"]
    target_column = data_config["target"]["column"]

    split_cfg = data_config["split"]

    train_df = load_source_dataframe(source_train_csv)
    test_df = load_source_dataframe(source_test_csv)

    train_part, val_part, test_part = official_train_test_with_internal_val_split(
        train_df=train_df,
        test_df=test_df,
        target_column=target_column,
        train_size_within_official_train=split_cfg["train_size_within_official_train"],
        random_seed=split_cfg["random_seed"],
    )

    return save_split_outputs(
        train_df=train_part,
        val_df=val_part,
        test_df=test_part,
        target_column=target_column,
        splits_dir=splits_dir,
    )


def create_and_save_splits(data_config: dict) -> Dict[str, Path]:
    strategy = data_config["split"]["strategy"]

    if strategy == "stratified":
        return create_and_save_stratified_splits(data_config)

    if strategy == "official_train_test_with_internal_val":
        return create_and_save_official_train_test_with_internal_val_splits(data_config)

    raise ValueError(f"Estrategia de split no soportada: {strategy}")

