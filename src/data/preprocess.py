from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.data.encoders import (
    encode_full_dataset_variant,
    encode_split_variant,
    fit_label_encoder,
    save_encoder_object,
    save_feature_list,
    save_label_encoder,
    save_label_mapping,
    transform_labels,
)
from src.utils.io import write_json


def load_source_csv(source_csv: str | Path) -> pd.DataFrame:
    return pd.read_csv(source_csv)


def load_split_csvs(splits_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits_dir = Path(splits_dir)

    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    return train_df, val_df, test_df


def maybe_build_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replica la lógica del notebook oficial:
    crea timestamp y timestamp2 si existen los componentes necesarios.
    """
    df = df.copy()

    ts_cols = ["ts_year", "ts_month", "ts_day", "ts_hour", "ts_min", "ts_second"]
    te_cols = ["te_year", "te_month", "te_day", "te_hour", "te_min", "te_second"]

    if all(c in df.columns for c in ts_cols) and "timestamp" not in df.columns:
        df["timestamp"] = (
            df["ts_year"].astype(str) + "-"
            + df["ts_month"].astype(str).str.pad(width=2, fillchar="0") + "-"
            + df["ts_day"].astype(str).str.pad(width=2, fillchar="0") + " "
            + df["ts_hour"].astype(str).str.pad(width=2, fillchar="0") + ":"
            + df["ts_min"].astype(str).str.pad(width=2, fillchar="0") + ":"
            + df["ts_second"].astype(str).str.pad(width=2, fillchar="0")
        )

    if all(c in df.columns for c in te_cols) and "timestamp2" not in df.columns:
        df["timestamp2"] = (
            df["te_year"].astype(str) + "-"
            + df["te_month"].astype(str).str.pad(width=2, fillchar="0") + "-"
            + df["te_day"].astype(str).str.pad(width=2, fillchar="0") + " "
            + df["te_hour"].astype(str).str.pad(width=2, fillchar="0") + ":"
            + df["te_min"].astype(str).str.pad(width=2, fillchar="0") + ":"
            + df["te_second"].astype(str).str.pad(width=2, fillchar="0")
        )

    return df


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    drop_columns: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    drop_columns = drop_columns or []

    cols_to_drop = [c for c in drop_columns if c in df.columns and c != target_column]

    X = df.drop(columns=cols_to_drop + [target_column])
    y = df[target_column].copy()

    return X, y


def validate_numeric(df: pd.DataFrame, name: str = "df") -> None:
    bad_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    if bad_cols:
        print(f"\n[ERROR] Columnas no numéricas en {name}:")
        for col in bad_cols:
            print(f" - {col} ({df[col].dtype})")

        raise TypeError(f"{name} contiene columnas no numéricas: {bad_cols}")


def dataframe_to_tensor(X: pd.DataFrame) -> torch.Tensor:
    validate_numeric(X, "X before tensor")
    return torch.from_numpy(X.to_numpy(dtype="float32"))


def series_to_tensor(y: pd.Series) -> torch.Tensor:
    return torch.tensor(y.values, dtype=torch.long)


def save_tensor_artifacts(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(X_train, output_dir / "X_train.pt")
    torch.save(y_train, output_dir / "y_train.pt")
    torch.save(X_val, output_dir / "X_val.pt")
    torch.save(y_val, output_dir / "y_val.pt")
    torch.save(X_test, output_dir / "X_test.pt")
    torch.save(y_test, output_dir / "y_test.pt")


def minmax_fit(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    df_min = df.min()
    df_max = df.max()
    return df_min, df_max


def minmax_transform(df: pd.DataFrame, df_min: pd.Series, df_max: pd.Series) -> pd.DataFrame:
    denom = (df_max - df_min).replace(0, 1.0)
    return (df - df_min) / denom


def apply_normalization(
    X_train_enc: pd.DataFrame,
    X_val_enc: pd.DataFrame,
    X_test_enc: pd.DataFrame,
    preprocess_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preprocess_mode = preprocess_mode.lower()

    if preprocess_mode == "clean":
        train_min, train_max = minmax_fit(X_train_enc)
        X_train_norm = minmax_transform(X_train_enc, train_min, train_max)
        X_val_norm = minmax_transform(X_val_enc, train_min, train_max)
        X_test_norm = minmax_transform(X_test_enc, train_min, train_max)
        return X_train_norm, X_val_norm, X_test_norm

    if preprocess_mode == "ieee_exact":
        all_encoded = pd.concat([X_train_enc, X_val_enc, X_test_enc], axis=0)
        global_min, global_max = minmax_fit(all_encoded)

        X_train_norm = minmax_transform(X_train_enc, global_min, global_max)
        X_val_norm = minmax_transform(X_val_enc, global_min, global_max)
        X_test_norm = minmax_transform(X_test_enc, global_min, global_max)
        return X_train_norm, X_val_norm, X_test_norm

    raise ValueError(
        f"preprocess_mode no soportado: {preprocess_mode}. Usa 'ieee_exact' o 'clean'."
    )


def detect_categorical_columns(df: pd.DataFrame, configured_columns: List[str] | None) -> List[str]:
    if configured_columns:
        return [c for c in configured_columns if c in df.columns]
    return df.select_dtypes(include=["object", "string", "category"]).columns.tolist()


def split_ieee_exact(
    X: pd.DataFrame,
    y: pd.Series,
    split_cfg: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Replica el notebook oficial:
    - 70/30 estratificado
    - luego 50/50 del temporal => 15/15
    """
    random_seed = split_cfg.get("random_seed", 42)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=random_seed,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_seed,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_split_csvs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    splits_dir: str | Path,
) -> None:
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)


def preprocess_variant(data_config: Dict, variant_name: str) -> Path:
    paths_cfg = data_config["paths"]
    target_column = data_config["target"]["column"]
    features_cfg = data_config["features"]
    artifacts_cfg = data_config["artifacts"]
    variant_cfg = data_config["encoding_variants"][variant_name]
    split_cfg = data_config.get("split", {})

    preprocess_mode = data_config.get("preprocess_mode", "clean").lower()

    processed_root = Path(paths_cfg["processed_dir"])
    variant_dir = processed_root / f"{variant_name}__{preprocess_mode}"
    variant_dir.mkdir(parents=True, exist_ok=True)

    drop_columns = features_cfg.get("drop_columns", [])
    configured_categorical_columns = features_cfg.get("categorical_columns", [])

    print("\n[INFO] preprocess_mode:")
    print(preprocess_mode)

    # ------------------------------------------------------------
    # IEEE EXACT: source_csv -> encode global -> normalize global -> split
    # ------------------------------------------------------------
    if preprocess_mode == "ieee_exact":
        source_csv = paths_cfg["source_csv"]
        splits_dir = Path(paths_cfg["splits_dir"])

        df = load_source_csv(source_csv)
        df = maybe_build_timestamps(df)

        X_full, y_full = split_features_target(df, target_column, drop_columns)
        categorical_columns = detect_categorical_columns(X_full, configured_categorical_columns)

        print("\n[INFO] Columnas categóricas detectadas:")
        print(categorical_columns)

        # Baseline notebook: LabelEncoder global antes del split
        label_encoder = fit_label_encoder(y_full)
        y_full_enc = transform_labels(y_full, label_encoder)

        # Baseline notebook: encoding global antes del split
        X_full_enc, feature_encoder = encode_full_dataset_variant(
            X=X_full,
            y_encoded=y_full_enc,
            variant_config=variant_cfg,
            categorical_columns=categorical_columns,
        )

        validate_numeric(X_full_enc, "X_full_enc")

        # Baseline notebook: normalización global antes del split
        global_min, global_max = minmax_fit(X_full_enc)
        X_full_norm = minmax_transform(X_full_enc, global_min, global_max)

        # Baseline notebook: split al final
        X_train_enc, X_val_enc, X_test_enc, y_train_enc, y_val_enc, y_test_enc = split_ieee_exact(
            X_full_norm,
            y_full_enc,
            split_cfg,
        )

        # Guardar CSVs del split si se solicitó
        if split_cfg.get("save_split_csv", False):
            train_df = X_train_enc.copy()
            train_df[target_column] = y_train_enc.values

            val_df = X_val_enc.copy()
            val_df[target_column] = y_val_enc.values

            test_df = X_test_enc.copy()
            test_df[target_column] = y_test_enc.values

            save_split_csvs(train_df, val_df, test_df, splits_dir)

    # ------------------------------------------------------------
    # CLEAN: split first -> fit encoders on train -> transform val/test
    # ------------------------------------------------------------
    elif preprocess_mode == "clean":
        splits_dir = Path(paths_cfg["splits_dir"])
        train_df, val_df, test_df = load_split_csvs(splits_dir)

        X_train, y_train = split_features_target(train_df, target_column, drop_columns)
        X_val, y_val = split_features_target(val_df, target_column, drop_columns)
        X_test, y_test = split_features_target(test_df, target_column, drop_columns)

        categorical_columns = detect_categorical_columns(X_train, configured_categorical_columns)

        print("\n[INFO] Columnas categóricas detectadas:")
        print(categorical_columns)

        # Clean: LabelEncoder fit solo en train
        label_encoder = fit_label_encoder(y_train)
        y_train_enc = transform_labels(y_train, label_encoder)
        y_val_enc = transform_labels(y_val, label_encoder)
        y_test_enc = transform_labels(y_test, label_encoder)

        X_train_enc, X_val_enc, X_test_enc, feature_encoder = encode_split_variant(
            X_train=X_train,
            y_train_encoded=y_train_enc,
            X_val=X_val,
            X_test=X_test,
            variant_config=variant_cfg,
            categorical_columns=categorical_columns,
        )

        X_train_enc, X_val_enc, X_test_enc = apply_normalization(
            X_train_enc=X_train_enc,
            X_val_enc=X_val_enc,
            X_test_enc=X_test_enc,
            preprocess_mode=preprocess_mode,
        )

        validate_numeric(X_train_enc, "X_train_enc")
        validate_numeric(X_val_enc, "X_val_enc")
        validate_numeric(X_test_enc, "X_test_enc")

    else:
        raise ValueError(
            f"preprocess_mode no soportado: {preprocess_mode}. Usa 'ieee_exact' o 'clean'."
        )

    # ------------------------------------------------------------
    # Tensorización y artefactos comunes
    # ------------------------------------------------------------
    X_train_tensor = dataframe_to_tensor(X_train_enc)
    X_val_tensor = dataframe_to_tensor(X_val_enc)
    X_test_tensor = dataframe_to_tensor(X_test_enc)

    y_train_tensor = series_to_tensor(y_train_enc)
    y_val_tensor = series_to_tensor(y_val_enc)
    y_test_tensor = series_to_tensor(y_test_enc)

    save_tensor_artifacts(
        X_train=X_train_tensor,
        y_train=y_train_tensor,
        X_val=X_val_tensor,
        y_val=y_val_tensor,
        X_test=X_test_tensor,
        y_test=y_test_tensor,
        output_dir=variant_dir,
    )

    if artifacts_cfg.get("save_feature_list", False):
        save_feature_list(list(X_train_enc.columns), variant_dir / "feature_list.csv")

    if artifacts_cfg.get("save_label_mapping", False):
        save_label_mapping(label_encoder, variant_dir / "label_mapping.csv")

    if artifacts_cfg.get("save_encoder_objects", False):
        save_label_encoder(label_encoder, variant_dir / "label_encoder.joblib")
        save_encoder_object(feature_encoder, variant_dir / "feature_encoder.joblib")

    write_json(
        {
            "variant_name": variant_name,
            "preprocess_mode": preprocess_mode,
            "n_train": int(len(X_train_enc)),
            "n_val": int(len(X_val_enc)),
            "n_test": int(len(X_test_enc)),
            "n_features": int(X_train_enc.shape[1]),
            "categorical_encoding": variant_cfg["categorical_encoding"],
            "target_encoding_for_labels": variant_cfg["target_encoding_for_labels"],
            "categorical_columns_detected": categorical_columns,
        },
        variant_dir / "preprocess_metadata.json",
    )

    return variant_dir