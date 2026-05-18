# src/data/encoders.py Este archivo dejará lista la codificación para las variantes de ablación:
# label encoder para la variable objetivo
# opción de sin codificación categórica
# opción de CatBoost encoding para variables categóricas
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

try:
    from category_encoders import CatBoostEncoder
except ImportError:
    CatBoostEncoder = None


def fit_label_encoder(y: pd.Series) -> LabelEncoder:
    """
    Ajusta un LabelEncoder sobre la serie recibida.
    Nota:
    - En modo ieee_exact debe llamarse con y completo.
    - En modo clean debe llamarse con y_train.
    """
    encoder = LabelEncoder()
    encoder.fit(y)
    return encoder


def transform_labels(
    y: pd.Series,
    encoder: LabelEncoder,
) -> pd.Series:
    return pd.Series(encoder.transform(y), index=y.index, name=y.name)


def save_label_encoder(encoder: LabelEncoder, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, output_path)


def save_label_mapping(encoder: LabelEncoder, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mapping = pd.DataFrame({
        "class_name": encoder.classes_,
        "class_id": range(len(encoder.classes_)),
    })
    mapping.to_csv(output_path, index=False)


def save_feature_list(columns: List[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": columns}).to_csv(output_path, index=False)


def save_encoder_object(encoder: object, output_path: str | Path) -> None:
    if encoder is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, output_path)


def apply_no_categorical_encoding(
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, None]:
    return X.copy(), None


def apply_no_categorical_encoding_split(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, None]:
    return X_train.copy(), X_val.copy(), X_test.copy(), None


def label_encode_full_dataframe(
    X: pd.DataFrame,
    categorical_columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Para ieee_exact:
    ajusta el mapping sobre TODO el dataset antes del split.
    """
    X_enc = X.copy()
    encoders: Dict[str, Dict[str, int]] = {}

    for col in categorical_columns:
        values = X[col].astype(str)
        unique_values = sorted(values.unique().tolist())
        value_to_int = {value: idx for idx, value in enumerate(unique_values)}
        X_enc[col] = values.map(value_to_int).astype("int64")
        encoders[col] = value_to_int

    return X_enc, encoders


def label_encode_dataframe_split(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Para clean:
    ajusta mapping solo con train y aplica a val/test.
    unseen -> -1
    """
    X_train_enc = X_train.copy()
    X_val_enc = X_val.copy()
    X_test_enc = X_test.copy()

    encoders: Dict[str, Dict[str, int]] = {}

    for col in categorical_columns:
        train_values = X_train[col].astype(str)
        unique_train = sorted(train_values.unique().tolist())
        value_to_int = {value: idx for idx, value in enumerate(unique_train)}

        X_train_enc[col] = train_values.map(value_to_int).astype("int64")
        X_val_enc[col] = X_val[col].astype(str).map(value_to_int).fillna(-1).astype("int64")
        X_test_enc[col] = X_test[col].astype(str).map(value_to_int).fillna(-1).astype("int64")

        encoders[col] = value_to_int

    return X_train_enc, X_val_enc, X_test_enc, encoders


def apply_catboost_encoding_full(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_columns: List[str],
) -> Tuple[pd.DataFrame, object]:
    """
    Para ieee_exact:
    ajusta CatBoostEncoder sobre TODO el dataset antes del split.
    """
    if CatBoostEncoder is None:
        raise ImportError(
            "category_encoders no está instalado. Instálalo para usar CatBoostEncoder."
        )

    encoder = CatBoostEncoder(cols=categorical_columns)
    X_enc = encoder.fit_transform(X, y)

    return X_enc, encoder


def apply_catboost_encoding_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """
    Para clean:
    ajusta CatBoostEncoder solo con train y transforma val/test.
    """
    if CatBoostEncoder is None:
        raise ImportError(
            "category_encoders no está instalado. Instálalo para usar CatBoostEncoder."
        )

    encoder = CatBoostEncoder(cols=categorical_columns)

    X_train_enc = encoder.fit_transform(X_train, y_train)
    X_val_enc = encoder.transform(X_val)
    X_test_enc = encoder.transform(X_test)

    return X_train_enc, X_val_enc, X_test_enc, encoder


def encode_full_dataset_variant(
    X: pd.DataFrame,
    y_encoded: pd.Series,
    variant_config: Dict,
    categorical_columns: List[str],
) -> Tuple[pd.DataFrame, object]:
    """
    Entry point para ieee_exact.
    """
    encoding_type = variant_config["categorical_encoding"]

    if encoding_type == "none":
        return apply_no_categorical_encoding(X)

    if encoding_type == "label":
        return label_encode_full_dataframe(
            X=X,
            categorical_columns=categorical_columns,
        )

    if encoding_type == "catboost":
        return apply_catboost_encoding_full(
            X=X,
            y=y_encoded,
            categorical_columns=categorical_columns,
        )

    raise ValueError(f"Codificación categórica no soportada: {encoding_type}")


def encode_split_variant(
    X_train: pd.DataFrame,
    y_train_encoded: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    variant_config: Dict,
    categorical_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """
    Entry point para clean.
    """
    encoding_type = variant_config["categorical_encoding"]

    if encoding_type == "none":
        return apply_no_categorical_encoding_split(X_train, X_val, X_test)

    if encoding_type == "label":
        return label_encode_dataframe_split(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            categorical_columns=categorical_columns,
        )

    if encoding_type == "catboost":
        return apply_catboost_encoding_split(
            X_train=X_train,
            y_train=y_train_encoded,
            X_val=X_val,
            X_test=X_test,
            categorical_columns=categorical_columns,
        )

    raise ValueError(f"Codificación categórica no soportada: {encoding_type}")


# Compatibilidad hacia atrás con el nombre anterior
def encode_feature_variant(
    X_train: pd.DataFrame,
    y_train_encoded: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    variant_config: Dict,
    categorical_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    return encode_split_variant(
        X_train=X_train,
        y_train_encoded=y_train_encoded,
        X_val=X_val,
        X_test=X_test,
        variant_config=variant_config,
        categorical_columns=categorical_columns,
    )