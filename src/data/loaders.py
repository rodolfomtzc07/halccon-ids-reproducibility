from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset


def load_tensor(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el tensor: {path}")
    return torch.load(path)


def build_tensor_dataset(
    X_path: str | Path,
    y_path: str | Path,
) -> TensorDataset:
    X = load_tensor(X_path)
    y = load_tensor(y_path)

    return TensorDataset(X.float(), y.long())


def create_dataloader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def build_dataloaders_from_variant(
    processed_dir: str | Path,
    variant_name: str,
    train_config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    variant_dir = Path(processed_dir) / variant_name

    train_dataset = build_tensor_dataset(
        variant_dir / "X_train.pt",
        variant_dir / "y_train.pt",
    )

    val_dataset = build_tensor_dataset(
        variant_dir / "X_val.pt",
        variant_dir / "y_val.pt",
    )

    test_dataset = build_tensor_dataset(
        variant_dir / "X_test.pt",
        variant_dir / "y_test.pt",
    )

    dl_cfg = train_config["dataloader"]

    train_loader = create_dataloader(
        train_dataset,
        batch_size=dl_cfg["batch_size"],
        shuffle=True,
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
        drop_last=dl_cfg["drop_last"],
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=dl_cfg["batch_size"],
        shuffle=False,
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
        drop_last=False,
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=dl_cfg["batch_size"],
        shuffle=False,
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
        drop_last=False,
    )

    return train_loader, val_loader, test_loader