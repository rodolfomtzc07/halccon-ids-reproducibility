#04_src/training/seed.py Este archivo es clave para reproducibilidad científica:
#evita variaciones entre corridas
#asegura consistencia en GPU
#controla randomness en DataLoader
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True, benchmark: bool = False) -> None:
    """
    Fija la semilla para reproducibilidad completa en:
    - Python
    - NumPy
    - PyTorch (CPU y CUDA)
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = benchmark


def seed_worker(worker_id: int):
    """
    Para DataLoader (workers)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)