# 04_src/utils/paths.py Este archivo define las rutas principales del proyecto y funciones para crear directorios de forma consistente.
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = PROJECT_ROOT / "01_configs"
DATA_DIR = PROJECT_ROOT / "02_data"
NOTEBOOKS_DIR = PROJECT_ROOT / "03_notebook"
SRC_DIR = PROJECT_ROOT / "04_src"
SCRIPTS_DIR = PROJECT_ROOT / "05_scripts"
EXPERIMENTS_DIR = PROJECT_ROOT / "06_experiments"
RUNS_DIR = PROJECT_ROOT / "07_runs"
REPORTS_DIR = PROJECT_ROOT / "08_reports"
TESTS_DIR = PROJECT_ROOT / "09_tests"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_subdirs(dataset_name: str) -> dict:
    dataset_root = DATA_DIR
    return {
        "raw": dataset_root / "raw" / dataset_name,
        "interim": dataset_root / "interim" / dataset_name,
        "processed": dataset_root / "processed" / dataset_name,
        "splits": dataset_root / "splits" / dataset_name,
    }


def get_run_dir(experiment_name: str) -> Path:
    run_dir = RUNS_DIR / experiment_name
    return ensure_dir(run_dir)


def get_report_dir(experiment_name: str) -> Path:
    report_dir = REPORTS_DIR / experiment_name
    return ensure_dir(report_dir)