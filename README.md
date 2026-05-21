# halccon-ids-reproducibility

Reproducibility code and experiments for H.A.L.C.CO.N, a hierarchical attention-based CNN for real-world multiclass intrusion detection on LITNET-2020.

## Paper Information

**Title:** Hierarchical Attention-Based Convolutional Neural Network Model for Intrusion Detection  
**Manuscript ID:** 10406

## Authors

- **Rodolfo Martínez Cadena**  
  División Académica de Ciencias y Tecnologías de la Información (DACYTI), Universidad Juárez Autónoma de Tabasco, Tabasco, Mexico

- **José Adán Hernández-Nolasco**  
  División Académica de Ciencias y Tecnologías de la Información (DACYTI), Universidad Juárez Autónoma de Tabasco, Tabasco, Mexico

- **Noel Zacarias-Morales**  
  División Académica de Ciencias y Tecnologías de la Información (DACYTI), Universidad Juárez Autónoma de Tabasco, Tabasco, Mexico

## Overview

This repository contains the code, configuration files, preprocessing artifacts, split definitions, and experimental outputs required to reproduce the main results reported in the final accepted version of the paper.

The repository supports the reproduction of the following components:

- Main multiclass evaluation on **LITNET-2020**
- **Stratified 5-fold cross-validation**
- Comparison with **classical machine learning baselines**
- **Ablation study**
- **EQLv2 hyperparameter sensitivity analysis**
- **Cross-dataset evaluation on UNSW-NB15**

## Repository Structure

### `01_configs/`
Configuration files for the reproducibility workflow.

- `01_configs/data/`  
  Dataset-specific configuration files for **LITNET-2020**, **UNSW-NB15**, and other supported datasets.
- `01_configs/model/`  
  Model architecture definitions and model-specific YAML configuration files, including H.A.L.C.CO.N and CANET-related settings.
- `01_configs/train/`  
  Training configuration files, including default settings and experiment-specific training definitions.
- `01_configs/hpo/`  
  Hyperparameter optimization settings, including the Optuna configuration used in the study.

### `02_data/`
Data organization for raw, processed, and split metadata.

- `02_data/raw/`  
  Raw input data and source dataset files, including LITNET-2020 and UNSW-NB15 source files.
- `02_data/processed/`  
  Processed tensors, encoders, label mappings, feature lists, preprocessing metadata, and scaled datasets generated during preprocessing.
- `02_data/split/`  
  Train, validation, and test split definitions for LITNET-2020 and UNSW-NB15, including split summaries and class-distribution files.

### `03_notebook/`
Archived notebooks used during exploratory development, preprocessing audits, and intermediate experimentation. These notebooks are included for reference only and are not the primary reproducibility workflow for the final accepted manuscript.

### `05_scripts/`
Main executable scripts for preprocessing, split creation, experiment execution, and model testing.

Key scripts include:
- `create_splits.py`  
  Generates train, validation, and test partitions and corresponding metadata.
- `preprocess_variant.py`  
  Runs preprocessing variants, including encoding and tensor preparation.
- `run_experiment.py`  
  Main script for launching configured experiments.
- `halccontest.py`  
  Testing/evaluation script for H.A.L.C.CO.N variants.

### `06_experiments/`
Saved experimental outputs corresponding to the results reported in the paper.

This folder includes:
- trained checkpoints (`best.ckpt`, `last.ckpt`)
- summary files (`best_summary.json`, `results_summary.json`)
- training histories (`train_history.csv`, `history.csv`)
- confusion matrices
- per-class metrics
- classification reports
- Optuna trial outputs
- 5-fold cross-validation outputs
- ablation experiment results
- UNSW-NB15 cross-dataset evaluation outputs

### `07_runs/`
Execution run artifacts and intermediate run folders.

### `08_reports/`
Report-oriented outputs and supporting documentation for experiments.

### `09_tests/`
Testing-related files for repository validation and experimental checks.

## Main Reproducibility Workflow

The official reproducibility workflow is script-based and aligned with the final accepted manuscript.

A typical execution order is:

1. Prepare dataset paths and raw files in `02_data/raw/`
2. Generate dataset splits using `05_scripts/create_splits.py`
3. Run preprocessing using `05_scripts/preprocess_variant.py`
4. Execute the main LITNET-2020 experiment using `05_scripts/run_experiment.py`
5. Evaluate the main model outputs
6. Reproduce the ablation experiments from the corresponding experiment folders in `06_experiments/`
7. Reproduce the stratified 5-fold cross-validation results
8. Reproduce the EQLv2 hyperparameter optimization and sensitivity analysis
9. Reproduce the cross-dataset evaluation on UNSW-NB15

## File-by-File Reproducibility Notes

### Main configuration files
- `01_configs/data/litnet.yaml`: configuration for the LITNET-2020 dataset
- `01_configs/data/unsw_nb15.yaml`: configuration for the UNSW-NB15 dataset
- `01_configs/model/halccon.yaml`: main H.A.L.C.CO.N architecture configuration
- `01_configs/model/canet.yaml`: CANET-related reference configuration
- `01_configs/train/default.yaml`: default training configuration
- `01_configs/train/exp_litnet_multiclass.yaml`: training configuration for the main LITNET multiclass experiment
- `01_configs/hpo/halccon_optuna.yaml`: Optuna-based hyperparameter search configuration

### Main scripts
- `05_scripts/create_splits.py`: generates train/validation/test splits and related summaries
- `05_scripts/preprocess_variant.py`: preprocesses data and produces encoded/scaled tensors and metadata
- `05_scripts/run_experiment.py`: executes experiment pipelines based on configuration files
- `05_scripts/halccontest.py`: evaluates trained models and generates metrics/reports

### Main processed data artifacts
Each processed-data folder may contain:
- `feature_encoder.joblib`
- `label_encoder.joblib`
- `feature_list.csv`
- `label_mapping.csv`
- `preprocess_metadata.json`
- `X_train.pt`, `X_val.pt`, `X_test.pt`
- `y_train.pt`, `y_val.pt`, `y_test.pt`

### Main experimental output artifacts
Each experiment folder may contain:
- `best.ckpt`
- `last.ckpt`
- `best_summary.json`
- `results_summary.json`
- `classification_report.csv`
- `confusion_matrix.csv`
- `per_class_metrics.csv`
- `train_history.csv`
- `history.csv`

## Environment

- Python 3.10
- PyTorch 2.3.0+cu118

Additional dependencies are listed in `requirements.txt`.

## Data

The original full datasets are not redistributed in this repository.

### Required datasets
- **LITNET-2020**
- **UNSW-NB15** (required only for the cross-dataset evaluation)

### Included metadata
This repository includes:
- split summary files
- train/validation/test split definitions
- class-distribution summaries
- preprocessing metadata
- encoded feature metadata
- label mappings

These files are intended to facilitate reproducibility of the evaluation protocol.

## Reproducibility Notes

This repository is intended to document the workflow and artifacts associated with the final accepted version of the paper. The primary goal is to make the reported experiments easier to inspect, understand, and reproduce.

The notebooks in `03_notebook/` are retained as archived exploratory material, while the official reproducibility path is based on the configuration files, scripts, processed artifacts, and experiment outputs included in the repository.

## Contact

For questions regarding this repository or reproducibility details, please contact the authors through their institutional affiliation.
