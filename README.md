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

## Installing

The repository was developed and tested with the following environment:

- Python 3.10
- PyTorch 2.3.0+cu118

The environment specification is provided in:

- `environment.yml`

To create the environment with conda:

```bash
conda env create -f environment.yml
conda activate halccon
```
Alternatively, with micromamba:

```bash
micromamba env create -f environment.yml
micromamba activate halccon
```

## Repository Structure

### `configs/`

Configuration files for the reproducibility workflow.

* `configs/data/`
  Dataset-specific configuration files for LITNET-2020, UNSW-NB15, and other supported datasets.
* `configs/model/`
  Model architecture definitions and YAML configuration files, including H.A.L.C.CO.N and CANET-related settings.
* `configs/train/`
  Training configuration files, including default settings and experiment-specific definitions.
* `configs/hpo/`
  Hyperparameter optimization settings, including the Optuna configuration used in the study.


### `data/`

Data organization for raw, processed, and split metadata.

* `data/raw/`
  Raw input data and source dataset files, including LITNET-2020 and UNSW-NB15 source files.
* `data/processed/`
  Processed tensors, encoders, label mappings, feature lists, preprocessing metadata, and scaled datasets generated during preprocessing.
* `data/split/`
  Train, validation, and test split definitions for LITNET-2020 and UNSW-NB15, including split summaries and class-distribution files.
* `data/interim/`
  Intermediate data artifacts generated during processing.

### `notebook/`

Archived notebooks used during exploratory development, preprocessing audits, and intermediate experimentation. These notebooks are included for reference only and are not the primary reproducibility workflow for the final accepted manuscript.

### `scripts/`

Main executable scripts for split creation, preprocessing, experiment execution, and model testing.

### `experiments/`

Saved experimental outputs corresponding to the results reported in the paper.

This directory contains, depending on the experiment:

* trained checkpoints
* best and last model states
* result summaries
* classification reports
* confusion matrices
* per-class metrics
* training history files
* Optuna trial outputs
* 5-fold cross-validation outputs
* ablation experiment results
* UNSW-NB15 cross-dataset evaluation outputs


## Included Scripts

| Script / File                      | Description                                                                                                |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `scripts/create_splits.py`      | Generates train, validation, and test splits and their corresponding metadata.                             |
| `scripts/preprocess_variant.py` | Runs preprocessing, encoding, normalization, and tensor generation for the selected preprocessing variant. |
| `scripts/run_experiment.py`     | Executes the main configured experiments for H.A.L.C.CO.N variants.                                        |
| `scripts/halccontest.py`        | Evaluates trained models and generates result summaries, confusion matrices, and class-wise metrics.       |

## Main Configuration Files

| Configuration File                            | Description                                                       |
| --------------------------------------------- | ----------------------------------------------------------------- |
| `configs/data/litnet.yaml`                 | Dataset configuration for the main LITNET-2020 experiment.        |
| `configs/data/unsw_nb15.yaml`              | Dataset configuration for the UNSW-NB15 cross-dataset evaluation. |
| `configs/data/ciciot2023.yaml`             | Additional dataset configuration included in the project.         |
| `configs/model/halccon.yaml`               | Main H.A.L.C.CO.N architecture configuration.                     |
| `configs/model/canet.yaml`                 | CANET-related reference configuration.                            |
| `configs/model/architecture.py`            | Architecture-related implementation/configuration support.        |
| `configs/train/default.yaml`               | Default training configuration.                                   |
| `configs/train/exp_litnet_multiclass.yaml` | Training configuration for the main LITNET multiclass experiment. |
| `configs/hpo/halccon_optuna.yaml`          | Optuna-based hyperparameter optimization configuration.           |

## Executing

A typical execution order is:

1. Place the datasets in the expected local directories under `02_data/raw/`
2. Generate split metadata
3. Run preprocessing for the desired variant
4. Execute the main LITNET-2020 experiment
5. Evaluate the trained model
6. Reproduce the stratified 5-fold cross-validation results
7. Reproduce the ablation experiments
8. Reproduce the EQLv2 hyperparameter optimization and sensitivity analysis
9. Reproduce the cross-dataset evaluation on UNSW-NB15

Representative script usage:

```bash
python scripts/create_splits.py
python scripts/preprocess_variant.py
python scripts/run_experiment.py
python scripts/halccontest.py
```

## Data

The original full datasets are not redistributed in this repository.

### Required datasets

* **LITNET-2020**
* **UNSW-NB15** (required only for the cross-dataset evaluation)

### Included metadata and artifacts

This repository includes:

* split summary files
* train / validation / test split files
* attack distribution summaries
* preprocessing metadata
* label mappings
* feature lists
* processed tensors for selected experiment variants

Examples of included processed-data artifacts are:

* `feature_encoder.joblib`
* `label_encoder.joblib`
* `feature_list.csv`
* `label_mapping.csv`
* `preprocess_metadata.json`
* `X_train.pt`, `X_val.pt`, `X_test.pt`
* `y_train.pt`, `y_val.pt`, `y_test.pt`

## Experimental Outputs Included

The `experiments/` directory contains saved artifacts for the main experiments, including:

* `best.ckpt`
* `last.ckpt`
* `best_summary.json`
* `results_summary.json`
* `classification_report.csv`
* `confusion_matrix.csv`
* `per_class_metrics.csv`
* `train_history.csv`
* `history.csv`
* Optuna trial summaries
* 5-fold cross-validation summaries
* UNSW-NB15 evaluation outputs

Representative experiment folders include:

* attention + CatBoost + EQLv2 experiments
* no-attention ablation variants
* label-encoding ablation variants
* Optuna-based EQLv2 experiments
* k-fold evaluation outputs
* UNSW-NB15 cross-dataset outputs

## Reproducibility Notes

The official reproducibility workflow for the final accepted manuscript is script-based. The notebooks included in `03_notebook/` are retained only as archived exploratory material and are not the primary execution path for reproducing the final paper results.

This repository is intended to document the workflow and artifacts associated with the final accepted version of the paper and to make the reported experiments easier to inspect, understand, and reproduce.

## Contact

For questions regarding this repository or reproducibility details, please contact the authors through their institutional affiliation.
232H21006@alumno.ujat.mx; 252H23007@alumno.ujat.mx




