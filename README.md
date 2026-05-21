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

This repository contains the code and supporting materials required to reproduce the main experiments reported in the final accepted version of the paper.

The repository is intended to support the reproducibility of the following components:

- Main multiclass evaluation on **LITNET-2020**
- **Stratified 5-fold cross-validation**
- Comparison with **classical machine learning baselines**
- **Ablation study**
- **EQLv2 hyperparameter sensitivity analysis**
- **Cross-dataset evaluation on UNSW-NB15**

## Repository Structure

### `src/`
Core implementation of the methodology reported in the paper. This folder contains the main modules for preprocessing, model definition, loss implementation, training, evaluation, and auxiliary utilities.

### `scripts/`
Executable scripts used as entry points for the reproducibility workflow. These scripts are intended to run the main experiments and generate the reported outputs.

### `configs/`
Configuration files defining the parameters for preprocessing, training, evaluation, ablation experiments, and cross-dataset validation.

### `data/`
Documentation and metadata related to dataset organization. This folder includes instructions for dataset placement, split metadata in JSON format, and attack distribution summaries for train, validation, and test partitions.

### `results/`
Generated outputs, logs, tables, figures, and example artifacts associated with the experiments reported in the paper.

### `notebooks/archive/`
Archived exploratory notebooks kept only for reference. These notebooks are not the official reproducibility workflow for the final accepted manuscript.

## Main Reproducibility Workflow

The official reproducibility workflow in this repository is **script-based** and aligned with the final accepted manuscript.

A typical execution order is:

1. Obtain the required datasets
2. Place the datasets in the expected local directories
3. Run preprocessing
4. Train the main H.A.L.C.CO.N model on LITNET-2020
5. Evaluate the trained model on the LITNET-2020 test set
6. Run stratified 5-fold cross-validation
7. Run the ablation experiments
8. Run the EQLv2 sensitivity experiments
9. Run the cross-dataset evaluation on UNSW-NB15

## Environment

- Python 3.10
- PyTorch 2.3.0+cu118

Additional dependencies are listed in `requirements.txt`.

## Data

The full datasets are **not included** in this repository.

Users must obtain the datasets separately and place them in the expected local directories before running the pipeline.

### Required datasets
- **LITNET-2020**
- **UNSW-NB15** (required only for the cross-dataset evaluation)

### Included metadata
This repository includes:

- predefined dataset split files in **JSON format**
- attack distribution summaries for **train**, **validation**, and **test** partitions

These files are intended to facilitate reproducibility of the evaluation protocol.

## Reproducibility Notes

This repository is intended to document the experimental workflow associated with the final accepted version of the paper. The main objective is to make the reported results easier to inspect, understand, and reproduce.

The archived notebooks are provided only as supplementary historical material. The official workflow for reproducibility is the script-based implementation included in this repository.

## Contact

For questions regarding the repository or reproducibility details, please contact the authors through their institutional affiliation.
